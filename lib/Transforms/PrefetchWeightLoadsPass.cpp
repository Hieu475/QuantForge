//===----------------------------------------------------------------------===//
// quantforge-prefetch-weight-loads{distance=P} — multi-tile load pipeline
//
// A decode GEMM streams the INT4 weights exactly once, so its speed is the
// number of bytes in flight: the K loop issues the global loads of ONE tile
// (packed bytes, scale, zero point), waits for them, dequantises, stores to
// shared memory, runs a handful of MMAs and starts over. Per block there is
// never more than one tile of loads outstanding, which at ~1 us of DRAM latency
// caps the bandwidth far below the peak.
//
// This pass moves the loads P tiles ahead and carries the loaded registers
// through the loop (a rotating register file in scf.for iter_args):
//
//   prologue:  r_0..r_{P-1} = loads of the first P tiles
//   for k:     n = loads of tile (k + P)            // issued first
//              dequantise + store r_0               // loaded P iterations ago
//              MMAs ...
//              yield r_1..r_{P-1}, n
//
// so P+1 tiles of loads are in flight per block while one is consumed. The
// index arithmetic of the loads (a slice of the loop body that only depends on
// the induction variable) is cloned with  iv -> min(iv + P*step, ub - step),
// i.e. reads past the end re-read the last tile (harmless, never consumed).
// Works on both schedules (single buffer: loads feed the current tile;
// double buffer + sink-dequant-stores: loads feed the next tile) because only
// the *operands* of the dequantisation change.
//
// Cost: ~14 registers per thread per tile of distance (scale / zero point are
// de-duplicated by quantforge-vectorize-global-loads).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-prefetch-weight-loads"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

struct Plan {
  SmallVector<vector::LoadOp> loads;
  SmallVector<Operation *> slice; // program order, excludes the loads
};

static bool isGlobalLoad(Operation *op) {
  auto l = dyn_cast<vector::LoadOp>(op);
  return l && !l.getMemRefType().getMemorySpace();
}

/// Collect the global loads directly in the loop body and the ops of the body
/// their operands depend on. Fails when a slice op has side effects / regions
/// or depends on a loop-carried value.
static bool analyze(scf::ForOp forOp, Block *body, Value iv, Plan &plan) {
  for (Operation &op : body->without_terminator())
    if (isGlobalLoad(&op))
      plan.loads.push_back(cast<vector::LoadOp>(&op));
  if (plan.loads.empty())
    return false;
  llvm::SetVector<Operation *> slice;
  SmallVector<Value> work;
  for (auto l : plan.loads)
    work.append(l->operand_begin(), l->operand_end());
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() == body && arg != iv)
        return false; // iter_arg
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (def->getBlock() != body || slice.contains(def))
      continue;
    if (isa<vector::LoadOp>(def) || !isMemoryEffectFree(def) ||
        def->getNumRegions() != 0)
      return false;
    slice.insert(def);
    work.append(def->operand_begin(), def->operand_end());
  }
  plan.slice.assign(slice.begin(), slice.end());
  llvm::sort(plan.slice, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  (void)forOp;
  return true;
}

/// Clone the slice and the loads with iv -> ivVal; returns the new loads.
static SmallVector<Value> cloneLoads(OpBuilder &b, const Plan &plan, Value iv,
                                     Value ivVal) {
  IRMapping map;
  map.map(iv, ivVal);
  for (Operation *op : plan.slice)
    b.clone(*op, map);
  SmallVector<Value> out;
  for (auto l : plan.loads)
    out.push_back(b.clone(*l.getOperation(), map)->getResult(0));
  return out;
}

/// min(base + k * step, ub - step)  (index arithmetic, unsigned).
static Value clampedIv(OpBuilder &b, Location loc, Value base, int64_t k,
                       Value step, Value ub) {
  Value off = b.create<arith::MulIOp>(loc, step,
                                      b.create<arith::ConstantIndexOp>(loc, k));
  Value sum = b.create<arith::AddIOp>(loc, base, off);
  Value last = b.create<arith::SubIOp>(loc, ub, step);
  return b.create<arith::MinUIOp>(loc, sum, last);
}

struct PrefetchWeightLoadsPass
    : public PassWrapper<PrefetchWeightLoadsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchWeightLoadsPass)
  StringRef getArgument() const override {
    return "quantforge-prefetch-weight-loads";
  }
  StringRef getDescription() const override {
    return "Issue the weight (packed / scale / zero-point) global loads of the "
           "K loop `distance` tiles ahead, carried through iter_args.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }
  Option<int64_t> distance{
      *this, "distance",
      llvm::cl::desc("Tiles of extra look-ahead for the weight loads (0 = off)"),
      llvm::cl::init(1)};
  PrefetchWeightLoadsPass() = default;
  PrefetchWeightLoadsPass(const PrefetchWeightLoadsPass &other)
      : PassWrapper<PrefetchWeightLoadsPass, OperationPass<func::FuncOp>>(
            other) {
    distance.setValue(other.distance.getValue());
  }

  void processLoop(scf::ForOp forOp) {
    int64_t P = distance.getValue();
    if (P <= 0)
      return;
    Block *body = forOp.getBody();
    // The K loop: contains a barrier / async wait (the tile synchronisation).
    bool isKLoop = false;
    for (Operation &op : body->without_terminator())
      if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncWaitOp>(&op))
        isKLoop = true;
    if (!isKLoop)
      return;
    Plan plan;
    if (!analyze(forOp, body, forOp.getInductionVar(), plan))
      return;
    size_t L = plan.loads.size();

    IRRewriter rewriter(forOp.getContext());
    rewriter.setInsertionPoint(forOp);
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound(),
          step = forOp.getStep();

    // prologue: tiles 0..P-1 (relative to the loop's first iteration)
    SmallVector<Value> inits;
    for (int64_t i = 0; i < P; ++i) {
      Value ivi = clampedIv(rewriter, loc, lb, i, step, ub);
      SmallVector<Value> v =
          cloneLoads(rewriter, plan, forOp.getInductionVar(), ivi);
      inits.append(v.begin(), v.end());
    }

    // The callback runs while the new body is still empty (the old body is
    // merged in afterwards): yield the rotated carried values now and patch the
    // last slot with the fresh loads below.
    auto yieldFn = [&](OpBuilder &, Location,
                       ArrayRef<BlockArgument> args) -> SmallVector<Value> {
      SmallVector<Value> yields;
      for (int64_t i = 1; i < P; ++i)
        for (size_t l = 0; l < L; ++l)
          yields.push_back(args[i * L + l]);
      for (size_t l = 0; l < L; ++l)
        yields.push_back(args[(P - 1) * L + l]); // placeholder
      return yields;
    };
    auto res = cast<LoopLikeOpInterface>(forOp.getOperation())
                   .replaceWithAdditionalYields(rewriter, inits,
                                                /*replaceInitOperandUsesInLoop=*/false,
                                                yieldFn);
    if (succeeded(res)) {
      auto newFor = cast<scf::ForOp>(res->getOperation());
      Block *nb = newFor.getBody();
      Value iv = newFor.getInductionVar();
      Plan p2;
      if (!analyze(newFor, nb, iv, p2) || p2.loads.size() != L) {
        newFor.emitError("quantforge-prefetch-weight-loads: lost the loads");
        return signalPassFailure();
      }
      unsigned firstCarried = nb->getNumArguments() - P * L;
      OpBuilder tb(nb, nb->begin());
      Value ivShift = clampedIv(tb, loc, iv, P, step, ub);
      SmallVector<Value> fresh = cloneLoads(tb, p2, iv, ivShift);
      for (size_t l = 0; l < L; ++l) {
        p2.loads[l].getResult().replaceAllUsesWith(nb->getArgument(firstCarried + l));
        p2.loads[l]->erase();
      }
      auto yield = cast<scf::YieldOp>(nb->getTerminator());
      unsigned lastSlot = yield.getNumOperands() - L;
      for (size_t l = 0; l < L; ++l)
        yield->setOperand(lastSlot + l, fresh[l]);
    }
    if (failed(res))
      forOp.emitWarning("quantforge-prefetch-weight-loads: could not rewrite loop");
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp op) {
      if (op->getParentOfType<gpu::LaunchOp>())
        loops.push_back(op);
    });
    // innermost-last order does not matter: only the loop that owns the
    // synchronisation and the loads is rewritten.
    for (scf::ForOp op : loops)
      processLoop(op);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createPrefetchWeightLoadsPass() {
  return std::make_unique<PrefetchWeightLoadsPass>();
}
void mlir::quantforge::registerPrefetchWeightLoadsPass() {
  PassRegistration<PrefetchWeightLoadsPass>();
}
