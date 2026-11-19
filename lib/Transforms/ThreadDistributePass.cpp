//===----------------------------------------------------------------------===//
// ThreadDistributePass — spread the warp/lane tile loops over threadIdx
//
// Runs on bufferized IR inside gpu.launch, after quantforge-forall-to-gpu.
// quantforge-gpu-mapping / quantforge-tiling annotate sequential scf.for
// loops with *how* they should be parallelised:
//
//   mapping = [#gpu.thread<z|y|x>]   warp-level loops. The loop's iterations
//                                    are spread over that threadIdx axis.
//   quantforge.lane_m + lane_n       the (m16 × n8) MMA-tile loops of a warp
//                                    tile (nested pair). They are flattened
//                                    over the lanes of a warp:
//                                      lane = threadIdx.x
//                                      m = lane / trip_n,  n = lane % trip_n
//                                    (requires trip_m * trip_n == blockDim.x)
//
// Each mapped scf.for is *removed*: its body is inlined with the induction
// variable replaced by  lb + id * step.  If the trip count differs from the
// axis size the pass falls back to a guard (trip < dim) or a cyclic loop
// (trip > dim).  Loops that cannot be mapped are left sequential, which is
// always correct.
//
// Requirements: static lb/ub/step, no loop-carried values (bufferization +
// canonicalization remove the memref iter_args), no gpu.barrier in the body
// (barriers inside divergent code would deadlock).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-thread-distribute"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

struct StaticLoop {
  int64_t lb, ub, step;
  int64_t trip() const { return (ub - lb + step - 1) / step; }
};

static std::optional<StaticLoop> getStaticLoop(scf::ForOp op) {
  auto lb = getConstantIntValue(op.getLowerBound());
  auto ub = getConstantIntValue(op.getUpperBound());
  auto step = getConstantIntValue(op.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub < *lb)
    return std::nullopt;
  return StaticLoop{*lb, *ub, *step};
}

/// Loops we can legally remove: no results / iter_args, no barrier inside.
static bool isDistributable(scf::ForOp op) {
  if (op.getNumResults() != 0)
    return false;
  bool hasBarrier = false;
  op.walk([&](gpu::BarrierOp) { hasBarrier = true; });
  return !hasBarrier;
}

/// Returns 0/1/2 for DimX/DimY/DimZ if the loop carries a single
/// #gpu.thread mapping, else -1.
static int getThreadAxis(scf::ForOp op) {
  auto mapping = op->getAttrOfType<ArrayAttr>("mapping");
  if (!mapping || mapping.size() != 1)
    return -1;
  auto attr = dyn_cast<gpu::GPUThreadMappingAttr>(mapping[0]);
  if (!attr)
    return -1;
  switch (attr.getThread()) {
  case gpu::MappingId::DimX:
    return 0;
  case gpu::MappingId::DimY:
    return 1;
  case gpu::MappingId::DimZ:
    return 2;
  default:
    return -1;
  }
}

/// Replace `loop` by its body with the induction variable bound to `iv`.
static void inlineLoopBody(scf::ForOp loop, Value iv) {
  Block *body = loop.getBody();
  loop.getBody()->getTerminator()->erase();
  IRRewriter rewriter(loop.getContext());
  rewriter.inlineBlockBefore(body, loop, ValueRange{iv});
  rewriter.eraseOp(loop);
}

struct ThreadDistributePass
    : public PassWrapper<ThreadDistributePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadDistributePass)

  StringRef getArgument() const override {
    return "quantforge-thread-distribute";
  }
  StringRef getDescription() const override {
    return "Distribute warp/lane tile loops over threadIdx inside gpu.launch "
           "(replaces mapped scf.for loops by thread-id arithmetic).";
  }

  ThreadDistributePass() = default;
  ThreadDistributePass(const ThreadDistributePass &other)
      : PassWrapper<ThreadDistributePass, OperationPass<func::FuncOp>>(other) {
    distributeLanes.setValue(other.distributeLanes.getValue());
  }

  Option<bool> distributeLanes{
      *this, "distribute-lanes",
      llvm::cl::desc("Spread the m16 x n8 MMA-tile loops over the lanes of a "
                     "warp. Disable for warp-cooperative MMA (Tensor Core) "
                     "where all lanes execute the same tile loops together."),
      llvm::cl::init(true)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  /// iv = lb + id * step
  static Value makeIv(OpBuilder &b, Location loc, const StaticLoop &l,
                      Value id) {
    Value step = b.create<arith::ConstantIndexOp>(loc, l.step);
    Value lb = b.create<arith::ConstantIndexOp>(loc, l.lb);
    Value scaled = b.create<arith::MulIOp>(loc, id, step);
    return b.create<arith::AddIOp>(loc, lb, scaled);
  }

  void distribute(gpu::LaunchOp launch) {
    std::array<std::optional<int64_t>, 3> blockDim = {
        getConstantIntValue(launch.getBlockSizeX()),
        getConstantIntValue(launch.getBlockSizeY()),
        getConstantIntValue(launch.getBlockSizeZ())};
    gpu::KernelDim3 tids = launch.getThreadIds();
    std::array<Value, 3> tid = {tids.x, tids.y, tids.z};

    // ── Collect candidates (post-order: inner loops first) ───────────
    SmallVector<scf::ForOp> warpLoops;
    SmallVector<scf::ForOp> laneM;
    launch.walk<WalkOrder::PostOrder>([&](scf::ForOp op) {
      if (getThreadAxis(op) >= 0)
        warpLoops.push_back(op);
      if (op->hasAttr("quantforge.lane_m"))
        laneM.push_back(op);
    });

    OpBuilder b(launch.getContext());

    // ── Lane loops: nested (lane_m ⊃ lane_n) pair → threadIdx.x ──────
    for (scf::ForOp mLoop : laneM) {
      if (!distributeLanes)
        break;
      scf::ForOp nLoop;
      for (Operation &op : mLoop.getBody()->without_terminator())
        if (auto f = dyn_cast<scf::ForOp>(op))
          if (f->hasAttr("quantforge.lane_n"))
            nLoop = f;
      if (!nLoop)
        continue;
      auto m = getStaticLoop(mLoop);
      auto n = getStaticLoop(nLoop);
      if (!m || !n || !blockDim[0] || !isDistributable(mLoop) ||
          !isDistributable(nLoop))
        continue;
      if (m->trip() * n->trip() != *blockDim[0]) {
        mLoop.emitRemark("lane loops stay sequential: trip_m * trip_n != "
                         "blockDim.x");
        continue;
      }
      b.setInsertionPoint(mLoop);
      Location loc = mLoop.getLoc();
      Value tripN = b.create<arith::ConstantIndexOp>(loc, n->trip());
      Value mId = b.create<arith::DivUIOp>(loc, tid[0], tripN);
      Value nId = b.create<arith::RemUIOp>(loc, tid[0], tripN);
      Value mIv = makeIv(b, loc, *m, mId);
      Value nIv = makeIv(b, loc, *n, nId);
      inlineLoopBody(nLoop, nIv);
      inlineLoopBody(mLoop, mIv);
    }

    // ── Warp loops: one axis each ────────────────────────────────────
    for (scf::ForOp loop : warpLoops) {
      int axis = getThreadAxis(loop);
      auto sl = getStaticLoop(loop);
      if (!sl || !blockDim[axis] || !isDistributable(loop))
        continue;
      int64_t dim = *blockDim[axis];
      int64_t trip = sl->trip();
      b.setInsertionPoint(loop);
      Location loc = loop.getLoc();

      if (trip == dim) {
        inlineLoopBody(loop, makeIv(b, loc, *sl, tid[axis]));
      } else if (trip < dim) {
        // Guard: threads beyond the trip count idle.
        Value tripV = b.create<arith::ConstantIndexOp>(loc, trip);
        Value inRange = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, tid[axis], tripV);
        // The induction value must be materialised BEFORE the scf.if so that
        // it dominates the inlined body.
        Value iv = makeIv(b, loc, *sl, tid[axis]);
        auto ifOp = b.create<scf::IfOp>(loc, inRange, /*withElse=*/false);
        // Move the body into the `then` block.
        loop.getBody()->getTerminator()->erase();
        IRRewriter rw(loop.getContext());
        rw.inlineBlockBefore(loop.getBody(), ifOp.thenBlock()->getTerminator(),
                             ValueRange{iv});
        rw.eraseOp(loop);
      } else {
        // Cyclic: iv = lb + id*step, stride dim*step.
        Value lb = makeIv(b, loc, *sl, tid[axis]);
        Value newStep = b.create<arith::ConstantIndexOp>(loc, sl->step * dim);
        loop.getLowerBoundMutable().assign(lb);
        loop.getStepMutable().assign(newStep);
        loop->removeAttr("mapping");
      }
    }
  }

  void runOnOperation() override {
    SmallVector<gpu::LaunchOp> launches;
    getOperation().walk([&](gpu::LaunchOp op) { launches.push_back(op); });
    for (gpu::LaunchOp launch : launches)
      distribute(launch);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createThreadDistributePass() {
  return std::make_unique<ThreadDistributePass>();
}
void mlir::quantforge::registerThreadDistributePass() {
  PassRegistration<ThreadDistributePass>();
}
