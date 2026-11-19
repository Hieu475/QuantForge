//===----------------------------------------------------------------------===//
// quantforge-sink-dequant-stores — overlap weight-load latency with the MMAs
//
// The double-buffered K loop is laid out as
//
//   for k:  cp.async A(k+1)                      // async, fine
//           load packed/scale/zp B(k+1)          // global loads -> registers
//           dequantise B(k+1); store to smem     // needs the loads: STALLS
//           for ...: ldmatrix + mma on tile k    // the Tensor Core work
//           commit / wait / barrier
//
// so the dequantisation waits for the global loads immediately (ncu: ~24 % of
// all warp stalls sit on the first consumer of the weight loads, and the Tensor
// Cores idle ~37 % of the time on wide problems), while the MMAs that could
// hide that latency run afterwards. This pass moves everything that depends on
// the global loads (the dequantisation arithmetic and the shared-memory
// stores) to just before the commit / barrier at the end of the iteration:
//
//   for k:  cp.async A(k+1);  load B(k+1)        // issued first
//           for ...: ldmatrix + mma on tile k    // hides the load latency
//           dequantise B(k+1); store to smem
//           commit / wait / barrier
//
// Legal because the stores target the *next* stage's buffer, which the MMAs of
// tile k do not read; the barrier at the end still orders them before tile k+1.
// Costs a few registers (the loaded vectors live across the MMA loop).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-sink-dequant-stores"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

static bool isGlobalLoad(Operation *op) {
  auto l = dyn_cast<vector::LoadOp>(op);
  return l && !l.getMemRefType().getMemorySpace();
}

static bool isSharedStore(Operation *op) {
  auto s = dyn_cast<vector::StoreOp>(op);
  return s && s.getMemRefType().getMemorySpace();
}

struct SinkDequantStoresPass
    : public PassWrapper<SinkDequantStoresPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkDequantStoresPass)
  StringRef getArgument() const override {
    return "quantforge-sink-dequant-stores";
  }
  StringRef getDescription() const override {
    return "Move the dequantisation + shared-memory stores of the next B tile "
           "behind the MMA loop of the current tile (loads stay first).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, nvgpu::NVGPUDialect>();
  }

  void processLoop(scf::ForOp forOp) {
    Block *body = forOp.getBody();

    // The compute loop: the first nested scf.for after a global load.
    Operation *firstLoad = nullptr, *computeLoop = nullptr;
    for (Operation &op : body->without_terminator()) {
      if (!firstLoad && isGlobalLoad(&op))
        firstLoad = &op;
      else if (firstLoad && isa<scf::ForOp>(&op)) {
        computeLoop = &op;
        break;
      }
    }
    if (!firstLoad || !computeLoop)
      return;
    // Only the double-buffered schedule: there the stores target the *next*
    // stage and nothing synchronises between the copy and the MMA loop. With a
    // single buffer a barrier separates them (the stores feed the tile the MMAs
    // are about to read), and sinking would be a race.
    for (Operation *op = firstLoad; op && op != computeLoop; op = op->getNextNode())
      if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncWaitOp>(op))
        return;

    // End-of-iteration sync point after the compute loop.
    Operation *sync = nullptr;
    for (Operation *op = computeLoop->getNextNode(); op; op = op->getNextNode())
      if (isa<nvgpu::DeviceAsyncCreateGroupOp, gpu::BarrierOp>(op)) {
        sync = op;
        break;
      }
    if (!sync)
      return;

    // Forward closure of the global loads' results (ops before the compute loop).
    llvm::SetVector<Operation *> moved;
    SmallVector<Operation *> work;
    for (Operation &op : body->without_terminator()) {
      if (&op == computeLoop)
        break;
      if (isGlobalLoad(&op))
        work.push_back(&op);
    }
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      for (Value r : op->getResults())
        for (Operation *user : r.getUsers()) {
          if (user->getBlock() != body || !user->isBeforeInBlock(computeLoop) ||
              moved.contains(user))
            continue;
          if (!isSharedStore(user) && (!isMemoryEffectFree(user) ||
                                       user->getNumRegions() != 0))
            return; // something we cannot move: leave the loop alone
          moved.insert(user);
          work.push_back(user);
        }
    }
    moved.remove_if([](Operation *op) { return isGlobalLoad(op); });
    if (moved.empty() || !llvm::any_of(moved, isSharedStore))
      return;

    // The moved ops must only feed each other (or things after the sync).
    for (Operation *op : moved)
      for (Value r : op->getResults())
        for (Operation *user : r.getUsers())
          if (!moved.contains(user) && user->getBlock() == body &&
              user->isBeforeInBlock(sync))
            return;

    SmallVector<Operation *> ordered(moved.begin(), moved.end());
    llvm::sort(ordered, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    // Operands of moved ops that are defined by NOT-moved ops after... all
    // operands are defined before the originals, hence before `sync`.
    for (Operation *op : ordered)
      op->moveBefore(sync);
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp op) {
      if (op->getParentOfType<gpu::LaunchOp>())
        loops.push_back(op);
    });
    for (scf::ForOp op : loops)
      processLoop(op);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createSinkDequantStoresPass() {
  return std::make_unique<SinkDequantStoresPass>();
}
void mlir::quantforge::registerSinkDequantStoresPass() {
  PassRegistration<SinkDequantStoresPass>();
}
