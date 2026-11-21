//===----------------------------------------------------------------------===//
// quantforge-overwrite-output — C = A @ W instead of C += A @ W
//
// The generated GEMM starts its accumulator fragments from the current content of C
// (that is how bias / residual are fused). When the output is to be overwritten, those
// initial loads are pointless: they cost a full read of C, force the caller to zero the
// buffer first (one more kernel launch), and forbid kernels that write C at different
// coordinates than they read it (the gate/up epilogue stores the pair (2j, 2j+1) at j).
//
// For every K loop in a gpu.launch the initial values of the loop-carried accumulators are
// traced back through vector.insert chains; the global float vector.loads inserted there are
// replaced by zeros (loads that are loop-carried values themselves are left alone).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-overwrite-output"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

struct OverwriteOutputPass
    : public PassWrapper<OverwriteOutputPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OverwriteOutputPass)
  StringRef getArgument() const override { return "quantforge-overwrite-output"; }
  StringRef getDescription() const override {
    return "Start the accumulators from zero instead of the content of C (C = A @ W).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    SmallVector<vector::LoadOp> loads;
    getOperation().walk([&](scf::ForOp loop) {
      if (!loop->getParentOfType<gpu::LaunchOp>())
        return;
      bool isK = false;
      for (Operation &op : loop.getBody()->without_terminator())
        if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncWaitOp>(&op))
          isK = true;
      if (!isK)
        return;
      // The accumulators are assembled as insert(load rank-1 fragment, ...) chains. Loop-carried
      // values that are themselves a plain vector.load (the look-ahead registers of
      // quantforge-prefetch-weight-loads: packed bytes, scales, zero points) must not be touched,
      // so only loads that are the *source* of a vector.insert are zeroed.
      SmallVector<Value> work(loop.getInitArgs().begin(), loop.getInitArgs().end());
      while (!work.empty()) {
        Value v = work.pop_back_val();
        auto ins = v.getDefiningOp<vector::InsertOp>();
        if (!ins)
          continue;
        work.push_back(ins.getDest());
        if (auto ld = ins.getSource().getDefiningOp<vector::LoadOp>())
          if (!ld.getMemRefType().getMemorySpace() &&
              (isa<FloatType>(ld.getVectorType().getElementType()) ||
               ld.getVectorType().getElementType().isInteger(32))) // f32 / i32 accumulators
            loads.push_back(ld);
      }
    });
    for (vector::LoadOp ld : loads) {
      if (!ld->getParentOp()) // already handled
        continue;
      OpBuilder b(ld);
      Value zero = b.create<arith::ConstantOp>(
          ld.getLoc(), DenseElementsAttr::get(ld.getVectorType(),
                                              b.getZeroAttr(ld.getVectorType().getElementType())));
      ld.getResult().replaceAllUsesWith(zero);
      ld->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createOverwriteOutputPass() {
  return std::make_unique<OverwriteOutputPass>();
}
void mlir::quantforge::registerOverwriteOutputPass() {
  PassRegistration<OverwriteOutputPass>();
}
