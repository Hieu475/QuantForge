//===----------------------------------------------------------------------===//
// QuantForgeBufferizePass — Task 4.1: One-Shot Bufferize (Tensor → MemRef)
//
// Converts the entire module from value semantics (tensor) to pointer
// semantics (memref) using MLIR's One-Shot Bufferize infrastructure.
//
// This pass wraps bufferization::runOneShotModuleBufferize with
// QuantForge-specific configuration:
//   • bufferizeFunctionBoundaries = true  (cross-function analysis)
//   • allowReturnAllocs = true            (functions may return memref.alloc)
//   • copyBeforeWrite = true              (safe: insert copy when in-place
//                                          analysis cannot prove safety)
//   • IdentityLayoutMap at function boundaries
//
// After this pass, all tensor.empty → memref.alloc, all
// tensor.extract_slice → memref.subview, and linalg ops operate on memref.
// All data remains in Global Memory (HBM) at this stage.
//
// Pipeline position:
//   quantforge-tiling → quantforge-gpu-mapping → **quantforge-bufferize**
//     → quantforge-smem-promotion → quantforge-swizzle-load
//     → quantforge-lower-to-nvvm
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-bufferize"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

//===----------------------------------------------------------------------===//
// QuantForgeBufferizePass
//===----------------------------------------------------------------------===//

struct QuantForgeBufferizePass
    : public PassWrapper<QuantForgeBufferizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantForgeBufferizePass)

  //------------------------------------------------------------------
  // Pass metadata
  //------------------------------------------------------------------
  StringRef getArgument() const override { return "quantforge-bufferize"; }

  StringRef getDescription() const override {
    return "One-Shot Bufferize: convert all tensor ops to memref ops with "
           "in-place analysis. Converts tensor.empty to memref.alloc, "
           "tensor.extract_slice to memref.subview, etc.";
  }

  //------------------------------------------------------------------
  // Dependent dialects
  //------------------------------------------------------------------
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,
                    memref::MemRefDialect, func::FuncDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, gpu::GPUDialect>();

    // Register bufferization interfaces for all upstream dialects.
    // Without these, ops like tensor.extract_slice, linalg.matmul, etc.
    // would not know how to convert from tensor to memref semantics.
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // ── 1. Configure One-Shot Bufferization ─────────────────────────
    bufferization::OneShotBufferizationOptions options;

    // Enable cross-function bufferization: function arguments and results
    // are converted from tensor to memref types.
    options.bufferizeFunctionBoundaries = true;

    // Allow returning allocs from loops (needed for K-reduction loops).
    options.allowReturnAllocsFromLoops = true;

    // Safe default: insert copies when in-place analysis cannot prove
    // that a write is safe (no aliasing conflicts). Downstream
    // canonicalization passes can eliminate redundant copies.
    options.copyBeforeWrite = true;

    // Use identity layout maps at function boundaries to keep memref
    // types simple and predictable for downstream GPU lowering.
    options.setFunctionBoundaryTypeConversion(
        bufferization::LayoutMapOption::IdentityLayoutMap);

    LLVM_DEBUG(llvm::dbgs()
               << "[QuantForgeBufferize] running one-shot bufferize\n"
               << "  bufferizeFunctionBoundaries = true\n"
               << "  allowReturnAllocs = true\n"
               << "  copyBeforeWrite = true\n"
               << "  LayoutMap = IdentityLayoutMap\n");

    // ── 2. Run One-Shot Module Bufferize ────────────────────────────
    if (failed(bufferization::runOneShotModuleBufferize(moduleOp, options))) {
      moduleOp.emitError("QuantForge one-shot bufferization failed");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[QuantForgeBufferize] bufferization complete\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createQuantForgeBufferizePass() {
  return std::make_unique<QuantForgeBufferizePass>();
}

void mlir::quantforge::registerQuantForgeBufferizePass() {
  PassRegistration<QuantForgeBufferizePass>();
}
