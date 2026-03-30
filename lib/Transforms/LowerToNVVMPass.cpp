//===----------------------------------------------------------------------===//
// LowerToNVVMPass.cpp — Task 3.2: Lower GPU/Arith/Vector to NVVM/LLVM
//
// Targets: gpu::GPUModuleOp
// Converts: gpu.func -> llvm.func (ptx_kernel)
//           arith.* -> llvm.*
//           gpu.barrier -> nvvm.barrier0
//           gpu.thread_id -> nvvm.read.ptx.sreg.tid
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-lower-to-nvvm"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

struct LowerToNVVMPass
    : public PassWrapper<LowerToNVVMPass, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToNVVMPass)

  StringRef getArgument() const override { return "quantforge-lower-to-nvvm"; }
  StringRef getDescription() const override {
    return "Lower GPU, Arith, Vector, and MemRef ops to LLVM and NVVM dialects.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = &getContext();

    // ── 1. Configure Type Converter for GPU ──────────────────────
    LowerToLLVMOptions options(ctx);
    // Use bare pointer calling convention to reduce overhead for memref params
    options.useBarePtrCallConv = true;
    LLVMTypeConverter converter(ctx, options);

    // ── 2. Set up Conversion Target ─────────────────────────────
    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();

    // Exception: Allow gpu.module to survive as container for llvm.func
    target.addLegalOp<gpu::GPUModuleOp, gpu::ModuleEndOp>();

    // ── 3. Populate Lowering Patterns ───────────────────────────
    RewritePatternSet patterns(ctx);

    // SCF -> Control Flow (if any remain after Tiling/Mapping)
    populateSCFToControlFlowConversionPatterns(patterns);

    // GPU -> NVVM (SREG intrinsics, barrier, kernel attribute)
    populateGpuToNVVMConversionPatterns(converter, patterns);

    // Basic types & Control Flow -> LLVM
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);

    // Memory & Vector -> LLVM
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

    // ── 4. Execute Full Conversion ──────────────────────────────
    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createLowerToNVVMPass() {
  return std::make_unique<LowerToNVVMPass>();
}

void mlir::quantforge::registerLowerToNVVMPass() {
  PassRegistration<LowerToNVVMPass>();
}
