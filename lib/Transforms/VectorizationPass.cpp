//===----------------------------------------------------------------------===//
// VectorizationPass — Task 2.3.1: Vectorize tiled linalg operations
//
// Targets:
//   1) linalg.matmul  -> vector.contract
//   2) linalg.generic -> vectorized arithmetic (SIMD-friendly)
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-vectorization"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {
struct VectorizationPass
    : public PassWrapper<VectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizationPass)

  StringRef getArgument() const override { return "quantforge-vectorization"; }

  StringRef getDescription() const override {
    return "Vectorize innermost linalg ops (matmul, generic) into "
           "vector.contract and SIMD arithmetic for GPU mapping.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    IRRewriter rewriter(ctx);
    // Vectorize from innermost ops first to maximize success rate.
    funcOp.walk<WalkOrder::PostOrder>([&](linalg::LinalgOp linalgOp) {
      // Dynamic shapes often need prior tiling/padding; skip to avoid noisy
      // failures.
      if (linalgOp.hasDynamicShape()) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[QuantForge-Vectorization] Skipping dynamic-shaped op: ";
          linalgOp->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        return;
      }

      rewriter.setInsertionPoint(linalgOp);
      if (failed(linalg::vectorize(rewriter, linalgOp.getOperation()))) {
        linalgOp->emitWarning(
            "Vectorization failed on static-shaped LinalgOp. Kernel will "
            "fallback to scalar loops, leading to severe GPU performance "
            "degradation.");
        LLVM_DEBUG({
          llvm::dbgs() << "[QuantForge-Vectorization] Failed to vectorize op: ";
          linalgOp->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
      } else {
        LLVM_DEBUG(
            llvm::dbgs()
            << "[QuantForge-Vectorization] Successfully vectorized op.\n");
      }
    });

    RewritePatternSet patterns(ctx);

    // Clean up transfer/reduction forms that often appear post-vectorization.
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::quantforge::createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

void mlir::quantforge::registerVectorizationPass() {
  PassRegistration<VectorizationPass>();
}
