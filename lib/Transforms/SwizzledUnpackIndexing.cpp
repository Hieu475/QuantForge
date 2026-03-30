//===----------------------------------------------------------------------===//
// SwizzledUnpackIndexing Pass — Shared Memory Bank Conflict Prevention
//
// Rewrites the column index calculation in unpack SCF loop bodies to apply
// XOR swizzling, preventing shared memory bank conflicts when weights are
// loaded from HBM into shared memory before unpacking.
//
// Problem:
//   Shared memory has 32 banks × 4 bytes. With dense INT4 packing, the
//   probability of bank conflicts when reading aligned i32 chunks with
//   ldmatrix or ld.shared is 100% for consecutive rows of the same column.
//   This reduces SRAM throughput by up to 32×.
//
// Solution — XOR swizzle pattern (used in Cutlass):
//   original_col = ch * 4 + byte_offset       (linear)
//   swizzled_col = original_col ^ (k % 8)     (swizzled)
//
//   This ensures that for the 8-row period (rows k, k+1, ..., k+7),
//   each row's i32 chunks land in a different set of banks, eliminating
//   conflicts:
//     row k=0: col XOR 0 → no shift
//     row k=1: col XOR 1 → 4-byte shift
//     row k=2: col XOR 2 → 8-byte shift
//     ...
//     row k=7: col XOR 7 → 28-byte shift
//
// Implementation:
//   This pass runs AFTER LowerUnpackToNVVM (or LowerUnpackToPRMT) and
//   BEFORE bufferization. It finds the characteristic scf.for loop nest
//   produced by those passes and rewrites the tensor.extract column index.
//
//   The input_index and output_index swizzle must be consistent:
//   - Input (packed weights, shape [K, N/2]): swizzle read index
//   - Output (unpacked nibbles, shape [K, N]): output index is NOT swizzled
//     because nibbles must remain in linear order for downstream ops.
//
// Guard:
//   - N (packed last dimension) must be >= 32 to justify swizzle overhead.
//   - Loop must not already have "swizzled" attribute (idempotent).
//
// Note: At tensor dialect level, this transform is heuristic — swizzling
// only provides hardware benefit after bufferization maps tensors to shared
// memory memrefs. This pass is designed to be a "pre-pass" that emits the
// correct index patterns to match swizzled smem layout used by Cutlass/cuBLAS.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "swizzled-unpack-indexing"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper: check if an scf.for loop nest looks like an unpack loop
//===----------------------------------------------------------------------===//

/// Returns the direct inner loop in the outer unpack loop body.
static scf::ForOp getDirectInnerLoop(scf::ForOp outerFor) {
  for (Operation &op : outerFor.getBody()->without_terminator()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(op))
      return innerFor;
  }
  return {};
}

/// Returns true if the loop body contains a 2D tensor.extract that reads the
/// packed tensor payload.
static bool has2DTensorExtract(scf::ForOp loop) {
  bool found = false;
  loop.getBody()->walk([&](tensor::ExtractOp ex) {
    if (ex.getIndices().size() == 2)
      found = true;
  });
  return found;
}

/// Returns true if `forOp` is the outer (K-dimension) loop of an unpack SCF
/// nest: not swizzled, has a direct inner loop, and the inner loop reads from
/// a 2D packed tensor.
static bool isUnpackOuterLoop(scf::ForOp forOp) {
  if (forOp->hasAttr("swizzled"))
    return false;

  scf::ForOp innerFor = getDirectInnerLoop(forOp);
  if (!innerFor)
    return false;

  return has2DTensorExtract(innerFor);
}

//===----------------------------------------------------------------------===//
// SwizzleUnpackPattern
//===----------------------------------------------------------------------===//

namespace {

/// Rewrites tensor.extract column indices inside the inner SCF for-loop of
/// an unpack nest to use XOR swizzling:
///
///   Before: col_idx = base + offset        (linear)
///   After:  col_idx = (base + offset) XOR (k % 8)
///
/// Also marks the outer loop with "swizzled" attribute to prevent
/// repeated application.
struct SwizzleUnpackPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  SwizzleUnpackPattern(MLIRContext *ctx, int64_t minN)
      : OpRewritePattern<scf::ForOp>(ctx), minNThreshold(minN) {}

  LogicalResult matchAndRewrite(scf::ForOp outerFor,
                                PatternRewriter &rewriter) const override {
    // ── Guard: must look like an unpack outer loop ─────────────────
    if (!isUnpackOuterLoop(outerFor))
      return rewriter.notifyMatchFailure(
          outerFor, "not an unpack outer loop or already swizzled");

    // ── Guard: N must be large enough for swizzle to be worthwhile
    // We infer N from the inner loop upper bound (should be N/4).
    // Check if we can find the inner loop bound >= minNThreshold / 4.
    scf::ForOp innerFor = getDirectInnerLoop(outerFor);
    if (!innerFor)
      return rewriter.notifyMatchFailure(outerFor, "no inner for found");

    // Try to get static upper bound of inner loop
    if (auto ubOp =
            innerFor.getUpperBound().getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t n4 = ubOp.value();
      if (n4 * 4 < minNThreshold)
        return rewriter.notifyMatchFailure(outerFor,
                                           "N too small for swizzling");
    }

    Location loc = outerFor.getLoc();

    // ── Compute k % 8 at the OUTER loop body level ────────────────
    // k_mod8 must dominate all uses in the inner loop body.
    // Set insertion point to the start of the outer loop body.
    Value k = outerFor.getInductionVar();
    {
      OpBuilder::InsertionGuard outerGuard(rewriter);
      rewriter.setInsertionPointToStart(outerFor.getBody());

      Value c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);
      Value k_mod8 = rewriter.create<arith::RemUIOp>(loc, k, c8);

      // ── Rewrite each tensor.extract in the inner loop body ────────
      // Collect extracts first (walk, not mutate during walk)
      SmallVector<tensor::ExtractOp> extractOps;
      innerFor.getBody()->walk([&](tensor::ExtractOp ex) {
        if (ex.getIndices().size() == 2)
          extractOps.push_back(ex);
      });

      if (extractOps.empty())
        return rewriter.notifyMatchFailure(outerFor,
                                           "no tensor.extract with 2 indices");

      for (tensor::ExtractOp ex : extractOps) {
        // Index layout from LowerUnpackToNVVM:
        //   indices[0] = %k  (row — do NOT swizzle)
        //   indices[1] = base + byte_offset  (column — SWIZZLE THIS)
        Value oldColIdx = ex.getIndices()[1];

        // Insert the XOR immediately before the extract op
        OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPoint(ex);

        // swizzled_col = old_col XOR k_mod8
        Value newColIdx =
            rewriter.create<arith::XOrIOp>(ex.getLoc(), oldColIdx, k_mod8);

        // Mutate the extract's column index operand
        ex.getIndicesMutable()[1].assign(newColIdx);

        LLVM_DEBUG(llvm::dbgs()
                   << "SwizzledUnpackIndexing: swizzled column index of " << ex
                   << "\n");
      }
    }

    // ── Mark the outer loop as swizzled (idempotency) ─────────────
    outerFor->setAttr("swizzled", rewriter.getUnitAttr());

    return success();
  }

private:
  int64_t minNThreshold; // Minimum N to justify swizzle (default: 32)
};

//===----------------------------------------------------------------------===//
// SwizzledUnpackIndexingPass
//===----------------------------------------------------------------------===//

struct SwizzledUnpackIndexingPass
    : public PassWrapper<SwizzledUnpackIndexingPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwizzledUnpackIndexingPass)

  // Minimum N (packed last dim) threshold to apply swizzling.
  // Passes with options cannot use the default clonePass because
  // mlir::Pass::Option is non-copyable; we override clonePass explicitly.
  int64_t minNValue = 32;

  SwizzledUnpackIndexingPass() = default;
  explicit SwizzledUnpackIndexingPass(int64_t minN) : minNValue(minN) {}

  // Required because we have a custom constructor:
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<SwizzledUnpackIndexingPass>(minNValue);
  }

  StringRef getArgument() const override { return "swizzled-unpack-indexing"; }
  StringRef getDescription() const override {
    return "Rewrite tensor.extract column indices in unpack SCF loops "
           "with XOR swizzling (col ^= k%8) to prevent shared memory "
           "bank conflicts on Ampere/Hopper GPUs.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<SwizzleUnpackPattern>(ctx, minNValue);

    // Use greedy driver with a single-round walk to avoid infinite
    // reapplication (SwizzleUnpackPattern is already idempotent via attr).
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace mlir::quantforge {

std::unique_ptr<Pass> createSwizzledUnpackIndexingPass() {
  return std::make_unique<SwizzledUnpackIndexingPass>();
}

void registerSwizzledUnpackIndexingPass() {
  PassRegistration<SwizzledUnpackIndexingPass>();
}

} // namespace mlir::quantforge
