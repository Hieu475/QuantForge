//===----------------------------------------------------------------------===//
// CanonicalizeDequantZeroPoint Pass — Symmetric Quantization Fast-Path
//
// Detects zero_point == 0 (symmetric quantization) and eliminates redundant
// arith.sitofp + arith.subf instructions from the dequantization pipeline.
//
// This pass operates at two levels:
//
// (A) Pre-lowering: Matches qf.dequant with constant-zero zero_point,
//     replaces with simplified dequant: output = sitofp(input) * scale
//     (eliminating sitofp(zp) + subf entirely)
//
// (B) Post-lowering: Matches arith.subf %x, <const 0.0> inside
//     linalg.generic bodies and folds to %x, then DCEs dead sitofp.
//
// In practice, symmetric quantization (zp==0) is the dominant scheme used
// in production LLM inference (vLLM, TensorRT-LLM, etc.), so this saves:
//   - 2 instructions per element (sitofp + subf)
//   - 1 register (no need to hold zero_point)
//   - Reduced register pressure in fused unpack+dequant kernels
//
// This pass is safe to run at any point in the pipeline.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "canonicalize-dequant-zp"

#include "QuantForge/Transforms/Passes.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

//===----------------------------------------------------------------------===//
// Helper: check if a Value is a constant tensor/scalar containing all zeros
//===----------------------------------------------------------------------===//

/// Returns true if `val` is defined by arith.constant whose attribute
/// is an integer (or dense integer) that is all zeros.
static bool isZeroConstant(Value val)
{
    auto defOp = val.getDefiningOp<arith::ConstantOp>();
    if (!defOp)
        return false;

    Attribute attr = defOp.getValue();

    // Case 1: DenseElementsAttr (tensor of constants)
    if (auto dense = dyn_cast<DenseElementsAttr>(attr))
    {
        // Must be integer type (i8 for zero_point)
        if (!isa<IntegerType>(dense.getElementType()))
            return false;
        // Check if all elements are zero
        return dense.isSplat() && dense.getSplatValue<APInt>().isZero();
    }

    // Case 2: IntegerAttr (scalar constant)
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getValue().isZero();

    // Case 3: DenseFPElementsAttr (tensor of FP constants)
    if (auto denseFP = dyn_cast<DenseFPElementsAttr>(attr))
        return denseFP.isSplat() && denseFP.getSplatValue<APFloat>().isZero();

    // Case 4: FloatAttr (scalar FP constant)
    if (auto fpAttr = dyn_cast<FloatAttr>(attr))
        return fpAttr.getValue().isZero();

    return false;
}

//===----------------------------------------------------------------------===//
// Pattern A: Pre-lowering — DequantOp with zp==0
//===----------------------------------------------------------------------===//

namespace
{

    /// Matches qf.dequant where zero_point is a constant zero tensor.
    /// Replaces with a simplified version: output = sitofp(input) * scale
    /// (no sitofp(zp), no subf).
    ///
    /// Before:
    ///   %zp = arith.constant dense<0> : tensor<i8>
    ///   %out = qf.dequant %input, %scale, %zp
    ///
    /// After:
    ///   %out = qf.dequant %input, %scale, %zp  →  simplified linalg.generic
    ///     with body: output = sitofp(input) * scale  (2 fewer ops)
    struct DequantZeroPointFoldPattern : public OpRewritePattern<DequantOp>
    {
        using OpRewritePattern<DequantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(DequantOp dequantOp,
                                      PatternRewriter &rewriter) const override
        {
            // ── Check: zero_point must be constant zero ───────────────────────
            Value zpVal = dequantOp.getZeroPoint();
            if (!isZeroConstant(zpVal))
                return rewriter.notifyMatchFailure(
                    dequantOp, "zero_point is not constant zero");

            // ── Guard: types ──────────────────────────────────────────────────
            auto inputTy =
                dyn_cast<RankedTensorType>(dequantOp.getInput().getType());
            auto outputTy =
                dyn_cast<RankedTensorType>(dequantOp.getOutput().getType());

            if (!inputTy || !outputTy)
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected ranked tensor types");

            if (!inputTy.getElementType().isInteger(8))
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected i8 input element type");

            if (!outputTy.getElementType().isF16())
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected f16 output element type");

            unsigned rank = inputTy.getRank();
            if (rank == 0)
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected non-scalar input");

            Location loc = dequantOp.getLoc();
            MLIRContext *ctx = rewriter.getContext();
            auto f16Ty = rewriter.getF16Type();

            // ── Allocate output tensor ────────────────────────────────────────
            SmallVector<Value> dynSizes;
            for (unsigned i = 0; i < rank; ++i)
                if (ShapedType::isDynamic(inputTy.getShape()[i]))
                    dynSizes.push_back(
                        rewriter.create<tensor::DimOp>(
                            loc, dequantOp.getInput(), i));

            Value emptyTensor = rewriter.create<tensor::EmptyOp>(
                loc, outputTy.getShape(), f16Ty, dynSizes);

            // ── Build affine maps ─────────────────────────────────────────────
            AffineMap inputMap =
                AffineMap::getMultiDimIdentityMap(rank, ctx);
            AffineMap scaleMap = AffineMap::get(rank, 0, {}, ctx);
            AffineMap outputMap =
                AffineMap::getMultiDimIdentityMap(rank, ctx);

            SmallVector<utils::IteratorType> iterTypes(
                rank, utils::IteratorType::parallel);

            // ── Build simplified linalg.generic ───────────────────────────────
            // Body: output = sitofp(input) * scale
            // (NO sitofp(zp), NO subf)
            auto genericOp = rewriter.create<linalg::GenericOp>(
                loc,
                /*resultTypes=*/TypeRange{outputTy},
                /*inputs=*/
                ValueRange{dequantOp.getInput(), dequantOp.getScale()},
                /*outputs=*/ValueRange{emptyTensor},
                /*indexingMaps=*/
                ArrayRef<AffineMap>{inputMap, scaleMap, outputMap},
                /*iteratorTypes=*/iterTypes,
                /*bodyBuilder=*/
                [f16Ty](OpBuilder &b, Location bodyLoc, ValueRange args)
                {
                    Value intVal = args[0];   // i8
                    Value scaleVal = args[1]; // f16

                    // sitofp i8 → f16
                    Value intFP =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, intVal);

                    // Directly: result = input_fp * scale
                    // (skip: zpFP = sitofp(zp), diff = subf(intFP, zpFP))
                    Value result =
                        b.create<arith::MulFOp>(bodyLoc, intFP, scaleVal);

                    b.create<linalg::YieldOp>(bodyLoc, result);
                });

            rewriter.replaceOp(dequantOp, genericOp.getResults());
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // Pattern B: Post-lowering — fold arith.subf %x, 0.0 → %x
    //===----------------------------------------------------------------------===//

    /// Matches arith.subf where the RHS is a constant-zero f16/f32 value,
    /// and replaces the result with the LHS.
    /// This catches zero_point folding opportunities that survived through
    /// fusion passes (e.g., fuse-unpack-dequant output).
    struct FoldSubfZeroPattern : public OpRewritePattern<arith::SubFOp>
    {
        using OpRewritePattern<arith::SubFOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::SubFOp subOp,
                                      PatternRewriter &rewriter) const override
        {
            Value rhs = subOp.getRhs();

            // Check if RHS is a constant zero (floating-point)
            APFloat zeroVal(0.0f);
            if (!matchPattern(rhs, m_AnyZeroFloat()))
                return rewriter.notifyMatchFailure(
                    subOp, "RHS is not constant zero");

            // subf(x, 0.0) → x
            rewriter.replaceOp(subOp, subOp.getLhs());

            LLVM_DEBUG(llvm::dbgs()
                       << "CanonicalizeDequantZP: folded subf(x, 0.0) → x\n");

            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // Pattern C: DCE dead sitofp producing the folded-away zero
    //===----------------------------------------------------------------------===//

    /// After FoldSubfZeroPattern removes subf(x, sitofp(0)), the sitofp
    /// producing 0.0 may become dead. This pattern eagerly removes it
    /// (standard GreedyPatternRewriter DCE also handles this, but being
    /// explicit makes the intent clear).
    struct DCEDeadSIToFPPattern : public OpRewritePattern<arith::SIToFPOp>
    {
        using OpRewritePattern<arith::SIToFPOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::SIToFPOp op,
                                      PatternRewriter &rewriter) const override
        {
            // Only remove if the result has no users
            if (!op->use_empty())
                return failure();

            rewriter.eraseOp(op);
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // CanonicalizeDequantZeroPointPass
    //===----------------------------------------------------------------------===//

    struct CanonicalizeDequantZeroPointPass
        : public PassWrapper<CanonicalizeDequantZeroPointPass,
                             OperationPass<func::FuncOp>>
    {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
            CanonicalizeDequantZeroPointPass)

        StringRef getArgument() const override
        {
            return "canonicalize-dequant-zp";
        }
        StringRef getDescription() const override
        {
            return "Eliminate arith.sitofp + arith.subf when zero_point == 0 "
                   "(symmetric quantization fast-path). Operates at both "
                   "pre-lowering (qf.dequant) and post-lowering (arith.subf) "
                   "levels.";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                            tensor::TensorDialect>();
        }

        void runOnOperation() override
        {
            func::FuncOp funcOp = getOperation();
            MLIRContext *ctx = &getContext();

            RewritePatternSet patterns(ctx);
            // Pre-lowering: fold qf.dequant with zp==0
            patterns.add<DequantZeroPointFoldPattern>(ctx);
            // Post-lowering: fold arith.subf(x, 0.0) → x
            patterns.add<FoldSubfZeroPattern>(ctx);
            // DCE dead sitofp after subf fold
            patterns.add<DCEDeadSIToFPPattern>(ctx);

            if (failed(applyPatternsAndFoldGreedily(funcOp,
                                                     std::move(patterns))))
                signalPassFailure();
        }
    };

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace mlir::quantforge
{

    std::unique_ptr<Pass> createCanonicalizeDequantZeroPointPass()
    {
        return std::make_unique<CanonicalizeDequantZeroPointPass>();
    }

    void registerCanonicalizeDequantZeroPointPass()
    {
        PassRegistration<CanonicalizeDequantZeroPointPass>();
    }

} // namespace mlir::quantforge
