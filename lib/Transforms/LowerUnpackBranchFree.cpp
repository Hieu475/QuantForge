//===----------------------------------------------------------------------===//
// LowerUnpackBranchFree Pass — Phase 1: Vectorized Unpacking
//
// Replaces qf.unpack (doubling-shape) with two branch-free linalg.generic ops
// — one for the low nibble, one for the high nibble — interleaved via
// tensor.insert_slice with stride-2 on the last dimension.
//
// This eliminates:
//   (1) linalg.index  — used to decide whether to extract the low or high nibble
//   (2) arith.select  — conditional selection in the inner loop body
//
// Both of these are sources of GPU warp divergence and prevent the LLVM /
// PTX backend from generating SIMD instructions for the inner loop.
// After this pass every linalg.generic is a purely *pointwise* (one-to-one)
// operation — the ideal shape for GPU vectorisation.
//
// Generated IR (2-D case, static K × N):
//
//   %lo_empty  = tensor.empty() : tensor<K×N×i8>
//   %hi_empty  = tensor.empty() : tensor<K×N×i8>
//   %out_empty = tensor.empty() : tensor<K×2N×i8>
//
//   %lows  = linalg.generic
//       {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]}
//       ins(%packed : tensor<K×N×i8>) outs(%lo_empty) {
//     ^bb0(%b: i8, %_: i8):
//       %m = arith.constant 15 : i8
//       linalg.yield  arith.andi %b, %m
//   }
//
//   %highs = linalg.generic
//       {same maps}
//       ins(%packed) outs(%hi_empty) {
//     ^bb0(%b: i8, %_: i8):
//       %m = arith.constant 15 : i8 ; %s = arith.constant 4 : i8
//       linalg.yield  arith.andi (arith.shrui %b, %s), %m
//   }
//
//   // Interleave: even columns ← lows, odd columns ← highs
//   %r0  = tensor.insert_slice %lows  into %out_empty [0,0]    [K,N] [1,2]
//   %out = tensor.insert_slice %highs into %r0         [0,1]   [K,N] [1,2]
//
// Compare with LowerUnpackToArith (before this pass):
//   • linalg.generic over (K, N, 2) with linalg.index + arith.select → BRANCH
// After this pass:
//   • two (K, N) linalg.generics, zero branches, pure data-parallel kernels
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lower-unpack-branch-free"

#include "QuantForge/Transforms/Passes.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantforge;

//===----------------------------------------------------------------------===//
// UnpackBranchFreePattern
//===----------------------------------------------------------------------===//

namespace
{

    /// Matches qf.unpack with the doubling-shape contract and rewrites it to a
    /// pair of branch-free linalg.generics interleaved via stride-2
    /// tensor.insert_slice, eliminating all linalg.index / arith.select ops.
    struct UnpackBranchFreePattern : public OpRewritePattern<UnpackOp>
    {
        using OpRewritePattern<UnpackOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(UnpackOp unpackOp,
                                      PatternRewriter &rewriter) const override
        {
            Location loc = unpackOp.getLoc();

            auto inputTy =
                dyn_cast<RankedTensorType>(unpackOp.getInput().getType());
            auto outputTy =
                dyn_cast<RankedTensorType>(unpackOp.getOutput().getType());
            if (!inputTy || !outputTy)
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "expected ranked tensor types");

            if (!inputTy.getElementType().isInteger(8))
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "expected i8 element type");

            unsigned rank = inputTy.getRank();
            if (rank == 0)
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "expected non-scalar input");

            ArrayRef<int64_t> inShape = inputTy.getShape();
            ArrayRef<int64_t> outShape = outputTy.getShape();

            // ── Rank and leading dims must match ─────────────────────────────────
            if ((int64_t)outShape.size() != rank)
                return rewriter.notifyMatchFailure(unpackOp, "rank mismatch");

            for (unsigned i = 0; i + 1 < rank; ++i)
                if (inShape[i] != outShape[i])
                    return rewriter.notifyMatchFailure(unpackOp,
                                                       "leading dim mismatch");

            // ── Skip same-shape (semantic-marker) mode ────────────────────────────
            if (inShape.back() == outShape.back())
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "same-shape mode — skip");

            // ── Last output dim must be 2× last input dim (static check) ──────────
            if (!ShapedType::isDynamic(inShape.back()) &&
                !ShapedType::isDynamic(outShape.back()) &&
                outShape.back() != 2 * inShape.back())
                return rewriter.notifyMatchFailure(
                    unpackOp, "output last dim != 2 * input last dim");

            MLIRContext *ctx = rewriter.getContext();
            auto i8Ty = rewriter.getIntegerType(8);
            Value packed = unpackOp.getInput();

            // ── Compute dynamic input sizes ───────────────────────────────────────
            SmallVector<Value> dynInSizes;
            for (unsigned i = 0; i < rank; ++i)
                if (ShapedType::isDynamic(inShape[i]))
                    dynInSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, packed, i));

            // ── Helper: build linalg.generic for one nibble ───────────────────────
            AffineMap identMap =
                AffineMap::getMultiDimIdentityMap(rank, ctx);
            SmallVector<utils::IteratorType> iterTypes(
                rank, utils::IteratorType::parallel);

            auto buildNibbleGeneric =
                [&](bool isHigh) -> linalg::GenericOp
            {
                Value emptySlice = rewriter.create<tensor::EmptyOp>(
                    loc,
                    SmallVector<int64_t>(inShape.begin(), inShape.end()),
                    i8Ty, dynInSizes);

                return rewriter.create<linalg::GenericOp>(
                    loc, TypeRange{inputTy},
                    ValueRange{packed}, ValueRange{emptySlice},
                    ArrayRef<AffineMap>{identMap, identMap},
                    iterTypes,
                    [isHigh](OpBuilder &b, Location bl, ValueRange args)
                    {
                        Value mask =
                            b.create<arith::ConstantIntOp>(bl, 0x0F, 8);
                        Value nibble = args[0];
                        if (isHigh)
                        {
                            Value shift =
                                b.create<arith::ConstantIntOp>(bl, 4, 8);
                            nibble = b.create<arith::ShRUIOp>(
                                bl, nibble, shift);
                        }
                        nibble = b.create<arith::AndIOp>(bl, nibble, mask);
                        b.create<linalg::YieldOp>(bl, nibble);
                    });
            };

            linalg::GenericOp loGen = buildNibbleGeneric(/*isHigh=*/false);
            linalg::GenericOp hiGen = buildNibbleGeneric(/*isHigh=*/true);

            // ── Allocate flat output tensor ───────────────────────────────────────
            SmallVector<int64_t> outStaticShape(outShape.begin(),
                                                outShape.end());
            SmallVector<Value> dynOutSizes;
            for (unsigned i = 0; i + 1 < rank; ++i)
                if (ShapedType::isDynamic(outShape[i]))
                    dynOutSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, packed, i));

            if (ShapedType::isDynamic(outShape.back()))
            {
                Value inN =
                    rewriter.create<tensor::DimOp>(loc, packed, rank - 1);
                Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
                dynOutSizes.push_back(
                    rewriter.create<arith::MulIOp>(loc, inN, c2));
            }

            Value outFlat = rewriter.create<tensor::EmptyOp>(
                loc, outStaticShape, i8Ty, dynOutSizes);

            // ── Build insert_slice sizes and strides ──────────────────────────────
            // sizes  = input shape (K, …, N)
            // strides = (1, …, 1, 2)  — stride-2 on the last dimension only
            SmallVector<OpFoldResult> sizes;
            for (unsigned i = 0; i < rank; ++i)
            {
                if (ShapedType::isDynamic(inShape[i]))
                    sizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, packed, i)
                            .getResult());
                else
                    sizes.push_back(rewriter.getIndexAttr(inShape[i]));
            }

            SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
            strides.back() = rewriter.getIndexAttr(2); // stride-2 on last dim

            // Low nibbles at even output positions:  offsets = (0, …, 0)
            SmallVector<OpFoldResult> loOffsets(rank, rewriter.getIndexAttr(0));
            Value r0 = rewriter.create<tensor::InsertSliceOp>(
                loc, loGen.getResult(0), outFlat, loOffsets, sizes, strides);

            // High nibbles at odd output positions:  last offset = 1
            SmallVector<OpFoldResult> hiOffsets(rank, rewriter.getIndexAttr(0));
            hiOffsets.back() = rewriter.getIndexAttr(1);
            Value result = rewriter.create<tensor::InsertSliceOp>(
                loc, hiGen.getResult(0), r0, hiOffsets, sizes, strides);

            rewriter.replaceOp(unpackOp, result);
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // LowerUnpackBranchFreePass
    //===----------------------------------------------------------------------===//

    struct LowerUnpackBranchFreePass
        : public PassWrapper<LowerUnpackBranchFreePass,
                             OperationPass<func::FuncOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerUnpackBranchFreePass)

        StringRef getArgument() const override
        {
            return "lower-unpack-branch-free";
        }
        StringRef getDescription() const override
        {
            return "Lower qf.unpack to two branch-free linalg.generics + "
                   "stride-2 tensor.insert_slice (Phase 1 vectorized unpacking)";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                            tensor::TensorDialect>();
        }

        void runOnOperation() override
        {
            MLIRContext *ctx = &getContext();
            RewritePatternSet patterns(ctx);
            patterns.add<UnpackBranchFreePattern>(ctx);
            if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                    std::move(patterns))))
                signalPassFailure();
        }
    };

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::quantforge::createLowerUnpackBranchFreePass()
{
    return std::make_unique<LowerUnpackBranchFreePass>();
}

void mlir::quantforge::registerLowerUnpackBranchFreePass()
{
    PassRegistration<LowerUnpackBranchFreePass>();
}
