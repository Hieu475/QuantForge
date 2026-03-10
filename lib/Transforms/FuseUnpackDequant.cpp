//===----------------------------------------------------------------------===//
// FuseUnpackDequant Pass
//
// Fuses the pattern   qf.dequant(qf.unpack(%packed), %scale, %zp)
// into a single linalg.generic that performs INT4 bit-extraction AND
// dequantization in one loop body, eliminating the intermediate i8 tensor.
//
// Before (two separate ops):
//   %unpacked = qf.unpack %packed : tensor<KxNxi8> -> tensor<Kx2Nxi8>
//   %fp       = qf.dequant %unpacked, %scale, %zp
//                            : tensor<Kx2Nxi8>, ... -> tensor<Kx2Nxf16>
//
// After (one fused generic):
//   %fp = linalg.generic
//       ins(%packed, %scale, %zp)
//       outs(%empty : tensor<K x N x 2 x f16>) {
//     ^bb0(%byte: i8, %s: f16, %z: i8, %out: f16):
//       // Unpack
//       %low  = arith.andi  %byte, 0x0F
//       %high = arith.shrui %byte, 4; arith.andi ..., 0x0F
//       %sel  = arith.select (nibble_idx == 0), %low, %high
//       // Dequant
//       %fp   = arith.sitofp %sel : i8 to f16
//       %zfp  = arith.sitofp %z   : i8 to f16
//       %diff = arith.subf %fp, %zfp
//       %out  = arith.mulf %diff, %s
//       linalg.yield %out
//   }
//   %result = tensor.collapse_shape ... -> tensor<K x 2N x f16>
//
// This is the core optimization of QuantForge: by keeping data in i8 (packed)
// all the way until the fused kernel, we avoid materializing the unpacked
// i8 tensor into HBM/Shared Memory.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "fuse-unpack-dequant"

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
// FuseUnpackDequantPattern
//===----------------------------------------------------------------------===//

namespace
{

    /// Matches: %dq = qf.dequant(qf.unpack(%packed), %scale, %zp)
    /// and replaces both ops with a single fused linalg.generic.
    ///
    /// Two sub-cases:
    ///   A) Same-shape unpack (semantic marker): pointwise mask + dequant
    ///   B) Doubling-shape unpack (actual INT4): nibble extraction + collapse
    struct FuseUnpackDequantPattern : public OpRewritePattern<DequantOp>
    {
        using OpRewritePattern<DequantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(DequantOp dequantOp,
                                      PatternRewriter &rewriter) const override
        {
            // ── Match: dequant's input must come from an unpack ───────────────────
            auto unpackOp = dequantOp.getInput().getDefiningOp<UnpackOp>();
            if (!unpackOp)
                return rewriter.notifyMatchFailure(
                    dequantOp, "input is not produced by qf.unpack");

            // ── Guard: types ──────────────────────────────────────────────────────
            Value packed = unpackOp.getInput();
            auto packedTy = dyn_cast<RankedTensorType>(packed.getType());
            auto unpackedTy =
                dyn_cast<RankedTensorType>(unpackOp.getOutput().getType());
            auto outputTy =
                dyn_cast<RankedTensorType>(dequantOp.getOutput().getType());

            if (!packedTy || !unpackedTy || !outputTy)
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected ranked tensor types");

            if (!packedTy.getElementType().isInteger(8))
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected i8 packed element type");

            if (!outputTy.getElementType().isF16())
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected f16 output");

            unsigned rank = packedTy.getRank();
            if (rank == 0)
                return rewriter.notifyMatchFailure(dequantOp,
                                                   "expected non-scalar input");

            // Determine mode: same-shape or doubling-shape
            ArrayRef<int64_t> inShape = packedTy.getShape();
            ArrayRef<int64_t> unpShape = unpackedTy.getShape();

            bool sameShape = (inShape == unpShape);
            if (!sameShape)
            {
                // Verify doubling contract: last dim doubles, leading dims match
                if ((int64_t)unpShape.size() != rank)
                    return rewriter.notifyMatchFailure(dequantOp, "rank mismatch");
                for (unsigned i = 0; i < rank - 1; ++i)
                    if (inShape[i] != unpShape[i])
                        return rewriter.notifyMatchFailure(dequantOp,
                                                           "leading dim mismatch");
                if (!ShapedType::isDynamic(inShape[rank - 1]) &&
                    !ShapedType::isDynamic(unpShape[rank - 1]) &&
                    unpShape[rank - 1] != 2 * inShape[rank - 1])
                    return rewriter.notifyMatchFailure(
                        dequantOp, "unpacked last dim != 2 * packed last dim");
            }

            Location loc = dequantOp.getLoc();
            MLIRContext *ctx = rewriter.getContext();
            auto f16Ty = rewriter.getF16Type();

            Value result;
            if (sameShape)
                result = buildSameShapeFused(rewriter, loc, ctx, f16Ty,
                                             packed, dequantOp, packedTy,
                                             outputTy, rank);
            else
                result = buildDoublingShapeFused(rewriter, loc, ctx, f16Ty,
                                                 packed, dequantOp, packedTy,
                                                 outputTy, rank);

            rewriter.replaceOp(dequantOp, result);
            if (unpackOp->use_empty())
                rewriter.eraseOp(unpackOp);
            return success();
        }

    private:
        /// Case A: Same-shape unpack — pointwise (mask & 0x0F) + dequant
        Value buildSameShapeFused(PatternRewriter &rewriter, Location loc,
                                  MLIRContext *ctx, Type f16Ty,
                                  Value packed, DequantOp dequantOp,
                                  RankedTensorType packedTy,
                                  RankedTensorType outputTy,
                                  unsigned rank) const
        {
            SmallVector<Value> dynSizes;
            ArrayRef<int64_t> shape = packedTy.getShape();
            for (unsigned i = 0; i < rank; ++i)
                if (ShapedType::isDynamic(shape[i]))
                    dynSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, packed, i));

            Value emptyTensor = rewriter.create<tensor::EmptyOp>(
                loc, outputTy.getShape(), f16Ty, dynSizes);

            AffineMap identityMap =
                AffineMap::getMultiDimIdentityMap(rank, ctx);
            AffineMap scalarMap = AffineMap::get(rank, 0, {}, ctx);

            SmallVector<utils::IteratorType> iterTypes(
                rank, utils::IteratorType::parallel);

            auto genericOp = rewriter.create<linalg::GenericOp>(
                loc, TypeRange{outputTy},
                ValueRange{packed, dequantOp.getScale(),
                           dequantOp.getZeroPoint()},
                ValueRange{emptyTensor},
                ArrayRef<AffineMap>{identityMap, scalarMap, scalarMap,
                                    identityMap},
                iterTypes,
                [f16Ty](OpBuilder &b, Location bodyLoc, ValueRange args)
                {
                    Value elem = args[0];     // i8
                    Value scaleVal = args[1]; // f16
                    Value zpVal = args[2];    // i8

                    // Mask lower nibble
                    Value mask =
                        b.create<arith::ConstantIntOp>(bodyLoc, 0x0F, 8);
                    Value masked =
                        b.create<arith::AndIOp>(bodyLoc, elem, mask);

                    // Dequant
                    Value fp =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, masked);
                    Value zpFP =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, zpVal);
                    Value diff = b.create<arith::SubFOp>(bodyLoc, fp, zpFP);
                    Value result =
                        b.create<arith::MulFOp>(bodyLoc, diff, scaleVal);

                    b.create<linalg::YieldOp>(bodyLoc, result);
                });

            return genericOp.getResult(0);
        }

        /// Case B: Doubling-shape unpack — extract both nibbles + collapse
        Value buildDoublingShapeFused(PatternRewriter &rewriter, Location loc,
                                      MLIRContext *ctx, Type f16Ty,
                                      Value packed, DequantOp dequantOp,
                                      RankedTensorType packedTy,
                                      RankedTensorType outputTy,
                                      unsigned rank) const
        {
            ArrayRef<int64_t> inShape = packedTy.getShape();

            // Intermediate: [..., N, 2] of f16
            SmallVector<int64_t> intermediateShape(inShape.begin(),
                                                   inShape.end());
            intermediateShape.push_back(2);
            auto intermediateTy =
                RankedTensorType::get(intermediateShape, f16Ty);

            SmallVector<Value> dynSizes;
            for (unsigned i = 0; i < rank; ++i)
                if (ShapedType::isDynamic(inShape[i]))
                    dynSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, packed, i));

            Value emptyTensor = rewriter.create<tensor::EmptyOp>(
                loc, intermediateShape, f16Ty, dynSizes);

            unsigned genericRank = rank + 1;

            SmallVector<AffineExpr> packedExprs;
            for (unsigned i = 0; i < rank; ++i)
                packedExprs.push_back(getAffineDimExpr(i, ctx));
            AffineMap packedMap =
                AffineMap::get(genericRank, 0, packedExprs, ctx);
            AffineMap scalarMap = AffineMap::get(genericRank, 0, {}, ctx);
            AffineMap outputMap =
                AffineMap::getMultiDimIdentityMap(genericRank, ctx);

            SmallVector<utils::IteratorType> iterTypes(
                genericRank, utils::IteratorType::parallel);

            auto genericOp = rewriter.create<linalg::GenericOp>(
                loc, TypeRange{intermediateTy},
                ValueRange{packed, dequantOp.getScale(),
                           dequantOp.getZeroPoint()},
                ValueRange{emptyTensor},
                ArrayRef<AffineMap>{packedMap, scalarMap, scalarMap, outputMap},
                iterTypes,
                [rank, f16Ty](OpBuilder &b, Location bodyLoc, ValueRange args)
                {
                    Value packedElem = args[0];
                    Value scaleVal = args[1];
                    Value zpVal = args[2];

                    Value mask =
                        b.create<arith::ConstantIntOp>(bodyLoc, 0x0F, 8);
                    Value shift =
                        b.create<arith::ConstantIntOp>(bodyLoc, 4, 8);

                    Value low =
                        b.create<arith::AndIOp>(bodyLoc, packedElem, mask);
                    Value shifted =
                        b.create<arith::ShRUIOp>(bodyLoc, packedElem, shift);
                    Value high =
                        b.create<arith::AndIOp>(bodyLoc, shifted, mask);

                    Value nibbleIdx =
                        b.create<linalg::IndexOp>(bodyLoc, rank);
                    Value cZero =
                        b.create<arith::ConstantIndexOp>(bodyLoc, 0);
                    Value isLow = b.create<arith::CmpIOp>(
                        bodyLoc, arith::CmpIPredicate::eq, nibbleIdx, cZero);
                    Value nibble = b.create<arith::SelectOp>(
                        bodyLoc, isLow, low, high);

                    Value fp =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, nibble);
                    Value zpFP =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, zpVal);
                    Value diff =
                        b.create<arith::SubFOp>(bodyLoc, fp, zpFP);
                    Value result =
                        b.create<arith::MulFOp>(bodyLoc, diff, scaleVal);

                    b.create<linalg::YieldOp>(bodyLoc, result);
                });

            // Collapse [..., N, 2] → [..., 2*N]
            SmallVector<ReassociationIndices> reassoc;
            for (unsigned i = 0; i < rank - 1; ++i)
                reassoc.push_back({static_cast<int64_t>(i)});
            reassoc.push_back({static_cast<int64_t>(rank - 1),
                               static_cast<int64_t>(rank)});

            return rewriter.create<tensor::CollapseShapeOp>(
                               loc, outputTy, genericOp.getResult(0), reassoc)
                .getResult();
        }
    };

    //===----------------------------------------------------------------------===//
    // FuseUnpackDequantPass
    //===----------------------------------------------------------------------===//

    struct FuseUnpackDequantPass
        : public PassWrapper<FuseUnpackDequantPass, OperationPass<func::FuncOp>>
    {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseUnpackDequantPass)

        StringRef getArgument() const override { return "fuse-unpack-dequant"; }
        StringRef getDescription() const override
        {
            return "Fuse qf.unpack + qf.dequant into a single linalg.generic "
                   "performing on-the-fly INT4 unpacking and dequantization";
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
            patterns.add<FuseUnpackDequantPattern>(ctx);

            if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
                signalPassFailure();
        }
    };

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace mlir::quantforge
{

    std::unique_ptr<Pass> createFuseUnpackDequantPass()
    {
        return std::make_unique<FuseUnpackDequantPass>();
    }

    void registerFuseUnpackDequantPass()
    {
        PassRegistration<FuseUnpackDequantPass>();
    }

} // namespace mlir::quantforge
