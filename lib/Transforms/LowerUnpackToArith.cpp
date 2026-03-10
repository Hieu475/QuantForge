//===----------------------------------------------------------------------===//
// LowerUnpackToArith Pass
//
// Lowers qf.unpack → arith.shrui + arith.andi via linalg.generic
//
// Semantics of qf.unpack:
//   Each input i8 element contains two packed INT4 nibbles:
//     bits [3:0] → low  nibble
//     bits [7:4] → high nibble
//
// Shape contract:
//   input:  tensor<D0 x D1 x … x N   x i8>
//   output: tensor<D0 x D1 x … x 2*N x i8>
//
// Lowering strategy (3-step):
//   1. linalg.generic  (D0…DN, 2)  →  each packed element emits two nibbles
//      using arith.andi / arith.shrui + arith.andi, selected by linalg.index.
//   2. tensor.collapse_shape  merges the two trailing dims back to one.
//
// Generated IR sketch (2-D case):
//   %empty = tensor.empty() : tensor<K x N x 2 x i8>
//   %tmp = linalg.generic
//       {indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1)>,
//                         affine_map<(d0,d1,d2) -> (d0,d1,d2)>],
//        iterator_types = ["parallel","parallel","parallel"]}
//       ins(%in : tensor<K x N x i8>)
//       outs(%empty : tensor<K x N x 2 x i8>) {
//     ^bb0(%e: i8, %_: i8):
//       %mask  = arith.constant 15 : i8
//       %four  = arith.constant  4 : i8
//       %low   = arith.andi  %e, %mask : i8
//       %sh    = arith.shrui %e, %four : i8
//       %high  = arith.andi  %sh, %mask : i8
//       %d2    = linalg.index 2 : index
//       %c0    = arith.constant 0 : index
//       %cond  = arith.cmpi eq, %d2, %c0 : index
//       %r     = arith.select %cond, %low, %high : i8
//       linalg.yield %r : i8
//   }
//   %out = tensor.collapse_shape %tmp [[0],[1,2]]
//                : tensor<K x N x 2 x i8> into tensor<K x 2*N x i8>
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lower-unpack-to-arith"

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
// UnpackOpLoweringPattern
//===----------------------------------------------------------------------===//

namespace
{

    /// Matches qf.unpack where the output tensor's last dimension is exactly
    /// twice the input's last dimension, and lowers it to a linalg.generic
    /// that extracts the low and high nibbles using arith.andi / arith.shrui.
    struct UnpackOpLoweringPattern : public OpRewritePattern<UnpackOp>
    {
        using OpRewritePattern<UnpackOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(UnpackOp unpackOp,
                                      PatternRewriter &rewriter) const override
        {
            Location loc = unpackOp.getLoc();

            // ── Guard: input must be a ranked i8 tensor ──────────────────────────
            auto inputTy = dyn_cast<RankedTensorType>(unpackOp.getInput().getType());
            auto outputTy = dyn_cast<RankedTensorType>(unpackOp.getOutput().getType());
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

            // ── Guard: last output dim must be 2× last input dim ─────────────────
            ArrayRef<int64_t> inShape = inputTy.getShape();
            ArrayRef<int64_t> outShape = outputTy.getShape();

            if ((int64_t)outShape.size() != rank)
                return rewriter.notifyMatchFailure(unpackOp, "rank mismatch");

            // Check all leading dims match.
            for (unsigned i = 0; i < rank - 1; ++i)
                if (inShape[i] != outShape[i])
                    return rewriter.notifyMatchFailure(unpackOp,
                                                       "leading dim mismatch");

            // Last dim: statically, outShape[rank-1] == 2 * inShape[rank-1].
            // We accept ShapedType::kDynamic for dynamic shapes too.
            if (!ShapedType::isDynamic(inShape[rank - 1]) &&
                !ShapedType::isDynamic(outShape[rank - 1]))
            {
                if (outShape[rank - 1] != 2 * inShape[rank - 1])
                    return rewriter.notifyMatchFailure(unpackOp,
                                                       "output last dim != 2 * input last dim");
            }

            auto i8Ty = rewriter.getIntegerType(8);
            MLIRContext *ctx = rewriter.getContext();

            // ── Step 1: Build intermediate type: [..., N, 2] ─────────────────────
            SmallVector<int64_t> intermediateShape(inShape.begin(), inShape.end());
            intermediateShape.push_back(2);
            auto intermediateTy = RankedTensorType::get(intermediateShape, i8Ty);

            // ── Step 2: Allocate empty intermediate tensor ────────────────────────
            // Use mixed static/dynamic sizes.
            SmallVector<Value> dynSizes;
            SmallVector<int64_t> staticSizes(intermediateShape);

            for (unsigned i = 0; i < rank; ++i)
            {
                if (ShapedType::isDynamic(inShape[i]))
                {
                    staticSizes[i] = ShapedType::kDynamic;
                    dynSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, unpackOp.getInput(), i));
                }
            }
            // The trailing "2" is always static.

            Value emptyTensor = rewriter.create<tensor::EmptyOp>(
                loc, staticSizes, i8Ty, dynSizes);

            // ── Step 3: Build affine maps for the linalg.generic ─────────────────
            unsigned genericRank = rank + 1; // extra dim for nibble index

            // Input map:  (d0, d1, …, d_{rank-1}, d_rank) → (d0, …, d_{rank-1})
            //   (Broadcasts over the extra "nibble" dimension.)
            SmallVector<AffineExpr> inputExprs;
            for (unsigned i = 0; i < rank; ++i)
                inputExprs.push_back(getAffineDimExpr(i, ctx));
            AffineMap inputMap = AffineMap::get(genericRank, 0, inputExprs, ctx);

            // Output map: identity over all genericRank dimensions.
            AffineMap outputMap = AffineMap::getMultiDimIdentityMap(genericRank, ctx);

            SmallVector<utils::IteratorType> iterTypes(genericRank,
                                                       utils::IteratorType::parallel);

            // ── Step 4: Create linalg.generic with arith bit-ops in its body ─────
            auto genericOp = rewriter.create<linalg::GenericOp>(
                loc,
                /*resultTypes=*/TypeRange{intermediateTy},
                /*inputs=*/ValueRange{unpackOp.getInput()},
                /*outputs=*/ValueRange{emptyTensor},
                /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, outputMap},
                /*iteratorTypes=*/iterTypes,
                /*bodyBuilder=*/
                [rank](OpBuilder &b, Location bodyLoc, ValueRange args)
                {
                    Value packedElem = args[0]; // i8

                    // Constants
                    Value maskConst =
                        b.create<arith::ConstantIntOp>(bodyLoc, 0x0F, 8);
                    Value shiftConst =
                        b.create<arith::ConstantIntOp>(bodyLoc, 4, 8);

                    // low  = packed & 0x0F
                    Value lowNibble =
                        b.create<arith::AndIOp>(bodyLoc, packedElem, maskConst);

                    // high = (packed >> 4) & 0x0F
                    Value shifted =
                        b.create<arith::ShRUIOp>(bodyLoc, packedElem, shiftConst);
                    Value highNibble =
                        b.create<arith::AndIOp>(bodyLoc, shifted, maskConst);

                    // Select low or high based on the nibble index dimension.
                    Value nibbleIdx = b.create<linalg::IndexOp>(bodyLoc, rank);
                    Value cZero = b.create<arith::ConstantIndexOp>(bodyLoc, 0);
                    Value isLow = b.create<arith::CmpIOp>(
                        bodyLoc, arith::CmpIPredicate::eq, nibbleIdx, cZero);
                    Value result =
                        b.create<arith::SelectOp>(bodyLoc, isLow, lowNibble, highNibble);

                    b.create<linalg::YieldOp>(bodyLoc, result);
                });

            // ── Step 5: Collapse [..., N, 2] → [..., 2*N] ────────────────────────
            // Reassociation: every leading dim maps 1-to-1, last two dims merge.
            SmallVector<ReassociationIndices> reassoc;
            for (unsigned i = 0; i < rank - 1; ++i)
                reassoc.push_back({static_cast<int64_t>(i)});
            // Merge dim[rank-1] and dim[rank] (the extra "2").
            reassoc.push_back({static_cast<int64_t>(rank - 1),
                               static_cast<int64_t>(rank)});

            Value collapsed = rewriter.create<tensor::CollapseShapeOp>(
                loc, outputTy, genericOp.getResult(0), reassoc);

            rewriter.replaceOp(unpackOp, collapsed);
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // LowerUnpackToArithPass
    //===----------------------------------------------------------------------===//

    struct LowerUnpackToArithPass
        : public PassWrapper<LowerUnpackToArithPass, OperationPass<func::FuncOp>>
    {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerUnpackToArithPass)

        StringRef getArgument() const override { return "lower-unpack-to-arith"; }
        StringRef getDescription() const override
        {
            return "Lower qf.unpack to arith.shrui + arith.andi via linalg.generic";
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
            patterns.add<UnpackOpLoweringPattern>(ctx);

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

    std::unique_ptr<Pass> createLowerUnpackToArithPass()
    {
        return std::make_unique<LowerUnpackToArithPass>();
    }

    void registerLowerUnpackToArithPass()
    {
        PassRegistration<LowerUnpackToArithPass>();
    }

} // namespace mlir::quantforge
