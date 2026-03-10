//===----------------------------------------------------------------------===//
// LowerDequantToArith Pass
//
// Lowers qf.dequant → arith.sitofp + arith.subf + arith.mulf
// via linalg.generic.
//
// Semantics of qf.dequant:
//   output = (sitofp(input) - sitofp(zero_point)) * scale
//
// Operands:
//   input:      tensor<D0 x D1 x … x DN x i8>
//   scale:      tensor<f16>       (scalar, broadcast over all dims)
//   zero_point: tensor<i8>        (scalar, broadcast over all dims)
//   output:     tensor<D0 x D1 x … x DN x f16>
//
// Generated IR sketch (2-D case):
//   %empty = tensor.empty() : tensor<K x N x f16>
//   %result = linalg.generic
//       {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>,   // input
//                         affine_map<(d0,d1) -> ()>,         // scale
//                         affine_map<(d0,d1) -> ()>,         // zp
//                         affine_map<(d0,d1) -> (d0,d1)>],  // output
//        iterator_types = ["parallel","parallel"]}
//       ins(%input, %scale, %zp : ...) outs(%empty : ...) {
//     ^bb0(%int: i8, %s: f16, %z: i8, %out: f16):
//       %int_fp = arith.sitofp %int : i8 to f16
//       %zp_fp  = arith.sitofp %z   : i8 to f16
//       %diff   = arith.subf %int_fp, %zp_fp : f16
//       %result = arith.mulf %diff, %s : f16
//       linalg.yield %result : f16
//   }
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lower-dequant-to-arith"

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
// DequantOpLoweringPattern
//===----------------------------------------------------------------------===//

namespace
{

    struct DequantOpLoweringPattern : public OpRewritePattern<DequantOp>
    {
        using OpRewritePattern<DequantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(DequantOp dequantOp,
                                      PatternRewriter &rewriter) const override
        {
            Location loc = dequantOp.getLoc();

            // ── Guard: ranked tensor types ────────────────────────────────────────
            auto inputTy =
                dyn_cast<RankedTensorType>(dequantOp.getInput().getType());
            auto outputTy =
                dyn_cast<RankedTensorType>(dequantOp.getOutput().getType());
            auto scaleTy =
                dyn_cast<RankedTensorType>(dequantOp.getScale().getType());
            auto zpTy =
                dyn_cast<RankedTensorType>(dequantOp.getZeroPoint().getType());

            if (!inputTy || !outputTy || !scaleTy || !zpTy)
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

            MLIRContext *ctx = rewriter.getContext();
            auto f16Ty = rewriter.getF16Type();

            // ── Step 1: Allocate empty output tensor ──────────────────────────────
            SmallVector<Value> dynSizes;
            for (unsigned i = 0; i < rank; ++i)
            {
                if (ShapedType::isDynamic(inputTy.getShape()[i]))
                {
                    dynSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, dequantOp.getInput(), i));
                }
            }

            Value emptyTensor = rewriter.create<tensor::EmptyOp>(
                loc, outputTy.getShape(), f16Ty, dynSizes);

            // ── Step 2: Build affine maps ─────────────────────────────────────────
            // Input: identity map  (d0, d1, …) → (d0, d1, …)
            AffineMap inputMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

            // Scale: scalar → broadcast over all dims: (d0, d1, …) → ()
            AffineMap scaleMap = AffineMap::get(rank, 0, {}, ctx);

            // ZeroPoint: scalar → broadcast: (d0, d1, …) → ()
            AffineMap zpMap = AffineMap::get(rank, 0, {}, ctx);

            // Output: identity
            AffineMap outputMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

            SmallVector<utils::IteratorType> iterTypes(
                rank, utils::IteratorType::parallel);

            // ── Step 3: Create linalg.generic ─────────────────────────────────────
            auto genericOp = rewriter.create<linalg::GenericOp>(
                loc,
                /*resultTypes=*/TypeRange{outputTy},
                /*inputs=*/
                ValueRange{dequantOp.getInput(), dequantOp.getScale(),
                           dequantOp.getZeroPoint()},
                /*outputs=*/ValueRange{emptyTensor},
                /*indexingMaps=*/
                ArrayRef<AffineMap>{inputMap, scaleMap, zpMap, outputMap},
                /*iteratorTypes=*/iterTypes,
                /*bodyBuilder=*/
                [&](OpBuilder &b, Location bodyLoc, ValueRange args)
                {
                    Value intVal = args[0];   // i8
                    Value scaleVal = args[1]; // f16
                    Value zpVal = args[2];    // i8

                    // sitofp i8 → f16
                    Value intFP =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, intVal);
                    Value zpFP =
                        b.create<arith::SIToFPOp>(bodyLoc, f16Ty, zpVal);

                    // (input - zero_point) * scale
                    Value diff = b.create<arith::SubFOp>(bodyLoc, intFP, zpFP);
                    Value result =
                        b.create<arith::MulFOp>(bodyLoc, diff, scaleVal);

                    b.create<linalg::YieldOp>(bodyLoc, result);
                });

            rewriter.replaceOp(dequantOp, genericOp.getResults());
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // LowerDequantToArithPass
    //===----------------------------------------------------------------------===//

    struct LowerDequantToArithPass
        : public PassWrapper<LowerDequantToArithPass, OperationPass<func::FuncOp>>
    {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDequantToArithPass)

        StringRef getArgument() const override { return "lower-dequant-to-arith"; }
        StringRef getDescription() const override
        {
            return "Lower qf.dequant to arith.sitofp + arith.subf + arith.mulf "
                   "via linalg.generic";
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
            patterns.add<DequantOpLoweringPattern>(ctx);

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

    std::unique_ptr<Pass> createLowerDequantToArithPass()
    {
        return std::make_unique<LowerDequantToArithPass>();
    }

    void registerLowerDequantToArithPass()
    {
        PassRegistration<LowerDequantToArithPass>();
    }

} // namespace mlir::quantforge
