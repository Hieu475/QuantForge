//===----------------------------------------------------------------------===//
// ConvertLinalgToQuantForge Pass
//
// Recognizes linalg.matmul patterns where the RHS weight tensor has element
// type i8 (representing INT8-quantized or INT4-packed weights) and rewrites:
//
//   linalg.matmul(activation, weight_i8)
//     ↓
//   %unpacked = qf.unpack  weight_i8   : tensor<KxNxi8> -> tensor<KxNxi8>
//   %fp       = qf.dequant %unpacked, %scale, %zp
//                                       : tensor<KxNxi8> -> tensor<KxNxf16>
//   %out      = linalg.matmul(act_f16, %fp) -> tensor<MxNxf16>
//
// qf.unpack is a semantic marker that signals downstream lowering passes to
// perform on-the-fly INT4 bit-extraction in registers.  Shape stays the same
// so the greedy rewrite preserves SSA types.
//===----------------------------------------------------------------------===//

#include "QuantForge/Transforms/Passes.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantforge;

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Return true if `ty` is an integer type with the given bit width.
static bool isIntOfWidth(Type ty, unsigned width)
{
    if (auto intTy = dyn_cast<IntegerType>(ty))
        return intTy.getWidth() == width;
    return false;
}

/// Return true if the tensor element type is i8 (INT8/INT4-packed weights).
static bool isInt8PackedWeight(Value weight)
{
    auto tensorTy = dyn_cast<RankedTensorType>(weight.getType());
    if (!tensorTy)
        return false;
    return isIntOfWidth(tensorTy.getElementType(), 8);
}

//===----------------------------------------------------------------------===//
// ConvertMatmulToUnpackPattern
//===----------------------------------------------------------------------===//

/// Matches `linalg.matmul` whose RHS operand is a rank-2 INT8 tensor.
///
/// The rewrite inserts:
///   1. qf.unpack  – marks the tensor for INT4 unpacking (same shape)
///   2. qf.dequant – converts i8 → f16 with scale & zero_point
///   3. A new linalg.matmul operating entirely in FP16
///
/// Placeholder scale=1.0 and zero_point=0 are emitted; a real pipeline would
/// propagate calibration metadata from the model.
///
struct ConvertMatmulToUnpackPattern
    : public OpRewritePattern<linalg::MatmulOp>
{
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const override
    {
        // --- Step 0: Operand extraction ----------------------------------------
        Value lhs = matmulOp.getDpsInputOperand(0)->get();    // activation
        Value weight = matmulOp.getDpsInputOperand(1)->get(); // weight (RHS)

        // --- Step 1: Check if weight is INT8-packed ----------------------------
        if (!isInt8PackedWeight(weight))
            return rewriter.notifyMatchFailure(matmulOp,
                                               "RHS weight is not INT8-packed");

        auto weightTy = cast<RankedTensorType>(weight.getType());
        if (weightTy.getRank() != 2)
            return rewriter.notifyMatchFailure(matmulOp,
                                               "expected rank-2 weight tensor");

        Location loc = matmulOp.getLoc();

        // --- Step 2: Types & shapes --------------------------------------------
        // linalg.matmul:  A[M,K] x B[K,N] -> C[M,N]
        // Weight shape: B = [K, N] i8
        int64_t dimK = weightTy.getDimSize(0);
        int64_t dimN = weightTy.getDimSize(1);

        auto i8Ty = rewriter.getIntegerType(8);
        auto f16Ty = rewriter.getF16Type();

        // Unpack & dequant keep the same spatial shape [K, N]
        RankedTensorType unpackedI8Ty =
            RankedTensorType::get({dimK, dimN}, i8Ty);
        RankedTensorType dequantFP16Ty =
            RankedTensorType::get({dimK, dimN}, f16Ty);

        // --- Step 3: Insert qf.unpack ------------------------------------------
        // Marks the i8 data for on-the-fly INT4 bit-extraction at lower levels.
        auto unpackOp = rewriter.create<UnpackOp>(loc, unpackedI8Ty, weight);

        // --- Step 4: Create default scale (1.0) and zero_point (0) constants ---
        auto scaleTy = RankedTensorType::get({}, f16Ty);
        auto zpTy = RankedTensorType::get({}, i8Ty);

        auto scaleAttr = DenseElementsAttr::get(
            scaleTy,
            APFloat(APFloat::IEEEhalf(), APInt(16, 0x3C00)));    // 1.0 in FP16
        auto zpAttr = DenseElementsAttr::get(zpTy, APInt(8, 0)); // zero_point = 0

        auto scaleConst =
            rewriter.create<arith::ConstantOp>(loc, scaleTy, scaleAttr);
        auto zpConst =
            rewriter.create<arith::ConstantOp>(loc, zpTy, zpAttr);

        // --- Step 5: Insert qf.dequant -----------------------------------------
        auto dequantOp = rewriter.create<DequantOp>(
            loc, dequantFP16Ty, unpackOp.getResult(), scaleConst.getResult(),
            zpConst.getResult());

        // --- Step 6: Cast LHS to FP16 if necessary -----------------------------
        Value lhsFP16 = lhs;
        auto lhsTy = cast<RankedTensorType>(lhs.getType());
        if (lhsTy.getElementType() != f16Ty)
        {
            auto lhsFP16Ty = RankedTensorType::get(lhsTy.getShape(), f16Ty);
            lhsFP16 = rewriter.create<arith::SIToFPOp>(loc, lhsFP16Ty, lhs);
        }

        // --- Step 7: Create new FP16 output init (same shape as original) ------
        Value origInit = matmulOp.getDpsInitOperand(0)->get();
        auto origInitTy = cast<RankedTensorType>(origInit.getType());
        auto newInitTy = RankedTensorType::get(origInitTy.getShape(), f16Ty);

        Value emptyTensor =
            rewriter.create<tensor::EmptyOp>(loc, newInitTy.getShape(), f16Ty);
        auto zeroF16 = rewriter.create<arith::ConstantOp>(
            loc, f16Ty, rewriter.getF16FloatAttr(0.0));
        Value filledInit =
            rewriter
                .create<linalg::FillOp>(loc, ValueRange{zeroF16},
                                        ValueRange{emptyTensor})
                .getResult(0);

        // --- Step 8: Build new linalg.matmul in FP16 ---------------------------
        auto newMatmul = rewriter.create<linalg::MatmulOp>(
            loc, TypeRange{newInitTy},
            ValueRange{lhsFP16, dequantOp.getResult()}, ValueRange{filledInit});

        // --- Step 9: Replace original op ----------------------------------------
        rewriter.replaceOp(matmulOp, newMatmul.getResults());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ConvertLinalgToQuantForge Pass Definition
//===----------------------------------------------------------------------===//

namespace
{
    struct ConvertLinalgToQuantForgePass
        : public PassWrapper<ConvertLinalgToQuantForgePass,
                             OperationPass<func::FuncOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToQuantForgePass)

        StringRef getArgument() const final
        {
            return "convert-linalg-to-quantforge";
        }

        StringRef getDescription() const final
        {
            return "Convert linalg.matmul with INT8-packed weights to "
                   "qf.unpack + qf.dequant + linalg.matmul(FP16)";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<quantforge::QuantForgeDialect,
                            arith::ArithDialect,
                            linalg::LinalgDialect,
                            tensor::TensorDialect>();
        }

        void runOnOperation() override
        {
            RewritePatternSet patterns(&getContext());
            patterns.add<ConvertMatmulToUnpackPattern>(&getContext());

            if (failed(
                    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
                signalPassFailure();
        }
    };
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createConvertLinalgToQuantForgePass()
{
    return std::make_unique<ConvertLinalgToQuantForgePass>();
}

void mlir::quantforge::registerConvertLinalgToQuantForgePass()
{
    PassRegistration<ConvertLinalgToQuantForgePass>();
}
