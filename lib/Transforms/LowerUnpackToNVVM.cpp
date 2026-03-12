//===----------------------------------------------------------------------===//
// LowerUnpackToNVVM Pass — Phase 2: i32-Chunk SCF Lowering
//
// Direct lowering of qf.unpack (doubling-shape, 2-D, static N divisible by 4)
// to SCF loops that read four packed bytes at once, reconstruct a 32-bit word,
// and extract all eight nibbles with CONSTANT shift amounts.
//
// This pass replaces linalg.generic entirely with explicit scf.for loops and
// scalar tensor.extract / tensor.insert ops, expressing the exact memory-access
// and bit-manipulation patterns required by GPU PTX code generation:
//
//   ld.global.u32  →  load four INT8 bytes (one i32 chunk)  ← 1 mem transaction
//   shr.u32 + and.b32  ×8  →  shift 0,4,8,…,28 (immediate) + mask 0xF
//   st.global.u8   ×8  →  store eight nibble bytes
//
// The eight extractions use STATIC (compile-time) shift immediates (0,4,…,28),
// so the PTX backend emits "shr.u32 reg, 4" etc. — no branch, no index op.
//
// On Ampere / Ada with warp-level coalescing, one warp (32 threads) reading 32
// consecutive i32 chunks covers 128 bytes of packed INT4 = 256 unpacked INT4.
//
// Guards:
//   • rank must be exactly 2 (K × N weight matrix)
//   • inShape[1] (N) must be static and divisible by 4
//   • doubling-shape only (same-shape marker is skipped)
//
// Generated IR sketch (static K × N, N%4==0):
//
//   %out = tensor.empty() : tensor<K × 2N × i8>
//
//   %r = scf.for %k = 0 to K step 1 iter_args(%out_k = %out) {
//     %r2 = scf.for %ch = 0 to N/4 step 1 iter_args(%cur = %out_k) {
//       // Read 4 packed bytes as one conceptual i32
//       %b0 = tensor.extract %packed[%k, %ch*4+0]
//       %b1 = tensor.extract %packed[%k, %ch*4+1]
//       %b2 = tensor.extract %packed[%k, %ch*4+2]
//       %b3 = tensor.extract %packed[%k, %ch*4+3]
//       // Pack to i32: b0 | (b1<<8) | (b2<<16) | (b3<<24)
//       %w = arith.ori arith.ori arith.ori ... : i32
//       // Extract 8 nibbles with CONSTANT shifts
//       %n0 = trunci (andi  %w,       0xF)   // shift 0  (PTX: and.b32)
//       %n1 = trunci (andi (shrui %w,  4), 0xF)  // shift 4
//       …
//       %n7 = trunci (andi (shrui %w, 28), 0xF)  // shift 28
//       // Write 8 nibbles
//       %t0 = tensor.insert %n0 into %cur[%k, %ch*8+0]
//       …
//       scf.yield %t7
//     }
//     scf.yield %r2
//   }
//
// Phase 3 direction (not implemented here):
//   Replace tensor.extract/insert with vector.transfer_read/write after
//   bufferization to emit ld.global.v4.u8 + st.global.v4.u8 for full
//   coalesced GPU loads; integrate dequant into the same loop body to
//   feed mma.sync Tensor Core ops without materialising an intermediate
//   f16 tensor.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lower-unpack-to-nvvm"

#include "QuantForge/Transforms/Passes.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantforge;

//===----------------------------------------------------------------------===//
// UnpackNVVMPattern
//===----------------------------------------------------------------------===//

namespace
{

    /// Matches qf.unpack (2-D, doubling-shape, static N divisible by 4) and
    /// lowers it to nested scf.for loops that process one i32 chunk per inner
    /// iteration.  The result is PTX-ready: eight constant-shift extractions
    /// per chunk map directly to "shr.u32 / and.b32" PTX instructions.
    struct UnpackNVVMPattern : public OpRewritePattern<UnpackOp>
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

            // ── Guards ────────────────────────────────────────────────────────
            if (!inputTy || !outputTy)
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "expected ranked tensor types");

            if (!inputTy.getElementType().isInteger(8))
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "expected i8 element type");

            if (inputTy.getRank() != 2)
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "only 2-D tensors supported");

            ArrayRef<int64_t> inShape = inputTy.getShape();
            ArrayRef<int64_t> outShape = outputTy.getShape();

            // Must be doubling-shape (skip same-shape semantic marker)
            if (inShape[1] == outShape[1])
                return rewriter.notifyMatchFailure(unpackOp,
                                                   "same-shape mode — skip");

            if (!ShapedType::isDynamic(inShape[1]) &&
                !ShapedType::isDynamic(outShape[1]) &&
                outShape[1] != 2 * inShape[1])
                return rewriter.notifyMatchFailure(
                    unpackOp, "output dim[1] != 2 * input dim[1]");

            // N (packed last dim) must be static and divisible by 4
            if (ShapedType::isDynamic(inShape[1]))
                return rewriter.notifyMatchFailure(
                    unpackOp, "dynamic N not supported by this pattern");

            if (inShape[1] % 4 != 0)
                return rewriter.notifyMatchFailure(
                    unpackOp, "N must be divisible by 4 for i32-chunk lowering");

            // ── Derived constants ─────────────────────────────────────────────
            int64_t N = inShape[1];
            int64_t N4 = N / 4; // number of i32 chunks per row
            int64_t N2 = N * 2; // output last dim

            auto i8Ty = rewriter.getIntegerType(8);
            auto i32Ty = rewriter.getIntegerType(32);
            Value packed = unpackOp.getInput();

            // ── All scalar constants (created OUTSIDE loops for CSE) ──────────
            Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            Value c4 = rewriter.create<arith::ConstantIndexOp>(loc, 4);
            Value c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);
            Value mask = rewriter.create<arith::ConstantIntOp>(loc, 0xF, 32);

            // Pack-shift constants used to reconstruct the i32 from four i8s
            Value psh8 = rewriter.create<arith::ConstantIntOp>(loc, 8, 32);
            Value psh16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 32);
            Value psh24 = rewriter.create<arith::ConstantIntOp>(loc, 24, 32);

            // Nibble-extract: shifts 0,4,8,12,16,20,24,28 (constant immediates)
            // shift[0] = 0 is never used (we skip the shrui for nibble 0).
            int nibbleShiftAmounts[8] = {0, 4, 8, 12, 16, 20, 24, 28};
            SmallVector<Value> nibbleShifts(8);
            for (int s = 1; s < 8; ++s)
                nibbleShifts[s] = rewriter.create<arith::ConstantIntOp>(
                    loc, nibbleShiftAmounts[s], 32);

            // Output-index offsets 0..7 within an 8-nibble block
            SmallVector<Value> outOffsets(8);
            for (int o = 0; o < 8; ++o)
                outOffsets[o] =
                    rewriter.create<arith::ConstantIndexOp>(loc, o);

            // ── K upper bound (static or dynamic) ────────────────────────────
            Value kUpper = ShapedType::isDynamic(inShape[0])
                               ? rewriter.create<tensor::DimOp>(loc, packed, 0)
                                     .getResult()
                               : rewriter.create<arith::ConstantIndexOp>(
                                             loc, inShape[0])
                                     .getResult();

            Value n4Val = rewriter.create<arith::ConstantIndexOp>(loc, N4);

            // ── Output tensor ─────────────────────────────────────────────────
            SmallVector<int64_t> outStaticShape = {inShape[0], N2};
            SmallVector<Value> dynOutSizes;
            if (ShapedType::isDynamic(inShape[0]))
                dynOutSizes.push_back(
                    rewriter.create<tensor::DimOp>(loc, packed, 0));

            Value outInit = rewriter.create<tensor::EmptyOp>(
                loc, outStaticShape, i8Ty, dynOutSizes);

            // ── Outer scf.for over rows (%k) ──────────────────────────────────
            auto outerFor = rewriter.create<scf::ForOp>(
                loc, c0, kUpper, c1, SmallVector<Value>{outInit});

            {
                OpBuilder::InsertionGuard outerGuard(rewriter);
                rewriter.setInsertionPointToStart(outerFor.getBody());

                Value k = outerFor.getInductionVar();
                Value outerArg = outerFor.getRegionIterArgs()[0];

                // ── Inner scf.for over i32 chunks (%chunk) ──────────────────
                auto innerFor = rewriter.create<scf::ForOp>(
                    loc, c0, n4Val, c1, SmallVector<Value>{outerArg});

                {
                    OpBuilder::InsertionGuard innerGuard(rewriter);
                    rewriter.setInsertionPointToStart(innerFor.getBody());

                    Value chunk = innerFor.getInductionVar();
                    Value curTens = innerFor.getRegionIterArgs()[0];

                    // Input base index: chunk * 4
                    Value base = rewriter.create<arith::MulIOp>(loc, chunk, c4);

                    // ── Read 4 consecutive i8 bytes ──────────────────────────
                    auto readByte = [&](Value extraOffset) -> Value
                    {
                        Value idx = rewriter.create<arith::AddIOp>(
                            loc, base, extraOffset);
                        return rewriter.create<tensor::ExtractOp>(
                            loc, packed, SmallVector<Value>{k, idx});
                    };
                    Value b0 = readByte(outOffsets[0]); // outOffsets[0..3] = 0..3
                    Value b1 = readByte(outOffsets[1]);
                    Value b2 = readByte(outOffsets[2]);
                    Value b3 = readByte(outOffsets[3]);

                    // ── Pack four i8 → one i32: b0|(b1<<8)|(b2<<16)|(b3<<24) ─
                    auto ext32 = [&](Value v) -> Value
                    {
                        return rewriter.create<arith::ExtUIOp>(loc, i32Ty, v);
                    };
                    Value b1_sh = rewriter.create<arith::ShLIOp>(
                        loc, ext32(b1), psh8);
                    Value b2_sh = rewriter.create<arith::ShLIOp>(
                        loc, ext32(b2), psh16);
                    Value b3_sh = rewriter.create<arith::ShLIOp>(
                        loc, ext32(b3), psh24);

                    Value acc = rewriter.create<arith::OrIOp>(
                        loc, ext32(b0), b1_sh);
                    acc = rewriter.create<arith::OrIOp>(loc, acc, b2_sh);
                    Value chunk32 = rewriter.create<arith::OrIOp>(
                        loc, acc, b3_sh);

                    // ── Extract 8 nibbles with CONSTANT shifts ────────────────
                    // nibble[n] = trunci( (chunk32 >> nibbleShiftAmounts[n]) & 0xF )
                    // PTX:  shr.u32 %r, %chunk32, <imm>  +  and.b32 %r, %r, 0xF
                    SmallVector<Value> nibbles(8);
                    for (int n = 0; n < 8; ++n)
                    {
                        Value shifted = chunk32;
                        if (n > 0)
                            shifted = rewriter.create<arith::ShRUIOp>(
                                loc, chunk32, nibbleShifts[n]);
                        Value masked =
                            rewriter.create<arith::AndIOp>(loc, shifted, mask);
                        nibbles[n] =
                            rewriter.create<arith::TruncIOp>(loc, i8Ty, masked);
                    }

                    // ── Write 8 nibbles via chained tensor.insert ─────────────
                    // Output positions: chunk*8 + 0, chunk*8 + 1, …, chunk*8 + 7
                    Value outBase =
                        rewriter.create<arith::MulIOp>(loc, chunk, c8);

                    Value updated = curTens;
                    for (int n = 0; n < 8; ++n)
                    {
                        Value outIdx = rewriter.create<arith::AddIOp>(
                            loc, outBase, outOffsets[n]);
                        updated = rewriter.create<tensor::InsertOp>(
                            loc, nibbles[n], updated,
                            SmallVector<Value>{k, outIdx});
                    }

                    rewriter.create<scf::YieldOp>(
                        loc, SmallVector<Value>{updated});
                }

                // Outer yield: pass the inner-loop result to the next row
                rewriter.setInsertionPointAfter(innerFor);
                rewriter.create<scf::YieldOp>(
                    loc, SmallVector<Value>{innerFor.getResult(0)});
            }

            rewriter.replaceOp(unpackOp, outerFor.getResult(0));
            return success();
        }
    };

    //===----------------------------------------------------------------------===//
    // LowerUnpackToNVVMPass
    //===----------------------------------------------------------------------===//

    struct LowerUnpackToNVVMPass
        : public PassWrapper<LowerUnpackToNVVMPass, OperationPass<func::FuncOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerUnpackToNVVMPass)

        StringRef getArgument() const override
        {
            return "lower-unpack-to-nvvm";
        }
        StringRef getDescription() const override
        {
            return "Lower qf.unpack to SCF loops processing one i32 chunk "
                   "(4 packed bytes = 8 INT4 nibbles) per iteration with "
                   "constant-shift extractions — PTX-ready IR (Phase 2)";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<arith::ArithDialect, scf::SCFDialect,
                            tensor::TensorDialect>();
        }

        void runOnOperation() override
        {
            MLIRContext *ctx = &getContext();
            RewritePatternSet patterns(ctx);
            patterns.add<UnpackNVVMPattern>(ctx);
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
mlir::quantforge::createLowerUnpackToNVVMPass()
{
    return std::make_unique<LowerUnpackToNVVMPass>();
}

void mlir::quantforge::registerLowerUnpackToNVVMPass()
{
    PassRegistration<LowerUnpackToNVVMPass>();
}
