//===----------------------------------------------------------------------===//
// RegisterLayoutAwareUnpack Pass — mma.sync Fragment Layout Optimization
//
// Rewrites the unpacking SCF loop's output indexing so that each GPU thread
// (warp lane) directly produces and holds the exact INT4 nibbles required by
// the mma.sync Tensor Core instruction — eliminating the need for costly
// shfl.sync (warp shuffle) instructions to redistribute data.
//
// Background — mma.sync Fragment Layout:
//   When nvgpu.mma.sync / mma.sync.aligned.m16n8k16 loads an operand (Matrix A),
//   it does NOT read a linear memory array. Instead, data is "distributed"
//   across the 32 threads of a warp in a specific "fragment layout":
//
//   For m16n8k16, Matrix A (16×16, row-major), each lane holds 4 elements:
//     lane t (0–31):
//       rows held: {t / 4, t / 4 + 8}
//       cols held: {(t % 4) * 2, (t % 4) * 2 + 1}
//
//   Fragment indices (0–3) map as:
//     frag[0] = A[row0, col0]
//     frag[1] = A[row0, col1]
//     frag[2] = A[row1, col0]
//     frag[3] = A[row1, col1]
//
//   If the unpacking loop fills a TEMPORARY buffer linearly (idx = chunk*8+n),
//   the Tensor Core will need extra shfl.sync instructions at runtime to
//   redistribute data to the correct lanes — expensive warp-level communication.
//
// This pass:
//   1. Detects unpack SCF loops marked with the "mma_consumer" attribute
//      (set by the user or a preceding analysis pass) to indicate that the
//      unpacked output feeds into an mma.sync op.
//   2. Reads the tile shape (M, N, K) from attached attributes.
//   3. Looks up the hardcoded fragment mapping table for the given tile shape.
//   4. Rewrites tensor.insert indices from linear to fragment-layout positions.
//   5. Also emits gpu.thread_id to compute per-lane offsets at runtime.
//
// Status: SKELETON — Currently supports only m16n8k16 with hardcoded mapping.
//   Full support requires:
//   - MLIRGPUDialect (gpu.thread_id for threadIdx.x)
//   - nvgpu or gpu.launch wrapper (to ensure we're inside a kernel)
//   - Dynamic tile-shape dispatch (multiple mma tile sizes)
//
// How to trigger:
//   Annotate the intended qf.unpack op or its containing scf.for with:
//     {mma_consumer, mma_m = 16 : i64, mma_n = 8 : i64, mma_k = 16 : i64}
//
//   Then run: quantforge-opt --register-layout-aware-unpack
//
// Pipeline position:
//   Run AFTER LowerUnpackToNVVM (or LowerUnpackToPRMT),
//   BEFORE bufferization and GPU mapping.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "register-layout-aware-unpack"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Fragment Layout Tables
//
// These tables encode which elements each warp lane must hold for a given
// mma.sync tile configuration.  Derived from NVIDIA PTX ISA documentation,
// Table 91 ("mma.sync.aligned.m16n8k16.f16 operand A layout").
//
// Layout format per lane t (0–31):
//   row0 = t / 4          (integer division)
//   row1 = t / 4 + 8
//   col0 = (t % 4) * 2
//   col1 = (t % 4) * 2 + 1
//
// For fragment index f (0–3):
//   f=0: element at (row0, col0)
//   f=1: element at (row0, col1)
//   f=2: element at (row1, col0)
//   f=3: element at (row1, col1)
//===----------------------------------------------------------------------===//

struct FragmentMapping
{
    int mmaM, mmaN, mmaK;
    int elemsPerLane; // fragments per lane for matrix A
};

/// Hardcoded fragment layout for m16n8k16, Matrix A operand.
/// Returns the linear element index (in the 16×16 matrix A tile) that
/// fragment slot `fragIdx` (0–3) of lane `lane` (0–31) should hold.
static int64_t getFragmentLinearIndex_m16n8k16(int lane, int fragIdx)
{
    // Each lane t holds: rows {t/4, t/4+8}, cols {(t%4)*2, (t%4)*2+1}
    int row0 = lane / 4;
    int row1 = lane / 4 + 8;
    int col0 = (lane % 4) * 2;
    int col1 = (lane % 4) * 2 + 1;

    // mmaK = 16, so matrix A is (mmaM=16) × (mmaK=16)
    // Linear index in A[16][16]: row * 16 + col
    switch (fragIdx)
    {
    case 0: return row0 * 16 + col0;
    case 1: return row0 * 16 + col1;
    case 2: return row1 * 16 + col0;
    case 3: return row1 * 16 + col1;
    default: return -1;
    }
}

//===----------------------------------------------------------------------===//
// RegisterLayoutAwareUnpackPass implementation
//===----------------------------------------------------------------------===//

namespace
{

    struct RegisterLayoutAwareUnpackPass
        : public PassWrapper<RegisterLayoutAwareUnpackPass,
                             OperationPass<func::FuncOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegisterLayoutAwareUnpackPass)

        StringRef getArgument() const override
        {
            return "register-layout-aware-unpack";
        }
        StringRef getDescription() const override
        {
            return "Rewrite unpack SCF loop output indices to match mma.sync "
                   "fragment layout, eliminating shfl.sync warp shuffles. "
                   "Requires 'mma_consumer' attribute on the outer scf.for. "
                   "Currently supports m16n8k16 (hardcoded mapping).";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<arith::ArithDialect, scf::SCFDialect,
                            tensor::TensorDialect, gpu::GPUDialect>();
        }

        void runOnOperation() override
        {
            func::FuncOp funcOp  = getOperation();
            MLIRContext  *ctx    = &getContext();
            OpBuilder     builder(ctx);

            // Walk all outer scf.for loops looking for "mma_consumer" attribute
            funcOp.walk([&](scf::ForOp outerFor) -> WalkResult
            {
                // ── Check for mma_consumer marker ────────────────────────────
                if (!outerFor->hasAttr("mma_consumer"))
                    return WalkResult::skip();

                // ── Check for already-transformed marker ─────────────────────
                if (outerFor->hasAttr("layout_aware"))
                    return WalkResult::skip();

                // ── Read tile configuration from attributes ───────────────────
                int64_t mmaM = 16, mmaN = 8, mmaK = 16; // defaults

                if (auto mAttr = outerFor->getAttrOfType<IntegerAttr>("mma_m"))
                    mmaM = mAttr.getInt();
                if (auto nAttr = outerFor->getAttrOfType<IntegerAttr>("mma_n"))
                    mmaN = nAttr.getInt();
                if (auto kAttr = outerFor->getAttrOfType<IntegerAttr>("mma_k"))
                    mmaK = kAttr.getInt();

                // ── Currently only support m16n8k16 ──────────────────────────
                if (mmaM != 16 || mmaN != 8 || mmaK != 16)
                {
                    LLVM_DEBUG(llvm::dbgs()
                               << "RegisterLayoutAwareUnpack: unsupported tile "
                               << mmaM << "x" << mmaN << "x" << mmaK
                               << " — skipping\n");
                    return WalkResult::skip();
                }

                Location loc = outerFor.getLoc();

                LLVM_DEBUG(llvm::dbgs()
                           << "RegisterLayoutAwareUnpack: transforming loop at "
                           << loc << " for m16n8k16\n");

                // ── Emit threadIdx.x ─────────────────────────────────────────
                // Must be inside a gpu.launch; here we emit gpu::ThreadIdOp.
                // The op is placed at the beginning of the function (hoisted).
                builder.setInsertionPointToStart(&funcOp.getBody().front());

                Value tidX = builder.create<gpu::ThreadIdOp>(
                    loc, builder.getIndexType(), gpu::Dimension::x);

                // lane = tidX % 32 (warp lane ID)
                Value c32  = builder.create<arith::ConstantIndexOp>(loc, 32);
                Value lane = builder.create<arith::RemUIOp>(loc, tidX, c32);

                // ── Find inner scf.for and collect tensor.insert ops ─────────
                scf::ForOp innerFor;
                outerFor.getBody()->walk([&](scf::ForOp f)
                {
                    if (!innerFor) innerFor = f;
                });

                if (!innerFor)
                    return WalkResult::skip();

                // Collect all tensor.insert ops in inner body
                SmallVector<tensor::InsertOp> insertOps;
                innerFor.getBody()->walk([&](tensor::InsertOp ins)
                {
                    if (ins.getIndices().size() == 2)
                        insertOps.push_back(ins);
                });

                if (insertOps.size() != 8)
                {
                    LLVM_DEBUG(llvm::dbgs()
                               << "RegisterLayoutAwareUnpack: expected 8 tensor.insert "
                               << "ops, found " << insertOps.size() << " — skipping\n");
                    return WalkResult::skip();
                }

                // ── Rewrite output indices to fragment layout ─────────────────
                // The 8 nibble inserts (0–7) need to be remapped so that
                // each thread stores nibbles at the fragment positions it owns.
                //
                // For m16n8k16 Matrix A: 4 fragments per lane.
                // But we have 8 nibbles because we process 2 consecutive columns.
                // Convention: inserts[0..3] = fragment indices 0..3 for chunk tile 0
                //             inserts[4..7] = fragment indices 0..3 for chunk tile 1
                //
                // Fragment linear index = getFragmentLinearIndex_m16n8k16(lane, frag)
                // We compute this at runtime using the formula:
                //   row0 = lane / 4     col0 = (lane % 4) * 2
                //   row1 = lane/4 + 8   col1 = (lane % 4) * 2 + 1

                // Build per-lane row/col values (index type)
                builder.setInsertionPointToStart(innerFor.getBody());

                Value c4  = builder.create<arith::ConstantIndexOp>(loc,  4);
                Value c2  = builder.create<arith::ConstantIndexOp>(loc,  2);
                Value c8  = builder.create<arith::ConstantIndexOp>(loc,  8);
                Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);

                // row0 = lane / 4
                Value row0 = builder.create<arith::DivUIOp>(loc, lane, c4);
                // row1 = row0 + 8
                Value row1 = builder.create<arith::AddIOp>(loc, row0, c8);
                // lane_mod4 = lane % 4
                Value lane_mod4 = builder.create<arith::RemUIOp>(loc, lane, c4);
                // col0 = lane_mod4 * 2
                Value col0 = builder.create<arith::MulIOp>(loc, lane_mod4, c2);
                // col1 = col0 + 1
                Value c1impl  = builder.create<arith::ConstantIndexOp>(loc, 1);
                Value col1 = builder.create<arith::AddIOp>(loc, col0, c1impl);

                // Fragment → (row, col) pairs
                Value fragRows[4] = {row0, row0, row1, row1};
                Value fragCols[4] = {col0, col1, col0, col1};

                // Rewrite 8 inserts:
                //   inserts[f] (f=0..3): linearIdx = fragRows[f] * mmaK + fragCols[f]
                //   inserts[f+4] (f=0..3): linearIdx = fragRows[f] * mmaK + fragCols[f]
                //                          + mmaK/2 (shift by half-tile columns)
                for (int f = 0; f < 4; ++f)
                {
                    for (int tile = 0; tile < 2; ++tile)
                    {
                        int insertIdx = tile * 4 + f;
                        if (insertIdx >= static_cast<int>(insertOps.size()))
                            break;

                        tensor::InsertOp ins = insertOps[insertIdx];
                        OpBuilder::InsertionGuard guard(builder);
                        builder.setInsertionPoint(ins);
                        Location insertLoc = ins.getLoc();

                        // linearIdx = fragRows[f] * mmaK + fragCols[f]
                        Value rowIdx = fragRows[f];
                        Value colIdx = fragCols[f];

                        // For tile 1, offset columns by mmaK/2 = 8
                        if (tile == 1)
                        {
                            Value colOffset = builder.create<arith::ConstantIndexOp>(
                                insertLoc, mmaK / 2);
                            colIdx = builder.create<arith::AddIOp>(
                                insertLoc, colIdx, colOffset);
                        }

                        Value linearIdx = builder.create<arith::AddIOp>(
                            insertLoc,
                            builder.create<arith::MulIOp>(insertLoc, rowIdx, c16),
                            colIdx);

                        // Rewrite both indices
                        ins.getIndicesMutable()[0].assign(rowIdx);
                        ins.getIndicesMutable()[1].assign(linearIdx);

                        LLVM_DEBUG(llvm::dbgs()
                                   << "RegisterLayoutAwareUnpack: rewrite insert["
                                   << insertIdx << "] to fragment layout\n");
                    }
                }

                // ── Mark loop as transformed ──────────────────────────────────
                outerFor->setAttr("layout_aware", builder.getUnitAttr());
                outerFor->removeAttr("mma_consumer"); // consumed

                return WalkResult::advance();
            });
        }
    };

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createRegisterLayoutAwareUnpackPass()
{
    return std::make_unique<RegisterLayoutAwareUnpackPass>();
}

void mlir::quantforge::registerRegisterLayoutAwareUnpackPass()
{
    PassRegistration<RegisterLayoutAwareUnpackPass>();
}
