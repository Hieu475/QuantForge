//===----------------------------------------------------------------------===//
// TilingPass — Task 2.2: Tile linalg ops for GPU shared-memory blocking
//
// Splits a 4096×4096 matrix operation into 128×128 tiles by wrapping
// linalg.matmul (and linalg.generic) operations with scf.for loops.
//
// Strategy:
//   • Walk the FuncOp to collect all linalg.LinalgOp instances.
//   • For each, call scf::tileUsingSCFForOp with tile sizes [M, N, K].
//   • Default tile sizes: M=128, N=128, K=64  (configurable via options).
//   • Tile size 0 means "do not tile that dimension" (used to skip K for
//     purely parallel generic ops that lack a reduction dimension).
//
// Output IR structure (matmul example):
//   scf.for %m = 0 to 4096 step 128 {          // M tiles
//     scf.for %n = 0 to 4096 step 128 {        // N tiles
//       scf.for %k = 0 to 4096 step 64 {       // K reduction tiles
//         linalg.matmul on 128×64 × 64×128 slices
//       }
//     }
//   }
//
// Memory estimate (Ampere SM, 164 KB shared memory):
//   Tile A [128×64] × f16 = 16 KB
//   Tile B [64×128] × f16 = 16 KB
//   Total  = 32 KB  (well within 164 KB limit)
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-tiling"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace
{

//===----------------------------------------------------------------------===//
// TilingPass
//===----------------------------------------------------------------------===//

struct TilingPass
    : public PassWrapper<TilingPass, OperationPass<func::FuncOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)

    //------------------------------------------------------------------
    // Constructors
    //------------------------------------------------------------------
    TilingPass() = default;

    /// Required explicit copy constructor because Option<> is non-copyable.
    /// The MLIR pass infrastructure calls this when cloning a pipeline.
    TilingPass(const TilingPass &other)
        : PassWrapper<TilingPass, OperationPass<func::FuncOp>>(other)
    {
        blockTileM.setValue(other.blockTileM.getValue());
        blockTileN.setValue(other.blockTileN.getValue());
        
        warpTileM.setValue(other.warpTileM.getValue());
        warpTileN.setValue(other.warpTileN.getValue());
        
        threadTileM.setValue(other.threadTileM.getValue());
        threadTileN.setValue(other.threadTileN.getValue());
        threadTileK.setValue(other.threadTileK.getValue());
    }

    //------------------------------------------------------------------
    // Pass metadata
    //------------------------------------------------------------------
    StringRef getArgument() const override { return "quantforge-tiling"; }

    StringRef getDescription() const override
    {
        return "Tile linalg.matmul / linalg.generic operations into "
               "128×128 (×64) tiles using scf.for loops for GPU "
               "shared-memory blocking (Task 2.2).";
    }

    //------------------------------------------------------------------
    // Configurable tile sizes (command-line options)
    //------------------------------------------------------------------

    Option<int64_t> blockTileM{*this, "block-tile-m", llvm::cl::desc("Block tile size M"), llvm::cl::init(128)};
    Option<int64_t> blockTileN{*this, "block-tile-n", llvm::cl::desc("Block tile size N"), llvm::cl::init(128)};
    
    Option<int64_t> warpTileM{*this, "warp-tile-m", llvm::cl::desc("Warp tile size M"), llvm::cl::init(64)};
    Option<int64_t> warpTileN{*this, "warp-tile-n", llvm::cl::desc("Warp tile size N"), llvm::cl::init(64)};
    
    Option<int64_t> threadTileM{*this, "thread-tile-m", llvm::cl::desc("Thread tile size M"), llvm::cl::init(16)};
    Option<int64_t> threadTileN{*this, "thread-tile-n", llvm::cl::desc("Thread tile size N"), llvm::cl::init(8)};
    Option<int64_t> threadTileK{*this, "thread-tile-k", llvm::cl::desc("Reduction tile size K"), llvm::cl::init(64)};

    //------------------------------------------------------------------
    // Dependent dialects
    //------------------------------------------------------------------
    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<scf::SCFDialect, linalg::LinalgDialect,
                        func::FuncDialect>();
        linalg::registerTilingInterfaceExternalModels(registry);
    }

    // Helper to create OpFoldResult from integer sizes
    SmallVector<OpFoldResult> getAsFoldResult(OpBuilder &b, ArrayRef<int64_t> sizes, unsigned rank)
    {
        SmallVector<OpFoldResult> res;
        for (unsigned i = 0; i < std::min<unsigned>(rank, sizes.size()); ++i)
        {
            res.push_back(b.getIndexAttr(sizes[i]));
        }
        return res;
    }

    //------------------------------------------------------------------
    // runOnOperation
    //------------------------------------------------------------------
    void runOnOperation() override
    {
        func::FuncOp funcOp = getOperation();
        MLIRContext *ctx     = &getContext();

        // ── Collect ops to tile ────────────────────────────────────────
        SmallVector<linalg::LinalgOp> worklist;
        funcOp.walk([&](linalg::LinalgOp op)
        {
            // Only tile ops that implement TilingInterface
            if (isa<TilingInterface>(op.getOperation()))
                worklist.push_back(op);
        });

        if (worklist.empty())
        {
            LLVM_DEBUG(llvm::dbgs() << "[TilingPass] no linalg ops found.\n");
            return;
        }

        IRRewriter rewriter(ctx);

        for (linalg::LinalgOp linalgOp : worklist)
        {
            unsigned iterRank = linalgOp.getNumLoops();

            rewriter.setInsertionPoint(linalgOp);
            auto tilingIface = cast<TilingInterface>(linalgOp.getOperation());

            // ── Level 1: Block Level (scf.forall) ──────────────────────
            scf::SCFTilingOptions blockOpts;
            blockOpts.setTileSizes(getAsFoldResult(rewriter, {blockTileM, blockTileN, 0}, iterRank));
            
            FailureOr<scf::SCFTilingResult> blockResult = 
                scf::tileUsingSCFForallOp(rewriter, tilingIface, blockOpts);
                
            if (failed(blockResult)) {
                linalgOp.emitWarning("Block level scf.forall tiling failed");
                continue;
            }
            
            // ── Level 2: Warp Level (scf.for) ───────────────────────
            auto blockTiledOp = cast<TilingInterface>(blockResult->tiledOps.back());
            scf::SCFTilingOptions warpOpts;
            warpOpts.setTileSizes(getAsFoldResult(rewriter, {warpTileM, warpTileN, 0}, iterRank));
            
            FailureOr<scf::SCFTilingResult> warpResult = 
                scf::tileUsingSCFForOp(rewriter, blockTiledOp, warpOpts);
                
            if (failed(warpResult)) {
                blockTiledOp.emitWarning("Warp level scf.for tiling failed");
                continue;
            }

            // ── Level 3: Thread Level & K Reduction (scf.for) ──────────
            auto warpTiledOp = cast<TilingInterface>(warpResult->tiledOps.back());
            scf::SCFTilingOptions threadOpts;
            threadOpts.setTileSizes(getAsFoldResult(rewriter, {threadTileM, threadTileN, threadTileK}, iterRank));
            
            FailureOr<scf::SCFTilingResult> threadResult = 
                scf::tileUsingSCFForOp(rewriter, warpTiledOp, threadOpts);
                
            if (failed(threadResult)) {
                warpTiledOp.emitWarning("Thread level scf.for tiling failed");
                continue;
            }

            // ── Replacements (Innermost out to outermost) ──────────────
            rewriter.replaceOp(warpTiledOp, threadResult->replacements);
            rewriter.replaceOp(blockTiledOp, warpResult->replacements);
            rewriter.replaceOp(linalgOp, blockResult->replacements);
        }
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createTilingPass()
{
    return std::make_unique<TilingPass>();
}

void mlir::quantforge::registerTilingPass()
{
    PassRegistration<TilingPass>();
}
