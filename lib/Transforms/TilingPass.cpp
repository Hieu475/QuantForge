//===----------------------------------------------------------------------===//
// TilingPass — Task 2.2: Tile linalg ops for GPU shared-memory blocking
//
// Splits a 4096×4096 matrix operation into 128×128 tiles by wrapping
// linalg.matmul (and linalg.generic) operations with scf.for loops.
//
// Strategy:
//   • Walk the FuncOp to collect all linalg.LinalgOp instances.
//   • For each, apply a 4-level hierarchy:
//       1) Block grid        : scf.forall [blockM, blockN, 0]
//       2) Block reduction K : scf.for    [0, 0, blockK]
//       3) Warp distribution : scf.for    [warpM, warpN, 0]
//       4) MMA instruction   : scf.for    [threadM, threadN, threadK]
//   • Tile size 0 means "do not tile that dimension".
//
// Output IR structure (matmul example):
//   scf.forall (%m_blk, %n_blk) in (...) {     // Block grid tiles
//     scf.for %k_blk = 0 to 4096 step 64 {     // Cooperative K chunks
//       scf.for %m_warp = ... {                 // Warp distribution M
//         scf.for %n_warp = ... {               // Warp distribution N
//           scf.for %k = 0 to 64 step 16 {     // MMA K tile
//             scf.for %m = 0 to 64 step 16 {
//               scf.for %n = 0 to 64 step 8 {
//                 linalg.matmul on MMA-sized slices
//             }
//           }
//         }
//         }
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

//===----------------------------------------------------------------------===//
// TilingPass
//===----------------------------------------------------------------------===//

struct TilingPass
    : public PassWrapper<TilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass)

  //------------------------------------------------------------------
  // Constructors
  //------------------------------------------------------------------
  TilingPass() = default;

  /// Required explicit copy constructor because Option<> is non-copyable.
  /// The MLIR pass infrastructure calls this when cloning a pipeline.
  TilingPass(const TilingPass &other)
      : PassWrapper<TilingPass, OperationPass<func::FuncOp>>(other) {
    blockTileM.setValue(other.blockTileM.getValue());
    blockTileN.setValue(other.blockTileN.getValue());
    blockTileK.setValue(other.blockTileK.getValue());

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

  StringRef getDescription() const override {
    return "Tile linalg.matmul / linalg.generic operations into "
           "a 4-level hardware-aware hierarchy (block/"
           "reduction/warp/mma) using scf.forall + scf.for loops "
           "for GPU shared-memory + Tensor Core lowering (Task 2.2).";
  }

  //------------------------------------------------------------------
  // Configurable tile sizes (command-line options)
  //------------------------------------------------------------------

  Option<int64_t> blockTileM{*this, "block-tile-m",
                             llvm::cl::desc("Block tile size M"),
                             llvm::cl::init(128)};
  Option<int64_t> blockTileN{*this, "block-tile-n",
                             llvm::cl::desc("Block tile size N"),
                             llvm::cl::init(128)};
  Option<int64_t> blockTileK{
      *this, "block-tile-k",
      llvm::cl::desc("Block reduction size K (SRAM load size)"),
      llvm::cl::init(64)};

  Option<int64_t> warpTileM{*this, "warp-tile-m",
                            llvm::cl::desc("Warp tile size M"),
                            llvm::cl::init(64)};
  Option<int64_t> warpTileN{*this, "warp-tile-n",
                            llvm::cl::desc("Warp tile size N"),
                            llvm::cl::init(64)};

  Option<int64_t> threadTileM{*this, "thread-tile-m",
                              llvm::cl::desc("Thread tile size M"),
                              llvm::cl::init(16)};
  Option<int64_t> threadTileN{*this, "thread-tile-n",
                              llvm::cl::desc("Thread tile size N"),
                              llvm::cl::init(8)};
  Option<int64_t> threadTileK{*this, "thread-tile-k",
                              llvm::cl::desc("Tensor Core mma.sync size K"),
                              llvm::cl::init(16)};

  //------------------------------------------------------------------
  // Dependent dialects
  //------------------------------------------------------------------
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<scf::SCFDialect, linalg::LinalgDialect, func::FuncDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  // Helper to create OpFoldResult from integer sizes
  SmallVector<OpFoldResult>
  getAsFoldResult(OpBuilder &b, ArrayRef<int64_t> sizes, unsigned rank) {
    SmallVector<OpFoldResult> res;
    for (unsigned i = 0; i < std::min<unsigned>(rank, sizes.size()); ++i) {
      res.push_back(b.getIndexAttr(sizes[i]));
    }
    return res;
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ── Collect ops to tile ────────────────────────────────────────
    SmallVector<linalg::LinalgOp> worklist;
    funcOp.walk([&](linalg::LinalgOp op) {
      // Only tile ops that implement TilingInterface
      if (isa<TilingInterface>(op.getOperation()))
        worklist.push_back(op);
    });

    if (worklist.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[TilingPass] no linalg ops found.\n");
      return;
    }

    IRRewriter rewriter(ctx);

    for (linalg::LinalgOp linalgOp : worklist) {
      unsigned iterRank = linalgOp.getNumLoops();

      rewriter.setInsertionPoint(linalgOp);
      auto tilingIface = cast<TilingInterface>(linalgOp.getOperation());

      // ── Level 1: Block Level (Grid, scf.forall) ────────────────
      scf::SCFTilingOptions blockOpts;
      blockOpts.setTileSizes(
          getAsFoldResult(rewriter, {blockTileM, blockTileN, 0}, iterRank));

      FailureOr<scf::SCFTilingResult> blockResult =
          scf::tileUsingSCFForallOp(rewriter, tilingIface, blockOpts);

      if (failed(blockResult)) {
        linalgOp.emitWarning("Block level scf.forall tiling failed");
        continue;
      }

      // ── Level 2: Block Reduction (K loop, scf.for) ─────────────
      auto l2Op = cast<TilingInterface>(blockResult->tiledOps.front());
      FailureOr<scf::SCFTilingResult> reductionResult;
      TilingInterface l3Op = l2Op;

      // Only apply explicit reduction tiling if a K-loop exists.
      if (iterRank >= 3) {
        scf::SCFTilingOptions reductionOpts;
        reductionOpts.setTileSizes(
            getAsFoldResult(rewriter, {0, 0, blockTileK}, iterRank));

        reductionResult = scf::tileUsingSCFForOp(rewriter, l2Op, reductionOpts);

        if (failed(reductionResult)) {
          l2Op.emitWarning("Block reduction scf.for tiling failed");
          continue;
        }

        l3Op = cast<TilingInterface>(reductionResult->tiledOps.front());
      }

      // ── Level 3: Warp Level (scf.for) ────────────────────────────
      // Use scf.for instead of scf.forall to avoid dominance issues:
      // scf.forall creates an isolated region that cannot capture values
      // from the enclosing scf.for (K-loop) body.
      scf::SCFTilingOptions warpOpts;
      warpOpts.setTileSizes(
          getAsFoldResult(rewriter, {warpTileM, warpTileN, 0}, iterRank));

      FailureOr<scf::SCFTilingResult> warpResult =
          scf::tileUsingSCFForOp(rewriter, l3Op, warpOpts);

      if (failed(warpResult)) {
        l3Op.emitWarning("Warp level scf.for tiling failed");
        continue;
      }

      // ── Level 4: Instruction Level (scf.for) ───────────────────
      auto l4Op = cast<TilingInterface>(warpResult->tiledOps.front());
      scf::SCFTilingOptions threadOpts;
      threadOpts.setTileSizes(getAsFoldResult(
          rewriter, {threadTileM, threadTileN, threadTileK}, iterRank));

      FailureOr<scf::SCFTilingResult> threadResult =
          scf::tileUsingSCFForOp(rewriter, l4Op, threadOpts);

      if (failed(threadResult)) {
        l4Op.emitWarning("Instruction level scf.for tiling failed");
        continue;
      }

      // ── Replacements (Innermost out to outermost) ──────────────
      rewriter.replaceOp(l4Op, threadResult->replacements);
      rewriter.replaceOp(l3Op, warpResult->replacements);
      if (iterRank >= 3)
        rewriter.replaceOp(l2Op, reductionResult->replacements);
      rewriter.replaceOp(linalgOp, blockResult->replacements);
    }

    // Semantic tags for downstream synchronization pass:
    // - quantforge.sram_load: operations in the HBM->SRAM load phase
    // - quantforge.compute: first compute loop in each K-loop body
    funcOp.walk([&](scf::ForallOp blockForall) {
      if (blockForall->getParentOfType<scf::ForallOp>())
        return;

      for (Operation &op : blockForall.getBody()->without_terminator()) {
        auto kLoop = dyn_cast<scf::ForOp>(op);
        if (!kLoop)
          continue;

        bool seenCompute = false;
        bool taggedAnyLoad = false;
        Operation *lastPreCompute = nullptr;
        bool computeTagged = false;
        for (Operation &bodyOp : kLoop.getBody()->without_terminator()) {
          if (!seenCompute)
            lastPreCompute = &bodyOp;

          if (isa<tensor::ExtractSliceOp>(bodyOp)) {
            bodyOp.setAttr("quantforge.sram_load", UnitAttr::get(ctx));
            taggedAnyLoad = true;
            continue;
          }

          if (!computeTagged && isa<scf::ForOp>(bodyOp)) {
            bodyOp.setAttr("quantforge.compute", UnitAttr::get(ctx));
            computeTagged = true;
            seenCompute = true;
            continue;
          }

          seenCompute = seenCompute || computeTagged;
        }

        // Keep synchronization semantics robust even when canonicalization
        // changes the exact load ops in the K-loop body.
        if (!taggedAnyLoad && computeTagged && lastPreCompute)
          lastPreCompute->setAttr("quantforge.sram_load", UnitAttr::get(ctx));

        // Last-resort fallback for unusual loop shapes: still mark first loop
        // as compute so GPUMapping can place the boundary barrier.
        if (!computeTagged) {
          for (Operation &bodyOp : kLoop.getBody()->without_terminator()) {
            if (isa<scf::ForOp>(bodyOp)) {
              bodyOp.setAttr("quantforge.compute", UnitAttr::get(ctx));
              break;
            }
          }
        }
      }
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createTilingPass() {
  return std::make_unique<TilingPass>();
}

void mlir::quantforge::registerTilingPass() { PassRegistration<TilingPass>(); }
