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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;


/// Split-K: tileReductionUsingForall leaves `forall (z) { ... forall (m, n) { ... } }` over a partial-result tensor
/// P[M, N, S]. The launch lowering maps a single forall to the grid, so fold the two into `forall (m, n, z)`. Two more
/// things change on the way: (1) P becomes [S, M, N], so that the tile of a block is contiguous like an ordinary C tile
/// (the Tensor Core path does not recognise a tile with stride S along N), and the reduce is re-indexed accordingly;
/// (2) `zeroInit` false (the kernel overwrites its output) drops the zero fill, which would otherwise become a huge
/// host-side vector store. Blocks read / write exactly their own tile P[z, m, n].
static LogicalResult mergeSplitKForall(RewriterBase &rewriter, scf::ForallOp outer, linalg::GenericOp reduceOp,
                                       bool zeroInit) {
  if (outer.getRank() != 1 || outer.getNumResults() != 1)
    return failure();
  scf::ForallOp inner;
  for (scf::ForallOp f : outer.getBody()->getOps<scf::ForallOp>()) {
    if (inner)
      return failure();
    inner = f;
  }
  if (!inner || inner.getRank() != 2 || inner.getNumResults() != 1)
    return failure();
  auto outerYield = outer.getTerminator();
  if (std::distance(outerYield.getYieldingOps().begin(), outerYield.getYieldingOps().end()) != 1)
    return failure();
  auto outerInsert = dyn_cast<tensor::ParallelInsertSliceOp>(*outerYield.getYieldingOps().begin());
  auto innerYield = inner.getTerminator();
  if (!outerInsert || outerInsert.getSource() != inner.getResult(0) ||
      std::distance(innerYield.getYieldingOps().begin(), innerYield.getYieldingOps().end()) != 1)
    return failure();
  auto innerInsert = dyn_cast<tensor::ParallelInsertSliceOp>(*innerYield.getYieldingOps().begin());
  if (!innerInsert)
    return failure();
  for (Operation *user : inner.getResult(0).getUsers())
    if (user != outerInsert.getOperation())
      return failure();
  auto pType = dyn_cast<RankedTensorType>(outer.getOutputs()[0].getType());
  if (!pType || pType.getRank() != 3 || !pType.hasStaticShape() || reduceOp.getNumDpsInputs() != 1)
    return failure();
  const int64_t M = pType.getDimSize(0), N = pType.getDimSize(1), S = pType.getDimSize(2);

  Location loc = outer.getLoc();
  SmallVector<OpFoldResult> lbs = inner.getMixedLowerBound(), ubs = inner.getMixedUpperBound(),
                            steps = inner.getMixedStep();
  lbs.push_back(outer.getMixedLowerBound()[0]);
  ubs.push_back(outer.getMixedUpperBound()[0]);
  steps.push_back(outer.getMixedStep()[0]);

  // P' [S, M, N]
  rewriter.setInsertionPoint(outer);
  Value pNew = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{S, M, N}, pType.getElementType());
  if (zeroInit) {
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(pType.getElementType()));
    pNew = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{pNew}).getResult(0);
  }
  auto merged = rewriter.create<scf::ForallOp>(loc, lbs, ubs, steps, ValueRange{pNew}, std::nullopt);
  Value p = merged.getRegionOutArgs()[0];
  Value z = merged.getInductionVars()[2];

  IRMapping map;
  map.map(outer.getInductionVars()[0], z);
  rewriter.setInsertionPoint(merged.getTerminator());
  for (Operation &op : outer.getBody()->without_terminator()) {
    if (&op == inner.getOperation())
      continue;
    // the whole-slice view of the old P: the new one is P'[z, :, :]
    if (auto ex = dyn_cast<tensor::ExtractSliceOp>(&op); ex && ex.getSource() == outer.getRegionOutArgs()[0]) {
      auto view = rewriter.create<tensor::ExtractSliceOp>(
          op.getLoc(), ex.getType(), p, ArrayRef<OpFoldResult>{z, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(M), rewriter.getIndexAttr(N)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
      map.map(ex.getResult(), view.getResult());
      continue;
    }
    rewriter.clone(op, map);
  }
  map.map(inner.getInductionVars()[0], merged.getInductionVars()[0]);
  map.map(inner.getInductionVars()[1], merged.getInductionVars()[1]);
  map.map(inner.getRegionOutArgs()[0], map.lookup(inner.getOutputs()[0]));
  for (Operation &op : inner.getBody()->without_terminator())
    rewriter.clone(op, map);

  // A block reads its accumulator tile through the whole-slice view P'[z]; other blocks write other tiles of that slice in
  // parallel, so bufferization would copy the full [M, N] slice per block. Read the tile straight from P'.
  {
    Value init = map.lookup(inner.getOutputs()[0]);
    if (auto id = init.getDefiningOp<tensor::ExtractSliceOp>())
      if (id.getSourceType() == id.getType() && llvm::all_of(id.getMixedOffsets(), [](OpFoldResult o) {
            return isConstantIntValue(o, 0);
          })) {
        init = id.getSource();
        rewriter.replaceOp(id, init);
      }
    if (auto view = init.getDefiningOp<tensor::ExtractSliceOp>(); view && view.getSource() == p) {
      for (OpOperand &use : llvm::make_early_inc_range(view->getUses())) {
        auto ex = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
        if (!ex || ex.getSource() != view.getResult())
          continue;
        rewriter.setInsertionPoint(ex);
        SmallVector<OpFoldResult> o = {z}, sz = {rewriter.getIndexAttr(1)}, st = {rewriter.getIndexAttr(1)};
        for (OpFoldResult x : ex.getMixedOffsets()) o.push_back(x);
        for (OpFoldResult x : ex.getMixedSizes()) sz.push_back(x);
        for (OpFoldResult x : ex.getMixedStrides()) st.push_back(x);
        auto direct = rewriter.create<tensor::ExtractSliceOp>(ex.getLoc(), ex.getType(), p, o, sz, st);
        rewriter.replaceOp(ex, direct.getResult());
      }
      if (view->use_empty())
        rewriter.eraseOp(view);
    }
  }

  auto mapOfr = [&](OpFoldResult o) -> OpFoldResult {
    if (auto v = dyn_cast<Value>(o))
      return OpFoldResult(map.lookupOrDefault(v));
    return o;
  };
  SmallVector<OpFoldResult> offs = {z}, sizes = {rewriter.getIndexAttr(1)}, strides = {rewriter.getIndexAttr(1)};
  for (OpFoldResult o : innerInsert.getMixedOffsets())
    offs.push_back(mapOfr(o));
  for (OpFoldResult o : innerInsert.getMixedSizes())
    sizes.push_back(mapOfr(o));
  for (OpFoldResult o : innerInsert.getMixedStrides())
    strides.push_back(mapOfr(o));
  rewriter.setInsertionPointToStart(&merged.getTerminator().getRegion().front());
  rewriter.create<tensor::ParallelInsertSliceOp>(loc, map.lookupOrDefault(innerInsert.getSource()), p, offs, sizes,
                                                 strides);

  // The reduce of the partial results reads P'[s, m, n] now: (m, n, s) -> (s, m, n).
  Operation *oldFill = outer.getOutputs()[0].getDefiningOp();
  rewriter.modifyOpInPlace(reduceOp, [&] {
    SmallVector<AffineMap> maps = reduceOp.getIndexingMapsArray();
    MLIRContext *ctx = rewriter.getContext();
    maps[0] = AffineMap::get(3, 0, {getAffineDimExpr(2, ctx), getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx);
    reduceOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    reduceOp->setOperand(0, merged.getResult(0));
  });
  rewriter.eraseOp(outer);
  if (oldFill && oldFill->use_empty())
    rewriter.eraseOp(oldFill);
  return success();
}

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
    splitK.setValue(other.splitK.getValue());
    splitKInit.setValue(other.splitKInit.getValue());
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
  Option<int64_t> splitK{*this, "split-k",
                         llvm::cl::desc("Split the K reduction over this many "
                                        "scf.forall iterations (partial "
                                        "results + a final reduce); 1 = off"),
                         llvm::cl::init(1)};

  Option<bool> splitKInit{*this, "split-k-init",
                          llvm::cl::desc("split-k: zero-fill the partial results "
                                         "(needed unless the kernel overwrites "
                                         "its output)"),
                          llvm::cl::init(true)};

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
      // The fused-epilogue elementwise op (int8-dequant) is folded into the GEMM store by
      // quantforge-fuse-epilogue; it is not tiled.
      if (op->hasAttr("quantforge.no_tile"))
        return;
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

      // ── Level 0 (optional): split-K. The reduction is cut into `split-k`
      // chunks run by a scf.forall; every chunk writes its own partial result
      // and a linalg reduce adds them up. The tiled op inside the forall is
      // an ordinary (smaller-K) matmul, tiled by the levels below.
      scf::ForallOp splitLoop;
      linalg::GenericOp splitReduce;
      if (splitK > 1 && iterRank >= 3) {
        rewriter.setInsertionPoint(linalgOp);
        auto partial = dyn_cast<PartialReductionOpInterface>(linalgOp.getOperation());
        if (!partial) {
          linalgOp.emitWarning("split-k: op does not implement PartialReductionOpInterface");
        } else {
          SmallVector<OpFoldResult> numThreads(iterRank, rewriter.getIndexAttr(0));
          numThreads[2] = rewriter.getIndexAttr(splitK);
          FailureOr<linalg::ForallReductionTilingResult> split =
              linalg::tileReductionUsingForall(rewriter, partial, numThreads);
          if (failed(split)) {
            linalgOp.emitWarning("split-k: tileReductionUsingForall failed");
          } else {
            splitLoop = split->loops;
            splitReduce = dyn_cast_or_null<linalg::GenericOp>(split->mergeOp);
            if (splitReduce)
              splitReduce->setAttr("quantforge.no_vectorize", UnitAttr::get(ctx));
            linalgOp = cast<linalg::LinalgOp>(split->parallelTiledOp);
            iterRank = linalgOp.getNumLoops();
          }
        }
      }

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

      // The instruction-level M and N loops enumerate the (m16 × n8) MMA
      // tiles of a warp tile; they are distributed over the lanes of a warp
      // (threadIdx.x) by quantforge-thread-distribute. The K loop is a true
      // reduction and stays sequential.
      if (iterRank >= 2 && threadResult->loops.size() >= 2) {
        threadResult->loops[0]->setAttr("quantforge.lane_m",
                                        UnitAttr::get(ctx));
        threadResult->loops[1]->setAttr("quantforge.lane_n",
                                        UnitAttr::get(ctx));
      }

      // ── Replacements (Innermost out to outermost) ──────────────
      rewriter.replaceOp(l4Op, threadResult->replacements);
      rewriter.replaceOp(l3Op, warpResult->replacements);
      if (iterRank >= 3)
        rewriter.replaceOp(l2Op, reductionResult->replacements);
      rewriter.replaceOp(linalgOp, blockResult->replacements);
      if (splitLoop && !splitReduce)
        splitLoop.emitWarning("split-k: the reduce of the partial results is not a linalg.generic");
      else if (splitLoop && failed(mergeSplitKForall(rewriter, splitLoop, splitReduce, splitKInit)))
        splitLoop.emitWarning("split-k: could not merge the K forall with the block forall");
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
