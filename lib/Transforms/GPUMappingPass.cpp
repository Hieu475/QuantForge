//===----------------------------------------------------------------------===//
// GPUMappingPass — Annotate scf.forall with GPU mapping + insert barriers
//
// Takes the multi-level tiled IR produced by quantforge-tiling and:
//
//   1. Sets GPUBlockMappingAttr on the outer scf.forall (block grid).
//      These annotations are used by MLIR's ForallToGPU lowering to
//      generate gpu.launch / gpu.block_id / gpu.thread_id.
//
//   2. Inserts gpu.barrier at two positions in the K-loop (scf.for):
//      - After SRAM load position (before compute)
//      - After compute (before yield)
//
// This is a "Phase 4" annotation pass. The actual lowering from
// annotated scf.forall → gpu.launch is handled by the standard MLIR
// -map-forall-to-gpu pass (or equivalent) in a later pipeline stage
// after bufferization.
//
// Hardware strategy (embedded in attributes):
//   Block:  scf.forall [M_blk, N_blk] → GPUBlockMapping [DimY, DimX]
//   Lanes:  threadIdx.x = 0..31 reserved for tensor core
//
// Configurable block/warp tile sizes are recorded as string attributes
// on the function for downstream passes to consume.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-gpu-mapping"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

//===----------------------------------------------------------------------===//
// GPUMappingPass
//===----------------------------------------------------------------------===//

struct GPUMappingPass
    : public PassWrapper<GPUMappingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUMappingPass)

  //------------------------------------------------------------------
  // Constructors
  //------------------------------------------------------------------
  GPUMappingPass() = default;

  GPUMappingPass(const GPUMappingPass &other)
      : PassWrapper<GPUMappingPass, OperationPass<func::FuncOp>>(other) {
    blockTileM.setValue(other.blockTileM.getValue());
    blockTileN.setValue(other.blockTileN.getValue());
    warpTileM.setValue(other.warpTileM.getValue());
    warpTileN.setValue(other.warpTileN.getValue());
    numLanes.setValue(other.numLanes.getValue());
  }

  //------------------------------------------------------------------
  // Pass metadata
  //------------------------------------------------------------------
  StringRef getArgument() const override { return "quantforge-gpu-mapping"; }

  StringRef getDescription() const override {
    return "Annotate tiled scf.forall with GPU mapping attributes and "
           "insert gpu.barrier for shared-memory synchronization.";
  }

  //------------------------------------------------------------------
  // Configurable options
  //------------------------------------------------------------------
  Option<int64_t> blockTileM{*this, "block-tile-m",
                             llvm::cl::desc("Block tile size M"),
                             llvm::cl::init(128)};
  Option<int64_t> blockTileN{*this, "block-tile-n",
                             llvm::cl::desc("Block tile size N"),
                             llvm::cl::init(128)};
  Option<int64_t> warpTileM{*this, "warp-tile-m",
                            llvm::cl::desc("Warp tile size M"),
                            llvm::cl::init(64)};
  Option<int64_t> warpTileN{*this, "warp-tile-n",
                            llvm::cl::desc("Warp tile size N"),
                            llvm::cl::init(64)};
  Option<int64_t> numLanes{
      *this, "num-lanes",
      llvm::cl::desc("Threads per warp (tensor core lanes, threadIdx.x)"),
      llvm::cl::init(32)};

  //------------------------------------------------------------------
  // Dependent dialects
  //------------------------------------------------------------------
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  //------------------------------------------------------------------
  // Helpers
  //------------------------------------------------------------------

  /// Collect all top-level block foralls (not nested inside another forall).
  SmallVector<scf::ForallOp> collectTopLevelBlockForalls(func::FuncOp funcOp) {
    SmallVector<scf::ForallOp> result;
    funcOp.walk([&](scf::ForallOp op) {
      if (!op->getParentOfType<scf::ForallOp>())
        result.push_back(op);
      return WalkResult::advance();
    });
    return result;
  }

  static std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
    std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!lb || !ub || !step || *step <= 0 || *ub < *lb)
      return std::nullopt;
    return (*ub - *lb) / *step;
  }

  static bool hasUnitAttr(Operation *op, StringRef attrName) {
    return op->hasAttr(attrName);
  }

  static bool isSemanticSramLoad(Operation *op) {
    return hasUnitAttr(op, "quantforge.sram_load") ||
           isa<tensor::ExtractSliceOp>(op);
  }

  static bool isSemanticCompute(Operation *op) {
    return hasUnitAttr(op, "quantforge.compute") || isa<scf::ForOp>(op) ||
           isa<linalg::LinalgOp>(op);
  }

  static SmallVector<gpu::MappingId> getWarpMappingDimsForRank(unsigned rank) {
    if (rank == 0)
      return {};
    if (rank == 1)
      return {gpu::MappingId::DimY};
    if (rank == 2)
      return {gpu::MappingId::DimZ, gpu::MappingId::DimY};
    return {gpu::MappingId::DimZ, gpu::MappingId::DimY, gpu::MappingId::DimX};
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    Location loc = funcOp.getLoc();

    // ── Step 1: Collect top-level block scf.forall ────────────────
    SmallVector<scf::ForallOp> blockForalls =
        collectTopLevelBlockForalls(funcOp);
    if (blockForalls.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[GPUMappingPass] no block-level scf.forall found\n");
      return;
    }

    // ── Step 3: Record hardware config as function attributes ────
    // Downstream passes (e.g. ForallToGPU) can read these.
    int64_t numWarpsM = blockTileM / warpTileM; // e.g., 128/64 = 2
    int64_t numWarpsN = blockTileN / warpTileN; // e.g., 128/64 = 2
    int64_t blockSizeX = numLanes;
    int64_t blockSizeY = numWarpsN;
    int64_t blockSizeZ = numWarpsM;

    funcOp->setAttr("quantforge.block_size_x",
                    IntegerAttr::get(IndexType::get(ctx), blockSizeX));
    funcOp->setAttr("quantforge.block_size_y",
                    IntegerAttr::get(IndexType::get(ctx), blockSizeY));
    funcOp->setAttr("quantforge.block_size_z",
                    IntegerAttr::get(IndexType::get(ctx), blockSizeZ));

    LLVM_DEBUG(llvm::dbgs()
               << "[GPUMappingPass] block size: (" << blockSizeX << ", "
               << blockSizeY << ", " << blockSizeZ << ")\n");

    // ── Step 2: Annotate all block foralls with GPU block mapping ─
    // dim0 (%m_blk) → gpu.block_id.y
    // dim1 (%n_blk) → gpu.block_id.x
    SmallVector<Attribute> blockMapping = {
        gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimY),
        gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimX)};
    for (scf::ForallOp blockForall : blockForalls) {
      blockForall.setMappingAttr(ArrayAttr::get(ctx, blockMapping));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[GPUMappingPass] annotated " << blockForalls.size()
               << " block forall(s) with GPUBlockMappingAttr [DimY, DimX]\n");

    IRRewriter rewriter(ctx);
    int64_t insertedKLoopBarriers = 0;

    for (scf::ForallOp blockForall : blockForalls) {
      // ── Step 3: Insert gpu.barrier in K-loop(s) semantically ───
      // Process direct scf.for children of this block forall.
      SmallVector<scf::ForOp> kLoops;
      for (Operation &op : blockForall.getBody()->without_terminator()) {
        if (auto forOp = dyn_cast<scf::ForOp>(op))
          kLoops.push_back(forOp);
      }

      for (scf::ForOp kLoop : kLoops) {
        Block &kBody = *kLoop.getBody();

        // Map async-copy loop nests emitted by SharedMemoryPromotionPass.
        // This prevents each thread from executing the full sequential copy.
        kLoop.walk([&](scf::ForOp forOp) {
          if (hasUnitAttr(forOp, "quantforge.sram_copy_outer")) {
            forOp->setAttr(
                "mapping",
                ArrayAttr::get(ctx, {gpu::GPUThreadMappingAttr::get(
                                        ctx, gpu::MappingId::DimY)}));
          }
          if (hasUnitAttr(forOp, "quantforge.sram_copy_inner")) {
            forOp->setAttr(
                "mapping",
                ArrayAttr::get(ctx, {gpu::GPUThreadMappingAttr::get(
                                        ctx, gpu::MappingId::DimX)}));
          }
        });

        // Skip if barriers already exist to keep pass idempotent.
        bool hasBarrier = false;
        for (Operation &op : kBody.without_terminator()) {
          if (isa<gpu::BarrierOp>(op)) {
            hasBarrier = true;
            break;
          }
        }
        if (hasBarrier)
          continue;

        // Semantic synchronization boundary:
        //   1) last op of SRAM load phase (quantforge.sram_load)
        //   2) first op of compute phase (quantforge.compute)
        // If no tags exist in the loop, fallback to structural heuristics.
        Operation *lastLoadOp = nullptr;
        Operation *firstComputeOp = nullptr;
        bool hasTaggedLoad = false;
        bool hasTaggedCompute = false;
        for (Operation &op : kBody.without_terminator()) {
          hasTaggedLoad =
              hasTaggedLoad || hasUnitAttr(&op, "quantforge.sram_load");
          hasTaggedCompute =
              hasTaggedCompute || hasUnitAttr(&op, "quantforge.compute");
        }

        // Copy-only tagged regions are synchronized explicitly by
        // SharedMemoryPromotionPass via device_async_wait + gpu.barrier.
        // Skip extra barrier insertion here to avoid over-synchronization.
        if (hasTaggedLoad && !hasTaggedCompute)
          continue;

        const bool useTaggedSemantics = hasTaggedLoad || hasTaggedCompute;
        for (Operation &op : kBody.without_terminator()) {
          if (useTaggedSemantics) {
            if (hasUnitAttr(&op, "quantforge.sram_load"))
              lastLoadOp = &op;
            if (!firstComputeOp && hasUnitAttr(&op, "quantforge.compute"))
              firstComputeOp = &op;
            continue;
          }

          if (isSemanticSramLoad(&op))
            lastLoadOp = &op;
          if (!firstComputeOp && isSemanticCompute(&op))
            firstComputeOp = &op;
        }

        if (firstComputeOp) {
          rewriter.setInsertionPoint(firstComputeOp);
          rewriter.create<gpu::BarrierOp>(loc);
        } else if (lastLoadOp) {
          rewriter.setInsertionPointAfter(lastLoadOp);
          rewriter.create<gpu::BarrierOp>(loc);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "[GPUMappingPass] no semantic load/compute boundary "
                        "found in K-loop\n");
          continue;
        }

        Operation *terminator = kBody.getTerminator();
        rewriter.setInsertionPoint(terminator);
        rewriter.create<gpu::BarrierOp>(loc);
        ++insertedKLoopBarriers;
      }

      // ── Step 4: Annotate warp-level forall (Thread Mapping) ───
      bool warpForallAnnotated = false;
      blockForall.walk([&](scf::ForallOp warpForall) {
        if (warpForall == blockForall)
          return WalkResult::advance();

        unsigned rank = warpForall.getInductionVars().size();
        SmallVector<gpu::MappingId> dims = getWarpMappingDimsForRank(rank);
        if (dims.empty())
          return WalkResult::advance();

        SmallVector<Attribute> threadMapping;
        for (gpu::MappingId dim : dims)
          threadMapping.push_back(gpu::GPUThreadMappingAttr::get(ctx, dim));
        warpForall.setMappingAttr(ArrayAttr::get(ctx, threadMapping));
        warpForallAnnotated = true;
        return WalkResult::advance();
      });

      // ── Step 5: Stable fallback mapping for nested scf.for ─────
      // Only apply when static trip counts match the expected warp topology.
      if (!warpForallAnnotated && !kLoops.empty()) {
        for (scf::ForOp kLoop : kLoops) {
          Block &kBody = *kLoop.getBody();
          scf::ForOp outerWarpFor;

          for (Operation &op : kBody.without_terminator()) {
            if (auto forOp = dyn_cast<scf::ForOp>(op)) {
              if (hasUnitAttr(forOp, "quantforge.compute") || !outerWarpFor) {
                outerWarpFor = forOp;
                if (hasUnitAttr(forOp, "quantforge.compute"))
                  break;
              }
            }
          }
          if (!outerWarpFor)
            continue;

          SmallVector<scf::ForOp> loopChain;
          loopChain.push_back(outerWarpFor);
          scf::ForOp current = outerWarpFor;
          while (loopChain.size() < 3) {
            scf::ForOp nested;
            for (Operation &op : current.getBody()->without_terminator()) {
              if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                nested = forOp;
                break;
              }
            }
            if (!nested)
              break;
            loopChain.push_back(nested);
            current = nested;
          }

          SmallVector<gpu::MappingId> dims;
          if (loopChain.size() == 1) {
            auto t0 = getStaticTripCount(loopChain[0]);
            if (t0 && *t0 == numWarpsM * numWarpsN)
              dims = {gpu::MappingId::DimY};
          } else if (loopChain.size() >= 2) {
            auto t0 = getStaticTripCount(loopChain[0]);
            auto t1 = getStaticTripCount(loopChain[1]);
            if (t0 && t1 && *t0 == numWarpsM && *t1 == numWarpsN)
              dims = {gpu::MappingId::DimZ, gpu::MappingId::DimY};
            if (dims.empty() && loopChain.size() >= 3) {
              auto t2 = getStaticTripCount(loopChain[2]);
              if (t0 && t1 && t2 && *t0 == numWarpsM && *t1 == numWarpsN &&
                  *t2 == numLanes)
                dims = {gpu::MappingId::DimZ, gpu::MappingId::DimY,
                        gpu::MappingId::DimX};
            }
          }

          if (dims.empty())
            continue;

          for (auto [idx, dim] : llvm::enumerate(dims)) {
            loopChain[idx]->setAttr(
                "mapping",
                ArrayAttr::get(ctx,
                               {gpu::GPUThreadMappingAttr::get(ctx, dim)}));
          }
          break;
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[GPUMappingPass] inserted barriers in "
                            << insertedKLoopBarriers << " K-loop(s)\n");

    LLVM_DEBUG(llvm::dbgs() << "[GPUMappingPass] annotation complete\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createGPUMappingPass() {
  return std::make_unique<GPUMappingPass>();
}

void mlir::quantforge::registerGPUMappingPass() {
  PassRegistration<GPUMappingPass>();
}
