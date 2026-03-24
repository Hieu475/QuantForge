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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
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

  /// Find the first scf.forall in the function (block-level).
  scf::ForallOp findBlockForall(func::FuncOp funcOp) {
    scf::ForallOp result;
    funcOp.walk([&](scf::ForallOp op) {
      if (!result)
        result = op;
      return result ? WalkResult::interrupt() : WalkResult::advance();
    });
    return result;
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    Location loc = funcOp.getLoc();

    // ── Step 1: Find block-level scf.forall ──────────────────────
    scf::ForallOp blockForall = findBlockForall(funcOp);
    if (!blockForall) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[GPUMappingPass] no block-level scf.forall found\n");
      return;
    }

    // ── Step 2: Annotate block forall with GPU mapping attributes ─
    // dim0 (%m_blk) → gpu.block_id.y
    // dim1 (%n_blk) → gpu.block_id.x
    SmallVector<Attribute> blockMapping = {
        gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimY),
        gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimX)};
    blockForall.setMappingAttr(ArrayAttr::get(ctx, blockMapping));

    LLVM_DEBUG(llvm::dbgs()
               << "[GPUMappingPass] annotated block forall with "
                  "GPUBlockMappingAttr [DimY, DimX]\n");

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

    // ── Step 4: Insert gpu.barrier in K-loop ────────────────────
    // Walk all scf.for ops that are directly inside the blockForall body.
    // The first scf.for is the K-reduction loop.
    IRRewriter rewriter(ctx);
    bool barrierInserted = false;

    blockForall.walk([&](scf::ForOp forOp) {
      // Only process the outermost scf.for in the forall body (the K-loop).
      // Its parent region should be the blockForall's body.
      if (barrierInserted)
        return;

      // Check that this for loop is directly parented by the blockForall
      Operation *parent = forOp->getParentOp();
      if (parent != blockForall.getOperation())
        return;

      Block &kBody = *forOp.getBody();

      // Barrier 1: at start of K-loop body (after load, before compute)
      rewriter.setInsertionPointToStart(&kBody);
      // Skip past the IV block argument — insert after existing ops
      // that are part of the "load phase". For now, just insert at start.
      rewriter.create<gpu::BarrierOp>(loc);

      // Barrier 2: at end of K-loop body (after compute, before yield)
      Operation *terminator = kBody.getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.create<gpu::BarrierOp>(loc);

      barrierInserted = true;
      LLVM_DEBUG(llvm::dbgs()
                 << "[GPUMappingPass] inserted 2 gpu.barrier ops in K-loop\n");
    });

    if (!barrierInserted) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[GPUMappingPass] warning: no K-loop found for barrier insertion\n");
    }

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
