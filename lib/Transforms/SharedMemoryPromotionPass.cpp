//===----------------------------------------------------------------------===//
// SharedMemoryPromotionPass — Task 4.2 + 4.4: HBM → SRAM Promotion
//
// GPU does not automatically cache HBM into SRAM efficiently for matmul.
// This pass explicitly promotes tile-sized memref.subview reads from
// Global Memory (HBM) into Shared Memory (SRAM) allocations within
// K-reduction loops.
//
// Algorithm:
//   1. Walk scf.for loops inside scf.forall (K-reduction loops)
//   2. Find memref.subview ops that extract tiles matching --tile-m/n/k
//   3. For each qualifying subview:
//      a) Allocate SRAM: memref.alloc with memory_space = 3 (workgroup)
//      b) Copy:          memref.copy from HBM subview → SRAM
//      c) Barrier:       gpu.barrier after copy (RAW hazard prevention)
//      d) Replace:       all uses of HBM subview → SRAM alloc
//      e) Barrier:       gpu.barrier before yield (WAR hazard prevention)
//      f) Dealloc:       memref.dealloc for MLIR liveness analysis
//
// Memory space convention:
//   0 = Global (HBM) — default
//   3 = Shared (SRAM) — #gpu.address_space<workgroup>
//
// Pipeline position:
//   quantforge-bufferize → **quantforge-smem-promotion** → quantforge-swizzle-load
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-smem-promotion"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return true if the given memref type has the default (global/HBM) memory
/// space (either no memory space attribute, or memory space = 0).
static bool isGlobalMemorySpace(MemRefType memrefTy) {
  Attribute space = memrefTy.getMemorySpace();
  if (!space)
    return true; // default = global
  if (auto intAttr = dyn_cast<IntegerAttr>(space))
    return intAttr.getInt() == 0;
  // gpu::AddressSpaceAttr with Global
  if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(space))
    return gpuSpace.getValue() == gpu::AddressSpace::Global;
  return false;
}

/// Return true if the memref.subview result shape matches one of the expected
/// tile dimensions for promotion.
static bool isTileSizedSubview(memref::SubViewOp subview,
                               ArrayRef<int64_t> tileSizesM,
                               ArrayRef<int64_t> tileSizesN,
                               int64_t tileK) {
  MemRefType resultTy = subview.getResult().getType();

  // We only handle 2D tiles.
  if (resultTy.getRank() != 2)
    return false;

  ArrayRef<int64_t> shape = resultTy.getShape();
  int64_t dimM = shape[0];
  int64_t dimN = shape[1];

  // Check if it's a tile A [M×K] or tile B [K×N] shape
  for (int64_t m : tileSizesM) {
    if (dimM == m && dimN == tileK)
      return true; // Tile A: [blockM × blockK] or [warpM × blockK]
  }
  for (int64_t n : tileSizesN) {
    if (dimM == tileK && dimN == n)
      return true; // Tile B: [blockK × blockN] or [blockK × warpN]
  }

  return false;
}

/// Check if any gpu.barrier already exists immediately after the given op.
static bool hasBarrierAfter(Operation *op) {
  Operation *next = op->getNextNode();
  return next && isa<gpu::BarrierOp>(next);
}

/// Check if any gpu.barrier already exists immediately before the given op.
static bool hasBarrierBefore(Operation *op) {
  Operation *prev = op->getPrevNode();
  return prev && isa<gpu::BarrierOp>(prev);
}

//===----------------------------------------------------------------------===//
// SharedMemoryPromotionPass
//===----------------------------------------------------------------------===//

struct SharedMemoryPromotionPass
    : public PassWrapper<SharedMemoryPromotionPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SharedMemoryPromotionPass)

  //------------------------------------------------------------------
  // Constructors
  //------------------------------------------------------------------
  SharedMemoryPromotionPass() = default;

  SharedMemoryPromotionPass(const SharedMemoryPromotionPass &other)
      : PassWrapper<SharedMemoryPromotionPass,
                    OperationPass<func::FuncOp>>(other) {
    blockTileM.setValue(other.blockTileM.getValue());
    blockTileN.setValue(other.blockTileN.getValue());
    blockTileK.setValue(other.blockTileK.getValue());
    warpTileM.setValue(other.warpTileM.getValue());
    warpTileN.setValue(other.warpTileN.getValue());
    smemSpace.setValue(other.smemSpace.getValue());
  }

  //------------------------------------------------------------------
  // Pass metadata
  //------------------------------------------------------------------
  StringRef getArgument() const override {
    return "quantforge-smem-promotion";
  }

  StringRef getDescription() const override {
    return "Promote tile-sized HBM memref.subview reads into Shared Memory "
           "(SRAM) allocations with gpu.barrier synchronization and "
           "memref.dealloc for liveness analysis.";
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
  Option<int64_t> smemSpace{
      *this, "smem-space",
      llvm::cl::desc("Memory space integer for shared memory (default 3)"),
      llvm::cl::init(3)};

  //------------------------------------------------------------------
  // Dependent dialects
  //------------------------------------------------------------------
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, gpu::GPUDialect,
                    scf::SCFDialect, arith::ArithDialect>();
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Valid tile sizes for promotion detection
    SmallVector<int64_t> tileSizesM = {blockTileM, warpTileM};
    SmallVector<int64_t> tileSizesN = {blockTileN, warpTileN};

    // Memory space attribute for SRAM
    auto smemSpaceAttr =
        gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    IRRewriter rewriter(ctx);
    int64_t promotedCount = 0;

    // ── Walk K-reduction loops ──────────────────────────────────────
    funcOp.walk([&](scf::ForOp kLoop) {
      // Only process K-loops that are inside a scf.forall (block grid).
      if (!kLoop->getParentOfType<scf::ForallOp>())
        return;

      Block &kBody = *kLoop.getBody();
      Location loc = kLoop.getLoc();

      // Collect subview ops in the K-loop body that qualify for promotion.
      SmallVector<memref::SubViewOp> subviewsToPromote;
      for (Operation &op : kBody.without_terminator()) {
        auto subview = dyn_cast<memref::SubViewOp>(op);
        if (!subview)
          continue;

        MemRefType srcTy = subview.getSource().getType();
        if (!isGlobalMemorySpace(srcTy))
          continue;

        if (isTileSizedSubview(subview, tileSizesM, tileSizesN, blockTileK))
          subviewsToPromote.push_back(subview);
      }

      if (subviewsToPromote.empty())
        return;

      LLVM_DEBUG(llvm::dbgs()
                 << "[SmemPromotion] found " << subviewsToPromote.size()
                 << " subview(s) to promote in K-loop\n");

      // ── Promote each subview ────────────────────────────────────
      SmallVector<Value> sramAllocs;
      Operation *lastStoreOp = nullptr;

      for (memref::SubViewOp subview : subviewsToPromote) {
        MemRefType hbmTy = subview.getResult().getType();
        ArrayRef<int64_t> shape = hbmTy.getShape();
        Type elemTy = hbmTy.getElementType();

        // 1. Create SRAM memref type with shared memory space
        auto sramTy =
            MemRefType::get(shape, elemTy, AffineMap(), smemSpaceAttr);

        // 2. Allocate SRAM buffer (insert right after the subview)
        rewriter.setInsertionPointAfter(subview);
        auto sramAlloc = rewriter.create<memref::AllocOp>(loc, sramTy);
        sramAllocs.push_back(sramAlloc.getResult());

        // 3. Explicit copy loop: HBM → SRAM
        //    CRITICAL: We must NOT use memref.copy here because it writes
        //    linearly. The downstream SwizzleLoadPass will XOR-swizzle the
        //    memref.store indices (write to SRAM) AND the memref.load
        //    indices (read from SRAM), ensuring both sides are consistent.
        //    If we used memref.copy, the write would be linear but the
        //    read would be swizzled, causing DATA CORRUPTION.
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value dimM  = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
        Value dimN  = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
        Value one  = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        // Outer loop: rows
        auto outerLoop =
            rewriter.create<scf::ForOp>(loc, zero, dimM, one);
        rewriter.setInsertionPointToStart(outerLoop.getBody());
        Value rowIV = outerLoop.getInductionVar();

        // Inner loop: columns
        auto innerLoop =
            rewriter.create<scf::ForOp>(loc, zero, dimN, one);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        Value colIV = innerLoop.getInductionVar();

        // Load from HBM subview (linear addressing)
        Value val = rewriter.create<memref::LoadOp>(
            loc, subview.getResult(), ValueRange{rowIV, colIV});

        // Store to SRAM (linear here, but SwizzleLoadPass will XOR
        // the column index of this store op in a later pass)
        auto storeOp = rewriter.create<memref::StoreOp>(
            loc, val, sramAlloc.getResult(), ValueRange{rowIV, colIV});
        lastStoreOp = storeOp;

        // Move insertion point after the outer loop for the next subview
        rewriter.setInsertionPointAfter(outerLoop);

        // 4. Replace all uses of HBM subview with the SRAM alloc
        //    (the load from HBM inside the copy loop is excluded via
        //    subview → sramAlloc replacement not affecting the copy loop,
        //    since the loop uses subview.getResult() directly)
        subview.getResult().replaceUsesWithIf(
            sramAlloc.getResult(), [&](OpOperand &use) {
              // Don't replace uses inside the copy loops we just created
              Operation *user = use.getOwner();
              return !outerLoop->isProperAncestor(user);
            });

        LLVM_DEBUG(llvm::dbgs()
                   << "[SmemPromotion] promoted subview "
                   << shape[0] << "x" << shape[1] << " to SRAM\n");
        ++promotedCount;
      }

      // ── Insert gpu.barrier after all copy loops (RAW hazard) ────
      if (lastStoreOp) {
        // Find the outermost scf.for that contains lastStoreOp
        Operation *outerLoop = lastStoreOp->getParentOp()->getParentOp();
        if (!hasBarrierAfter(outerLoop)) {
          rewriter.setInsertionPointAfter(outerLoop);
          rewriter.create<gpu::BarrierOp>(loc);
        }
      }

      // ── Insert gpu.barrier before yield (WAR hazard) ────────────
      Operation *terminator = kBody.getTerminator();
      if (!hasBarrierBefore(terminator)) {
        rewriter.setInsertionPoint(terminator);
        rewriter.create<gpu::BarrierOp>(loc);
      }

      // ── Insert memref.dealloc for liveness analysis ─────────────
      // Place deallocs just before the WAR barrier (or before yield).
      Operation *warBarrier = terminator->getPrevNode();
      for (Value sram : sramAllocs) {
        rewriter.setInsertionPoint(warBarrier);
        rewriter.create<memref::DeallocOp>(loc, sram);
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "[SmemPromotion] promoted " << promotedCount
                            << " subview(s) to shared memory\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createSharedMemoryPromotionPass() {
  return std::make_unique<SharedMemoryPromotionPass>();
}

void mlir::quantforge::registerSharedMemoryPromotionPass() {
  PassRegistration<SharedMemoryPromotionPass>();
}
