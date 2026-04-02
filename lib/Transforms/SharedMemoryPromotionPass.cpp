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
//      b) Copy:          nvgpu.device_async_copy from HBM subview → SRAM
//                       in vectorized chunks (16-byte path for f16)
//      c) Sync:          nvgpu.device_async_create_group +
//                       nvgpu.device_async_wait (RAW hazard prevention)
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
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
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

/// Check if any gpu.barrier already exists immediately before the given op.
static bool hasBarrierBefore(Operation *op) {
  Operation *prev = op->getPrevNode();
  return prev && isa<gpu::BarrierOp>(prev);
}

/// Clone a subview and replace K induction variable occurrences in offsets
/// with `newK`. Returns empty op if no K-dependent offset exists.
static memref::SubViewOp cloneSubviewWithKReplacement(IRRewriter &rewriter,
                                                      memref::SubViewOp base,
                                                      Value oldK,
                                                      Value newK) {
  SmallVector<OpFoldResult> mixedOffsets = base.getMixedOffsets();
  SmallVector<OpFoldResult> offsets(mixedOffsets.begin(), mixedOffsets.end());
  bool replaced = false;
  for (OpFoldResult &ofr : offsets) {
    if (auto v = dyn_cast<Value>(ofr); v && v == oldK) {
      ofr = newK;
      replaced = true;
    }
  }
  if (!replaced)
    return {};

  SmallVector<OpFoldResult> mixedSizes = base.getMixedSizes();
  SmallVector<OpFoldResult> sizes(mixedSizes.begin(), mixedSizes.end());
  SmallVector<OpFoldResult> mixedStrides = base.getMixedStrides();
  SmallVector<OpFoldResult> strides(mixedStrides.begin(), mixedStrides.end());

  return rewriter.create<memref::SubViewOp>(base.getLoc(), base.getType(),
                                             base.getSource(), offsets, sizes,
                                             strides);
}

/// Emit nested row/col async copy loops from `src` to `dst`.
static Operation *emitAsyncCopyLoopNest(IRRewriter &rewriter, Location loc,
                                        Value src, Value dst,
                                        ArrayRef<int64_t> shape,
                                        int64_t chunkElems,
                                        Type asyncTokenTy) {
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value dimM = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
  Value dimN = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
  Value colStep = rewriter.create<arith::ConstantIndexOp>(loc, chunkElems);

  auto outer = rewriter.create<scf::ForOp>(loc, zero, dimM, one);
  outer->setAttr("quantforge.sram_load", rewriter.getUnitAttr());
  outer->setAttr("quantforge.sram_copy_outer", rewriter.getUnitAttr());
  rewriter.setInsertionPointToStart(outer.getBody());
  Value rowIV = outer.getInductionVar();

  auto inner = rewriter.create<scf::ForOp>(loc, zero, dimN, colStep);
  inner->setAttr("quantforge.sram_load", rewriter.getUnitAttr());
  inner->setAttr("quantforge.sram_copy_inner", rewriter.getUnitAttr());
  rewriter.setInsertionPointToStart(inner.getBody());
  Value colIV = inner.getInductionVar();

  rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, asyncTokenTy, dst, ValueRange{rowIV, colIV}, src,
      ValueRange{rowIV, colIV}, rewriter.getIndexAttr(chunkElems), Value(),
      UnitAttr());

  rewriter.setInsertionPointAfter(outer);
  return outer;
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
           "(SRAM) allocations with nvgpu.device_async_copy synchronization and "
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
                    scf::SCFDialect, arith::ArithDialect,
                    nvgpu::NVGPUDialect>();
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

      // ── Promote each subview with ping-pong double buffering ───────────
      SmallVector<Value> sramAllocs;
      auto asyncTokenTy = nvgpu::DeviceAsyncTokenType::get(ctx);
      Value kLower = kLoop.getLowerBound();
      Value kUpper = kLoop.getUpperBound();
      Value kStep = kLoop.getStep();
      Value kIV = kLoop.getInductionVar();

      // Prolog insertion point: preload K=0 tile(s) into buffer[0].
      rewriter.setInsertionPoint(kLoop);

      for (memref::SubViewOp subview : subviewsToPromote) {
        // Always allocate in K-loop preheader so buffers dominate entire loop.
        rewriter.setInsertionPoint(kLoop);

        MemRefType hbmTy = subview.getResult().getType();
        ArrayRef<int64_t> shape = hbmTy.getShape();
        Type elemTy = hbmTy.getElementType();

        // 1. Create SRAM memref type with shared memory space
        auto sramTy =
            MemRefType::get(shape, elemTy, AffineMap(), smemSpaceAttr);

        // 2. Allocate double SRAM buffers outside the K-loop body.
        auto sramAlloc0 = rewriter.create<memref::AllocOp>(loc, sramTy);
        auto sramAlloc1 = rewriter.create<memref::AllocOp>(loc, sramTy);
        sramAllocs.push_back(sramAlloc0.getResult());
        sramAllocs.push_back(sramAlloc1.getResult());

        int64_t chunkElems = 1;
        if (elemTy.isF16() && shape[1] % 8 == 0)
          chunkElems = 8;

        // 3. Prolog preload: copy tile at K=lower_bound into buffer[0].
        auto prologSubview =
            cloneSubviewWithKReplacement(rewriter, subview, kIV, kLower);
        if (!prologSubview)
          continue;
        emitAsyncCopyLoopNest(rewriter, loc, prologSubview.getResult(),
                              sramAlloc0.getResult(), shape, chunkElems,
                              asyncTokenTy);

        // 4. Main-loop rewrite inside K body:
        //    - compute buffer = iter%2 ? buf1 : buf0
        //    - fetch next tile into opposite buffer
        //    - wait(1) to keep one async group in flight
        rewriter.setInsertionPointAfter(subview);

        Value kDelta = rewriter.create<arith::SubIOp>(loc, kIV, kLower);
        Value iterIdx = rewriter.create<arith::DivUIOp>(loc, kDelta, kStep);
        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value iterParity = rewriter.create<arith::RemUIOp>(loc, iterIdx, c2);
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value isEven =
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           iterParity, c0);

        auto curBufIf =
            rewriter.create<scf::IfOp>(loc, TypeRange{sramTy}, isEven,
                                       /*withElseRegion=*/true);
        {
          rewriter.setInsertionPointToStart(curBufIf.thenBlock());
          rewriter.create<scf::YieldOp>(loc, sramAlloc0.getResult());
          rewriter.setInsertionPointToStart(curBufIf.elseBlock());
          rewriter.create<scf::YieldOp>(loc, sramAlloc1.getResult());
        }
        rewriter.setInsertionPointAfter(curBufIf);

        auto nextBufIf =
            rewriter.create<scf::IfOp>(loc, TypeRange{sramTy}, isEven,
                                       /*withElseRegion=*/true);
        {
          rewriter.setInsertionPointToStart(nextBufIf.thenBlock());
          rewriter.create<scf::YieldOp>(loc, sramAlloc1.getResult());
          rewriter.setInsertionPointToStart(nextBufIf.elseBlock());
          rewriter.create<scf::YieldOp>(loc, sramAlloc0.getResult());
        }
        rewriter.setInsertionPointAfter(nextBufIf);

        Value kNext = rewriter.create<arith::AddIOp>(loc, kIV, kStep);
        Value hasNext = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, kNext, kUpper);

        auto fetchIf = rewriter.create<scf::IfOp>(loc, hasNext,
                                                   /*withElseRegion=*/false);
        {
          rewriter.setInsertionPointToStart(fetchIf.thenBlock());
          auto nextSubview =
              cloneSubviewWithKReplacement(rewriter, subview, kIV, kNext);
          if (nextSubview) {
            emitAsyncCopyLoopNest(rewriter, loc, nextSubview.getResult(),
                                  nextBufIf.getResult(0), shape, chunkElems,
                                  asyncTokenTy);
          }
        }

        // Compute for current iteration reads from current ping-pong buffer.
        subview.getResult().replaceUsesWithIf(
            curBufIf.getResult(0), [&](OpOperand &use) {
              Operation *user = use.getOwner();
              return !fetchIf->isProperAncestor(user);
            });

        LLVM_DEBUG(llvm::dbgs()
                   << "[SmemPromotion] ping-pong promoted subview "
                   << shape[0] << "x" << shape[1] << " to SRAM\n");
        ++promotedCount;
      }

      // Prolog wait(0): first tile must be ready before K-loop compute starts.
      rewriter.setInsertionPoint(kLoop);
      auto prologGroup = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
          loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange{});
      rewriter.create<nvgpu::DeviceAsyncWaitOp>(
          loc, prologGroup.getAsyncToken(), rewriter.getI32IntegerAttr(0));
        rewriter.create<gpu::BarrierOp>(loc);

      // Main-loop wait(1): keep one async group in flight.
      rewriter.setInsertionPoint(kBody.getTerminator());
      auto mainGroup = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
          loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange{});
      rewriter.create<nvgpu::DeviceAsyncWaitOp>(
          loc, mainGroup.getAsyncToken(), rewriter.getI32IntegerAttr(1));
        rewriter.create<gpu::BarrierOp>(loc);

      // Epilog wait(0): drain remaining in-flight copies.
      rewriter.setInsertionPointAfter(kLoop);
      auto epilogGroup = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
          loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange{});
      rewriter.create<nvgpu::DeviceAsyncWaitOp>(
          loc, epilogGroup.getAsyncToken(), rewriter.getI32IntegerAttr(0));

      // Deallocate ping-pong buffers after loop completion.
      for (Value sram : sramAllocs) {
        rewriter.setInsertionPointAfter(kLoop);
        rewriter.create<memref::DeallocOp>(loc, sram);
      }

      // ── Insert gpu.barrier before yield (WAR hazard) ────────────
      Operation *terminator = kBody.getTerminator();
      if (!hasBarrierBefore(terminator)) {
        rewriter.setInsertionPoint(terminator);
        rewriter.create<gpu::BarrierOp>(loc);
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
