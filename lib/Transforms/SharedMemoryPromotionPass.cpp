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
//   2. Find memref.subview ops that extract READ-ONLY tiles matching
//      --tile-m/n/k (written tiles such as the C accumulator are skipped)
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
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <functional>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

/// Static shared memory per block that every sm_80+ GPU grants without the
/// dynamic-smem opt-in.
static constexpr int64_t kMaxStaticSmemBytes = 48 * 1024;

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

/// True if `value` (a tile) is only ever read. Promoting a tile that is also
/// written (e.g. the C accumulator, whose shape can coincide with a warp tile)
/// would redirect the writes into the SRAM copy and lose them: the pass copies
/// HBM → SRAM but never writes back.
static bool isReadOnlyTile(Value value) {
  for (Operation *user : value.getUsers()) {
    if (isa<memref::LoadOp, vector::TransferReadOp, vector::LoadOp,
            memref::DeallocOp>(user))
      continue;
    if (auto sub = dyn_cast<memref::SubViewOp>(user)) {
      if (!isReadOnlyTile(sub.getResult()))
        return false;
      continue;
    }
    return false; // store / transfer_write / copy / linalg / unknown
  }
  return true;
}

/// True if `v` is (a chain of subviews of) a function/kernel argument, i.e. a
/// buffer that already lives in HBM when the kernel starts. Buffers written
/// in-kernel or by a producer kernel (allocs) are not.
static bool isArgumentRooted(Value v) {
  while (auto sub = v.getDefiningOp<memref::SubViewOp>())
    v = sub.getSource();
  return isa<BlockArgument>(v);
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

/// Replace `oldVal` (a global-memory tile) by `newVal` (its SRAM copy) in all
/// users except those under `skip`. memref.subview results encode the source
/// layout and memory space in their type, so a subview of the promoted tile
/// has to be re-created with an inferred type instead of just swapping the
/// operand (that produces invalid IR: "different memory spaces specified for
/// base memref type and subview memref type").
static LogicalResult replaceAndRetype(Value oldVal, Value newVal,
                                      Operation *skip, IRRewriter &rewriter) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : oldVal.getUses())
    if (!skip || !skip->isProperAncestor(use.getOwner()))
      uses.push_back(&use);

  for (OpOperand *use : uses) {
    Operation *user = use->getOwner();
    if (auto sub = dyn_cast<memref::SubViewOp>(user)) {
      rewriter.setInsertionPoint(sub);
      auto newSrcTy = cast<MemRefType>(newVal.getType());
      auto newTy = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          sub.getType().getShape(), newSrcTy, sub.getMixedOffsets(),
          sub.getMixedSizes(), sub.getMixedStrides()));
      auto newSub = rewriter.create<memref::SubViewOp>(
          sub.getLoc(), newTy, newVal, sub.getMixedOffsets(),
          sub.getMixedSizes(), sub.getMixedStrides());
      Value oldRes = sub.getResult();
      if (failed(replaceAndRetype(oldRes, newSub.getResult(), skip, rewriter)))
        return failure();
      rewriter.eraseOp(sub);
      continue;
    }
    if (isa<memref::LoadOp, memref::StoreOp, vector::TransferReadOp,
            vector::TransferWriteOp, vector::LoadOp, vector::StoreOp>(user)) {
      use->set(newVal); // no result type depends on the memref type
      continue;
    }
    return user->emitError("cannot retarget user of promoted tile to shared "
                           "memory (unsupported op)");
  }
  return success();
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
    smemPad.setValue(other.smemPad.getValue());
    copyUnroll.setValue(other.copyUnroll.getValue());
    smemStages.setValue(other.smemStages.getValue());
    vectorCopy.setValue(other.vectorCopy.getValue());
    asyncCopy.setValue(other.asyncCopy.getValue());
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
  Option<int64_t> smemPad{
      *this, "smem-pad",
      llvm::cl::desc("Extra elements appended to every row of a promoted tile "
                     "(0 = none). Row stride 128 B maps all 8 rows of an "
                     "ldmatrix / column access onto the same banks; 8 f16 "
                     "elements (16 B) of padding removes the conflict. This "
                     "is a layout-preserving alternative to XOR swizzling: "
                     "every access goes through the same padded buffer."),
      llvm::cl::init(0)};
  Option<bool> vectorCopy{
      *this, "smem-vector-copy",
      llvm::cl::desc("Copy 16 bytes (8 x f16) per work item and write them to "
                     "shared memory with one vector.store (st.shared.v4) "
                     "instead of one scalar store per element"),
      llvm::cl::init(true)};
  Option<bool> asyncCopy{
      *this, "smem-async-copy",
      llvm::cl::desc("Use cp.async (nvgpu.device_async_copy, 16 B) for tiles "
                     "whose source is a kernel argument (no register "
                     "round-trip, no per-element index math). Tiles that are "
                     "produced in-kernel (fused dequantization) keep the "
                     "vector register path. Needs sm_80+"),
      llvm::cl::init(true)};
  Option<int64_t> smemStages{
      *this, "smem-stages",
      llvm::cl::desc("Number of shared-memory stages of the K loop. 1 = copy, "
                     "wait, compute (no overlap). 2 = double buffering: the "
                     "tile for iteration i+1 is copied (cp.async / register "
                     "path) while iteration i computes. Only applied inside "
                     "gpu.launch with a static K-loop trip count."),
      llvm::cl::init(1)};
  Option<int64_t> copyUnroll{
      *this, "copy-unroll",
      llvm::cl::desc("Unroll factor of the cooperative HBM->SRAM copy loop "
                     "(1 = no unrolling)"),
      llvm::cl::init(8)};
  Option<int64_t> smemSpace{
      *this, "smem-space",
      llvm::cl::desc("Memory space integer for shared memory (default 3)"),
      llvm::cl::init(3)};

  //------------------------------------------------------------------
  // Dependent dialects
  //------------------------------------------------------------------
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, gpu::GPUDialect, scf::SCFDialect,
                    arith::ArithDialect, vector::VectorDialect,
                    nvgpu::NVGPUDialect>();
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  struct CopyResult {
    scf::ForOp loop;
    Operation *store = nullptr;
    bool async = false;      // cp.async copies were emitted
    bool unrollable = false; // constant trip count: candidate for unrolling
  };

  /// Emit the cooperative copy of the global tile `src` into the shared-memory
  /// view `dst` at the rewriter's insertion point and leave the insertion
  /// point after the emitted loop. `srcArgRooted` selects cp.async (the tile
  /// lives in a kernel argument, not in a producer-written buffer).
  CopyResult emitCopy(IRRewriter &rewriter, Location loc, gpu::LaunchOp launch,
                      std::optional<int64_t> numThreads, Value src, Value dst,
                      bool srcArgRooted) {
    MLIRContext *ctx = rewriter.getContext();
    Value sramView = dst;
    auto hbmTy = cast<MemRefType>(src.getType());
    ArrayRef<int64_t> shape = hbmTy.getShape();
    Type elemTy = hbmTy.getElementType();
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value dimM  = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
        Value dimN  = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
        Value one  = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        scf::ForOp outerLoop;
        Operation *storeOp;
        CopyResult res;
        // 16-byte work items: W elements per item.
        int64_t elemBytes = elemTy.isIntOrFloat()
                                ? std::max<int64_t>(1, elemTy.getIntOrFloatBitWidth() / 8)
                                : 0;
        int64_t W = elemBytes ? 16 / elemBytes : 1;
        bool vecOK = numThreads && vectorCopy && W > 1 &&
                     shape[1] % W == 0 && smemPad % W == 0 &&
                     ((shape[0] * (shape[1] / W)) % *numThreads == 0);
        bool asyncOK = vecOK && asyncCopy && elemBytes == 2 &&
                       srcArgRooted;
        if (numThreads) {
          // Distributed copy: the linearised tile is split into work items
          // (one element, or W = 16 B worth of elements) handed out as
          //   item = tid + t * nthreads.
          // When the item count divides evenly by the thread count the trip
          // count of `t` is a compile-time constant, which lets us unroll so
          // the global loads of consecutive iterations are issued back to back
          // (a rolled loop serialises one HBM round-trip per element).
          gpu::KernelDim3 tids = launch.getThreadIds();
          gpu::KernelDim3 bs = launch.getBlockSize();
          Value linTid = rewriter.create<arith::AddIOp>(
              loc, tids.x,
              rewriter.create<arith::MulIOp>(
                  loc, bs.x,
                  rewriter.create<arith::AddIOp>(
                      loc, tids.y,
                      rewriter.create<arith::MulIOp>(loc, bs.y, tids.z))));
          Value stride =
              rewriter.create<arith::ConstantIndexOp>(loc, *numThreads);
          int64_t itemWidth = vecOK ? W : 1;
          int64_t totalItems = shape[0] * shape[1] / itemWidth;
          bool evenSplit = totalItems % *numThreads == 0;
          Value itemsPerRow =
              rewriter.create<arith::ConstantIndexOp>(loc, shape[1] / itemWidth);
          if (evenSplit) {
            Value trips = rewriter.create<arith::ConstantIndexOp>(
                loc, totalItems / *numThreads);
            outerLoop = rewriter.create<scf::ForOp>(loc, zero, trips, one);
            rewriter.setInsertionPointToStart(outerLoop.getBody());
          } else {
            Value total =
                rewriter.create<arith::ConstantIndexOp>(loc, totalItems);
            outerLoop = rewriter.create<scf::ForOp>(loc, linTid, total, stride);
            rewriter.setInsertionPointToStart(outerLoop.getBody());
          }
          Value idx = outerLoop.getInductionVar();
          if (evenSplit) {
            Value off = rewriter.create<arith::MulIOp>(loc, idx, stride);
            idx = rewriter.create<arith::AddIOp>(loc, linTid, off);
          }
          Value rowIV = rewriter.create<arith::DivUIOp>(loc, idx, itemsPerRow);
          Value colItem = rewriter.create<arith::RemUIOp>(loc, idx, itemsPerRow);
          if (asyncOK) {
            // cp.async: HBM -> SRAM directly, 16 B per thread-item.
            Value c0 = rewriter.create<arith::MulIOp>(
                loc, colItem, rewriter.create<arith::ConstantIndexOp>(loc, W));
            auto tokTy = nvgpu::DeviceAsyncTokenType::get(ctx);
            storeOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
                loc, tokTy, sramView, ValueRange{rowIV, c0},
                src, ValueRange{rowIV, c0},
                rewriter.getIndexAttr(W), Value(), UnitAttr());
            res.async = true;
          } else if (vecOK) {
            // 8 scalar loads (the producer of a fused tile is recomputed per
            // element later) packed into one vector and stored with a single
            // 128-bit st.shared.
            Value c0 = rewriter.create<arith::MulIOp>(
                loc, colItem, rewriter.create<arith::ConstantIndexOp>(loc, W));
            SmallVector<Value> elems;
            for (int64_t j = 0; j < W; ++j) {
              Value cj = j == 0 ? c0
                                : rewriter.create<arith::AddIOp>(
                                      loc, c0,
                                      rewriter.create<arith::ConstantIndexOp>(loc, j));
              elems.push_back(rewriter.create<memref::LoadOp>(
                  loc, src, ValueRange{rowIV, cj}));
            }
            // (vector.from_elements does not exist in MLIR 18.)
            auto vecTy = VectorType::get({W}, elemTy);
            Value vec = rewriter.create<arith::ConstantOp>(
                loc, vecTy, rewriter.getZeroAttr(vecTy));
            for (int64_t j = 0; j < W; ++j)
              vec = rewriter.create<vector::InsertOp>(loc, elems[j], vec,
                                                      ArrayRef<int64_t>{j});
            storeOp = rewriter.create<vector::StoreOp>(loc, vec, sramView,
                                                       ValueRange{rowIV, c0});
          } else {
            Value val = rewriter.create<memref::LoadOp>(
                loc, src, ValueRange{rowIV, colItem});
            storeOp = rewriter.create<memref::StoreOp>(
                loc, val, sramView, ValueRange{rowIV, colItem});
          }
          res.unrollable = evenSplit && copyUnroll > 1;
        } else {
          outerLoop = rewriter.create<scf::ForOp>(loc, zero, dimM, one);
          rewriter.setInsertionPointToStart(outerLoop.getBody());
          Value rowIV = outerLoop.getInductionVar();
          auto innerLoop = rewriter.create<scf::ForOp>(loc, zero, dimN, one);
          rewriter.setInsertionPointToStart(innerLoop.getBody());
          Value colIV = innerLoop.getInductionVar();
          Value val = rewriter.create<memref::LoadOp>(
              loc, src, ValueRange{rowIV, colIV});
          storeOp = rewriter.create<memref::StoreOp>(
              loc, val, sramView, ValueRange{rowIV, colIV});
        }

    res.loop = outerLoop;
    res.store = storeOp;
    rewriter.setInsertionPointAfter(outerLoop);
    return res;
  }

  //------------------------------------------------------------------
  // Double-buffered K loop
  //------------------------------------------------------------------
  struct PipeTile {
    memref::SubViewOp sv; // global tile of the current iteration
    MemRefType tileTy;    // [R x C]
    Value buf;            // [stages x R x (C+pad)] shared buffer
    bool argRooted;
  };

  /// The index computations inside the K-loop body that `sv` depends on, in
  /// topological order. Fails if any of them has side effects / regions (they
  /// could not be cloned for another iteration).
  static LogicalResult collectTileDeps(scf::ForOp kLoop, memref::SubViewOp sv,
                                       llvm::SetVector<Operation *> &deps) {
    Block *body = kLoop.getBody();
    bool ok = true;
    std::function<void(Value)> collect = [&](Value v) {
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != body || deps.contains(def))
        return;
      if (!isMemoryEffectFree(def) || def->getNumRegions() != 0) {
        ok = false;
        return;
      }
      for (Value operand : def->getOperands())
        collect(operand);
      deps.insert(def); // operands first => topological order
    };
    for (Value operand : sv->getOperands())
      collect(operand);
    return success(ok);
  }

  /// Clone `sv` (and its in-loop dependencies) at the rewriter's insertion
  /// point with the induction variable replaced by `newIv`.
  static Value cloneTileAt(IRRewriter &rewriter, scf::ForOp kLoop,
                           memref::SubViewOp sv, Value newIv) {
    llvm::SetVector<Operation *> deps;
    if (failed(collectTileDeps(kLoop, sv, deps)))
      return Value();
    IRMapping map;
    map.map(kLoop.getInductionVar(), newIv);
    for (Operation *dep : deps)
      rewriter.clone(*dep, map);
    return rewriter.clone(*sv.getOperation(), map)->getResult(0);
  }

  /// [R x C] window (rank-reduced) of stage `stage` of the 3-D buffer.
  Value stageView(IRRewriter &rewriter, Location loc, const PipeTile &t,
                  OpFoldResult stage) {
    auto bufTy = cast<MemRefType>(t.buf.getType());
    ArrayRef<int64_t> shape = t.tileTy.getShape();
    SmallVector<OpFoldResult> offs = {stage, rewriter.getIndexAttr(0),
                                      rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(shape[0]),
                                       rewriter.getIndexAttr(shape[1])};
    SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
    auto viewTy = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        shape, bufTy, offs, sizes, strides));
    return rewriter.create<memref::SubViewOp>(loc, viewTy, t.buf, offs, sizes,
                                              strides);
  }

  /// Double-buffer the K loop (see --smem-stages). Returns failure, without
  /// having modified the IR, when the loop does not qualify.
  LogicalResult pipelineKLoop(IRRewriter &rewriter, scf::ForOp kLoop,
                              gpu::LaunchOp launch, int64_t numThreads,
                              ArrayRef<memref::SubViewOp> tiles,
                              SmallVectorImpl<scf::ForOp> &unrollList) {
    auto lb = getConstantIntValue(kLoop.getLowerBound());
    auto ub = getConstantIntValue(kLoop.getUpperBound());
    auto stepC = getConstantIntValue(kLoop.getStep());
    if (!lb || !ub || !stepC || *stepC <= 0 || (*ub - *lb) % *stepC != 0 ||
        *ub <= *lb || kLoop.getNumRegionIterArgs() != 0)
      return failure();
    const int64_t S = smemStages;
    MLIRContext *ctx = rewriter.getContext();
    Location loc = kLoop.getLoc();
    Block &body = *kLoop.getBody();
    auto smemSpaceAttr =
        gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);

    // Pre-flight (no IR changes until it passes): every tile must be clonable
    // at another iteration value, and the stage buffers must fit the 48 KiB
    // static shared-memory limit.
    int64_t smemBytes = 0;
    for (memref::SubViewOp sv : tiles) {
      llvm::SetVector<Operation *> deps;
      if (failed(collectTileDeps(kLoop, sv, deps)))
        return failure();
      ArrayRef<int64_t> shape = sv.getType().getShape();
      Type et = sv.getType().getElementType();
      smemBytes += S * shape[0] * (shape[1] + smemPad) *
                   std::max<int64_t>(1, et.getIntOrFloatBitWidth() / 8);
    }
    if (smemBytes > kMaxStaticSmemBytes)
      return failure(); // fall back to the single-buffered schedule

    // 1. stage buffers, outside the loop.
    SmallVector<PipeTile> pipe;
    rewriter.setInsertionPoint(kLoop);
    for (memref::SubViewOp sv : tiles) {
      PipeTile t;
      t.sv = sv;
      t.tileTy = sv.getType();
      ArrayRef<int64_t> shape = t.tileTy.getShape();
      auto bufTy = MemRefType::get({S, shape[0], shape[1] + smemPad},
                                   t.tileTy.getElementType(), AffineMap(),
                                   smemSpaceAttr);
      t.buf = rewriter.create<memref::AllocOp>(loc, bufTy);
      t.argRooted = isArgumentRooted(sv.getSource());
      pipe.push_back(t);
    }

    bool anyAsync = false;
    auto commitAndWait = [&]() {
      auto group = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
          loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange{});
      rewriter.create<nvgpu::DeviceAsyncWaitOp>(loc, group.getResult(),
                                                rewriter.getI32IntegerAttr(0));
    };

    // 2. prologue: tile of the first iteration -> stage 0.
    rewriter.setInsertionPoint(kLoop);
    bool prologueAsync = false;
    Value zeroStage = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (PipeTile &t : pipe) {
      Value src0 = cloneTileAt(rewriter, kLoop, t.sv, kLoop.getLowerBound());
      Value dst0 = stageView(rewriter, loc, t, zeroStage);
      CopyResult cr = emitCopy(rewriter, loc, launch, numThreads, src0, dst0,
                               t.argRooted);
      prologueAsync |= cr.async;
      if (cr.unrollable)
        unrollList.push_back(cr.loop);
    }
    if (prologueAsync)
      commitAndWait();
    rewriter.create<gpu::BarrierOp>(loc);

    // 3. loop body: prefetch the next tile, then compute on the current one.
    rewriter.setInsertionPointToStart(&body);
    Value iv = kLoop.getInductionVar();
    Value stepV = kLoop.getStep();
    Value idx = rewriter.create<arith::DivUIOp>(
        loc, rewriter.create<arith::SubIOp>(loc, iv, kLoop.getLowerBound()),
        stepV);
    Value nStages = rewriter.create<arith::ConstantIndexOp>(loc, S);
    Value stageCur = rewriter.create<arith::RemUIOp>(loc, idx, nStages);
    Value stageNext = rewriter.create<arith::RemUIOp>(
        loc,
        rewriter.create<arith::AddIOp>(
            loc, idx, rewriter.create<arith::ConstantIndexOp>(loc, 1)),
        nStages);
    // The last iteration re-fetches the last tile (clamped) instead of using
    // a branch: no control flow keeps the loads and the MMAs schedulable
    // together by the PTX assembler, and the extra copy is harmless.
    Value lastIv = rewriter.create<arith::ConstantIndexOp>(loc, *ub - *stepC);
    Value nextIv = rewriter.create<arith::MinUIOp>(
        loc, rewriter.create<arith::AddIOp>(loc, iv, stepV), lastIv);

    SmallVector<Value> curViews;
    for (PipeTile &t : pipe) {
      curViews.push_back(stageView(rewriter, loc, t, stageCur));
      Value srcN = cloneTileAt(rewriter, kLoop, t.sv, nextIv);
      Value dstN = stageView(rewriter, loc, t, stageNext);
      CopyResult cr = emitCopy(rewriter, loc, launch, numThreads, srcN, dstN,
                               t.argRooted);
      anyAsync |= cr.async;
      if (cr.unrollable)
        unrollList.push_back(cr.loop);
    }

    // 4. compute reads the current stage.
    for (auto [t, view] : llvm::zip(pipe, curViews))
      if (failed(replaceAndRetype(t.sv.getResult(), view, nullptr, rewriter)))
        return failure();

    // 5. synchronisation. The pre-compute barrier of the single-buffered
    //    schedule is not needed any more (the previous iteration's trailing
    //    barrier / the prologue barrier already published this stage).
    SmallVector<gpu::BarrierOp> barriers;
    for (Operation &op : body.without_terminator())
      if (auto b = dyn_cast<gpu::BarrierOp>(op))
        barriers.push_back(b);
    if (barriers.size() >= 2)
      rewriter.eraseOp(barriers.front());
    Operation *tail = body.getTerminator();
    if (!barriers.empty())
      tail = barriers.back();
    else
      rewriter.setInsertionPoint(tail);
    rewriter.setInsertionPoint(tail);
    if (anyAsync)
      commitAndWait();
    if (barriers.empty())
      rewriter.create<gpu::BarrierOp>(loc);

    // 6. buffers die after the loop.
    rewriter.setInsertionPointAfter(kLoop);
    for (PipeTile &t : pipe)
      rewriter.create<memref::DeallocOp>(loc, t.buf);

    // original (now dead) tiles
    for (PipeTile &t : pipe)
      if (t.sv->use_empty())
        rewriter.eraseOp(t.sv);
    return success();
  }

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
    // Collect first, outermost loop first: promoting the K-loop tiles
    // re-sources the warp-level subviews to SRAM, so inner loops must not be
    // visited (and wrongly promoted again) before their parent.
    SmallVector<scf::ForOp> loops;
    funcOp.walk<WalkOrder::PreOrder>([&](scf::ForOp op) { loops.push_back(op); });
    for (scf::ForOp kLoop : loops) {
      // Only process K-loops that are inside the block grid: an scf.forall
      // (before quantforge-forall-to-gpu) or a gpu.launch (after it).
      auto launch = kLoop->getParentOfType<gpu::LaunchOp>();
      if (!kLoop->getParentOfType<scf::ForallOp>() && !launch)
        continue;

      // Inside a gpu.launch with a static block size the HBM→SRAM copy is
      // distributed over all threads of the block (grid-stride loop over the
      // linearised tile); otherwise every "thread" copies the whole tile.
      std::optional<int64_t> numThreads;
      if (launch) {
        auto bx = getConstantIntValue(launch.getBlockSizeX());
        auto by = getConstantIntValue(launch.getBlockSizeY());
        auto bz = getConstantIntValue(launch.getBlockSizeZ());
        if (bx && by && bz)
          numThreads = (*bx) * (*by) * (*bz);
      }

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

        if (isTileSizedSubview(subview, tileSizesM, tileSizesN, blockTileK) &&
            isReadOnlyTile(subview.getResult()))
          subviewsToPromote.push_back(subview);
      }

      // Keep only the outermost candidates: a candidate that is (transitively)
      // a subview of another candidate (e.g. a 64x64 warp tile carved out of
      // the 128x64 block tile) is re-sourced from the promoted parent and
      // must not be promoted on its own (it is erased when the parent is).
      {
        llvm::SmallPtrSet<Operation *, 4> candidates(subviewsToPromote.begin(),
                                                     subviewsToPromote.end());
        llvm::erase_if(subviewsToPromote, [&](memref::SubViewOp sv) {
          Value src = sv.getSource();
          while (Operation *def = src.getDefiningOp()) {
            if (candidates.contains(def))
              return true;
            auto parent = dyn_cast<memref::SubViewOp>(def);
            if (!parent)
              break;
            src = parent.getSource();
          }
          return false;
        });
      }

      if (subviewsToPromote.empty())
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "[SmemPromotion] found " << subviewsToPromote.size()
                 << " subview(s) to promote in K-loop\n");

      // The tiles must fit the static shared-memory budget; otherwise the PTX
      // assembler / driver rejects the kernel much later with a cryptic
      // "uses too much shared data" error.
      if (launch) {
        int64_t bytes = 0;
        for (memref::SubViewOp sv : subviewsToPromote) {
          ArrayRef<int64_t> shape = sv.getType().getShape();
          Type et = sv.getType().getElementType();
          bytes += shape[0] * (shape[1] + smemPad) *
                   std::max<int64_t>(1, et.getIntOrFloatBitWidth() / 8);
        }
        if (bytes > kMaxStaticSmemBytes) {
          kLoop.emitError("promoted tiles need ")
              << bytes << " bytes of shared memory (limit "
              << kMaxStaticSmemBytes
              << "); use a smaller block-tile-k / smem-pad";
          signalPassFailure();
          return;
        }
      }

      // ── Double-buffered schedule ────────────────────────────────
      if (smemStages >= 2 && launch && numThreads) {
        SmallVector<scf::ForOp> pipeUnroll;
        if (succeeded(pipelineKLoop(rewriter, kLoop, launch, *numThreads,
                                    subviewsToPromote, pipeUnroll))) {
          promotedCount += subviewsToPromote.size();
          for (scf::ForOp copy : pipeUnroll) {
            int64_t trips = 0;
            if (auto ub = getConstantIntValue(copy.getUpperBound()))
              trips = *ub;
            int64_t factor = std::min<int64_t>(copyUnroll, trips);
            while (factor > 1 && trips % factor != 0)
              --factor;
            if (factor > 1)
              (void)loopUnrollByFactor(copy, factor);
          }
          continue;
        }
      }

      // ── Promote each subview ────────────────────────────────────
      SmallVector<Value> sramAllocs;
      SmallVector<scf::ForOp> copyLoops;
      bool usedAsync = false;
      SmallVector<scf::ForOp> unrollCandidates;
      Operation *lastStoreOp = nullptr;

      for (memref::SubViewOp subview : subviewsToPromote) {
        MemRefType hbmTy = subview.getResult().getType();
        ArrayRef<int64_t> shape = hbmTy.getShape();
        Type elemTy = hbmTy.getElementType();

        // 1. Create SRAM memref type with shared memory space
        auto sramTy =
            MemRefType::get(shape, elemTy, AffineMap(), smemSpaceAttr);

        // 2. Allocate SRAM buffer (insert right after the subview). With
        //    --smem-pad the buffer is [rows, cols + pad] and the tile is the
        //    [rows, cols] window at the origin, so the row stride is not a
        //    multiple of the 32-bank width.
        rewriter.setInsertionPointAfter(subview);
        Value sramView; // what consumers of the tile are redirected to
        memref::AllocOp sramAlloc;
        if (smemPad > 0) {
          auto paddedTy = MemRefType::get({shape[0], shape[1] + smemPad},
                                          elemTy, AffineMap(), smemSpaceAttr);
          sramAlloc = rewriter.create<memref::AllocOp>(loc, paddedTy);
          SmallVector<OpFoldResult> offs(2, rewriter.getIndexAttr(0));
          SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(shape[0]),
                                             rewriter.getIndexAttr(shape[1])};
          SmallVector<OpFoldResult> strides(2, rewriter.getIndexAttr(1));
          auto viewTy = cast<MemRefType>(
              memref::SubViewOp::inferRankReducedResultType(
                  shape, paddedTy, offs, sizes, strides));
          sramView = rewriter.create<memref::SubViewOp>(
              loc, viewTy, sramAlloc.getResult(), offs, sizes, strides);
        } else {
          sramAlloc = rewriter.create<memref::AllocOp>(loc, sramTy);
          sramView = sramAlloc.getResult();
        }
        sramAllocs.push_back(sramAlloc.getResult());

        // 3. Explicit copy loop: HBM → SRAM
        //    CRITICAL: We must NOT use memref.copy here because it writes
        //    linearly. The downstream SwizzleLoadPass will XOR-swizzle the
        //    memref.store indices (write to SRAM) AND the memref.load
        //    indices (read from SRAM), ensuring both sides are consistent.
        //    If we used memref.copy, the write would be linear but the
        //    read would be swizzled, causing DATA CORRUPTION.
        CopyResult cr = emitCopy(rewriter, loc, launch, numThreads,
                                 subview.getResult(), sramView,
                                 isArgumentRooted(subview.getSource()));
        scf::ForOp outerLoop = cr.loop;
        Operation *storeOp = cr.store;
        usedAsync |= cr.async;
        if (cr.unrollable)
          unrollCandidates.push_back(outerLoop);
        lastStoreOp = storeOp;
        copyLoops.push_back(outerLoop);

        // Move insertion point after the outer loop for the next subview
        rewriter.setInsertionPointAfter(outerLoop);

        // 4. Replace all uses of the HBM subview with the SRAM alloc
        //    (except the copy loop itself), re-creating dependent subviews.
        if (failed(replaceAndRetype(subview.getResult(), sramView, outerLoop,
                                    rewriter))) {
          signalPassFailure();
          return;
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "[SmemPromotion] promoted subview "
                   << shape[0] << "x" << shape[1] << " to SRAM\n");
        ++promotedCount;
      }

      // cp.async completion: commit the outstanding copies of this thread
      // and wait for them before the barrier that publishes the tile.
      if (usedAsync) {
        rewriter.setInsertionPointAfter(copyLoops.back());
        auto group = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
            loc, nvgpu::DeviceAsyncTokenType::get(ctx), ValueRange{});
        rewriter.create<nvgpu::DeviceAsyncWaitOp>(loc, group.getResult(),
                                                  rewriter.getI32IntegerAttr(0));
      }

      // ── Insert gpu.barrier after all copy loops (RAW hazard) ────
      if (lastStoreOp) {
        // The last copy loop (outermost scf.for containing lastStoreOp)
        Operation *outerLoop = copyLoops.back();
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
      // Unroll the (constant-trip) copy loops. Done LAST: when the unroll
      // factor equals the trip count the loop is erased (promoted), so no
      // earlier step may still hold a pointer to it (e.g. the barrier anchor).
      for (scf::ForOp copy : unrollCandidates) {
        int64_t trips = 0;
        if (auto ub = getConstantIntValue(copy.getUpperBound()))
          trips = *ub;
        int64_t factor = std::min<int64_t>(copyUnroll, trips);
        while (factor > 1 && trips % factor != 0)
          --factor;
        if (factor > 1)
          (void)loopUnrollByFactor(copy, factor);
      }

    }

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
