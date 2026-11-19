//===----------------------------------------------------------------------===//
// GPULaunchLowering — bridge between tiled/bufferized IR and gpu.launch
//
// Two small passes that close the gap between the QuantForge tiling pipeline
// and the upstream GPU lowering (gpu-kernel-outlining → NVVM → PTX).
// Upstream MLIR 18 has no standalone `scf.forall → gpu.launch` pass (only the
// transform-dialect ops), so we provide one.
//
//   quantforge-forall-to-gpu
//     Rewrites every top-level, result-less (memref semantics) scf.forall
//     into a gpu.launch. The forall iteration space becomes the launch grid
//     (one block per iteration), and the induction variables are recomputed
//     from gpu.block_id as  iv = lb + block_id * step.
//     The grid axis of each forall dimension is taken from the
//     #gpu.block<x|y|z> entries of the `mapping` attribute (set by
//     quantforge-gpu-mapping); without a mapping the last dimension maps to x.
//     Block size comes from the function attributes
//     `quantforge.block_size_{x,y,z}` (set by quantforge-gpu-mapping), or the
//     block-dim-* options; it defaults to 1x1x1.
//
//   quantforge-host-alloc-to-gpu
//     Host code that survives outside gpu.launch must hand *device* memory to
//     the kernels. memref.alloc / memref.dealloc outside of device code are
//     rewritten to gpu.alloc / gpu.dealloc. Non-escaping buffers without an
//     explicit dealloc are freed before every func.return.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-gpu-launch-lowering"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

//===----------------------------------------------------------------------===//
// quantforge-forall-to-gpu
//===----------------------------------------------------------------------===//

struct ForallToGPUPass
    : public PassWrapper<ForallToGPUPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForallToGPUPass)

  ForallToGPUPass() = default;
  ForallToGPUPass(const ForallToGPUPass &other)
      : PassWrapper<ForallToGPUPass, OperationPass<func::FuncOp>>(other) {
    blockDimX.setValue(other.blockDimX.getValue());
    blockDimY.setValue(other.blockDimY.getValue());
    blockDimZ.setValue(other.blockDimZ.getValue());
    swizzleM.setValue(other.swizzleM.getValue());
  }

  StringRef getArgument() const override { return "quantforge-forall-to-gpu"; }
  StringRef getDescription() const override {
    return "Convert top-level memref-semantics scf.forall into gpu.launch "
           "(one block per forall iteration).";
  }

  // 0 = take the value from the function attribute
  // `quantforge.block_size_{x,y,z}` written by quantforge-gpu-mapping,
  // falling back to 1.
  Option<int64_t> blockDimX{
      *this, "block-dim-x",
      llvm::cl::desc("Threads per block (x); 0 = from quantforge.block_size_x"),
      llvm::cl::init(0)};
  Option<int64_t> blockDimY{
      *this, "block-dim-y",
      llvm::cl::desc("Threads per block (y); 0 = from quantforge.block_size_y"),
      llvm::cl::init(0)};
  Option<int64_t> blockDimZ{
      *this, "block-dim-z",
      llvm::cl::desc("Threads per block (z); 0 = from quantforge.block_size_z"),
      llvm::cl::init(0)};

  Option<int64_t> swizzleM{
      *this, "swizzle-m",
      llvm::cl::desc("2-D foralls: launch a 1-D grid and walk the tiles in "
                     "groups of this many M tiles (column-major inside a "
                     "group) so that the blocks resident at the same time "
                     "form a compact (M x N) patch: far fewer L2 misses on "
                     "wide problems. 0 = one grid axis per forall dimension"),
      llvm::cl::init(0)};

  int64_t resolveBlockDim(func::FuncOp func, int64_t opt, StringRef attr) {
    if (opt > 0)
      return opt;
    if (auto a = func->getAttrOfType<IntegerAttr>(attr))
      return a.getInt();
    return 1;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, arith::ArithDialect>();
  }

  /// Returns the launch axis (0=x,1=y,2=z) for forall dimension `dim`.
  static int64_t getAxis(scf::ForallOp op, unsigned dim) {
    if (std::optional<ArrayAttr> mapping = op.getMapping()) {
      if (dim < mapping->size()) {
        if (auto blockAttr =
                dyn_cast<gpu::GPUBlockMappingAttr>((*mapping)[dim])) {
          switch (blockAttr.getBlock()) {
          case gpu::MappingId::DimX:
            return 0;
          case gpu::MappingId::DimY:
            return 1;
          case gpu::MappingId::DimZ:
            return 2;
          default:
            break;
          }
        }
      }
    }
    // Default: last forall dim -> x, previous -> y, ...
    return op.getRank() - 1 - dim;
  }

  LogicalResult convert(scf::ForallOp op) {
    if (op.getNumResults() != 0)
      return op.emitError("scf.forall must have memref semantics (run "
                          "bufferization before quantforge-forall-to-gpu)");
    unsigned rank = op.getRank();
    if (rank > 3)
      return op.emitError("scf.forall rank > 3 cannot map to a 3-D grid");

    Location loc = op.getLoc();
    OpBuilder b(op);

    SmallVector<OpFoldResult> lbs = op.getMixedLowerBound();
    SmallVector<OpFoldResult> ubs = op.getMixedUpperBound();
    SmallVector<OpFoldResult> steps = op.getMixedStep();

    auto toValue = [&](OpFoldResult ofr) -> Value {
      return getValueOrCreateConstantIndexOp(b, loc, ofr);
    };

    // Grid size per forall dimension: ceildiv(ub - lb, step).
    SmallVector<Value> trip(rank);
    for (unsigned d = 0; d < rank; ++d) {
      Value lb = toValue(lbs[d]);
      Value ub = toValue(ubs[d]);
      Value st = toValue(steps[d]);
      Value diff = b.create<arith::SubIOp>(loc, ub, lb);
      trip[d] = b.create<arith::CeilDivUIOp>(loc, diff, st);
    }

    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    std::array<Value, 3> grid = {one, one, one};
    std::array<int64_t, 3> axisOfDim = {-1, -1, -1};
    const bool swizzle = swizzleM > 0 && rank == 2;
    if (swizzle) {
      grid[0] = b.create<arith::MulIOp>(loc, trip[0], trip[1]);
      axisOfDim = {0, 0, -1};
    } else {
      for (unsigned d = 0; d < rank; ++d) {
        int64_t axis = getAxis(op, d);
        if (axis < 0 || axis > 2 || grid[axis] != one)
          return op.emitError("invalid or duplicated block mapping axis");
        grid[axis] = trip[d];
        axisOfDim[d] = axis;
      }
    }

    auto func = op->getParentOfType<func::FuncOp>();
    Value bx = b.create<arith::ConstantIndexOp>(
        loc, resolveBlockDim(func, blockDimX, "quantforge.block_size_x"));
    Value by = b.create<arith::ConstantIndexOp>(
        loc, resolveBlockDim(func, blockDimY, "quantforge.block_size_y"));
    Value bz = b.create<arith::ConstantIndexOp>(
        loc, resolveBlockDim(func, blockDimZ, "quantforge.block_size_z"));

    auto launch = b.create<gpu::LaunchOp>(loc, grid[0], grid[1], grid[2], bx,
                                          by, bz);
    b.setInsertionPointToStart(&launch.getBody().front());

    gpu::KernelDim3 blockIds = launch.getBlockIds();
    std::array<Value, 3> blockIdVals = {blockIds.x, blockIds.y, blockIds.z};

    // Swizzled tile order (grouped, column-major inside a group):
    //   group = id / (G * Tn);  firstM = group * G;  gm = min(Tm - firstM, G)
    //   m = firstM + (id % (G * Tn)) % gm;  n = (id % (G * Tn)) / gm
    std::array<Value, 3> tileId = blockIdVals;
    if (swizzle) {
      Value id = blockIdVals[0];
      Value g = b.create<arith::ConstantIndexOp>(loc, swizzleM);
      Value groupSize = b.create<arith::MulIOp>(loc, g, trip[1]);
      Value group = b.create<arith::DivUIOp>(loc, id, groupSize);
      Value firstM = b.create<arith::MulIOp>(loc, group, g);
      Value remaining = b.create<arith::SubIOp>(loc, trip[0], firstM);
      Value gm = b.create<arith::MinUIOp>(loc, remaining, g);
      Value within = b.create<arith::RemUIOp>(loc, id, groupSize);
      Value m = b.create<arith::AddIOp>(
          loc, firstM, b.create<arith::RemUIOp>(loc, within, gm));
      Value n = b.create<arith::DivUIOp>(loc, within, gm);
      tileId = {m, n, blockIdVals[2]};
    }

    IRMapping mapping;
    for (unsigned d = 0; d < rank; ++d) {
      // The lb/step values are created outside the launch; gpu.launch is not
      // isolated-from-above so they can be used directly (outlining will
      // capture them as kernel operands).
      Value lb = toValue(lbs[d]);
      Value st = toValue(steps[d]);
      Value scaled = b.create<arith::MulIOp>(
          loc, swizzle ? tileId[d] : blockIdVals[axisOfDim[d]], st);
      Value iv = b.create<arith::AddIOp>(loc, lb, scaled);
      mapping.map(op.getInductionVars()[d], iv);
    }

    for (Operation &inner : op.getBody()->without_terminator())
      b.clone(inner, mapping);
    b.create<gpu::TerminatorOp>(loc);

    op.erase();
    return success();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<scf::ForallOp> foralls;
    funcOp.walk([&](scf::ForallOp op) {
      if (!op->getParentOfType<scf::ForallOp>() &&
          !op->getParentOfType<gpu::LaunchOp>())
        foralls.push_back(op);
    });
    for (scf::ForallOp op : foralls)
      if (failed(convert(op))) {
        signalPassFailure();
        return;
      }
  }
};

//===----------------------------------------------------------------------===//
// quantforge-host-alloc-to-gpu
//===----------------------------------------------------------------------===//

struct HostAllocToGPUPass
    : public PassWrapper<HostAllocToGPUPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HostAllocToGPUPass)

  StringRef getArgument() const override {
    return "quantforge-host-alloc-to-gpu";
  }
  StringRef getDescription() const override {
    return "Rewrite host-side memref.alloc/dealloc into gpu.alloc/dealloc so "
           "intermediate buffers live in device memory.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  static bool isHost(Operation *op) {
    return !op->getParentOfType<gpu::LaunchOp>() &&
           !op->getParentOfType<gpu::GPUFuncOp>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<memref::AllocOp> allocs;
    SmallVector<memref::DeallocOp> deallocs;
    funcOp.walk([&](Operation *op) {
      if (!isHost(op))
        return;
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        // Only default (global) memory space buffers belong to the device
        // heap; workgroup/private allocs are handled by kernel lowering.
        if (!alloc.getType().getMemorySpace())
          allocs.push_back(alloc);
      } else if (auto dealloc = dyn_cast<memref::DeallocOp>(op)) {
        if (!cast<MemRefType>(dealloc.getMemref().getType()).getMemorySpace())
          deallocs.push_back(dealloc);
      }
    });

    OpBuilder b(funcOp.getContext());
    SmallVector<func::ReturnOp> returns;
    funcOp.walk([&](func::ReturnOp ret) { returns.push_back(ret); });

    for (memref::AllocOp alloc : allocs) {
      // Buffers that are never freed explicitly and never leave the function
      // are released right before every return (bufferization does not emit
      // deallocs here). Escaping buffers (returned / passed to a call) are
      // left to the caller.
      bool hasDealloc = false;
      bool escapes = false;
      SmallVector<Value> worklist{alloc.getResult()};
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (isa<memref::DeallocOp>(user))
            hasDealloc = true;
          else if (isa<func::ReturnOp, func::CallOp>(user))
            escapes = true;
          else
            // Conservatively treat every memref-typed result (subview,
            // collapse_shape, cast, ...) as an alias of the buffer.
            for (Value res : user->getResults())
              if (isa<MemRefType>(res.getType()))
                worklist.push_back(res);
        }
      }
      bool needsDealloc = !hasDealloc && !escapes;

      b.setInsertionPoint(alloc);
      auto gpuAlloc = b.create<gpu::AllocOp>(
          alloc.getLoc(), alloc.getType(), /*asyncToken=*/Type(),
          /*asyncDependencies=*/ValueRange(), alloc.getDynamicSizes(),
          alloc.getSymbolOperands(), /*hostShared=*/false);
      alloc.replaceAllUsesWith(gpuAlloc.getMemref());
      if (needsDealloc)
        for (func::ReturnOp ret : returns) {
          b.setInsertionPoint(ret);
          b.create<gpu::DeallocOp>(ret.getLoc(), /*asyncToken=*/Type(),
                                   /*asyncDependencies=*/ValueRange(),
                                   gpuAlloc.getMemref());
        }
      alloc.erase();
    }
    for (memref::DeallocOp dealloc : deallocs) {
      b.setInsertionPoint(dealloc);
      b.create<gpu::DeallocOp>(dealloc.getLoc(), /*asyncToken=*/Type(),
                               /*asyncDependencies=*/ValueRange(),
                               dealloc.getMemref());
      dealloc.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// quantforge-workgroup-alloc-to-attribution
//===----------------------------------------------------------------------===//

/// Shared-memory buffers are created by quantforge-smem-promotion as
/// `memref.alloc : memref<..., #gpu.address_space<workgroup>>` inside the K
/// loop. Inside a gpu.func that would lower to a heap allocation. Workgroup
/// memory must instead be a *static* per-block buffer, i.e. a gpu.func
/// workgroup attribution (lowered to an addrspace(3) global). Each alloc op
/// becomes one attribution; the loop-iteration re-use is already protected by
/// the gpu.barrier pair the promotion pass inserts. Deallocs are dropped.
struct WorkgroupAllocToAttributionPass
    : public PassWrapper<WorkgroupAllocToAttributionPass,
                         OperationPass<gpu::GPUFuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WorkgroupAllocToAttributionPass)

  StringRef getArgument() const override {
    return "quantforge-workgroup-alloc-to-attribution";
  }
  StringRef getDescription() const override {
    return "Turn workgroup-space memref.alloc inside gpu.func into gpu.func "
           "workgroup attributions (static shared memory).";
  }

  static bool isWorkgroup(MemRefType ty) {
    auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(ty.getMemorySpace());
    return space && space.getValue() == gpu::AddressSpace::Workgroup;
  }

  void runOnOperation() override {
    gpu::GPUFuncOp func = getOperation();
    SmallVector<memref::AllocOp> allocs;
    func.walk([&](memref::AllocOp alloc) {
      if (isWorkgroup(alloc.getType()))
        allocs.push_back(alloc);
    });
    for (memref::AllocOp alloc : allocs) {
      if (!alloc.getType().hasStaticShape()) {
        alloc.emitError("dynamic-shaped shared memory cannot be made static");
        signalPassFailure();
        return;
      }
      BlockArgument attr =
          func.addWorkgroupAttribution(alloc.getType(), alloc.getLoc());
      SmallVector<Operation *> deallocs;
      for (Operation *user : alloc->getUsers())
        if (isa<memref::DeallocOp>(user))
          deallocs.push_back(user);
      for (Operation *d : deallocs)
        d->erase();
      alloc.replaceAllUsesWith(attr);
      alloc.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// quantforge-lower-vector
//===----------------------------------------------------------------------===//

/// Vector→vector lowering of the ops produced by vectorization, WITHOUT
/// converting to the LLVM dialect: contractions become dot products,
/// transfers become vector.load/store, broadcast/transpose/shape_cast are
/// simplified. `convert-vector-to-llvm` cannot be used on kernels that touch
/// shared memory because its type converter has no mapping for
/// #gpu.address_space<workgroup>; convert-gpu-to-nvvm (which does) then lowers
/// the remaining 1-D vector.load/store/fma/reduction ops.
struct LowerVectorPass
    : public PassWrapper<LowerVectorPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorPass)

  StringRef getArgument() const override { return "quantforge-lower-vector"; }
  StringRef getDescription() const override {
    return "Lower vector.contract/transfer/broadcast/transpose to simple 1-D "
           "vector ops without touching memref types.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    vector::VectorTransformsOptions opts;
    opts.setVectorTransformsOptions(vector::VectorContractLowering::Dot);
    vector::populateVectorContractLoweringPatterns(patterns, opts);
    vector::populateVectorOuterProductLoweringPatterns(patterns);
    vector::populateVectorTransferLoweringPatterns(patterns);
    vector::populateVectorBroadcastLoweringPatterns(patterns);
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    vector::populateVectorTransposeLoweringPatterns(patterns, opts);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// quantforge-hoist-vector-transfers
//===----------------------------------------------------------------------===//

/// The accumulator tile of the innermost K loop is read and written through
/// vector.transfer_read/write on every iteration. When its address is
/// loop-invariant this becomes a loop-carried vector value (register
/// accumulation) instead of a global-memory read-modify-write per MMA step.
struct HoistVectorTransfersPass
    : public PassWrapper<HoistVectorTransfersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistVectorTransfersPass)

  StringRef getArgument() const override {
    return "quantforge-hoist-vector-transfers";
  }
  StringRef getDescription() const override {
    return "Hoist loop-invariant vector.transfer_read/write pairs out of "
           "scf.for loops (accumulators live in registers across the loop).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, scf::SCFDialect,
                    memref::MemRefDialect>();
  }
  HoistVectorTransfersPass() = default;
  HoistVectorTransfersPass(const HoistVectorTransfersPass &other)
      : PassWrapper<HoistVectorTransfersPass, OperationPass<func::FuncOp>>(
            other) {
    unrollTiles.setValue(other.unrollTiles.getValue());
    unrollLimit.setValue(other.unrollLimit.getValue());
  }

  Option<bool> unrollTiles{
      *this, "unroll-tiles",
      llvm::cl::desc("Fully unroll the quantforge.lane_m / lane_n MMA-tile "
                     "loops first, so every (m16 x n8) accumulator becomes an "
                     "independent value that can be carried through the "
                     "enclosing K loops in registers (register tiling)."),
      llvm::cl::init(false)};
  Option<int64_t> unrollLimit{
      *this, "unroll-limit",
      llvm::cl::desc("Maximum trip count of a tile loop that is unrolled"),
      llvm::cl::init(16)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (unrollTiles) {
      SmallVector<scf::ForOp> tileLoops;
      func.walk<WalkOrder::PostOrder>([&](scf::ForOp op) {
        if (op->hasAttr("quantforge.lane_m") || op->hasAttr("quantforge.lane_n"))
          tileLoops.push_back(op);
      });
      for (scf::ForOp loop : tileLoops) { // inner (lane_n) before outer
        auto lb = getConstantIntValue(loop.getLowerBound());
        auto ub = getConstantIntValue(loop.getUpperBound());
        auto step = getConstantIntValue(loop.getStep());
        if (!lb || !ub || !step || *step <= 0)
          continue;
        int64_t trip = (*ub - *lb + *step - 1) / *step;
        // loopUnrollByFactor already promotes (erases) the resulting
        // single-iteration loop, so `loop` must not be touched afterwards.
        if (trip > 1 && trip <= unrollLimit)
          (void)loopUnrollByFactor(loop, trip);
      }
    }

    // Subviews carrying the accumulator address are created inside the loop;
    // hoist loop-invariant pure ops first so the transfers see invariant
    // sources.
    func.walk([&](LoopLikeOpInterface loop) {
      (void)moveLoopInvariantCode(loop);
    });
    // The upstream helper declines the accumulator pattern we generate, so
    // the read/write pairs are turned into loop-carried values here. Inner
    // loops first: the pair left behind by an inner hoist (read before the
    // loop, write after it) is then hoisted again out of the enclosing loop.
    SmallVector<scf::ForOp> loops;
    func.walk<WalkOrder::PostOrder>([&](scf::ForOp op) { loops.push_back(op); });
    for (scf::ForOp loop : loops) {
      // LICM again: previous hoists create new invariant subviews.
      (void)moveLoopInvariantCode(cast<LoopLikeOpInterface>(loop.getOperation()));
      hoistAccumulators(loop);
    }
    linalg::hoistRedundantVectorTransfers(func);
  }

  /// For every (transfer_read, transfer_write) pair on the same loop-invariant
  /// address directly in the body of `loop`:
  ///   scf.for { %v = transfer_read %S[idx]; ...; transfer_write %w, %S[idx] }
  ///   ==>  %v0 = transfer_read %S[idx]
  ///        %r = scf.for iter_args(%v = %v0) { ...; yield %w }
  ///        transfer_write %r, %S[idx]
  /// Preconditions: source and indices are loop-invariant, and nothing else in
  /// the loop touches the source.
  static void hoistAccumulators(scf::ForOp loop) {
    Block *body = loop.getBody();
    auto isInvariant = [&](Value v) {
      return !loop->isProperAncestor(v.getParentBlock()->getParentOp()) &&
             v.getParentBlock() != body;
    };

    struct Pair {
      vector::TransferReadOp read;
      vector::TransferWriteOp write;
    };
    SmallVector<Pair> pairs;
    for (auto read : body->getOps<vector::TransferReadOp>()) {
      if (!isa<MemRefType>(read.getSource().getType()) || read.getMask())
        continue;
      if (!isInvariant(read.getSource()) ||
          !llvm::all_of(read.getIndices(), isInvariant))
        continue;
      vector::TransferWriteOp write;
      bool ok = true;
      for (Operation *user : read.getSource().getUsers()) {
        if (!loop->isProperAncestor(user) || user == read.getOperation())
          continue;
        auto w = dyn_cast<vector::TransferWriteOp>(user);
        if (!w || write || w->getBlock() != body ||
            w.getIndices() != read.getIndices() ||
            w.getPermutationMap() != read.getPermutationMap() ||
            w.getVectorType() != read.getVectorType() || w.getMask() ||
            !read->isBeforeInBlock(w)) {
          ok = false;
          break;
        }
        write = w;
      }
      if (ok && write)
        pairs.push_back({read, write});
    }
    if (pairs.empty())
      return;

    IRRewriter rewriter(loop.getContext());
    rewriter.setInsertionPoint(loop);
    SmallVector<Value> inits;
    SmallVector<vector::TransferReadOp> initReads;
    for (Pair &p : pairs) {
      auto init = cast<vector::TransferReadOp>(rewriter.clone(*p.read));
      initReads.push_back(init);
      inits.push_back(init.getResult());
    }
    FailureOr<LoopLikeOpInterface> newLoop = loop.replaceWithAdditionalYields(
        rewriter, inits, /*replaceInitOperandUsesInLoop=*/false,
        [&](OpBuilder &, Location, ArrayRef<BlockArgument> newArgs) {
          SmallVector<Value> yields;
          for (auto [p, arg] : llvm::zip(pairs, newArgs)) {
            p.read.getResult().replaceAllUsesWith(arg);
            yields.push_back(p.write.getVector());
          }
          return yields;
        });
    if (failed(newLoop)) {
      for (auto init : initReads)
        rewriter.eraseOp(init);
      return;
    }
    auto newFor = cast<scf::ForOp>(newLoop->getOperation());
    unsigned firstNew = newFor->getNumResults() - pairs.size();
    rewriter.setInsertionPointAfter(newFor);
    for (auto [i, p] : llvm::enumerate(pairs)) {
      IRMapping map;
      map.map(p.write.getVector(), newFor->getResult(firstNew + i));
      rewriter.clone(*p.write, map);
      rewriter.eraseOp(p.write);
      rewriter.eraseOp(p.read);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createHoistVectorTransfersPass() {
  return std::make_unique<HoistVectorTransfersPass>();
}
void mlir::quantforge::registerHoistVectorTransfersPass() {
  PassRegistration<HoistVectorTransfersPass>();
}

std::unique_ptr<Pass> mlir::quantforge::createLowerVectorPass() {
  return std::make_unique<LowerVectorPass>();
}
void mlir::quantforge::registerLowerVectorPass() {
  PassRegistration<LowerVectorPass>();
}

std::unique_ptr<Pass> mlir::quantforge::createWorkgroupAllocToAttributionPass() {
  return std::make_unique<WorkgroupAllocToAttributionPass>();
}
void mlir::quantforge::registerWorkgroupAllocToAttributionPass() {
  PassRegistration<WorkgroupAllocToAttributionPass>();
}

std::unique_ptr<Pass> mlir::quantforge::createForallToGPUPass() {
  return std::make_unique<ForallToGPUPass>();
}
void mlir::quantforge::registerForallToGPUPass() {
  PassRegistration<ForallToGPUPass>();
}

std::unique_ptr<Pass> mlir::quantforge::createHostAllocToGPUPass() {
  return std::make_unique<HostAllocToGPUPass>();
}
void mlir::quantforge::registerHostAllocToGPUPass() {
  PassRegistration<HostAllocToGPUPass>();
}
