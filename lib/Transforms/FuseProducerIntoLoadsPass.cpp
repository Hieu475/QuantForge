//===----------------------------------------------------------------------===//
// FuseProducerIntoLoadsPass — recompute an elementwise producer at its loads
//
// After bufferization the dequantized weights are a global f16 buffer written
// by a standalone elementwise kernel and then re-read by the GEMM when it
// stages its B tile into shared memory:
//
//   %W = gpu.alloc : memref<128x64x2xf16>
//   gpu.launch {                                    // kernel 1 (dequant)
//     ...                                           //   slice: pure ops +
//     %v = <dequant of packed[i,j], scale, zp, p>   //   loads of the inputs
//     memref.store %v, %W[%i, %j, %p]
//   }
//   gpu.launch {                                    // kernel 2 (GEMM)
//     scf.for ... { %x = memref.load %W[%i,%j,%p] ... store to SRAM }
//   }
//
// This pass replaces every `memref.load %W[...]` by a clone of the producer
// slice with the store indices substituted by the load indices, so the GEMM
// reads the *packed* INT4 weights straight from HBM and dequantizes while
// staging to shared memory. When no load is left the producer kernel, the
// buffer and its alloc/dealloc are deleted: one kernel instead of two, and
// 4x less weight traffic (the f16 weight tensor is never materialised).
//
// Loads may address the buffer through memref.subview (unit strides) and
// memref.collapse_shape (static shapes) chains; the indices are composed here
// (offset addition / delinearisation), so no global alias folding is needed.
//
// Requirements (otherwise the buffer is left untouched):
//   * the buffer is a gpu.alloc with exactly one store, directly in the entry
//     block of a gpu.launch;
//   * the stored value's backward slice inside that launch consists of ops
//     without regions and only depends on the store's index *values* (these
//     are substituted, so they may be arbitrary expressions of block/thread
//     ids), values defined outside the launch, and loads from memrefs not
//     written there;
//   * all users of the buffer are that store, deallocs, or views whose only
//     users are loads.
//
// Loads from rank-0 memrefs inside the slice (per-tensor scale / zero point)
// are hoisted to the start of the consuming launch and shared.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-fuse-producer-into-loads"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

/// Everything needed to recompute the stored value for arbitrary indices.
struct Producer {
  gpu::AllocOp alloc;
  gpu::LaunchOp launch;
  memref::StoreOp store;
  SmallVector<Operation *> slice; // in program order
  SmallVector<Value> indexVals; // SSA values of the store's indices
  SmallVector<memref::LoadOp> loads; // loads reaching the buffer (via views)
  SmallVector<Operation *> views;    // subview / collapse_shape / dealloc
};

/// Walk from `load`'s memref up to `root`, composing the indices through
/// unit-stride subviews and static collapse_shapes. Materialises arith ops at
/// `b`'s insertion point. Returns false for unsupported chains.
static bool composeIndices(memref::LoadOp load, Value root,
                           SmallVectorImpl<Value> &out, OpBuilder &b) {
  Location loc = load.getLoc();
  Value cur = load.getMemRef();
  SmallVector<Value> idx(load.getIndices().begin(), load.getIndices().end());
  while (cur != root) {
    Operation *def = cur.getDefiningOp();
    if (auto sub = dyn_cast_or_null<memref::SubViewOp>(def)) {
      if (sub.getType().getRank() != sub.getSourceType().getRank())
        return false;
      for (OpFoldResult st : sub.getMixedStrides())
        if (getConstantIntValue(st) != 1)
          return false;
      auto offs = sub.getMixedOffsets();
      for (unsigned d = 0; d < idx.size(); ++d) {
        Value off = getValueOrCreateConstantIndexOp(b, loc, offs[d]);
        idx[d] = b.create<arith::AddIOp>(loc, idx[d], off);
      }
      cur = sub.getSource();
    } else if (auto col = dyn_cast_or_null<memref::CollapseShapeOp>(def)) {
      ArrayRef<int64_t> srcShape = col.getSrcType().getShape();
      SmallVector<Value> srcIdx;
      for (auto [g, group] : llvm::enumerate(col.getReassociationIndices())) {
        if (group.size() == 1) {
          srcIdx.push_back(idx[g]);
          continue;
        }
        SmallVector<Value> coords(group.size());
        Value rem = idx[g];
        for (int i = group.size() - 1; i >= 0; --i) {
          int64_t size = srcShape[group[i]];
          if (ShapedType::isDynamic(size))
            return false;
          if (i == 0) {
            coords[i] = rem;
          } else {
            Value sz = b.create<arith::ConstantIndexOp>(loc, size);
            coords[i] = b.create<arith::RemUIOp>(loc, rem, sz);
            rem = b.create<arith::DivUIOp>(loc, rem, sz);
          }
        }
        srcIdx.append(coords.begin(), coords.end());
      }
      idx = srcIdx;
      cur = col.getSrc();
    } else {
      return false;
    }
  }
  out.assign(idx.begin(), idx.end());
  return true;
}

/// Collect the slice computing `value` inside `launch`'s entry block.
/// Returns failure if the slice has unsupported ops.
static LogicalResult collectSlice(Value value, gpu::LaunchOp launch,
                                  ArrayRef<Value> indexVals,
                                  SetVector<Operation *> &slice) {
  Block *entry = &launch.getBody().front();
  SmallVector<Value> work{value};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    // The store's index values are *inputs* of the recomputation: they are
    // substituted by the load's indices, so the (possibly non-trivial)
    // expressions that computed them from block/thread ids are cut here.
    if (llvm::is_contained(indexVals, v))
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner() == entry)
        return failure(); // depends on a launch id that is not an index
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (def->getBlock() != entry)
      continue; // defined outside the launch (or in an outer region)
    if (slice.contains(def))
      continue;
    if (def->getNumRegions() != 0)
      return failure();
    if (auto load = dyn_cast<memref::LoadOp>(def)) {
      // The slice may read inputs but must not read what the launch writes.
      for (Operation *user : load.getMemRef().getUsers())
        if (isa<memref::StoreOp>(user) && user->getParentOp() == launch)
          return failure();
    } else if (!isMemoryEffectFree(def)) {
      return failure();
    }
    slice.insert(def);
    for (Value operand : def->getOperands())
      work.push_back(operand);
  }
  return success();
}

static std::optional<Producer> analyze(gpu::AllocOp alloc) {
  Value buf = alloc.getMemref();
  memref::StoreOp store;
  SmallVector<memref::LoadOp> loads;
  SmallVector<Operation *> views;
  SmallVector<Value> work{buf};
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (auto s = dyn_cast<memref::StoreOp>(user)) {
        // the single writer must store *to the buffer itself*
        if (v != buf || store)
          return std::nullopt;
        store = s;
      } else if (auto l = dyn_cast<memref::LoadOp>(user)) {
        loads.push_back(l);
      } else if (isa<gpu::DeallocOp>(user) && v == buf) {
        views.push_back(user);
      } else if (isa<memref::CollapseShapeOp, memref::SubViewOp>(user)) {
        views.push_back(user);
        work.push_back(user->getResult(0));
      } else {
        return std::nullopt;
      }
    }
  }
  if (!store)
    return std::nullopt;

  auto launch = store->getParentOfType<gpu::LaunchOp>();
  if (!launch || store->getBlock() != &launch.getBody().front())
    return std::nullopt;

  Producer p;
  p.loads = loads;
  p.views = views;
  p.alloc = alloc;
  p.launch = launch;
  p.store = store;
  for (Value idx : store.getIndices()) {
    if (llvm::is_contained(p.indexVals, idx))
      return std::nullopt; // repeated index value
    p.indexVals.push_back(idx);
  }

  SetVector<Operation *> slice;
  if (failed(collectSlice(store.getValue(), launch, p.indexVals, slice)))
    return std::nullopt;
  p.slice.assign(slice.begin(), slice.end());
  llvm::sort(p.slice, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return p;
}

struct FuseProducerIntoLoadsPass
    : public PassWrapper<FuseProducerIntoLoadsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseProducerIntoLoadsPass)

  StringRef getArgument() const override {
    return "quantforge-fuse-producer-into-loads";
  }
  StringRef getDescription() const override {
    return "Recompute the single elementwise producer kernel of a device "
           "buffer at each load of that buffer and delete the producer "
           "(fuses INT4 dequantization into the GEMM's shared-memory "
           "staging).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect>();
  }

  void fuse(Producer &p) {
    // Cache of hoisted scalar (rank-0) loads per consuming launch.
    DenseMap<std::pair<Operation *, Value>, Value> hoisted;

    SmallVector<memref::LoadOp> remaining;
    for (memref::LoadOp load : p.loads) {
      OpBuilder b(load);
      SmallVector<Value> idx;
      if (!composeIndices(load, p.alloc.getMemref(), idx, b) ||
          idx.size() != p.indexVals.size()) {
        remaining.push_back(load);
        continue;
      }
      auto consumer = load->getParentOfType<gpu::LaunchOp>();

      IRMapping mapping;
      for (auto [val, i] : llvm::zip(p.indexVals, idx))
        mapping.map(val, i);

      for (Operation *op : p.slice) {
        if (auto sl = dyn_cast<memref::LoadOp>(op);
            sl && sl.getIndices().empty() && consumer) {
          // per-tensor scalar (scale / zero point): load once per launch
          auto key = std::make_pair(consumer.getOperation(), sl.getMemRef());
          auto it = hoisted.find(key);
          if (it == hoisted.end()) {
            OpBuilder hb(&consumer.getBody().front(),
                         consumer.getBody().front().begin());
            Value v = hb.create<memref::LoadOp>(sl.getLoc(), sl.getMemRef(),
                                                ValueRange{});
            it = hoisted.try_emplace(key, v).first;
          }
          mapping.map(sl.getResult(), it->second);
          continue;
        }
        b.clone(*op, mapping);
      }
      Value replacement = mapping.lookupOrDefault(p.store.getValue());
      load.getResult().replaceAllUsesWith(replacement);
      load.erase();
    }
    p.loads = remaining;
  }

  /// Remove the producer kernel and the buffer once nothing reads it.
  void cleanup(Producer &p) {
    if (!p.loads.empty())
      return; // some load could not be fused: keep the producer + buffer

    p.store.erase();
    // erase the slice (reverse order) and the then-empty launch
    for (Operation *op : llvm::reverse(p.slice))
      if (op->use_empty())
        op->erase();
    Block &body = p.launch.getBody().front();
    if (llvm::hasSingleElement(body)) // only gpu.terminator left
      p.launch.erase();
    // views are now use-free; erase users before producers (reverse order)
    for (Operation *op : llvm::reverse(p.views))
      if (op->use_empty())
        op->erase();
    if (p.alloc->use_empty())
      p.alloc.erase();
  }

  void runOnOperation() override {
    SmallVector<gpu::AllocOp> allocs;
    getOperation().walk([&](gpu::AllocOp op) { allocs.push_back(op); });
    for (gpu::AllocOp alloc : allocs) {
      std::optional<Producer> p = analyze(alloc);
      if (!p)
        continue;
      fuse(*p);
      cleanup(*p);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createFuseProducerIntoLoadsPass() {
  return std::make_unique<FuseProducerIntoLoadsPass>();
}
void mlir::quantforge::registerFuseProducerIntoLoadsPass() {
  PassRegistration<FuseProducerIntoLoadsPass>();
}
