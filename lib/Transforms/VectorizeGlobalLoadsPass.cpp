//===----------------------------------------------------------------------===//
// Index simplification + vectorisation of adjacent global loads
//
// The fused dequantisation (FuseProducerIntoLoadsPass) recomputes the INT4
// producer once per *element*: for the 8 columns n0..n0+7 a thread stages in
// one 16-byte work item it issues 8 scalar loads of the scale, 8 of the zero
// point and 8 of the packed byte (4 distinct) -- 20 tiny global loads where 3
// wide ones would do. This file contains two passes that recover them:
//
//  quantforge-simplify-index-math
//      Expands affine.apply to arith and folds the index arithmetic of the
//      unrolled element copies using divisibility facts:
//          (x + c) / m  ->  x / m + c / m           (x % m == 0, m = 2^k)
//          (x + c) % m  ->  c % m
//          (x / m) * m  ->  x
//      so that e.g. the scale index (n / 2) * 2 + n % 2 collapses to n and the
//      nibble selects become constants.
//
//  quantforge-vectorize-global-loads
//      Inside gpu.launch, groups scalar memref.loads of the same memref with
//      identical leading indices whose last index is  base + j  (j = 0..W-1,
//      base provably a multiple of W, W * sizeof(elem) <= 16) and replaces each
//      complete group by one vector.load + extracts. A memref.assume_alignment
//      (W * sizeof) is emitted at the start of the launch for the memref, so
//      the NVPTX backend can emit ld.global.v4 / .v2.
//
//      The vector load ASSUMES the memref's base address is aligned to
//      W * sizeof(elem) bytes (<= 16): allocations from cudaMalloc / PyTorch
//      always are; callers passing interior views must check
//      (python/quantforge/e2e.py does).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-vectorize-global-loads"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include <map>
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

constexpr int64_t kMaxDivisor = 1024;

/// Largest power of two (<= kMaxDivisor) known to divide `v` (as an unsigned
/// quantity). 1 when nothing is known.
static int64_t knownDivisor(Value v, int depth = 0) {
  if (depth > 12)
    return 1;
  if (auto c = getConstantIntValue(v)) {
    if (*c == 0)
      return kMaxDivisor;
    return std::min<int64_t>(kMaxDivisor, *c & -*c);
  }
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == forOp.getInductionVar())
        return std::min(knownDivisor(forOp.getLowerBound(), depth + 1),
                        knownDivisor(forOp.getStep(), depth + 1));
    return 1;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return 1;
  if (auto mul = dyn_cast<arith::MulIOp>(def))
    return std::min<int64_t>(kMaxDivisor,
                             knownDivisor(mul.getLhs(), depth + 1) *
                                 knownDivisor(mul.getRhs(), depth + 1));
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return std::min(knownDivisor(add.getLhs(), depth + 1),
                    knownDivisor(add.getRhs(), depth + 1));
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return std::min(knownDivisor(sub.getLhs(), depth + 1),
                    knownDivisor(sub.getRhs(), depth + 1));
  if (auto rem = dyn_cast<arith::RemUIOp>(def)) {
    auto m = getConstantIntValue(rem.getRhs());
    if (m && *m > 0)
      return std::min(knownDivisor(rem.getLhs(), depth + 1),
                      std::min<int64_t>(kMaxDivisor, *m & -*m));
    return 1;
  }
  if (auto div = dyn_cast<arith::DivUIOp>(def)) {
    auto m = getConstantIntValue(div.getRhs());
    if (m && *m > 0 && (*m & (*m - 1)) == 0) {
      int64_t d = knownDivisor(div.getLhs(), depth + 1);
      return d >= *m ? d / *m : 1;
    }
    return 1;
  }
  if (auto shl = dyn_cast<arith::ShLIOp>(def)) {
    if (auto s = getConstantIntValue(shl.getRhs()); s && *s >= 0 && *s < 7)
      return std::min<int64_t>(kMaxDivisor,
                               knownDivisor(shl.getLhs(), depth + 1) << *s);
    return 1;
  }
  if (auto mn = dyn_cast<arith::MinUIOp>(def))
    return std::min(knownDivisor(mn.getLhs(), depth + 1),
                    knownDivisor(mn.getRhs(), depth + 1));
  if (auto mx = dyn_cast<arith::MaxUIOp>(def))
    return std::min(knownDivisor(mx.getLhs(), depth + 1),
                    knownDivisor(mx.getRhs(), depth + 1));
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return knownDivisor(cast.getIn(), depth + 1);
  if (auto cast = dyn_cast<arith::IndexCastUIOp>(def))
    return knownDivisor(cast.getIn(), depth + 1);
  return 1;
}

/// v = sum(leaves) + c, flattening nested arith.addi.
struct Linear {
  SmallVector<Value> leaves;
  int64_t c = 0;
};

static void decompose(Value v, Linear &out) {
  if (auto c = getConstantIntValue(v)) {
    out.c += *c;
    return;
  }
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    decompose(add.getLhs(), out);
    decompose(add.getRhs(), out);
    return;
  }
  out.leaves.push_back(v);
}

static bool leavesDivisibleBy(const Linear &l, int64_t m) {
  return llvm::all_of(l.leaves, [&](Value v) { return knownDivisor(v) % m == 0; });
}

/// Sum of the leaves (+ constant) as one value; leaves must be non-empty or the
/// constant is returned.
static Value buildSum(OpBuilder &b, Location loc, ArrayRef<Value> leaves,
                      int64_t c) {
  Value acc;
  for (Value v : leaves)
    acc = acc ? Value(b.create<arith::AddIOp>(loc, acc, v)) : v;
  if (!acc || c != 0) {
    Value cv = b.create<arith::ConstantIndexOp>(loc, c);
    acc = acc ? Value(b.create<arith::AddIOp>(loc, acc, cv)) : cv;
  }
  return acc;
}

//===----------------------------------------------------------------------===//
// Index simplification patterns
//===----------------------------------------------------------------------===//

static std::optional<int64_t> pow2Const(Value v) {
  auto c = getConstantIntValue(v);
  if (c && *c > 0 && (*c & (*c - 1)) == 0)
    return *c;
  return std::nullopt;
}

/// Split the leaves of `l` into those that are multiples of m and the rest.
static void splitByDivisor(const Linear &l, int64_t m, SmallVector<Value> &div,
                           SmallVector<Value> &rest) {
  for (Value v : l.leaves)
    (knownDivisor(v) % m == 0 ? div : rest).push_back(v);
}

/// (D + R + c) / m  ->  D / m + c1 / m + (R + c0) / m
///   D: leaves that are multiples of m, c = c1 + c0 with c1 a multiple of m.
/// e.g. (tid + 128) / 16 -> tid / 16 + 8, so the per-vector row / column
/// indices of one thread become common subexpressions.
struct DivOfMultiple : OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rw) const override {
    auto m = pow2Const(op.getRhs());
    if (!m || !op.getType().isIndex())
      return failure();
    Linear l;
    decompose(op.getLhs(), l);
    if (l.c < 0 || l.leaves.empty())
      return failure();
    SmallVector<Value> D, R;
    splitByDivisor(l, *m, D, R);
    int64_t c1 = (l.c / *m) * *m, c0 = l.c - c1;
    if (D.empty() && c1 == 0)
      return failure(); // nothing to split off
    if (R.empty() && l.c == 0)
      return failure(); // already D / m: rebuilding it would match forever
    Location loc = op.getLoc();
    Value mv = rw.create<arith::ConstantIndexOp>(loc, *m);
    Value acc;
    auto add = [&](Value v) {
      acc = acc ? Value(rw.create<arith::AddIOp>(loc, acc, v)) : v;
    };
    if (!D.empty())
      add(rw.create<arith::DivUIOp>(loc, buildSum(rw, loc, D, 0), mv));
    if (c1 != 0)
      add(rw.create<arith::ConstantIndexOp>(loc, c1 / *m));
    if (!R.empty())
      add(rw.create<arith::DivUIOp>(loc, buildSum(rw, loc, R, c0), mv));
    rw.replaceOp(op, acc);
    return success();
  }
};

/// (D + R + c) % m  ->  (R + c0) % m   (D multiples of m, c0 = c % m).
struct RemOfMultiple : OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rw) const override {
    auto m = pow2Const(op.getRhs());
    if (!m || !op.getType().isIndex())
      return failure();
    Linear l;
    decompose(op.getLhs(), l);
    if (l.c < 0)
      return failure();
    SmallVector<Value> D, R;
    splitByDivisor(l, *m, D, R);
    int64_t c0 = l.c % *m;
    if (R.empty()) {
      rw.replaceOpWithNewOp<arith::ConstantIndexOp>(op, c0);
      return success();
    }
    if (D.empty() && l.c == c0)
      return failure(); // nothing to drop
    Value inner = buildSum(rw, op.getLoc(), R, c0);
    rw.replaceOpWithNewOp<arith::RemUIOp>(op, inner, op.getRhs());
    return success();
  }
};

/// (x / m) * m  ->  x   when x is a multiple of m.
struct MulOfDiv : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rw) const override {
    for (int side = 0; side < 2; ++side) {
      Value a = side ? op.getRhs() : op.getLhs();
      Value b = side ? op.getLhs() : op.getRhs();
      auto m = pow2Const(b);
      auto div = a.getDefiningOp<arith::DivUIOp>();
      if (!m || !div || pow2Const(div.getRhs()) != m)
        continue;
      if (knownDivisor(div.getLhs()) % *m != 0)
        continue;
      rw.replaceOp(op, div.getLhs());
      return success();
    }
    return failure();
  }
};

/// (a + c) * m  ->  a * m + c * m   (constants c, m): exposes (x / m) * m.
struct MulOfAddConst : OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rw) const override {
    for (int side = 0; side < 2; ++side) {
      Value a = side ? op.getRhs() : op.getLhs();
      auto m = getConstantIntValue(side ? op.getLhs() : op.getRhs());
      auto add = a.getDefiningOp<arith::AddIOp>();
      if (!m || !add)
        continue;
      auto c = getConstantIntValue(add.getRhs());
      Value x = add.getLhs();
      if (!c) {
        c = getConstantIntValue(add.getLhs());
        x = add.getRhs();
      }
      if (!c)
        continue;
      Location loc = op.getLoc();
      Value mv = rw.create<arith::ConstantIndexOp>(loc, *m);
      Value xm = rw.create<arith::MulIOp>(loc, x, mv);
      rw.replaceOpWithNewOp<arith::AddIOp>(
          op, xm, rw.create<arith::ConstantIndexOp>(loc, *c * *m));
      return success();
    }
    return failure();
  }
};

/// Largest value `v` can take (all index values here are non-negative), or
/// nullopt. Knows constants, thread ids of an enclosing gpu.launch with static
/// block sizes, + * / % by constants, min and scf.for induction variables.
static std::optional<int64_t> upperBound(Value v, gpu::LaunchOp launch,
                                         int depth = 0) {
  if (depth > 16)
    return std::nullopt;
  if (auto c = getConstantIntValue(v))
    return *c;
  if (launch) {
    // getBlockSize() are the *body* block arguments (the launch's own block-size
    // values, not constants): the constants are the launch operands.
    gpu::KernelDim3 t = launch.getThreadIds(), nb = launch.getBlockSize(),
                    n = launch.getBlockSizeOperandValues();
    if (v == t.x) { if (auto d = getConstantIntValue(n.x)) return *d - 1; }
    if (v == t.y) { if (auto d = getConstantIntValue(n.y)) return *d - 1; }
    if (v == t.z) { if (auto d = getConstantIntValue(n.z)) return *d - 1; }
    if (v == nb.x) return getConstantIntValue(n.x);
    if (v == nb.y) return getConstantIntValue(n.y);
    if (v == nb.z) return getConstantIntValue(n.z);
  }
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == forOp.getInductionVar())
        if (auto ub = getConstantIntValue(forOp.getUpperBound()))
          return *ub - 1;
    return std::nullopt;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto ub2 = [&](Value a, Value b) {
    return std::make_pair(upperBound(a, launch, depth + 1), upperBound(b, launch, depth + 1));
  };
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto [a, b] = ub2(add.getLhs(), add.getRhs());
    if (a && b) return *a + *b;
  } else if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto [a, b] = ub2(mul.getLhs(), mul.getRhs());
    if (a && b) return *a * *b;
  } else if (auto div = dyn_cast<arith::DivUIOp>(def)) {
    auto c = getConstantIntValue(div.getRhs());
    auto a = upperBound(div.getLhs(), launch, depth + 1);
    if (a && c && *c > 0) return *a / *c;
  } else if (auto rem = dyn_cast<arith::RemUIOp>(def)) {
    auto c = getConstantIntValue(rem.getRhs());
    auto a = upperBound(rem.getLhs(), launch, depth + 1);
    if (c && *c > 0) return a ? std::min(*a, *c - 1) : *c - 1;
  } else if (auto mn = dyn_cast<arith::MinUIOp>(def)) {
    auto [a, b] = ub2(mn.getLhs(), mn.getRhs());
    if (a && b) return std::min(*a, *b);
    if (a) return a;
    if (b) return b;
  } else if (auto ic = dyn_cast<arith::IndexCastOp>(def)) {
    return upperBound(ic.getIn(), launch, depth + 1);
  }
  return std::nullopt;
}

/// `(a + r) floordiv G`  ->  `a floordiv G`  when a is a multiple of the K tile
/// (BK), BK divides G and 0 <= r < BK. Every row of a K tile then belongs to
/// the same quantisation group, so the scale / zero-point loads of the tile's
/// rows become identical (and CSE merges them).
struct GroupRowOfTile : OpRewritePattern<affine::AffineApplyOp> {
  GroupRowOfTile(MLIRContext *ctx, int64_t tileK)
      : OpRewritePattern(ctx, /*benefit=*/2), tileK(tileK) {}
  int64_t tileK;
  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rw) const override {
    if (tileK <= 0 || op.getAffineMap().getNumResults() != 1 ||
        op.getOperands().size() != 1)
      return failure();
    auto fd = dyn_cast<AffineBinaryOpExpr>(op.getAffineMap().getResult(0));
    if (!fd || fd.getKind() != AffineExprKind::FloorDiv)
      return failure();
    auto gc = dyn_cast<AffineConstantExpr>(fd.getRHS());
    if (!gc || gc.getValue() <= 0 || gc.getValue() % tileK != 0)
      return failure();
    if (!isa<AffineDimExpr, AffineSymbolExpr>(fd.getLHS()))
      return failure();
    Value x = op.getOperands()[0];
    Linear l;
    decompose(x, l);
    SmallVector<Value> aligned, rest;
    for (Value v : l.leaves)
      (tileK <= kMaxDivisor && knownDivisor(v) % tileK == 0 ? aligned : rest)
          .push_back(v);
    int64_t c = l.c, cAligned = (c / tileK) * tileK, cRest = c - cAligned;
    if (c < 0)
      return failure();
    gpu::LaunchOp launch = op->getParentOfType<gpu::LaunchOp>();
    int64_t ub = cRest;
    for (Value v : rest) {
      auto b = upperBound(v, launch);
      if (!b)
        return failure();
      ub += *b;
    }
    if (ub >= tileK)
      return failure();
    if (aligned.empty() && cAligned == 0) {
      // the whole index lies in [0, tileK) <= G: group row 0
      rw.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
      return success();
    }
    // Nothing to drop: the index already is `aligned + cAligned`. Rewriting it
    // would rebuild an identical sum + affine.apply and match again forever.
    if (rest.empty() && cRest == 0)
      return failure();
    OpBuilder::InsertionGuard g(rw);
    rw.setInsertionPoint(op);
    Value base = buildSum(rw, op.getLoc(), aligned, cAligned);
    rw.replaceOpWithNewOp<affine::AffineApplyOp>(op, op.getAffineMap(), ValueRange{base});
    return success();
  }
};

struct ExpandAffineApply : OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rw) const override {
    auto v = affine::expandAffineMap(rw, op.getLoc(), op.getAffineMap(),
                                     op.getOperands());
    if (!v)
      return failure();
    rw.replaceOp(op, *v);
    return success();
  }
};

struct SimplifyIndexMathPass
    : public PassWrapper<SimplifyIndexMathPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyIndexMathPass)
  StringRef getArgument() const override {
    return "quantforge-simplify-index-math";
  }
  StringRef getDescription() const override {
    return "Expand affine.apply and fold div/rem/mul index arithmetic using "
           "divisibility of the operands (prepares load vectorisation).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect>();
  }
  Option<int64_t> tileK{
      *this, "tile-k",
      llvm::cl::desc("K tile of the GEMM (0 = unknown): lets floor(k / G) of the "
                     "rows of one tile fold to a single group row when "
                     "tile-k divides the group size G"),
      llvm::cl::init(0)};
  SimplifyIndexMathPass() = default;
  SimplifyIndexMathPass(const SimplifyIndexMathPass &other)
      : PassWrapper<SimplifyIndexMathPass, OperationPass<func::FuncOp>>(other) {
    tileK.setValue(other.tileK.getValue());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandAffineApply, DivOfMultiple, RemOfMultiple, MulOfDiv,
                 MulOfAddConst>(
        &getContext());
    patterns.add<GroupRowOfTile>(&getContext(), tileK.getValue());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Load vectorisation
//===----------------------------------------------------------------------===//

/// Only memrefs that are never written (weights, scales, zero points) are
/// safe to vectorise: merging loads across a store to the same buffer would
/// change the values read.
static bool isReadOnly(Value memref) {
  return llvm::all_of(memref.getUsers(), [](Operation *user) {
    return isa<memref::LoadOp, vector::LoadOp, memref::AssumeAlignmentOp,
               memref::DimOp>(user);
  });
}

struct GroupKey {
  Value memref;
  SmallVector<Value> lead;   // indices before the last one
  SmallVector<Value> leaves; // sorted leaves of the last index
  bool operator<(const GroupKey &o) const {
    auto lt = [](ArrayRef<Value> a, ArrayRef<Value> b) {
      return std::lexicographical_compare(
          a.begin(), a.end(), b.begin(), b.end(), [](Value x, Value y) {
            return x.getAsOpaquePointer() < y.getAsOpaquePointer();
          });
    };
    if (memref != o.memref)
      return memref.getAsOpaquePointer() < o.memref.getAsOpaquePointer();
    if (lead != o.lead)
      return lt(lead, o.lead);
    return lt(leaves, o.leaves);
  }
};

struct VectorizeGlobalLoadsPass
    : public PassWrapper<VectorizeGlobalLoadsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeGlobalLoadsPass)
  StringRef getArgument() const override {
    return "quantforge-vectorize-global-loads";
  }
  StringRef getDescription() const override {
    return "Merge adjacent, aligned scalar global loads inside gpu.launch into "
           "vector.load (16-byte accesses; assumes aligned base addresses).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
  }

  /// Largest W (power of two, W * bytes <= 16) for which the group contains
  /// the aligned run [c0, c0 + W).
  void processBlock(Block &block, gpu::LaunchOp launch,
                    DenseMap<Value, int64_t> &alignNeeded) {
    struct Entry {
      memref::LoadOp load;
      int64_t c;
    };
    std::map<GroupKey, SmallVector<Entry>> groups;
    for (Operation &op : block) {
      auto load = dyn_cast<memref::LoadOp>(&op);
      if (!load || load.getIndices().empty())
        continue;
      auto ty = load.getMemRefType();
      if (!ty.getLayout().isIdentity() || !ty.getElementType().isIntOrFloat() ||
          !ty.hasStaticShape() || ty.getMemorySpace())
        continue;
      if (!isReadOnly(load.getMemRef()))
        continue;
      // the memref must be defined outside the launch (kernel argument)
      if (Operation *def = load.getMemRef().getDefiningOp())
        if (launch->isAncestor(def))
          continue;
      Linear l;
      decompose(load.getIndices().back(), l);
      if (l.leaves.empty() && l.c == 0)
        continue;
      llvm::sort(l.leaves, [](Value a, Value b) {
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      });
      GroupKey key{load.getMemRef(),
                   SmallVector<Value>(load.getIndices().drop_back()), l.leaves};
      groups[key].push_back({load, l.c});
    }

    for (auto &[key, entries] : groups) {
      auto ty = cast<MemRefType>(key.memref.getType());
      int64_t bytes = std::max<int64_t>(1, ty.getElementTypeBitWidth() / 8);
      if (ty.getElementTypeBitWidth() % 8)
        continue;
      int64_t rowLen = ty.getShape().back();
      // distinct constants -> first load with that constant
      std::map<int64_t, memref::LoadOp> byC;
      // further loads of the same element (e.g. the per-item scale loads of a
      // K tile that share one quantisation group): replaced by the same extract
      std::map<int64_t, SmallVector<memref::LoadOp>> dups;
      for (Entry &e : entries)
        if (!byC.emplace(e.c, e.load).second)
          dups[e.c].push_back(e.load);
      Linear shape;
      shape.leaves = key.leaves;
      for (int64_t W = 16 / bytes; W >= 2; W /= 2) {
        if (rowLen % W != 0 || !leavesDivisibleBy(shape, W))
          continue;
        bool progress = true;
        while (progress) {
          progress = false;
          for (auto &[c0, first] : byC) {
            if (c0 % W != 0)
              continue;
            bool all = true;
            for (int64_t j = 0; j < W && all; ++j)
              all = byC.count(c0 + j);
            if (!all)
              continue;
            emit(key, byC, dups, c0, W, launch, alignNeeded, bytes);
            for (int64_t j = 0; j < W; ++j)
              byC.erase(c0 + j);
            progress = true;
            break; // iterator invalidated
          }
        }
      }
    }
  }

  void emit(const GroupKey &key, std::map<int64_t, memref::LoadOp> &byC,
            std::map<int64_t, SmallVector<memref::LoadOp>> &dups, int64_t c0, int64_t W, gpu::LaunchOp launch,
            DenseMap<Value, int64_t> &alignNeeded, int64_t bytes) {
    // the vector load goes where the first (in program order) load of the run is
    memref::LoadOp anchor = byC[c0];
    for (int64_t j = 1; j < W; ++j)
      if (byC[c0 + j]->isBeforeInBlock(anchor))
        anchor = byC[c0 + j];
    OpBuilder b(anchor);
    Location loc = anchor.getLoc();
    Value base = buildSum(b, loc, key.leaves, c0);
    SmallVector<Value> idx(key.lead.begin(), key.lead.end());
    idx.push_back(base);
    auto elemTy = cast<MemRefType>(key.memref.getType()).getElementType();
    auto vecTy = VectorType::get({W}, elemTy);
    Value vec = b.create<vector::LoadOp>(loc, vecTy, key.memref, idx);
    for (int64_t j = 0; j < W; ++j) {
      memref::LoadOp ld = byC[c0 + j];
      OpBuilder eb(anchor);
      eb.setInsertionPointAfter(vec.getDefiningOp());
      Value e = eb.create<vector::ExtractOp>(ld.getLoc(), vec,
                                             ArrayRef<int64_t>{j});
      ld.getResult().replaceAllUsesWith(e);
      // loads of the same element: all later in program order than `anchor`
      for (memref::LoadOp d : dups[c0 + j])
        d.getResult().replaceAllUsesWith(e);
    }
    for (int64_t j = 0; j < W; ++j) {
      byC[c0 + j]->erase();
      for (memref::LoadOp d : dups[c0 + j])
        d->erase();
    }
    int64_t &a = alignNeeded[key.memref];
    a = std::max(a, W * bytes);
    (void)launch;
  }

  /// Loads of read-only buffers with identical operands inside one block read
  /// the same value no matter what is stored in between (CSE cannot know: the
  /// intervening shared-memory stores are opaque writes to it). After
  /// index simplification the 8 per-item scale / zero-point loads of a K tile
  /// are the same load.
  static void dedupeReadOnlyLoads(Block &block) {
    std::map<std::pair<const void *, std::pair<const void *, SmallVector<const void *>>>,
             Operation *>
        seen;
    SmallVector<Operation *> dead;
    for (Operation &op : block) {
      Value mem;
      Type ty;
      ValueRange idx;
      if (auto l = dyn_cast<vector::LoadOp>(&op)) {
        mem = l.getBase(); ty = l.getType(); idx = l.getIndices();
      } else if (auto l = dyn_cast<memref::LoadOp>(&op)) {
        mem = l.getMemRef(); ty = l.getType(); idx = l.getIndices();
      } else {
        continue;
      }
      if (!isReadOnly(mem) || cast<MemRefType>(mem.getType()).getMemorySpace())
        continue;
      SmallVector<const void *> k;
      for (Value i : idx)
        k.push_back(i.getAsOpaquePointer());
      auto key = std::make_pair(mem.getAsOpaquePointer(),
                                std::make_pair(ty.getAsOpaquePointer(), k));
      auto [it, inserted] = seen.try_emplace(key, &op);
      if (!inserted) {
        op.getResult(0).replaceAllUsesWith(it->second->getResult(0));
        dead.push_back(&op);
      }
    }
    for (Operation *op : dead)
      op->erase();
  }

  void runOnOperation() override {
    SmallVector<gpu::LaunchOp> launches;
    getOperation().walk([&](gpu::LaunchOp op) { launches.push_back(op); });
    for (gpu::LaunchOp launch : launches) {
      DenseMap<Value, int64_t> alignNeeded;
      SmallVector<Block *> blocks;
      launch.getBody().walk([&](Block *blk) { blocks.push_back(blk); });
      for (Block *blk : blocks) {
        processBlock(*blk, launch, alignNeeded);
        dedupeReadOnlyLoads(*blk);
      }
      OpBuilder b(&launch.getBody().front(), launch.getBody().front().begin());
      for (auto [memref, align] : alignNeeded)
        b.create<memref::AssumeAlignmentOp>(
            launch.getLoc(), memref, static_cast<uint32_t>(align));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createSimplifyIndexMathPass() {
  return std::make_unique<SimplifyIndexMathPass>();
}
void mlir::quantforge::registerSimplifyIndexMathPass() {
  PassRegistration<SimplifyIndexMathPass>();
}
std::unique_ptr<Pass> mlir::quantforge::createVectorizeGlobalLoadsPass() {
  return std::make_unique<VectorizeGlobalLoadsPass>();
}
void mlir::quantforge::registerVectorizeGlobalLoadsPass() {
  PassRegistration<VectorizeGlobalLoadsPass>();
}
