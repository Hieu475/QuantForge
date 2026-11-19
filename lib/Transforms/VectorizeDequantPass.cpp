//===----------------------------------------------------------------------===//
// quantforge-vectorize-dequant — packed f16x2 dequantisation
//
// After FuseProducerIntoLoads + VectorizeGlobalLoads the fused B-tile staging
// computes, for the 8 columns of one 16-byte work item,
//
//   P = vector.load packed : vector<4xi8>          (2 INT4 per byte)
//   S = vector.load scale  : vector<8xf16>         (or one scalar)
//   Z = vector.load zero   : vector<8xi8>          (or one scalar)
//   w[2i]   = (f16(P[i] & 15) - f16(Z[2i]))   * S[2i]
//   w[2i+1] = (f16(P[i] >> 4) - f16(Z[2i+1])) * S[2i+1]      (i8 -> f16 converts)
//   vector.store (insert chain of the 8 w) : vector<8xf16>
//
// as eight independent scalar expression trees: ~11 instructions per element,
// 8 `cvt` among them. This pass recognises exactly that tree and emits the
// packed form (the trick Marlin uses): a nibble q placed in the mantissa of the
// f16 constant 1152 = 0x6480 gives the f16 value 1152 + q EXACTLY, so
//
//   (1152 + q) - (1152 + zp) = q - zp         (one sub.f16x2 per 2 elements)
//   (q - zp) * s                              (one mul.f16x2 per 2 elements)
//
// with 1152 + zp built from the zero-point byte by flipping its sign bit
// (zp ^ 0x80 = zp + 128 as an unsigned byte) and OR-ing 0x6400. Both
// differences are exact integers < 2048 and the single rounding happens in the
// multiply, so the result is bit-identical to the scalar expression.
//
// int8 weights (W8A16) use the same form with the weight byte instead of a
// nibble ((w ^ 0x80) | 0x6400 = 1152 + w).
//
// FP8 E4M3 weights (`qf.dequant {fp8}`, lowered by lower-dequant-to-arith to
//   w = mulf(mulf(bitcast(((b & 0x7F) << 7) | ((b & 0x80) << 8)), 256), s) )
// are recognised too: two bytes are spread over the halves of an i32, the
// shifts / masks run on both halves at once, and the two multiplies are
// f16x2 (the x256 is exact, so the result is bit-identical).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-vectorize-dequant"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

constexpr int kLanes = 8;

/// A value that is element `j` of a vector (`vec`), or a lane-independent
/// scalar (`vec` null).
static bool isConst(Value v, int64_t c) {
  auto x = getConstantIntValue(v);
  return x && *x == c;
}

struct LaneSource {
  Value vec;     // vector the elements were extracted from (null: uniform)
  Value uniform; // the scalar when uniform
  bool valid = false;
};

/// Does `v` equal vector.extract(src)[j]?
static bool matchExtract(Value v, int64_t j, Value &src) {
  auto ex = v.getDefiningOp<vector::ExtractOp>();
  if (!ex || ex.getStaticPosition().size() != 1 || ex.getStaticPosition()[0] != j)
    return false;
  if (src && src != ex.getVector())
    return false;
  src = ex.getVector();
  return true;
}

/// Collect per-lane sources `vals[j]` for a vector operand (scale / zero point).
/// Either all lanes are extracts of the same vector, or all lanes are the same
/// scalar value.
static LaneSource collectLanes(ArrayRef<Value> vals) {
  LaneSource r;
  Value src;
  bool allExtract = true;
  for (int64_t j = 0; j < kLanes && allExtract; ++j)
    allExtract = matchExtract(vals[j], j, src);
  if (allExtract) {
    r.vec = src;
    r.valid = true;
    return r;
  }
  if (llvm::all_equal(vals) && !vals[0].getDefiningOp<vector::ExtractOp>()) {
    r.uniform = vals[0];
    r.valid = true;
  }
  return r;
}

/// Match `a op b` for a commutative op in either operand order.
template <typename OpT>
static bool matchCommutative(Value v, Value &x, Value &y,
                             function_ref<bool(Value)> pickFirst) {
  auto op = v.getDefiningOp<OpT>();
  if (!op)
    return false;
  if (pickFirst(op->getOperand(0))) {
    x = op->getOperand(0);
    y = op->getOperand(1);
    return true;
  }
  if (pickFirst(op->getOperand(1))) {
    x = op->getOperand(1);
    y = op->getOperand(0);
    return true;
  }
  return false;
}

struct Match {
  Value packed;         // vector<4xi8>   (INT4) — or null for int8 weights
  Value bytes;          // vector<8xi8>   (INT8 weights) — or null
  LaneSource scale, zp; // per-lane scale (f16) / zero point (i8)
};

/// The 8 scalar results `elems[j]` of the dequant expression trees.
static std::optional<Match> matchTrees(ArrayRef<Value> elems) {
  Value packed, bytes;
  SmallVector<Value> scales(kLanes), zps(kLanes);
  bool int4 = true, int8 = true;
  for (int64_t j = 0; j < kLanes; ++j) {
    // w = mulf(subf(sitofp(q), sitofp(z)), s)   (mulf commutes)
    Value diff, s;
    auto isSubf = [](Value v) { return v.getDefiningOp<arith::SubFOp>() != nullptr; };
    if (!matchCommutative<arith::MulFOp>(elems[j], diff, s, isSubf))
      return std::nullopt;
    auto sub = diff.getDefiningOp<arith::SubFOp>();
    auto qf = sub.getLhs().getDefiningOp<arith::SIToFPOp>();
    auto zf = sub.getRhs().getDefiningOp<arith::SIToFPOp>();
    if (!qf || !zf || !qf.getType().isF16() || !zf.getType().isF16())
      return std::nullopt;
    scales[j] = s;
    zps[j] = zf.getIn();

    Value q = qf.getIn();
    // INT4: q = andi(b, 15) (even j) / andi(shrui(b, 4), 15) (odd j), b = P[j/2]
    bool ok4 = false;
    if (auto an = q.getDefiningOp<arith::AndIOp>()) {
      Value x = an.getLhs(), m = an.getRhs();
      if (!isConst(m, 15))
        std::swap(x, m);
      if (isConst(m, 15)) {
        Value b = x;
        if (j % 2) {
          auto sh = x.getDefiningOp<arith::ShRUIOp>();
          ok4 = sh && isConst(sh.getRhs(), 4);
          if (ok4)
            b = sh.getLhs();
        } else {
          ok4 = true;
        }
        if (ok4)
          ok4 = matchExtract(b, j / 2, packed);
      }
    }
    int4 = int4 && ok4;
    // INT8: q = W[j]
    int8 = int8 && matchExtract(q, j, bytes);
  }
  if (int4 == int8) // neither (or, impossibly, both)
    return std::nullopt;
  Match m;
  if (int4) {
    auto ty = dyn_cast<VectorType>(packed.getType());
    if (!ty || ty.getNumElements() != kLanes / 2 || !ty.getElementType().isInteger(8))
      return std::nullopt;
    m.packed = packed;
  } else {
    auto ty = dyn_cast<VectorType>(bytes.getType());
    if (!ty || ty.getNumElements() != kLanes || !ty.getElementType().isInteger(8))
      return std::nullopt;
    m.bytes = bytes;
  }
  m.scale = collectLanes(scales);
  m.zp = collectLanes(zps);
  if (!m.scale.valid || !m.zp.valid)
    return std::nullopt;
  return m;
}

/// FP8 trees: elems[j] = mulf(mulf(bitcast(ori(shli(andi(E, 0x7F), 7),
/// shli(andi(E, 0x80), 8))), 256.0), s_j) with E = extui(W[j]).
struct Fp8Match {
  Value bytes; // vector<8xi8>
  LaneSource scale;
};

static bool isConstF(Value v, double c) {
  llvm::APFloat f(0.0);
  return matchPattern(v, m_ConstantFloat(&f)) && f.convertToDouble() == c;
}

static std::optional<Fp8Match> matchFp8Trees(ArrayRef<Value> elems) {
  Value bytes;
  SmallVector<Value> scales(kLanes);
  for (int64_t j = 0; j < kLanes; ++j) {
    Value inner, s;
    auto isMul = [](Value v) { return v.getDefiningOp<arith::MulFOp>() != nullptr; };
    if (!matchCommutative<arith::MulFOp>(elems[j], inner, s, isMul))
      return std::nullopt;
    auto m256 = inner.getDefiningOp<arith::MulFOp>();
    Value f;
    if (isConstF(m256.getRhs(), 256.0))
      f = m256.getLhs();
    else if (isConstF(m256.getLhs(), 256.0))
      f = m256.getRhs();
    else
      return std::nullopt;
    auto bc = f.getDefiningOp<arith::BitcastOp>();
    if (!bc || !bc.getType().isF16())
      return std::nullopt;
    auto orr = bc.getIn().getDefiningOp<arith::OrIOp>();
    if (!orr)
      return std::nullopt;
    // one operand: shli(andi(E, 0x7F), 7), the other shli(andi(E, 0x80), 8)
    Value e;
    auto side = [&](Value v, int64_t mask, int64_t sh) {
      auto shl = v.getDefiningOp<arith::ShLIOp>();
      if (!shl || !isConst(shl.getRhs(), sh))
        return false;
      auto an = shl.getLhs().getDefiningOp<arith::AndIOp>();
      if (!an)
        return false;
      Value x = an.getLhs(), m = an.getRhs();
      if (!isConst(m, mask))
        std::swap(x, m);
      if (!isConst(m, mask))
        return false;
      if (e && e != x)
        return false;
      e = x;
      return true;
    };
    bool ok = (side(orr.getLhs(), 0x7F, 7) && side(orr.getRhs(), 0x80, 8)) ||
              (side(orr.getRhs(), 0x7F, 7) && side(orr.getLhs(), 0x80, 8));
    if (!ok)
      return std::nullopt;
    auto ext = e.getDefiningOp<arith::ExtUIOp>();
    if (!ext || !matchExtract(ext.getIn(), j, bytes))
      return std::nullopt;
    scales[j] = s;
  }
  auto ty = dyn_cast<VectorType>(bytes.getType());
  if (!ty || ty.getNumElements() != kLanes || !ty.getElementType().isInteger(8))
    return std::nullopt;
  Fp8Match m;
  m.bytes = bytes;
  m.scale = collectLanes(scales);
  if (!m.scale.valid)
    return std::nullopt;
  return m;
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

struct Emitter {
  OpBuilder &b;
  Location loc;
  Type i32 = b.getI32Type();
  Type f16 = b.getF16Type();

  Value c32(int64_t v) {
    return b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(static_cast<int32_t>(v)));
  }
  Value shr(Value v, int s) { return s ? b.create<arith::ShRUIOp>(loc, v, c32(s)).getResult() : v; }
  Value shl(Value v, int s) { return s ? b.create<arith::ShLIOp>(loc, v, c32(s)).getResult() : v; }
  Value andi(Value v, int64_t m) { return b.create<arith::AndIOp>(loc, v, c32(m)); }
  Value ori(Value v, Value w) { return b.create<arith::OrIOp>(loc, v, w); }

  /// vector<NxI8> (N multiple of 4) -> its i32 words.
  SmallVector<Value> words(Value v) {
    auto ty = cast<VectorType>(v.getType());
    int64_t n = ty.getNumElements() / 4;
    Value w = b.create<vector::BitCastOp>(loc, VectorType::get({n}, i32), v);
    SmallVector<Value> out;
    for (int64_t i = 0; i < n; ++i)
      out.push_back(b.create<vector::ExtractOp>(loc, w, ArrayRef<int64_t>{i}));
    return out;
  }

  /// 4 i32 pair-words -> vector<8xf16>.
  Value asF16x8(ArrayRef<Value> pairs) {
    Value v = b.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(VectorType::get({4}, i32), b.getI32IntegerAttr(0)));
    for (int64_t i = 0; i < 4; ++i)
      v = b.create<vector::InsertOp>(loc, pairs[i], v, ArrayRef<int64_t>{i});
    return b.create<vector::BitCastOp>(loc, VectorType::get({kLanes}, f16), v);
  }

  /// INT4: 1152 + nibble pairs. Word w holds bytes b0..b3; pair k = (lo(b_k), hi(b_k)).
  Value nibbleMagic(Value packed) {
    Value w = words(packed)[0];
    SmallVector<Value> pairs;
    for (int k = 0; k < 4; ++k) {
      Value lo = andi(shr(w, 8 * k), 0xF);
      Value hi = shl(andi(shr(w, 8 * k + 4), 0xF), 16);
      pairs.push_back(ori(ori(lo, hi), c32(0x64806480)));
    }
    return asF16x8(pairs);
  }

  /// bytes (signed, vector<8xi8>) -> 1152 + value as f16: (v ^ 0x80) | 0x6400.
  Value byteMagic(Value bytes, bool withOffset128) {
    SmallVector<Value> ws = words(bytes), pairs;
    for (Value w : ws) {
      Value x = b.create<arith::XOrIOp>(loc, w, c32(0x80808080));
      for (int h = 0; h < 2; ++h) {
        Value lo = andi(shr(x, 16 * h), 0xFF);
        Value hi = shl(andi(shr(x, 16 * h + 8), 0xFF), 16);
        pairs.push_back(ori(ori(lo, hi), c32(0x64006400)));
      }
    }
    (void)withOffset128;
    return asF16x8(pairs);
  }

  /// FP8 E4M3 bytes (vector<8xi8>) -> vector<8xf16> holding 2^-8 x the value (see the header).
  Value fp8Raw(Value bytes) {
    SmallVector<Value> pairs;
    for (Value w : words(bytes)) {
      for (int h = 0; h < 2; ++h) {
        // spread bytes (2h, 2h+1) of the word over the two halves: x = b0 | b1 << 16
        Value ws = shr(w, 16 * h);
        Value x = ori(andi(ws, 0xFF), andi(shl(ws, 8), 0x00FF0000));
        Value mag = andi(shl(x, 7), 0x3F803F80);
        Value sgn = andi(shl(x, 8), 0x80008000);
        pairs.push_back(ori(mag, sgn));
      }
    }
    return asF16x8(pairs);
  }

  /// Uniform i8 scalar -> splat vector<8xf16> of 1152 + v.
  Value uniformByteMagic(Value z) {
    Value z32 = b.create<arith::ExtUIOp>(loc, i32, z);
    Value x = b.create<arith::XOrIOp>(loc, z32, c32(0x80));
    Value lane = andi(x, 0xFF);
    Value pair = ori(ori(lane, shl(lane, 16)), c32(0x64006400));
    return asF16x8({pair, pair, pair, pair});
  }

  Value scaleVec(const LaneSource &s) {
    if (s.vec)
      return s.vec;
    return b.create<vector::BroadcastOp>(loc, VectorType::get({kLanes}, f16), s.uniform);
  }
};

struct VectorizeDequantPass
    : public PassWrapper<VectorizeDequantPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeDequantPass)
  StringRef getArgument() const override { return "quantforge-vectorize-dequant"; }
  StringRef getDescription() const override {
    return "Replace the per-element INT4/INT8 dequantisation trees feeding a "
           "vector<8xf16> store by packed f16x2 arithmetic (magic-number "
           "int->f16 conversion; bit-identical results).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, arith::ArithDialect>();
  }

  bool tryRewrite(vector::StoreOp store) {
    auto vecTy = dyn_cast<VectorType>(store.getValueToStore().getType());
    if (!vecTy || vecTy.getNumElements() != kLanes || !vecTy.getElementType().isF16())
      return false;
    // the insert chain: insert(e7, insert(e6, ... insert(e0, zeros)))
    SmallVector<Value> elems(kLanes);
    Value cur = store.getValueToStore();
    for (int64_t j = kLanes - 1; j >= 0; --j) {
      auto ins = cur.getDefiningOp<vector::InsertOp>();
      if (!ins || ins.getStaticPosition().size() != 1 || ins.getStaticPosition()[0] != j)
        return false;
      elems[j] = ins.getSource();
      cur = ins.getDest();
    }
    if (std::optional<Fp8Match> f = matchFp8Trees(elems)) {
      OpBuilder b(store);
      Emitter e{b, store.getLoc()};
      Value raw = e.fp8Raw(f->bytes);
      Value k256 = b.create<arith::ConstantOp>(
          store.getLoc(),
          DenseElementsAttr::get(VectorType::get({kLanes}, b.getF16Type()),
                                 FloatAttr::get(b.getF16Type(), 256.0)));
      Value exact = b.create<arith::MulFOp>(store.getLoc(), raw, k256);
      Value w = b.create<arith::MulFOp>(store.getLoc(), exact, e.scaleVec(f->scale));
      store.getValueToStoreMutable().assign(w);
      return true;
    }
    std::optional<Match> m = matchTrees(elems);
    if (!m)
      return false;

    OpBuilder b(store);
    Emitter e{b, store.getLoc()};
    Value q = m->packed ? e.nibbleMagic(m->packed) : e.byteMagic(m->bytes, false);
    Value z = m->zp.vec ? e.byteMagic(m->zp.vec, false) : e.uniformByteMagic(m->zp.uniform);
    // int8 weights: both operands carry the same +128 bias ((v ^ 0x80) | 0x6400);
    // INT4 nibbles were built with 0x6480 = the same 1152 base.
    Value diff = b.create<arith::SubFOp>(store.getLoc(), q, z);
    Value w = b.create<arith::MulFOp>(store.getLoc(), diff, e.scaleVec(m->scale));
    store.getValueToStoreMutable().assign(w);
    return true;
  }

  void runOnOperation() override {
    SmallVector<vector::StoreOp> stores;
    getOperation().walk([&](vector::StoreOp s) { stores.push_back(s); });
    for (vector::StoreOp s : stores)
      (void)tryRewrite(s);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createVectorizeDequantPass() {
  return std::make_unique<VectorizeDequantPass>();
}
void mlir::quantforge::registerVectorizeDequantPass() {
  PassRegistration<VectorizeDequantPass>();
}
