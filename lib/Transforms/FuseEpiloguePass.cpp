//===----------------------------------------------------------------------===//
// quantforge-fuse-epilogue{kind=relu|silu|gelu} — activation on the accumulator
//
// The kernel computes C = C_init + A @ dequant(W). The final global store of
// every accumulator fragment is rewritten to store act(value) instead, so
// act(bias + A @ W) costs no extra pass over the output (one read + one write
// of the whole [M, N] tensor saved per activation).
//
//   relu : max(x, 0)
//   silu : x * sigmoid(x)
//   gelu : x * sigmoid(2*sqrt(2/pi) * (x + 0.044715 x^3))   (the tanh form:
//          0.5 (1 + tanh z) == sigmoid(2 z) exactly)
//   swiglu: fused MLP gate / up. The weights are column-interleaved (gate_j at column
//          2j, up_j at 2j+1), so the two columns of every f32 vector<2> accumulator
//          fragment are (gate_j, up_j): the store becomes  C[r, c / 2] = silu(v0) * v1
//          (to the ROOT output memref: the subview chain is folded into absolute
//          coordinates). The output is half as wide as the GEMM; needs
//          quantforge-overwrite-output (the pair of columns is not where it is read).
//
// sigmoid(z) = 1 / (1 + 2^(-z log2 e)) uses inline `ex2.approx.ftz.f32` (there is no
// libdevice, and convert-gpu-to-nvvm rejects llvm.intr.exp2). Only f32
// accumulators: stores of f32 values to a *global* memref inside gpu.launch are
// rewritten, which in the fused single-kernel pipeline are exactly the C tile
// (the unfused dequantisation kernel stores f16).
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-fuse-epilogue"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

struct FuseEpiloguePass
    : public PassWrapper<FuseEpiloguePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseEpiloguePass)
  StringRef getArgument() const override { return "quantforge-fuse-epilogue"; }
  StringRef getDescription() const override {
    return "Apply relu / silu / gelu to the accumulator fragments right before "
           "they are stored to the output.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect, vector::VectorDialect, scf::SCFDialect>();
  }
  Option<std::string> kind{*this, "kind",
                           llvm::cl::desc("relu | silu | gelu (none: no-op)"),
                           llvm::cl::init("none")};
  FuseEpiloguePass() = default;
  Option<int64_t> validRows{*this, "valid-rows",
                            llvm::cl::desc("f16-store / int8-dequant: only the first N rows of the output exist; the stores of the others are skipped "
                                           "(0 = all rows)"),
                            llvm::cl::init(0)};
  FuseEpiloguePass(const FuseEpiloguePass &other)
      : PassWrapper<FuseEpiloguePass, OperationPass<func::FuncOp>>(other) {
    kind.setValue(other.kind.getValue());
    validRows.setValue(other.validRows.getValue());
  }

  static Value constant(OpBuilder &b, Location loc, Type ty, double v) {
    if (auto vt = dyn_cast<VectorType>(ty))
      return b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vt, b.getF32FloatAttr(v)));
    return b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(v));
  }

  /// 2^x with the PTX approximation instruction (scalar f32; vectors are unrolled).
  static Value exp2(OpBuilder &b, Location loc, Value x) {
    if (auto vt = dyn_cast<VectorType>(x.getType())) {
      Value out = b.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vt, b.getF32FloatAttr(0.0f)));
      for (int64_t i = 0; i < vt.getNumElements(); ++i) {
        Value e = b.create<vector::ExtractOp>(loc, x, ArrayRef<int64_t>{i});
        out = b.create<vector::InsertOp>(loc, exp2(b, loc, e), out,
                                         ArrayRef<int64_t>{i});
      }
      return out;
    }
    return b
        .create<LLVM::InlineAsmOp>(
            loc, /*resultTypes=*/TypeRange{b.getF32Type()},
            /*operands=*/ValueRange{x}, "ex2.approx.ftz.f32 $0, $1;", "=f,f",
            /*has_side_effects=*/false, /*is_align_stack=*/false,
            LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
            /*operand_attrs=*/ArrayAttr())
        .getResult(0);
  }

  static Value sigmoid(OpBuilder &b, Location loc, Value z) {
    Type ty = z.getType();
    Value negLog2e = constant(b, loc, ty, -1.4426950408889634);
    Value e = exp2(b, loc, b.create<arith::MulFOp>(loc, z, negLog2e));
    Value one = constant(b, loc, ty, 1.0);
    return b.create<arith::DivFOp>(loc, one, b.create<arith::AddFOp>(loc, one, e));
  }

  Value apply(OpBuilder &b, Location loc, Value x) const {
    Type ty = x.getType();
    if (kind == "relu")
      return b.create<arith::MaximumFOp>(loc, x, constant(b, loc, ty, 0.0));
    if (kind == "silu")
      return b.create<arith::MulFOp>(loc, x, sigmoid(b, loc, x));
    // gelu (tanh form)
    Value x2 = b.create<arith::MulFOp>(loc, x, x);
    Value cubic = b.create<arith::MulFOp>(loc, x2, constant(b, loc, ty, 0.044715));
    Value inner = b.create<arith::AddFOp>(
        loc, x, b.create<arith::MulFOp>(loc, cubic, x));
    Value z = b.create<arith::MulFOp>(loc, inner,
                                      constant(b, loc, ty, 1.5957691216057308));
    return b.create<arith::MulFOp>(loc, x, sigmoid(b, loc, z));
  }


  /// int8-dequant: the kernel computes C[i32] = A[i8] @ Bt[i8]^T; the program then converts it to f16 with a row scale,
  /// a column scale and a bias in an elementwise kernel (`y = f16(c * sx[m] * sw[n] + bias[n])`). Fold that into the GEMM's
  /// final stores: every vector<2xi32> accumulator store becomes the dequantised pair, stored to the f16 output; the
  /// elementwise kernel and the int32 buffer disappear. The operands are the function arguments tagged
  /// quantforge.int8_row_scale / int8_col_scale / int8_bias (optional) / int8_out.
  /// f16-store (fMode): the same machinery for a GEMM with an f32 accumulator: y = f16(c + bias[n]) -- the f32 -> f16 conversion (and the bias
  /// add) of a plain W4A16 GEMM folded into its stores, so that the call is ONE kernel with no f32 buffer. Tags: quantforge.int8_bias
  /// (f16 or f32, optional) and quantforge.int8_out.
  /// atomic-add (atomic mode, implies fMode): the GEMM is split along K (`split-k=S`): every block of the K grid axis holds a partial
  /// result [m, n] in the partial buffer P[z, m, n]. Instead of storing it there (and running a second kernel that adds the S slices), each
  /// accumulator pair is added with red.global.add.f32 to the f32 output argument (tagged quantforge.int8_out, initialised by the caller: zeros,
  /// or a value to accumulate onto). P, its allocation and the reduce kernel disappear: one kernel, no scratch buffer (so it can be captured
  /// in a CUDA graph). The summation order across the K slices is not deterministic (float rounding differs by ~1 ulp from run to run).
  void runInt8Dequant(bool fMode = false, bool atomic = false) {
    func::FuncOp func = getOperation();
    Value sx, sw, bias, out;
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      if (func.getArgAttr(i, "quantforge.int8_row_scale")) sx = func.getArgument(i);
      if (func.getArgAttr(i, "quantforge.int8_col_scale")) sw = func.getArgument(i);
      if (func.getArgAttr(i, "quantforge.int8_bias")) bias = func.getArgument(i);
      if (func.getArgAttr(i, "quantforge.int8_out")) out = func.getArgument(i);
    }
    if (atomic) {
      auto ot = out ? dyn_cast<MemRefType>(out.getType()) : MemRefType();
      if (!ot || ot.getRank() != 2 || !ot.getElementType().isF32()) {
        func.emitError("epilogue=atomic-add needs a function argument tagged quantforge.int8_out that is a rank-2 f32 memref");
        return signalPassFailure();
      }
    }
    if (!out || (!fMode && (!sx || !sw))) {
      func.emitError("epilogue=int8-dequant / f16-store needs function arguments tagged quantforge.int8_row_scale, "
                     "quantforge.int8_col_scale and quantforge.int8_out");
      return signalPassFailure();
    }
    SmallVector<vector::StoreOp> accStores;
    func.walk([&](gpu::LaunchOp launch) {
      launch.walk([&](vector::StoreOp s) {
        auto vt = dyn_cast<VectorType>(s.getValueToStore().getType());
        if (!s.getMemRefType().getMemorySpace() && vt && (fMode ? vt.getElementType().isF32() : vt.getElementType().isInteger(32)))
          accStores.push_back(s);
      });
    });
    if (accStores.empty()) {
      func.emitError("epilogue=int8-dequant / f16-store found no accumulator stores (needs the Tensor Core path with hoist-acc)");
      return signalPassFailure();
    }
    Value cRoot;
    llvm::SmallPtrSet<Operation *, 4> gemmLaunches; // the kernel(s) whose stores are rewritten: they must survive
    for (vector::StoreOp st : accStores)
      gemmLaunches.insert(st->getParentOfType<gpu::LaunchOp>().getOperation());
    for (vector::StoreOp st : accStores) {
      auto vt = cast<VectorType>(st.getValueToStore().getType());
      if (vt.getNumElements() != 2 || vt.getRank() != 1) {
        st.emitError("epilogue=int8-dequant needs vector<2xi32> accumulator stores");
        return signalPassFailure();
      }
      OpBuilder b(st);
      Location loc = st.getLoc();
      Value memref = st.getBase();
      Value row = st.getIndices()[0], col = st.getIndices()[1];
      while (auto sv = memref.getDefiningOp<memref::SubViewOp>()) {
        auto offs = sv.getMixedOffsets();
        for (OpFoldResult stride : sv.getMixedStrides())
          if (getConstantIntValue(stride) != 1) {
            sv.emitError("int8-dequant epilogue needs unit strides");
            return signalPassFailure();
          }
        // A rank-reducing subview (the tile P[z, m0:m0+16, n0:n0+64] of the split-K partial buffer, seen as a [16, 64] view) drops unit
        // dimensions: the offsets of the kept dimensions are the row / column offsets, the dropped one (z) selects a slice that all
        // slices of the atomic epilogue share.
        llvm::SmallBitVector dropped = sv.getDroppedDims();
        SmallVector<OpFoldResult> kept;
        for (unsigned d = 0; d < offs.size(); ++d)
          if (!dropped[d])
            kept.push_back(offs[d]);
        if (kept.size() != 2 || (dropped.any() && !atomic)) {
          sv.emitError("int8-dequant / f16-store epilogue needs rank-2 subviews of the output");
          return signalPassFailure();
        }
        row = b.create<arith::AddIOp>(loc, row, getValueOrCreateConstantIndexOp(b, loc, kept[0]));
        col = b.create<arith::AddIOp>(loc, col, getValueOrCreateConstantIndexOp(b, loc, kept[1]));
        memref = sv.getSource();
      }
      if (cRoot && cRoot != memref) {
        st.emitError("int8-dequant epilogue: stores to more than one output buffer");
        return signalPassFailure();
      }
      cRoot = memref;
      Type f32 = b.getF32Type(), f16 = b.getF16Type();
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      if (atomic) {
        auto outTy = cast<MemRefType>(out.getType());
        Type i64 = b.getI64Type();
        Value baseIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, out);
        Value ld = b.create<arith::ConstantIndexOp>(loc, outTy.getDimSize(1));
        Value four = b.create<arith::ConstantIndexOp>(loc, 4);
        if (validRows > 0) {
          Value inside = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row,
                                                 b.create<arith::ConstantIndexOp>(loc, validRows));
          auto guard = b.create<scf::IfOp>(loc, inside, /*withElseRegion=*/false);
          b.setInsertionPointToStart(guard.thenBlock());
        }
        for (int64_t k = 0; k < 2; ++k) {
          Value ck = k ? b.create<arith::AddIOp>(loc, col, one).getResult() : col;
          Value e = b.create<vector::ExtractOp>(loc, st.getValueToStore(), ArrayRef<int64_t>{k});
          Value off = b.create<arith::MulIOp>(loc, b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, row, ld), ck), four);
          Value addr = b.create<arith::IndexCastOp>(loc, i64, b.create<arith::AddIOp>(loc, baseIdx, off));
          b.create<LLVM::InlineAsmOp>(loc, /*resultTypes=*/TypeRange{}, /*operands=*/ValueRange{addr, e},
                                      "red.global.add.f32 [$0], $1;", "l,f", /*has_side_effects=*/true, /*is_align_stack=*/false,
                                      LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
                                      /*operand_attrs=*/ArrayAttr());
        }
        st->erase();
        continue;
      }
      Value rs = fMode ? Value() : b.create<memref::LoadOp>(loc, sx, ValueRange{row});
      Value res = b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(VectorType::get({2}, f16), b.getF16FloatAttr(0.0f)));
      for (int64_t k = 0; k < 2; ++k) {
        Value ck = k ? b.create<arith::AddIOp>(loc, col, one).getResult() : col;
        Value e = b.create<vector::ExtractOp>(loc, st.getValueToStore(), ArrayRef<int64_t>{k});
        Value v = e;
        if (!fMode) {
          Value f = b.create<arith::SIToFPOp>(loc, f32, e);
          v = b.create<arith::MulFOp>(loc, b.create<arith::MulFOp>(loc, f, rs),
                                      b.create<memref::LoadOp>(loc, sw, ValueRange{ck}));
        }
        if (bias) {
          Value bv = b.create<memref::LoadOp>(loc, bias, ValueRange{ck});
          if (!bv.getType().isF32())
            bv = b.create<arith::ExtFOp>(loc, f32, bv);
          v = b.create<arith::AddFOp>(loc, v, bv);
        }
        Value h = b.create<arith::TruncFOp>(loc, f16, v);
        res = b.create<vector::InsertOp>(loc, h, res, ArrayRef<int64_t>{k});
      }
      // One 4-byte store for the pair of f16 (col is even): the generic lowering of vector<2xf16> stores gives two 2-byte stores
      // (the memref alignment is that of the element), which writes every 32-byte sector a quarter at a time.
      auto outTy = cast<MemRefType>(out.getType());
      Type i32 = b.getI32Type(), i64 = b.getI64Type();
      Value packed = b.create<vector::ExtractOp>(
          loc, b.create<vector::BitCastOp>(loc, VectorType::get({1}, i32), res), ArrayRef<int64_t>{0});
      Value baseIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, out);
      Value stride = b.create<arith::ConstantIndexOp>(loc, outTy.getDimSize(1));
      Value two = b.create<arith::ConstantIndexOp>(loc, 2);
      Value off = b.create<arith::MulIOp>(loc, b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, row, stride), col), two);
      Value addr = b.create<arith::IndexCastOp>(loc, i64, b.create<arith::AddIOp>(loc, baseIdx, off));
      if (validRows > 0) {                       // rows past the real row count do not exist in the output tensor
        Value inside = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, row,
                                               b.create<arith::ConstantIndexOp>(loc, validRows));
        auto guard = b.create<scf::IfOp>(loc, inside, /*withElseRegion=*/false);
        b.setInsertionPointToStart(guard.thenBlock());
      }
      b.create<LLVM::InlineAsmOp>(loc, /*resultTypes=*/TypeRange{}, /*operands=*/ValueRange{addr, packed},
                                  "st.global.b32 [$0], $1;", "l,r", /*has_side_effects=*/true, /*is_align_stack=*/false,
                                  LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
                                  /*operand_attrs=*/ArrayAttr());
      st->erase();
    }
    // The elementwise consumer kernel(s) of the int32 buffer and the buffer itself are dead now.
    SmallVector<gpu::LaunchOp> dead;
    func.walk([&](gpu::LaunchOp launch) {
      if (gemmLaunches.contains(launch.getOperation()))
        return;
      bool usesC = false;
      launch.walk([&](Operation *op) {
        for (Value v : op->getOperands()) {
          Value r = v;
          while (auto sv = r.getDefiningOp<memref::SubViewOp>())
            r = sv.getSource();
          if (r == cRoot)
            usesC = true;
        }
      });
      if (usesC)
        dead.push_back(launch);
    });
    for (gpu::LaunchOp l : dead)
      l->erase();
    // Subviews of the buffer that nothing reads any more (the address computations of the rewritten stores).
    for (bool changed = true; changed;) {
      changed = false;
      SmallVector<Operation *> users(cRoot.getUsers().begin(), cRoot.getUsers().end());
      for (Operation *u : users) {
        auto sv = dyn_cast<memref::SubViewOp>(u);
        if (!sv)
          continue;
        // erase a chain of dead subviews bottom-up
        SmallVector<Operation *> chain{sv};
        for (size_t i = 0; i < chain.size(); ++i)
          for (Operation *user : chain[i]->getResult(0).getUsers())
            if (isa<memref::SubViewOp>(user))
              chain.push_back(user);
        bool dead = llvm::all_of(chain, [&](Operation *c) {
          return llvm::all_of(c->getResult(0).getUsers(), [&](Operation *x) { return isa<memref::SubViewOp>(x); });
        });
        if (!dead)
          continue;
        for (auto it = chain.rbegin(); it != chain.rend(); ++it)
          (*it)->erase();
        changed = true;
        break;
      }
    }
    if (Operation *alloc = cRoot.getDefiningOp()) {
      SmallVector<Operation *> users(cRoot.getUsers().begin(), cRoot.getUsers().end());
      bool onlyDealloc = llvm::all_of(users, [](Operation *u) { return isa<gpu::DeallocOp, memref::DeallocOp>(u); });
      if (onlyDealloc && isa<gpu::AllocOp, memref::AllocOp>(alloc)) {
        for (Operation *u : users)
          u->erase();
        alloc->erase();
      }
    }
  }

  static bool isF32OrVecF32(Type t) {
    if (auto vt = dyn_cast<VectorType>(t))
      t = vt.getElementType();
    return t.isF32();
  }

  void runOnOperation() override {
    if (kind == "none" || kind.empty())
      return;
    if (kind != "relu" && kind != "silu" && kind != "gelu" && kind != "swiglu" && kind != "int8-dequant" && kind != "f16-store" && kind != "atomic-add") {
      getOperation().emitError("quantforge-fuse-epilogue: unknown kind '")
          << kind << "' (relu | silu | gelu | swiglu | int8-dequant | f16-store | atomic-add)";
      return signalPassFailure();
    }
    if (kind == "int8-dequant")
      return runInt8Dequant();
    if (kind == "f16-store")
      return runInt8Dequant(true);
    if (kind == "atomic-add")
      return runInt8Dequant(true, true);
    SmallVector<Operation *> stores;
    getOperation().walk([&](gpu::LaunchOp launch) {
      launch.walk([&](Operation *op) {
        if (auto s = dyn_cast<vector::StoreOp>(op)) {
          if (!s.getMemRefType().getMemorySpace() &&
              isF32OrVecF32(s.getValueToStore().getType()))
            stores.push_back(op);
        } else if (auto s = dyn_cast<memref::StoreOp>(op)) {
          if (!s.getMemRefType().getMemorySpace() &&
              isF32OrVecF32(s.getValue().getType()))
            stores.push_back(op);
        }
      });
    });
    if (kind == "swiglu") {
      for (Operation *op : stores) {
        auto st = dyn_cast<vector::StoreOp>(op);
        auto vt = st ? dyn_cast<VectorType>(st.getValueToStore().getType()) : VectorType();
        if (!st || !vt || vt.getNumElements() != 2) {
          op->emitError("epilogue=swiglu needs the Tensor Core path (vector<2xf32> "
                        "accumulator stores)");
          return signalPassFailure();
        }
        OpBuilder b(st);
        Location loc = st.getLoc();
        Value memref = st.getBase();
        Value row = st.getIndices()[0], col = st.getIndices()[1];
        while (auto sv = memref.getDefiningOp<memref::SubViewOp>()) {
          auto offs = sv.getMixedOffsets();
          for (OpFoldResult stride : sv.getMixedStrides())
            if (getConstantIntValue(stride) != 1) {
              sv.emitError("swiglu epilogue needs unit strides");
              return signalPassFailure();
            }
          row = b.create<arith::AddIOp>(loc, row, getValueOrCreateConstantIndexOp(b, loc, offs[0]));
          col = b.create<arith::AddIOp>(loc, col, getValueOrCreateConstantIndexOp(b, loc, offs[1]));
          memref = sv.getSource();
        }
        Value v = st.getValueToStore();
        Value gate = b.create<vector::ExtractOp>(loc, v, ArrayRef<int64_t>{0});
        Value up = b.create<vector::ExtractOp>(loc, v, ArrayRef<int64_t>{1});
        Value silu = b.create<arith::MulFOp>(loc, gate, sigmoid(b, loc, gate));
        Value res = b.create<arith::MulFOp>(loc, silu, up);
        Value half = b.create<arith::DivUIOp>(loc, col, b.create<arith::ConstantIndexOp>(loc, 2));
        b.create<memref::StoreOp>(loc, res, memref, ValueRange{row, half});
        st->erase();
      }
      return;
    }
    for (Operation *op : stores) {
      OpBuilder b(op);
      if (auto s = dyn_cast<vector::StoreOp>(op))
        s.getValueToStoreMutable().assign(apply(b, op->getLoc(), s.getValueToStore()));
      else if (auto s = dyn_cast<memref::StoreOp>(op))
        s.getValueMutable().assign(apply(b, op->getLoc(), s.getValue()));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createFuseEpiloguePass() {
  return std::make_unique<FuseEpiloguePass>();
}
void mlir::quantforge::registerFuseEpiloguePass() {
  PassRegistration<FuseEpiloguePass>();
}
