//===----------------------------------------------------------------------===//
// SwizzleLoadPass — Task 4.3: XOR Swizzling for Bank Conflict Elimination
//
// Shared memory on Ampere has 32 banks × 4 bytes/bank. With FP16 matrices
// (2 bytes/element), consecutive rows accessing the same column produce
// 32-way bank conflicts, reducing SRAM throughput by 32×.
//
// Solution: XOR swizzle addressing
//   Physical_Col = Logical_Col ^ (Logical_Row % phase)
//   where phase = 8 for FP16 (8 rows × 4 banks/element = 32 banks)
//
// This ensures that for any 8-row window, each row's elements land in a
// different set of banks:
//   row 0: col ^ 0 → no shift
//   row 1: col ^ 1 → 4-byte shift
//   row 2: col ^ 2 → 8-byte shift
//   ...
//   row 7: col ^ 7 → 28-byte shift
//
// Implementation:
//   Walks all memref.load and memref.store ops that access SRAM
//   (memory space = 3) and rewrites the column index with:
//     %row_mod  = arith.remui %row, %phase
//     %new_col  = arith.xori  %col, %row_mod
//
// This pass is the single source of truth for swizzle index rewriting,
// applied at memref pointer semantics where shared-memory layout is explicit.
//
// Pipeline position:
//   quantforge-smem-promotion → **quantforge-swizzle-load** → lower-to-nvvm
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-swizzle-load"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return true if the given memref type has Shared Memory space
/// (memory_space = 3 or #gpu.address_space<workgroup>).
static bool isSharedMemorySpace(MemRefType memrefTy) {
  Attribute space = memrefTy.getMemorySpace();
  if (!space)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(space))
    return intAttr.getInt() == 3;
  if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(space))
    return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
  return false;
}

/// Return the MemRefType of an SRAM access-like operation.
static MemRefType getMemRefType(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getMemRefType();
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getMemRefType();
  if (auto asyncCopy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op))
    return cast<MemRefType>(asyncCopy.getDst().getType());
  return {};
}

/// Return the number of destination indices for an SRAM access-like op.
static unsigned getNumIndices(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getIndices().size();
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getIndices().size();
  if (auto asyncCopy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op))
    return asyncCopy.getDstIndices().size();
  return 0;
}

/// Return the row index (index 0) of an SRAM access-like op.
static Value getRowIndex(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getIndices()[0];
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getIndices()[0];
  if (auto asyncCopy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op))
    return asyncCopy.getDstIndices()[0];
  return {};
}

/// Return the col index (index 1) of an SRAM access-like op.
static Value getColIndex(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getIndices()[1];
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getIndices()[1];
  if (auto asyncCopy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op))
    return asyncCopy.getDstIndices()[1];
  return {};
}

/// Get mutable indices for column swizzling.
static MutableOperandRange getMutableIndices(Operation *op) {
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getIndicesMutable();
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getIndicesMutable();
  if (auto asyncCopy = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op))
    return asyncCopy.getDstIndicesMutable();
  llvm_unreachable("expected load/store/async copy");
}

//===----------------------------------------------------------------------===//
// SwizzleLoadPass
//===----------------------------------------------------------------------===//

struct SwizzleLoadPass
    : public PassWrapper<SwizzleLoadPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwizzleLoadPass)

  //------------------------------------------------------------------
  // Constructors
  //------------------------------------------------------------------
  SwizzleLoadPass() = default;

  SwizzleLoadPass(const SwizzleLoadPass &other)
      : PassWrapper<SwizzleLoadPass, OperationPass<func::FuncOp>>(other) {
    swizzlePhase.setValue(other.swizzlePhase.getValue());
  }

  //------------------------------------------------------------------
  // Pass metadata
  //------------------------------------------------------------------
  StringRef getArgument() const override { return "quantforge-swizzle-load"; }

  StringRef getDescription() const override {
    return "XOR swizzle addressing for SRAM loads/stores to eliminate "
           "shared memory bank conflicts. Rewrites column indices: "
           "new_col = col ^ (row % phase).";
  }

  //------------------------------------------------------------------
  // Configurable options
  //------------------------------------------------------------------
  Option<int64_t> swizzlePhase{
      *this, "swizzle-phase",
      llvm::cl::desc("XOR swizzle period (default 8 for FP16, 32 banks)"),
      llvm::cl::init(8)};

  //------------------------------------------------------------------
  // Dependent dialects
  //------------------------------------------------------------------
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    gpu::GPUDialect, nvgpu::NVGPUDialect>();
  }

  //------------------------------------------------------------------
  // runOnOperation
  //------------------------------------------------------------------
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    IRRewriter rewriter(ctx);
    int64_t swizzledCount = 0;

    // Collect all SRAM accesses (memref.load/store and nvgpu async copies).
    SmallVector<Operation *> opsToSwizzle;
    funcOp.walk([&](Operation *op) {
      if (!isa<memref::LoadOp, memref::StoreOp, nvgpu::DeviceAsyncCopyOp>(op))
        return;

      // Skip already-swizzled ops
      if (op->hasAttr("swizzled"))
        return;

      MemRefType memrefTy = getMemRefType(op);
      if (!memrefTy || !isSharedMemorySpace(memrefTy))
        return;

      // Only swizzle 2D accesses (row, col)
      if (getNumIndices(op) != 2)
        return;

      opsToSwizzle.push_back(op);
    });

    if (opsToSwizzle.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[SwizzleLoad] no SRAM load/store ops to swizzle\n");
      return;
    }

    // ── Swizzle each op's column index ────────────────────────────
    for (Operation *op : opsToSwizzle) {
      Location loc = op->getLoc();
      Value rowIdx = getRowIndex(op);
      Value colIdx = getColIndex(op);

      rewriter.setInsertionPoint(op);

      // %phase_const = arith.constant <swizzlePhase>
      Value phaseConst =
          rewriter.create<arith::ConstantIndexOp>(loc, swizzlePhase);

      // %row_mod = arith.remui %row, %phase_const
      Value rowMod = rewriter.create<arith::RemUIOp>(loc, rowIdx, phaseConst);

      // %swizzled_col = arith.xori %col, %row_mod
      Value swizzledCol = rewriter.create<arith::XOrIOp>(loc, colIdx, rowMod);

      // Replace column index in the load/store
      MutableOperandRange mutIndices = getMutableIndices(op);
      mutIndices[1].assign(swizzledCol);

      // Mark as swizzled for idempotency
      op->setAttr("swizzled", rewriter.getUnitAttr());

      LLVM_DEBUG(llvm::dbgs() << "[SwizzleLoad] swizzled " << *op << "\n");
      ++swizzledCount;
    }

    LLVM_DEBUG(llvm::dbgs() << "[SwizzleLoad] swizzled " << swizzledCount
                            << " SRAM access(es)\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::quantforge::createSwizzleLoadPass() {
  return std::make_unique<SwizzleLoadPass>();
}

void mlir::quantforge::registerSwizzleLoadPass() {
  PassRegistration<SwizzleLoadPass>();
}
