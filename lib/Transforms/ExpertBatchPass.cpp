//===----------------------------------------------------------------------===//
// quantforge-expert-batch-launch / quantforge-expert-batch-pointers — grouped GEMM for
// mixture-of-experts layers
//
// A MoE layer is many small GEMMs with the same shape and different weights: rows are sorted by
// expert and padded to the row tile, so that every row block belongs to ONE expert. The kernel
// of such a "problem" is the ordinary dense kernel; what the grouped version adds is
//
//   * grid.z = P problems (one row block each),
//   * one extra kernel argument, `meta : memref<P x i32>`, holding the expert of each problem
//     (a negative value: padding, the block returns at once, so unused slots cost nothing),
//   * per-problem base pointers: an `expert` argument (packed weights, scales, zero points) is
//     advanced by `meta[z]` slabs, a `rows` argument (activations, output) by `z` slabs; a slab is
//     the memref the kernel was written for, e.g. memref<KxN/2xi8>.
//
// The host buffers are therefore [E, K, N/2] weights, [P*M, K] rows ..., while the memref types of
// the kernel stay those of ONE problem. Nothing else of the pipeline (tiling, shared memory,
// pipelining, epilogues) notices.
//
// Two passes because the pointers exist only after the memref descriptors are expanded:
//   quantforge-expert-batch-launch   (module, right after gpu-kernel-outlining)
//       grid.z := P, appends `meta` to the host function, the gpu.launch_func and the gpu.func,
//       records `quantforge.expert_layout = [role, slabBytes, rank]*` on the gpu.func.
//   quantforge-expert-batch-pointers (gpu.module, after convert-gpu-to-nvvm)
//       early exit on a negative meta[z]; GEP of the aligned pointer of every expert / rows argument.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "quantforge-expert-batch"

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::quantforge;

namespace {

constexpr StringLiteral kLayoutAttr = "quantforge.expert_layout";
enum Role : int64_t { kShared = 0, kExpert = 1, kRows = 2 };

static int64_t elementBytes(Type t) {
  if (t.isIntOrFloat())
    return std::max<int64_t>(1, t.getIntOrFloatBitWidth() / 8);
  return 1;
}

struct ExpertBatchLaunchPass
    : public PassWrapper<ExpertBatchLaunchPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpertBatchLaunchPass)
  Option<int64_t> problems{*this, "problems",
                           llvm::cl::desc("number of problems (grid.z); 0 = pass disabled"),
                           llvm::cl::init(0)};
  Option<std::string> roles{
      *this, "roles",
      llvm::cl::desc("one letter per argument of the host function: e = expert weights, "
                     "r = per-problem rows, s = shared (default)"),
      llvm::cl::init("")};
  ExpertBatchLaunchPass() = default;
  ExpertBatchLaunchPass(const ExpertBatchLaunchPass &other)
      : PassWrapper<ExpertBatchLaunchPass, OperationPass<ModuleOp>>(other) {
    problems.setValue(other.problems.getValue());
    roles.setValue(other.roles.getValue());
  }
  StringRef getArgument() const override { return "quantforge-expert-batch-launch"; }
  StringRef getDescription() const override {
    return "Grouped GEMM: grid.z problems, expert index array as an extra kernel argument.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    if (problems <= 0)
      return;
    SmallVector<gpu::LaunchFuncOp> launches;
    getOperation().walk([&](gpu::LaunchFuncOp l) { launches.push_back(l); });
    llvm::DenseMap<Operation *, int> perHost;
    for (gpu::LaunchFuncOp launch : launches)
      if (++perHost[launch->getParentOfType<func::FuncOp>()] > 1) {
        launch.emitError("expert batching needs a single fused kernel per function (is fuse-dequant "
                         "active, and accumulate=false or an epilogue given?)");
        return signalPassFailure();
      }
    for (gpu::LaunchFuncOp launch : launches)
      if (failed(rewrite(launch)))
        return signalPassFailure();
  }

  LogicalResult rewrite(gpu::LaunchFuncOp launch) {
    auto host = launch->getParentOfType<func::FuncOp>();
    auto kernel = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(launch, launch.getKernel());
    if (!host || !kernel)
      return launch.emitError("expert batching needs a gpu.launch_func inside a func.func");
    if (kernel->hasAttr(kLayoutAttr))
      return success();
    Location loc = launch.getLoc();

    SmallVector<int64_t> layout;
    for (Value operand : launch.getKernelOperands()) {
      auto mt = dyn_cast<MemRefType>(operand.getType());
      if (!mt || !mt.hasStaticShape())
        return launch.emitError("expert batching: every kernel operand must be a static memref");
      int64_t role = kShared;
      if (auto ba = dyn_cast<BlockArgument>(operand))
        if (ba.getOwner()->getParentOp() == host.getOperation() && ba.getArgNumber() < roles.size()) {
          char c = roles[ba.getArgNumber()];
          role = c == 'e' ? kExpert : c == 'r' ? kRows : kShared;
        }
      layout.push_back(role);
      layout.push_back(mt.getNumElements() * elementBytes(mt.getElementType()));
      layout.push_back(mt.getRank());
    }

    OpBuilder b(launch);
    Type i32 = b.getI32Type();
    auto metaTy = MemRefType::get({problems}, i32);
    host.insertArgument(host.getNumArguments(), metaTy, {}, loc);
    kernel.insertArgument(kernel.getNumArguments(), metaTy, {}, loc);
    Value meta = host.getArgument(host.getNumArguments() - 1);

    SmallVector<Value> operands(launch.getKernelOperands().begin(), launch.getKernelOperands().end());
    operands.push_back(meta);
    Value gridZ = b.create<arith::ConstantIndexOp>(loc, problems);
    gpu::KernelDim3 grid{launch.getGridSizeX(), launch.getGridSizeY(), gridZ};
    gpu::KernelDim3 block{launch.getBlockSizeX(), launch.getBlockSizeY(), launch.getBlockSizeZ()};
    auto fresh = b.create<gpu::LaunchFuncOp>(loc, kernel, grid, block, launch.getDynamicSharedMemorySize(),
                                             operands);
    (void)fresh;
    launch.erase();

    kernel->setAttr(kLayoutAttr, b.getDenseI64ArrayAttr(layout));
    if (auto g = kernel->getAttrOfType<DenseI32ArrayAttr>("gpu.known_grid_size")) {
      SmallVector<int32_t> v(g.asArrayRef());
      if (v.size() == 3)
        v[2] = static_cast<int32_t>(problems);
      kernel->setAttr("gpu.known_grid_size", b.getDenseI32ArrayAttr(v));
    }
    return success();
  }
};

struct ExpertBatchPointersPass
    : public PassWrapper<ExpertBatchPointersPass, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpertBatchPointersPass)
  StringRef getArgument() const override { return "quantforge-expert-batch-pointers"; }
  StringRef getDescription() const override {
    return "Grouped GEMM: per-problem base pointers and early exit of padding problems (LLVM level).";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    SmallVector<LLVM::LLVMFuncOp> kernels;
    getOperation().walk([&](LLVM::LLVMFuncOp f) {
      if (f->hasAttr(kLayoutAttr))
        kernels.push_back(f);
    });
    for (LLVM::LLVMFuncOp f : kernels)
      if (failed(rewrite(f)))
        return signalPassFailure();
  }

  LogicalResult rewrite(LLVM::LLVMFuncOp func) {
    auto layout = func->getAttrOfType<DenseI64ArrayAttr>(kLayoutAttr).asArrayRef();
    if (layout.size() % 3)
      return func.emitError("malformed quantforge.expert_layout");
    // Descriptor parameters per memref: allocated ptr, aligned ptr, offset, sizes[r], strides[r].
    struct Arg {
      int64_t role, bytes;
      unsigned alignedIdx;
    };
    SmallVector<Arg> args;
    unsigned idx = 0;
    for (size_t i = 0; i < layout.size(); i += 3) {
      args.push_back({layout[i], layout[i + 1], idx + 1});
      idx += 3 + 2 * static_cast<unsigned>(layout[i + 2]);
    }
    unsigned metaAligned = idx + 1; // the argument appended by the launch pass: memref<P x i32>
    if (metaAligned >= func.getNumArguments())
      return func.emitError("expert batching: the meta argument is missing");

    MLIRContext *ctx = func.getContext();
    Location loc = func.getLoc();
    Block *entry = &func.getBody().front();
    Block *rest = entry->splitBlock(entry->begin());
    OpBuilder b(ctx);
    Type i32 = b.getI32Type(), i64 = b.getI64Type(), i8 = b.getI8Type();
    Type ptrTy = LLVM::LLVMPointerType::get(ctx);

    Block *exit = b.createBlock(&func.getBody(), func.getBody().end());
    b.create<LLVM::ReturnOp>(loc, ValueRange{});

    b.setInsertionPointToEnd(entry);
    Value z32 = b.create<NVVM::BlockIdZOp>(loc, i32);
    Value z = b.create<LLVM::SExtOp>(loc, i64, z32);
    Value slot = b.create<LLVM::GEPOp>(loc, ptrTy, i32, func.getArgument(metaAligned), ValueRange{z});
    Value e32 = b.create<LLVM::LoadOp>(loc, i32, slot);
    Value zero = b.create<LLVM::ConstantOp>(loc, i32, b.getI32IntegerAttr(0));
    Value pad = b.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, e32, zero);
    b.create<LLVM::CondBrOp>(loc, pad, exit, ValueRange{}, rest, ValueRange{});

    b.setInsertionPointToStart(rest);
    Value e = b.create<LLVM::SExtOp>(loc, i64, e32);
    for (const Arg &a : args) {
      if (a.role == kShared)
        continue;
      Value index = a.role == kExpert ? e : z;
      Value bytes = b.create<LLVM::ConstantOp>(loc, i64, b.getI64IntegerAttr(a.bytes));
      Value off = b.create<LLVM::MulOp>(loc, index, bytes);
      BlockArgument base = func.getArgument(a.alignedIdx);
      auto gep = b.create<LLVM::GEPOp>(loc, ptrTy, i8, base, ValueRange{off});
      base.replaceAllUsesExcept(gep.getResult(), gep);
    }
    func->removeAttr(kLayoutAttr);
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::quantforge::createExpertBatchLaunchPass() {
  return std::make_unique<ExpertBatchLaunchPass>();
}
std::unique_ptr<Pass> mlir::quantforge::createExpertBatchPointersPass() {
  return std::make_unique<ExpertBatchPointersPass>();
}
void mlir::quantforge::registerExpertBatchPasses() {
  PassRegistration<ExpertBatchLaunchPass>();
  PassRegistration<ExpertBatchPointersPass>();
}
