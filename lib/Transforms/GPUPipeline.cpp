//===----------------------------------------------------------------------===//
// quantforge-gpu-pipeline — end-to-end lowering: qf/linalg IR  →  PTX
//
// Input : a builtin.module whose func.func ops compute a W4A16 GEMM on
//         tensors, e.g.
//             %w  = qf.unpack  %packed : tensor<KxN/2xi8> -> tensor<KxNxi8>
//             %wf = qf.dequant %w, %scale, %zp : ... -> tensor<KxNxf16>
//             %c  = linalg.matmul ins(%act, %wf) outs(%acc)
//         (tile sizes must divide the problem: M,N multiples of the block
//          tile (128), K a multiple of the K tile (64), N even.)
//
// Output: the same module lowered to LLVM with the kernels embedded as
//         `gpu.binary` (PTX text, format=isa by default). The host function
//         keeps its name and also gets a `_mlir_ciface_<name>` wrapper
//         (memref descriptors passed by pointer).
//
// Stages
//   1. Tensor level   : tiling → GPU mapping annotations → vectorization →
//                       unpack/dequant fusion
//   2. Bufferization  : One-Shot Bufferize, canonicalize, drop self-copies
//   3. GPU launch     : linalg → parallel loops, scf.forall / scf.parallel →
//                       gpu.launch, host buffers → gpu.alloc
//   4. Device lowering: kernel outlining, vector/scf/affine/memref → LLVM,
//                       gpu → NVVM
//   5. Binary         : nvvm target attach, host gpu-to-llvm, PTX emission
//
// Usage:
//   quantforge-opt in.mlir --quantforge-gpu-pipeline='chip=sm_80' -o out.mlir
//   mlir-translate --mlir-to-llvmir out.mlir | llc ...   (host object)
//===----------------------------------------------------------------------===//

#include "QuantForge/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {

struct GPUPipelineOptions
    : public PassPipelineOptions<GPUPipelineOptions> {
  Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("Target SM architecture for the NVVM target "
                     "(PTX is JIT-compiled forward-compatibly)"),
      llvm::cl::init("sm_80")};
  Option<std::string> features{
      *this, "features", llvm::cl::desc("PTX ISA feature string"),
      llvm::cl::init("+ptx80")};
  Option<int> optLevel{*this, "opt-level",
                       llvm::cl::desc("NVPTX backend optimization level"),
                       llvm::cl::init(3)};
  Option<std::string> format{
      *this, "format",
      llvm::cl::desc("gpu-module-to-binary output format: "
                     "offload | isa (PTX) | bin | fatbin"),
      llvm::cl::init("isa")};
  Option<bool> sharedMemory{
      *this, "smem",
      llvm::cl::desc("Stage A/B tiles through shared memory (cooperative "
                     "copy + barriers)"),
      llvm::cl::init(true)};
  Option<bool> tensorCore{
      *this, "tensor-core",
      llvm::cl::desc("Use warp-cooperative mma.sync (nvgpu) with ldmatrix for "
                     "the 16x8x16 tiles instead of per-lane FMA code. Needs "
                     "sm_80+; silently falls back to FMA code when "
                     "threads=false or smem=false"),
      llvm::cl::init(true)};
  Option<bool> fuseDequant{
      *this, "fuse-dequant",
      llvm::cl::desc("Fuse INT4 unpack+dequant into the GEMM's shared-memory "
                     "staging: one kernel, packed weights read from HBM "
                     "(requires smem=true)"),
      llvm::cl::init(true)};
  Option<int> blockTileK{
      *this, "block-tile-k",
      llvm::cl::desc("K extent of the shared-memory tiles (multiple of 16). "
                     "Smaller values use less shared memory -> more resident "
                     "blocks per SM"),
      llvm::cl::init(32)};
  Option<bool> asyncCopy{
      *this, "async-copy",
      llvm::cl::desc("cp.async (16 B) for global->shared tile copies whose "
                     "source is a kernel argument (sm_80+, needs smem=true)"),
      llvm::cl::init(true)};
  Option<bool> vectorCopy{
      *this, "vector-copy",
      llvm::cl::desc("16-byte vector stores into shared memory for the copies "
                     "that cannot use cp.async (e.g. fused dequantization)"),
      llvm::cl::init(true)};
  Option<bool> vectorizeLoads{
      *this, "vectorize-loads",
      llvm::cl::desc("Merge the per-element global loads of the fused "
                     "dequantisation (packed bytes, group-wise scale / zero "
                     "point) into 16-byte vector loads. ASSUMES the packed "
                     "weight, scale and zero-point buffers are 16-byte "
                     "aligned (cudaMalloc / PyTorch allocations are)"),
      llvm::cl::init(true)};
  Option<bool> vectorizeDequant{
      *this, "vectorize-dequant",
      llvm::cl::desc("Dequantise 8 weights at a time with packed f16x2 "
                     "arithmetic (magic-number int->f16 conversion, "
                     "bit-identical results). Needs vectorize-loads"),
      llvm::cl::init(true)};
  Option<int> indexBitwidth{
      *this, "index-bitwidth",
      llvm::cl::desc("Bitwidth of `index` inside the kernels (32 = 32-bit "
                     "address arithmetic, fewer IMAD/LEA/SGXT; 0 = 64). Needs "
                     "bare-kernel-pointers when 32"),
      llvm::cl::init(0)};
  Option<bool> bareKernelPointers{
      *this, "bare-kernel-pointers",
      llvm::cl::desc("Pass memrefs to the kernels as bare pointers (static "
                     "shapes only): 5 kernel parameters instead of ~35"),
      llvm::cl::init(false)};
  Option<int> blockSwizzle{
      *this, "block-swizzle",
      llvm::cl::desc("Walk the output tiles in groups of this many M tiles "
                     "(1-D grid, column-major inside a group) so that "
                     "concurrently resident blocks share A / B panels in L2. "
                     "Large win on wide N (e.g. 2048 x 4096 x 11008: L2 hit "
                     "rate 77% -> ~90%); 0 = plain 2-D grid"),
      llvm::cl::init(0)};
  Option<bool> accumulate{
      *this, "accumulate",
      llvm::cl::desc("true: C += A @ W (the accumulators start from C: bias / "
                     "residual by initialisation); false: C = A @ W (no initial "
                     "read of C, the caller need not zero the output). Tensor "
                     "Core + register-tile path only. Implied by epilogue=swiglu"),
      llvm::cl::init(true)};
  Option<std::string> epilogue{
      *this, "epilogue",
      llvm::cl::desc("Activation applied to the accumulator right before the "
                     "output store: none | relu | silu | gelu (tanh form) | swiglu (columns "
                     "2j, 2j+1 -> silu(c[2j]) * c[2j+1] stored at column j: fused MLP gate/up with "
                     "interleaved weights; implies accumulate=false). "
                     "out = act(out_init + A @ W). f32 accumulator only"),
      llvm::cl::init("none")};
  Option<int> prefetchDistance{
      *this, "prefetch-distance",
      llvm::cl::desc("Issue the weight loads (packed bytes, scale, zero point) "
                     "of the K loop this many tiles ahead, carried in "
                     "registers (rotating iter_args): more bytes in flight "
                     "per block for bandwidth-bound decode kernels. Costs "
                     "~14 registers / thread / tile. 0 = off. Needs "
                     "vectorize-loads"),
      llvm::cl::init(0)};
  Option<bool> sinkDequantStores{
      *this, "sink-dequant-stores",
      llvm::cl::desc("Experimental: in the double-buffered K loop issue the "
                     "weight loads first, run the MMAs of the current tile, "
                     "THEN dequantise + store the next B tile. Moves the "
                     "load-latency stall out of the way but costs registers: "
                     "measured -15% at 4096^3 and neutral on wide N, so off "
                     "by default. Needs vectorize-loads"),
      llvm::cl::init(false)};
  Option<bool> sinkIndex{
      *this, "sink-index",
      llvm::cl::desc("Sink index constants/arithmetic into gpu.launch before "
                     "outlining (otherwise they become kernel arguments, "
                     "opaque to the PTX backend). Only applied together with "
                     "the Tensor Core path: with the scalar FMA path LLVM 18's "
                     "NVPTX backend miscompiles the resulting vectorised "
                     "f16 shared-memory loads (wrong results for opt-level "
                     ">= 1, correct at 0)"),
      llvm::cl::init(true)};
  Option<int> smemStages{
      *this, "smem-stages",
      llvm::cl::desc("Shared-memory stages of the K loop: 1 = copy, wait, "
                     "compute; 2 = double buffering (prefetch the next tile "
                     "while computing the current one; falls back to 1 when "
                     "the buffers would exceed 48 KiB of shared memory)"),
      llvm::cl::init(2)};
  Option<int> blockTileM{
      *this, "block-tile-m",
      llvm::cl::desc("Rows of C per thread block (multiple of the warp tile). "
                     "Small values (16) give the decode / skinny-M kernels"),
      llvm::cl::init(128)};
  Option<int> blockTileN{
      *this, "block-tile-n",
      llvm::cl::desc("Columns of C per thread block (multiple of the warp "
                     "tile)"),
      llvm::cl::init(128)};
  Option<int> warpTileM{
      *this, "warp-tile-m",
      llvm::cl::desc("Rows of C owned by one warp (multiple of 16). Block "
                     "tile 128 / warp tile = warps along M"),
      llvm::cl::init(64)};
  Option<int> warpTileN{
      *this, "warp-tile-n",
      llvm::cl::desc("Columns of C owned by one warp (multiple of 8)"),
      llvm::cl::init(64)};
  Option<int> smemPad{
      *this, "smem-pad",
      llvm::cl::desc("Row padding (elements) of shared-memory tiles to avoid "
                     "bank conflicts"),
      llvm::cl::init(8)};
  Option<bool> registerTile{
      *this, "register-tile",
      llvm::cl::desc("Tensor Core only: unroll the warp's m16 x n8 tile loops "
                     "and keep the whole 64x64 accumulator in registers "
                     "across the K loop (needs hoist-acc=true)"),
      llvm::cl::init(true)};
  Option<bool> hoistAcc{
      *this, "hoist-acc",
      llvm::cl::desc("Keep the C accumulator tile in registers across the "
                     "innermost K loop"),
      llvm::cl::init(true)};
  Option<int> threadTileK{
      *this, "thread-tile-k",
      llvm::cl::desc("K extent of one mma.sync: 16 for f16 (m16n8k16), 32 for "
                     "int8 (m16n8k32)"),
      llvm::cl::init(16)};

  Option<int> validRows{
      *this, "valid-rows",
      llvm::cl::desc("Decode kernels: only this many rows of the activations / output exist (1..15 for a 16-row tile): "
                     "cp.async zero-fills the others and their stores are skipped, so the caller passes the [M, K] tensor "
                     "itself instead of a padded copy (0 = all rows exist)"),
      llvm::cl::init(0)};

  Option<int> splitK{
      *this, "split-k",
      llvm::cl::desc("Cut the K reduction into this many chunks run by "
                     "separate blocks (partial results + a reduce kernel); "
                     "1 = off"),
      llvm::cl::init(1)};

  Option<bool> decode{
      *this, "decode",
      llvm::cl::desc("Skinny-M (decode, M <= 16) schedule: 16x64 block tile, "
                     "16x16 warp tile, K tile 128, single stage. Reads the "
                     "INT4 weights once at ~70-85% of peak DRAM bandwidth. "
                     "Explicitly given tile/stage options still win. M and N "
                     "must be multiples of 16 and 64"),
      llvm::cl::init(false)};
  Option<bool> threads{
      *this, "threads",
      llvm::cl::desc("Distribute warp/lane tile loops over threadIdx "
                     "(false = one thread per block)"),
      llvm::cl::init(true)};
};

void buildGPUPipeline(OpPassManager &pm, const GPUPipelineOptions &opts) {
  // `decode` supplies defaults for the tile options the user did not set.
  auto eff = [&](const auto &o, int decodeValue) -> int {
    return opts.decode && o.getNumOccurrences() == 0 ? decodeValue
                                                     : o.getValue();
  };
  const int bm = eff(opts.blockTileM, 16), bn = eff(opts.blockTileN, 64),
            bk = eff(opts.blockTileK, 128), wm = eff(opts.warpTileM, 16),
            wn = eff(opts.warpTileN, 16), stages = eff(opts.smemStages, 1);
  // Tensor Core lowering needs warp-level thread mapping and shared memory.
  const bool tensorCore = opts.tensorCore && opts.threads && opts.sharedMemory;
  // nvgpu ops (ldmatrix/mma.sync, device_async_copy) need their NVVM lowering.
  const bool needNvgpu =
      tensorCore || (opts.sharedMemory && opts.threads && opts.asyncCopy);
  // Plain concatenation (not formatv) because the pass-option syntax is full
  // of braces.
  std::string text;
  // 1. Tensor level
  text += "func.func(quantforge-tiling{block-tile-m=" +
          std::to_string(bm) + " block-tile-n=" +
          std::to_string(bn) + " block-tile-k=" +
          std::to_string(bk) + " warp-tile-m=" +
          std::to_string(wm) + " warp-tile-n=" +
          std::to_string(wn) +
          " thread-tile-k=" + std::to_string(opts.threadTileK.getValue()) +
          " split-k=" + std::to_string(opts.splitK.getValue()) +
          // overwrite mode: every block writes its whole partial tile, no zero fill needed
          " split-k-init=" +
          ((!opts.accumulate || opts.epilogue == "swiglu" || opts.epilogue == "int8-dequant" || opts.epilogue == "f16-store" || opts.epilogue == "atomic-add") ? "false" : "true") +
          "},quantforge-gpu-mapping{block-tile-m=" +
          std::to_string(bm) + " block-tile-n=" +
          std::to_string(bn) + " warp-tile-m=" +
          std::to_string(wm) + " warp-tile-n=" +
          std::to_string(wn) + "},"
          "quantforge-vectorization{fold-extension=" +
          (tensorCore ? "true" : "false") +
          "},fuse-unpack-dequant,lower-dequant-to-arith),";
  // 2. Bufferization
  text += "quantforge-bufferize,canonicalize,cse,";
  // 3. GPU launch construction
  // The dequant generic (K × N/2 × 2) becomes one scf.parallel; tiling it by
  // (1, 64, 2) gives 128 threads per block over the inner loops.
  text += "func.func(convert-linalg-to-parallel-loops,"
          "scf-parallel-loop-tiling{parallel-loop-tile-sizes=1,64,2 "
          "no-min-max-bounds=true},gpu-map-parallel-loops,"
          "quantforge-forall-to-gpu{block-dim-x=" +
          std::string(opts.threads ? "0" : "1") + " block-dim-y=" +
          std::string(opts.threads ? "0" : "1") + " block-dim-z=" +
          std::string(opts.threads ? "0" : "1") +
          " swizzle-m=" + std::to_string(opts.blockSwizzle.getValue()) +
          "},convert-parallel-loops-to-gpu,quantforge-host-alloc-to-gpu),"
          "canonicalize,";
  if (opts.threads)
    text += std::string("func.func(quantforge-thread-distribute{"
                        "distribute-lanes=") +
            (tensorCore ? "false" : "true") + "}),canonicalize,";
  if (opts.sharedMemory)
    text += "func.func(quantforge-smem-promotion{smem-pad=" +
            std::to_string(opts.smemPad.getValue()) + " block-tile-m=" +
            std::to_string(bm) + " block-tile-n=" +
            std::to_string(bn) + " warp-tile-m=" +
            std::to_string(wm) + " warp-tile-n=" +
            std::to_string(wn) + " block-tile-k=" +
            std::to_string(bk) +
            " smem-stages=" + std::to_string(stages) +
            " smem-async-copy=" + (opts.asyncCopy ? "true" : "false") +
            " smem-vector-copy=" + (opts.vectorCopy ? "true" : "false") +
            " valid-rows=" + std::to_string(opts.validRows.getValue()) +
            "}),canonicalize,cse,";
  if (opts.sharedMemory && opts.fuseDequant)
    text += "func.func(quantforge-fuse-producer-into-loads),canonicalize,cse,";
  if (opts.sharedMemory && opts.fuseDequant && opts.vectorizeLoads)
    text += "func.func(quantforge-simplify-index-math{tile-k=" +
            std::to_string(bk) + "}),canonicalize,cse,"
            "func.func(quantforge-vectorize-global-loads),canonicalize,cse,";
  if (opts.sharedMemory && opts.fuseDequant && opts.vectorizeLoads &&
      opts.vectorizeDequant)
    text += "func.func(quantforge-vectorize-dequant),canonicalize,cse,";
  if (opts.sharedMemory && opts.fuseDequant && opts.vectorizeLoads &&
      opts.sinkDequantStores)
    text += "func.func(quantforge-sink-dequant-stores),canonicalize,cse,";
  if (opts.hoistAcc)
    text += std::string("func.func(quantforge-hoist-vector-transfers{"
                        "unroll-tiles=") +
            (tensorCore && opts.registerTile ? "true" : "false") +
            "}),canonicalize,cse,";
  if (tensorCore)
    // vector.contract (m16n8k16 f16) -> nvgpu.ldmatrix + nvgpu.mma.sync
    text += "func.func(convert-vector-to-gpu{use-nvgpu=true}),canonicalize,cse,";
  if (opts.prefetchDistance > 0 && opts.sharedMemory && opts.fuseDequant &&
      opts.vectorizeLoads)
    text += "func.func(quantforge-prefetch-weight-loads{distance=" +
            std::to_string(opts.prefetchDistance.getValue()) +
            "}),canonicalize,cse,";
  if (!opts.accumulate || opts.epilogue == "swiglu" || opts.epilogue == "int8-dequant" || opts.epilogue == "f16-store" || opts.epilogue == "atomic-add") {
    if (!tensorCore || !opts.hoistAcc)
      llvm::report_fatal_error("quantforge-gpu-pipeline: accumulate=false / "
                               "epilogue=swiglu need the Tensor Core path with "
                               "hoist-acc");
    text += "func.func(quantforge-overwrite-output),canonicalize,";
  }
  if (opts.epilogue != "none") {
    if (opts.epilogue != "relu" && opts.epilogue != "silu" &&
        opts.epilogue != "gelu" && opts.epilogue != "swiglu" &&
        opts.epilogue != "int8-dequant" && opts.epilogue != "f16-store" && opts.epilogue != "atomic-add")
      llvm::report_fatal_error("quantforge-gpu-pipeline: epilogue must be "
                               "none | relu | silu | gelu | swiglu | int8-dequant | f16-store | atomic-add");
    text += "func.func(quantforge-fuse-epilogue{kind=" +
            opts.epilogue.getValue() +
            " valid-rows=" + std::to_string(opts.validRows.getValue()) + "}),canonicalize,";
  }
  // 4. Device lowering (outlined kernels + host-side prep)
  // Constants / index arithmetic created outside gpu.launch would otherwise be
  // passed as kernel *arguments*: opaque to the PTX backend (div.u64 by a
  // runtime value, no strength reduction, extra param loads). Sink them in.
  if (opts.sinkIndex && tensorCore)
    text += "func.func(gpu-launch-sink-index-computations),";
  text += "gpu-kernel-outlining,"
          "gpu.module(gpu.func(quantforge-workgroup-alloc-to-attribution)),"
          + std::string(needNvgpu ? "convert-nvgpu-to-nvvm," : "") +
          "lower-affine,"
          "convert-vector-to-scf{full-unroll=true},"
          "quantforge-lower-vector,"
          "expand-strided-metadata,lower-affine,convert-scf-to-cf,"
          "convert-math-to-llvm,canonicalize,cse,";
  // 5. NVVM / PTX
  text += "func.func(llvm-request-c-wrappers),"
          "nvvm-attach-target{chip=" +
          opts.chip.getValue() + " features=" + opts.features.getValue() +
          " O=" + std::to_string(opts.optLevel.getValue()) +
          "},"
          "gpu.module(convert-gpu-to-nvvm{index-bitwidth=" +
          std::to_string(opts.indexBitwidth.getValue()) +
          (opts.bareKernelPointers ? " use-bare-ptr-memref-call-conv=true" : "") +
          "},canonicalize,cse),"
          "func.func(gpu-async-region),gpu-to-llvm" +
          std::string(opts.bareKernelPointers
                          ? "{use-bare-pointers-for-kernels=true}"
                          : "") +
          ","
          "finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
          "convert-arith-to-llvm,canonicalize,cse,reconcile-unrealized-casts,"
          "gpu-module-to-binary{format=" +
          opts.format.getValue() + "}";

  if (failed(parsePassPipeline(text, pm)))
    llvm::report_fatal_error("quantforge-gpu-pipeline: internal pipeline "
                             "string failed to parse");
}

} // namespace

void mlir::quantforge::registerGPUPipeline() {
  PassPipelineRegistration<GPUPipelineOptions>(
      "quantforge-gpu-pipeline",
      "End-to-end lowering of QuantForge W4A16 GEMM IR to PTX "
      "(tiling → bufferize → gpu.launch → NVVM → gpu.binary)",
      buildGPUPipeline);
}
