# `--quantforge-gpu-pipeline`: qf IR → PTX

```
quantforge-opt in.mlir --quantforge-gpu-pipeline='chip=sm_80' -o lowered.mlir
```

Input: tensor-level IR with `qf.unpack` / `qf.dequant` / `linalg.matmul`
(see `python/quantforge/e2e.py::w4a16_gemm_ir`). `qf.dequant` takes per-tensor
(rank-0) **or group-wise** scale / zero point (`tensor<K/G x N>` against a
`tensor<K x N>` weight; the group extent must divide K). `qf.unpack` may be
omitted: `qf.dequant` of an `i8` `[K, N]` tensor is the W8A16 kernel. Output: LLVM-dialect module with
the kernel(s) embedded as `gpu.binary` (PTX) and a host function +
`_mlir_ciface_<name>` wrapper calling the `mgpu*` runtime
(`python/quantforge/runtime/mgpu_cuda_runtime.c`).

## Stages

| # | What | Passes |
|---|------|--------|
| 1 | tensor level | `quantforge-tiling` (4-level) → `quantforge-gpu-mapping` → `quantforge-vectorization` → `fuse-unpack-dequant` → `lower-dequant-to-arith` (a `qf.dequant` without unpack) |
| 2 | bufferization | `quantforge-bufferize` (+ self-copy removal), `canonicalize`, `cse` |
| 3 | GPU launch | `convert-linalg-to-parallel-loops` + `scf-parallel-loop-tiling` (dequant), `quantforge-forall-to-gpu`, `convert-parallel-loops-to-gpu`, `quantforge-host-alloc-to-gpu` |
| 4 | threads / smem | `quantforge-thread-distribute`, `quantforge-smem-promotion`, `quantforge-fuse-producer-into-loads`, `quantforge-simplify-index-math` → `quantforge-vectorize-global-loads`, `quantforge-hoist-vector-transfers` |
| 5 | device lowering | `convert-vector-to-gpu{use-nvgpu}`, `gpu-kernel-outlining`, `quantforge-workgroup-alloc-to-attribution`, `convert-nvgpu-to-nvvm`, `quantforge-lower-vector`, scf/memref/math → LLVM |
| 6 | binary | `nvvm-attach-target`, `convert-gpu-to-nvvm`, `gpu-to-llvm`, `gpu-module-to-binary{format=isa}` |

## Options

| option | default | meaning |
|---|---|---|
| `chip`, `features`, `opt-level`, `format` | `sm_80`, `+ptx80`, 3, `isa` | NVVM target / output format (PTX is JIT-compiled forward-compatibly) |
| `threads` | true | 128 threads per block (32 lanes × 2 × 2 warps); false = 1 thread/block |
| `smem` | true | stage A/B tiles through shared memory (cooperative copy + barriers) |
| `tensor-core` | true | `mma.sync.m16n8k16` + `ldmatrix`; falls back to FMA code if `threads`/`smem` are off |
| `fuse-dequant` | true | recompute unpack+dequant while staging B → one kernel, packed INT4 read from HBM |
| `async-copy` | true | `cp.async` (16 B) for tiles whose source is a kernel argument (the A tile) |
| `vector-copy` | true | 16-byte work items + one `st.shared.v4` for copies that cannot use `cp.async` (the fused B tile) |
| `decode` | false | skinny-M schedule (M ≤ 16): 16×64 block tile, 16×16 warp tile, K tile 128, 1 stage; explicitly given tile/stage options still win. M a multiple of 16, N of 64 (pad with `pad=True`) |
| `block-tile-m`, `block-tile-n` | 128, 128 | C tile per thread block (multiples of the warp tile; M, N must be multiples) |
| `vectorize-loads` | true | merge the per-element global loads of the fused dequantisation (packed bytes, group-wise scale / zero point) into 16-byte vector loads. **Assumes the weight / scale / zero-point buffers are 16-byte aligned** (the Python harness checks) |
| `accumulate` | true | `true`: `C += A @ W` (accumulators start from C: bias / residual by initialisation). `false`: `C = A @ W`: `quantforge-overwrite-output` zeroes the initial accumulator loads (only loads that feed `vector.insert` chains, never the loop-carried look-ahead registers), so the caller need not zero the output and no read of C happens. Tensor Core + `hoist-acc` path only |
| `epilogue` | none | `none` \| `relu` \| `silu` \| `gelu` \| `swiglu`: `quantforge-fuse-epilogue` applies the activation to the f32 accumulator fragments right before their store, `out = act(out_init + A @ W)`. silu / gelu use the PTX `ex2.approx.ftz.f32` (inline asm: `convert-gpu-to-nvvm` rejects `llvm.intr.exp2` and there is no libdevice); gelu is the tanh form written as `x * sigmoid(2c (x + 0.044715 x^3))`. f32 accumulators only. `swiglu`: gate and up weights interleaved by column (`quant.interleave_gate_up`), so the two columns of each MMA accumulator pair a thread holds are (gate_j, up_j); the store becomes `C[r, c/2] = silu(v0) * v1` into the root output memref (subview chain folded), output half as wide; implies `accumulate=false` (the pair is written where another block may still read it) |
| `prefetch-distance` | 0 | `quantforge-prefetch-weight-loads`: issue the weight loads (packed bytes, scale, zero point) of the K loop this many tiles ahead, carried in registers through `iter_args` (~14 regs / thread / tile). Bit-identical results. Measured (decode, group-wise): 1 → −19 % on K ≫ N (1 × 11008 × 4096: 196 → 158 µs), neutral on 4096², ≥ 2 never helps; the autotuner tries distance 1. Needs `vectorize-loads` |
| `smem-stages` | 2 | K-loop buffering: 2 = prefetch tile i+1 while computing tile i; 1 = copy, wait, compute. Falls back to 1 if the buffers exceed 48 KiB |
| `register-tile` | true | Tensor Core only: unroll the warp's m16×n8 loops and keep its whole 64×64 accumulator in registers across the K loop |
| `hoist-acc` | true | accumulator tile lives in registers across the innermost K loop |
| `smem-pad` | 8 | row padding (elements) of shared tiles — removes ldmatrix bank conflicts |
| `block-tile-k` | 32 | K extent of the shared tiles |
| `warp-tile-m`, `warp-tile-n` | 64, 64 | C tile per warp (block is 128×128) |
| `sink-index` | true | sink index constants into `gpu.launch` before outlining (Tensor Core path only, see below) |

## Performance (RTX 4050 Laptop, sm_89, f32 accumulate, best of N)

| size | default | `smem-stages=1` | no cp.async / vector copy | FMA (`tensor-core=false`) | cuBLAS fp16 |
|---|---|---|---|---|---|
| 1024³ | 6.8 | 6.8 | 5.7 | 1.2 | 21.0 |
| 2048³ | 17.5 | 16.6 | 13.4 | 1.9 | 25.1 |
| 4096³ | 23.5 | 21.8 | 16.2 | 2.1 | 25.5 |

TFLOP/s. With an f16 accumulator (`QF_ACC=f16`) 4096³ reaches ~38 TFLOP/s. The
INT4 weights are read packed from HBM; cuBLAS reads pre-dequantised fp16.

What got the speed (each measured, 2048³): one thread per block → 128 threads
(×20), constants sunk into the kernel instead of passed as arguments (PTX
926 → 538 lines, `div.u64` gone), 64×64 accumulator kept in registers across
the K loop (×3.4), vector/`cp.async` tile copies (×1.2–1.3), double buffering
(×1.05–1.08 at ≥2048).

## Measured against Marlin

See README.md / `benchmarks/compare_marlin.py`: 0.7–0.9× Marlin's speed on the RTX 4050
(group size 128), 2–2.5× faster than a pre-dequantised fp16 cuBLAS GEMM at decode.
The vectorised scale / zero-point / packed-weight loads (`vectorize-loads`) took the
group-wise decode kernel from 118 to 83 µs (1 × 4096 × 4096, warm L2) and the group-wise
4096³ GEMM from 25.7 to 33.4 TFLOP/s.

## What limits the decode kernel

Measured with `ncu` and a micro-kernel that only reads the packed weights (RTX 4050, 22.5 MB):

* The read ceiling is ~183 GB/s. 64-byte row segments of the row-major `[K, N/2]` layout reach it
  as soon as a few loads are in flight; **32-byte segments (`block-tile-n=64`) drop to ~75 GB/s**
  unless the tile is stored contiguously (183 GB/s) — the case for repacking the weights.
* The kernel is *instruction-bound before it is latency-bound*: at 183 GB/s there are only ~10
  thread-instructions per weight to spend. A scalar per-weight `I2F` (1 per weight, ~7.7
  instructions per weight in total) was the dominant cost, which is why deeper load pipelining
  (`prefetch-distance` ≥ 2), 2-stage buffering, `sink-dequant-stores` and split-K did nothing
  on their own.
* Regression found by that analysis: after `GroupRowOfTile` started folding the group row of a
  K tile (tile-k > 64 previously never matched), the 8 per-item scale / zero-point scalar loads
  became identical and only the first was vectorised, so `quantforge-vectorize-dequant` stopped
  matching 7 of 8 items. Fixing it (all duplicate loads are replaced) took 1 × 4096 × 4096 from
  53 µs to 38 µs (Marlin: 49.6 µs). `test/Transforms/gpu_pipeline_groupwise.mlir` now checks that
  no `cvt.rn.f16` is left in the PTX.

* **Wave quantization.** Effective bandwidth by N (K = 4096, `block-tile-n=128`): 183.6 GB/s at
  N = 8192 (64 blocks), 169 at N = 10240 (80 blocks = 4 resident blocks × 20 SMs), ~151 for every
  N whose grid exceeds the resident-block count (86, 96, 112 blocks): the tail wave runs with
  little parallelism. A smaller K tile (`block-tile-k=64`: less shared memory per block) together
  with `prefetch-distance=1` (which keeps the bytes in flight) moved 1 × 4096 × 11008 from
  153 to 140 µs; the autotuner now tries it.
* Benchmark pitfall: with the L2 flushed, a harness that happens to copy the activations (the
  padding path for M < 16) hands the kernel hot A while M = 16 gets cold A (63 vs 38 µs on the
  same kernel). `compare_marlin.py` now re-touches the activations after the flush for every
  kernel.

Untried: stream-K / persistent scheduling for the tail wave; Marlin-style offline repacking (contiguous 16-byte-per-thread reads, dequantisation
straight into the MMA fragment layout, no shared-memory round trip for B).

## Prefill (M ≥ 256)

`ncu` on 512 × 4096 × 4096 (128 × 128 block tiles): Tensor Core pipe 70 % active, dominant stall
`math_pipe_throttle`, and a grid of 128 blocks for 60 resident slots (3 blocks / SM, limited by 167
registers and shared memory): 2.13 waves run as 3 (71 % at best), the same wave quantization as in
decode. 64 × 128 block tiles with a 64-wide K tile double the block count: 491 → 457 µs
(0.79× → 0.85× Marlin); at M = 2048 the 128 × 128 tile stays best. The autotuner tries both and its
cache now keys M in finer buckets (≤ 16, 32, …, 4096). Remaining gap to cuBLAS / Marlin: MMA issue
gaps between the dequantisation and the HMMA stream (no software pipelining of the two).

## Group-wise quantization

`qf.dequant %w, %scale, %zp` with `scale`, `zp : tensor<(K/G) x N>`: element `[k, n]`
uses parameter `[k / G, n]`. The fused kernel recomputes the dequantisation while
staging the B tile, so the scale / zero point of a thread's 8 columns are one
16-byte + one 8-byte load per work item (`vectorize-loads`). Cost relative to
per-tensor parameters at 4096³: ~12 % (2048³: ~10 %), decode ~15–20 %.
Checkpoints in AWQ / GPTQ layout are repacked by `quantforge.quant.from_awq` /
`from_gptq` (act-order GPTQ is rejected).

## Decode (M ≤ 16)

`decode=true` keeps one 16-row tile per block and spreads the N dimension over
blocks (64 columns each), so a single read of the INT4 weights is the whole cost:
the kernel is bandwidth-bound. `python/quantforge/autotune.py` searches tile /
stage options per shape and caches the winner. Split-K was not needed for the
shapes measured (4096 × 4096..11008); K ≫ N (e.g. 11008 × 4096) leaves only 64
blocks and is the case where it would help.

## Constraints

* M, N multiples of the block tile (128, or 16 / 64 with `decode=true`) and K a
  multiple of `block-tile-k` (and of the group size) — use
  `W4A16Gemm.build(..., pad=True)` for other shapes (exact zero padding).
* Accumulator may be f32 (default, exact vs. reference) or f16.
* The scalar FMA path (`tensor-core=false`) is correct but slow, and
  `gpu-launch-sink-index-computations` is deliberately not applied to it: with
  the sunk constants LLVM 18's NVPTX backend miscompiles the vectorised f16
  shared-memory loads that feed `fpext` (wrong results for `opt-level` ≥ 1,
  correct at 0; `compute-sanitizer` reports no race/memory error).
* `quantforge-swizzle-load` is **not** in the pipeline: it only rewrites 2-D
  `memref.load/store`, while the compute stage reads shared memory through
  `ldmatrix` / `vector.transfer_read`. Mixing would corrupt data, so the pass now
  skips such buffers with a warning; padding (`smem-pad`) removes the same bank
  conflicts without changing addressing.

## Testing

```
lit test/Transforms                       # QUANTFORGE_BUILD_DIR=<build>, FILECHECK=FileCheck-18
python test/E2E/test_w4a16_e2e.py         # runs on the GPU, compares with PyTorch
python python/quantforge/benchmark.py 2048
```

## Runtime (host side)

`python/quantforge/runtime/mgpu_cuda_runtime.c` implements the `mgpu*` calls of the generated
host code on the CUDA driver API. The PTX module is JIT-compiled once and cached by the address
of its embedded image; `W4A16Gemm.run` hands it PyTorch's current stream (`mgpuSetStream`), the
stream is neither created nor synchronised per call, so a call is an ordinary asynchronous op
ordered after the caller's pending work. Per-call wall time of `run` (1 × 4096 × 4096, group-wise):
211–272 µs before, ~45–50 µs now (kernel 38 µs). An M < 16 call only pads the activations and the
output (the weights, scales and zero points are not copied).

## Compile cache and row buckets

`compile_ir` publishes `libqf_kernel.so` + `lowered.mlir` under `~/.cache/quantforge/kernels/<key>/`
(atomic rename, safe for concurrent processes); the key hashes the IR, the target and pipeline
options, the path / size / mtime of `quantforge-opt`, `mlir-translate` and `llc`, and the source of the
CUDA shim, so a rebuilt compiler never serves a stale kernel. A compile takes ~0.3 s for a decode kernel
(more for large prefill tiles); a hit costs a file read. `torch_ops` additionally maps the row count of
a call to a bucket (`bucket_rows`) so that a server sees a handful of kernels per layer shape, not one
per prompt length; rows above the real M are zero activations and are dropped.
