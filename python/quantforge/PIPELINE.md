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
  matching 7 of 8 items. Fixing it (all duplicate loads are replaced) restored the packed dequantisation.
  `test/Transforms/gpu_pipeline_groupwise.mlir` now checks that no `cvt.rn.f16` is left in the PTX.
  (An earlier version of this note quoted 53 → 38 µs for that fix and "faster than Marlin"; those timings came from
  development runs that are not reproducible: the same kernels measure 50 µs cold, on par with Marlin's 49.6 µs, see
  the ablation below.)
* **Ablation of the decode kernel, cold L2, 1 × 4096 × 4096 (2026-11-20):** all passes 50 µs; scalar dequantisation
  (`vectorize-dequant=false`) 58 µs (+16 %); no look-ahead (`prefetch-distance=0`) 61 µs (+22 %); scalar scale / zero-point /
  weight loads (`vectorize-loads=false`) 156 µs (3.1×). With the weights L2-resident (no flush) the full kernel is 49 µs:
  it is latency-bound, not bandwidth-bound, at this size.

* **Wave quantization.** Effective bandwidth by N (K = 4096, `block-tile-n=128`): 183.6 GB/s at
  N = 8192 (64 blocks), 169 at N = 10240 (80 blocks = 4 resident blocks × 20 SMs), ~151 for every
  N whose grid exceeds the resident-block count (86, 96, 112 blocks): the tail wave runs with
  little parallelism. A smaller K tile (`block-tile-k=64`: less shared memory per block) together
  with `prefetch-distance=1` (which keeps the bytes in flight) moved 1 × 4096 × 11008 from
  153 to 140 µs; the autotuner now tries it.
* Benchmark pitfall: with the L2 flushed, a harness that happens to copy the activations (the
  padding path for M < 16) hands the kernel hot A while M = 16 gets cold A (an earlier note quoted 63 vs 38 µs on the
  same kernel; with the current harness M = 1, 4 and 16 all measure 50 µs). `compare_marlin.py` now re-touches the activations after the flush for every
  kernel.

Untried: stream-K / persistent scheduling for the tail wave. Marlin-style offline repacking (contiguous 16-byte-per-thread
reads, dequantisation straight into the MMA fragment layout, no shared-memory round trip for B) was only prototyped by
hand in CUDA, see "Small batch" below.

## Small batch (16 < M ≤ 256)

Measured with `compare_marlin.py --autotune` (RTX 4050, cold weights): M = 32 0.88x, 64 0.67x, 128 0.69x, 256 0.76x
of Marlin at 4096 x 4096. A sweep over block tile M (16 / 32 / 64 / 128), warp tile M / N, K tile, stages and look-ahead
found the best skinny schedules: 32 rows `block-tile-m=32 warp-tile-n=32 block-tile-n=128 block-tile-k=128` (58.6 us, the
earlier default 68.6 us), 64 rows `block-tile-m=64 warp-tile-m=32 warp-tile-n=64 block-tile-k=64 prefetch-distance=1`
(87 us, was 97 us); they are the `torch_ops` defaults (a 32-row bucket was added) and the autotune candidates (the tune
cache key now carries a candidates version). 16-row block tiles (more blocks, 4x the weight traffic) are 30 % slower,
and more resident blocks do not help (64 x 1024 x 16384 with 256 blocks takes as long as 64 x 4096 x 4096 with 64): at
M = 64 `ncu` shows the Tensor pipe 51 % active, issue slots 36 %, DRAM 52 %, stalls spread over `mio_throttle`,
`math_pipe_throttle`, `wait`, `long_scoreboard`, `barrier` (1.0-1.4 each): the kernel is bound by the shared-memory /
`ldmatrix` / MMA issue path, not by the weight stream -- A Marlin-style nibble layout (pair nibble p with p + 4 so that one SHF + one LOP3 unpacks a pair, ~7 instead of ~16
integer ops per 8 weights; dequantisation is ~60 % of the executed instructions at M = 64) was ablated by emitting
it with the wrong (but timed) layout: only 3 % at M = 64, 2.6 % at 512 and 1.2 % at 2048 x 4096 x 4096, so the
integer work of the dequantisation is not what limits these kernels. Marlin keeps B in registers (dequantised straight into the MMA
fragments), which is the repacking work listed under "Untried".

**Prototype of that design** (`benchmarks/proto/bir_w4a16.cu`, hand-written CUDA, verified against an f32
reference, RTX 4050, cold weights, 4096 x 4096, group 128): offline repack so that each thread loads 16 B = its
B fragments of a 64-column strip per k16 step, dequantisation (lop3 + sub.f16x2 + mul.f16x2) straight into the
`mma.m16n8k16` operands, A through cp.async + ldmatrix, K slices accumulated with atomics. Best of a sweep over
warps per block (2 / 4 / 8), K slices (1-8), A stages (2-4) and B prefetch distance (2-8): **M = 32: 57.3 us, M = 64:
75.8 us** -- against the compiler's 58 / 87 us and Marlin's 51 / 59 us. So keeping B in registers is worth ~13 % at
M = 64 and nothing at M = 32 in a straightforward implementation (254 registers, one warp per scheduler, Tensor-pipe
bound at ~60 % of its 47 us floor); closing the rest of the gap to Marlin is occupancy / scheduling tuning of a
hand-written kernel, not something a rewrite of the MLIR smem path would deliver by itself. Not pursued.. M = 128 / 256 sweeps (64 / 128-row tiles, warp tiles up to
64 x 64) improve only 3 %.

## Prefill (M ≥ 256)

`ncu` on 512 × 4096 × 4096 (128 × 128 block tiles): Tensor Core pipe 70 % active, dominant stall
`math_pipe_throttle`, and a grid of 128 blocks for 60 resident slots (3 blocks / SM, limited by 167
registers and shared memory): 2.13 waves run as 3 (71 % at best), the same wave quantization as in
decode. 64 × 128 block tiles with a 64-wide K tile double the block count: 491 → 457 µs
(0.79× → 0.85× Marlin); at M = 2048 the 128 × 128 tile stays best. The autotuner tries both and its
cache now keys M in finer buckets (≤ 16, 32, …, 4096). Large weights (K x N >= 2**25) at M >= 512 lose ~20 % throughput with the default
128 x 128 x 32, 2-stage schedule (31 vs 38 TFLOP/s at 2048 x 4096 x 11008 / 4096 x 14336); a 3-stage pipeline
(`block-tile-k=32 smem-stages=3`) is 8-12 % faster there (and 4 % slower at 4096 x 4096, 10 % slower at
256 x 11008 x 4096): `torch_ops._prefill_opts` uses it for M >= 512 and K x N >= 2**25, the autotuner has
it as a candidate. `prefetch-distance=1` does nothing for prefill (0.464 vs 0.467 ms at 512 x 4096 x 4096), and
64 x 64 / 128 x 64 block tiles, 4 stages and 64-deep K tiles are all slower. The cause is L2 locality of the
plain 2-D grid: `block-swizzle=16` gives the same gain on the large weights (2048 x 4096 x 11008: 5.80 -> 5.19 ms, vs
5.14 with 3 stages; not additive) and costs 8-14 % at 4096 x 4096. Decode: wider block tiles to fit the grid in one
wave (`block-tile-n=256`, 43 blocks for N = 11008) do not help (1 x 4096 x 11008: 149 -> 169 us; 1 x 11008 x 4096:
132 -> 129 us, noise level), 192-wide tiles are 2-4x slower. ncu at 2048 x 4096 x 4096 (default schedule, block 32 x 16 threads = 16 warps): Tensor pipe 79 % of peak
active, 16.8 M HMMA of 74 M warp instructions (23 %), issue slots 22 % busy, 1 resident block per SM (512 threads x
<= 128 registers) -- the dequantisation is already the packed f16x2 form (no I2F), so the instruction count is
not the problem; what is left is barrier-synchronised phases of a single block per SM (all 16 warps dequantise,
then all multiply), i.e. at most ~1.2x of headroom (79 % -> ~95 % tensor activity), consistent with Marlin /
cuBLAS at 0.89 / 0.84 of our time. Remaining gap to cuBLAS / Marlin: MMA issue
gaps between the dequantisation and the HMMA stream (no software pipelining of the two).

## FP8 (E4M3) weights

`qf.dequant %w, %scale, %zp {fp8}` reads the i8 tensor as E4M3 bit patterns: the 7 magnitude bits shifted left
by 7 are an f16 whose exponent / mantissa fields are the E4M3 ones, i.e. exactly 2^-8 times the FP8 value
(subnormals too; LLVM 18's NVPTX has no e4m3 type or `cvt` and none is needed); the sign bit moves to bit 15;
`x 256` is exact (|w| <= 448) and the group scale is the one rounding. `zero_point` is not read (its loads
are dead). `W4A16Gemm.build(bits=8, fp8=True)`; the torch op selects it from a `float8_e4m3fn` weight.
`quantforge-vectorize-dequant` recognises the tree and emits the packed form (bytes spread over the two halves of
an i32, shifts / masks on both halves at once, two f16x2 multiplies; bit-identical to the scalar tree, guarded by
`gpu_pipeline_fp8.mlir`: no `shl.b16` left, the scalar form has 128 of them in the decode kernel). Speed: decode
1 x 4096 x 4096 99 vs 101 us for int8, 1 x 4096 x 11008 271 vs 272 us, prefill 512 x 4096 x 4096 4 % slower
(584 vs 564 us); whole model (Qwen2.5-0.5B, RTN group 128, `hf_quantize_and_run.py --weights fp8`): ppl 20.10
(int8 19.91, fp16 19.93), CUDA-graph decode 165.5 tok/s (int8 165, fp16 129; the scalar tree gave 155). FP8 and INT8
cost the same; INT4 is still the speed format (~220 tok/s with an INT8 head).

## Group-wise quantization

`qf.dequant %w, %scale, %zp` with `scale`, `zp : tensor<(K/G) x N>`: element `[k, n]`
uses parameter `[k / G, n]`. The fused kernel recomputes the dequantisation while
staging the B tile, so the scale / zero point of a thread's 8 columns are one
16-byte + one 8-byte load per work item (`vectorize-loads`). Cost relative to
per-tensor parameters at 4096³: ~12 % (2048³: ~10 %), decode ~15–20 %.
Checkpoints in AWQ / GPTQ layout are repacked by `quantforge.quant.from_awq` /
`from_gptq`; act-order (`desc_act`) GPTQ goes through `quant.from_gptq_act_order`, which permutes the rows offline and returns the activation permutation (`perm=` in the torch op).

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
211–272 µs before, ~45–50 µs now (kernel ~50 µs). An M < 16 call only pads the activations and the
output (the weights, scales and zero points are not copied).

## Compile cache and row buckets

`compile_ir` publishes `libqf_kernel.so` + `lowered.mlir` under `~/.cache/quantforge/kernels/<key>/`
(atomic rename, safe for concurrent processes); the key hashes the IR, the target and pipeline
options, the path / size / mtime of `quantforge-opt`, `mlir-translate` and `llc`, and the source of the
CUDA shim, so a rebuilt compiler never serves a stale kernel. A compile takes ~0.3 s for a decode kernel
(more for large prefill tiles); a hit costs a file read. `torch_ops` additionally maps the row count of
a call to a bucket (`bucket_rows`) so that a server sees a handful of kernels per layer shape, not one
per prompt length; rows above the real M are zero activations and are dropped.

## Launches around the GEMM (torch op)

A call with M < 16 rows used to launch: pad the activations (`at::constant_pad_nd` = allocate + fill +
copy), the GEMM, the f32 -> f16 conversion of the result (and, before the overwrite kernel, a zeroing
of the accumulator). The C++ hot path now copies the rows into a reusable zero-initialised
`[Mp, K]` buffer (one launch; stale rows past M are harmless because a GEMM row only reads its own
activation row, tested with NaN / inf poison), and the kernel overwrites its output. What remains is
three launches per Linear. Removing the other two was measured to be worth little (a fused QKV and
gate/up already cut the number of Linears per layer from 7 to 4) and costly: no pad needs the kernel
to read only M activation rows (cp.async with a source size of 0 for the missing rows, one kernel per
M; reading them unpadded faults with `illegal address`), and no convert needs an f16 output memref
with an f32 accumulator, which the tensor-level IR cannot express without a temporary.

## Profile of a 7B decode step (Mistral-7B-Instruct, RTX 4050)

`torch.profiler` + the Chrome trace (kernel name, grid, duration) of one batch-1 decode step: grid 224
= the fused gate/up GEMM (N = 28672, 32 calls), 32 = o_proj and down_proj (alternating 54 / 180 us), 48 =
the fused q/k/v. Per call the kernels reach 165-175 GB/s when measured alone and ~150 GB/s in the model; the
step is 22 ms of ideal weight traffic, 29 ms measured. The default K tile (64) was the main loss for the
wide layers (gate/up 447 us vs 364 us with 128, via the torch op); `torch_ops._decode_opts` picks the
tile by N. In the model the large kernel runs ~8 % slower than alone (385-395 us vs 353-364 us) and shows
rare 0.5-1.4 ms outliers (another process using the GPU), not clock throttling (sm clock 2565 MHz, no
throttle flags during the run). Remaining: fp16 `lm_head` (7 % of the step). `hf_quantize_and_run.py --quant-head GROUP`
RTN-quantises it (Qwen2.5-0.5B-Instruct, RTN decoder, wikitext-2 8 x 512; fp16 head: ppl 24.09, CUDA-graph
decode 184 tok/s): group 128 -> ppl 26.29, 223 tok/s; group 64 -> 25.91, 221; group 32 -> 25.19, 220. So
the speed-up is +20 % for any group, and a smaller group only buys back part of the quality (the tied
head is also the embedding table, which RTN handles badly); kernel == dequantised reference in every case
(ppl within 0.002, 48/48 tokens). An INT8 head (`--quant-head 128 --head-bits 8`; the torch op takes a plain int8 [K, N]
weight and picks the W8A16 kernel from `packed.shape[1] == scale.shape[1]`) is the lossless route:
ppl 24.08 with group 128 (fp16 head 24.09), 214 tok/s (+16 %); group 32: 24.09, 209 tok/s, Mistral-7B-Instruct-v0.3 (NF4 -> INT4, `hf_bnb_to_quantforge.py --quant-head 128 --head-bits 8`, back to
back runs): ppl 8.507 vs 8.506, 35.3 vs 34.4 tok/s (HF generate), 34.9 vs 34.2 (graph) = +2-3 %, as predicted
by 134 MB / 160 GB/s ~ 0.8 ms of a 28 ms step, and 0.27 GB less VRAM (4.07 GB), the 1400 unfused HF kernels (10 %).

## Wave-aware K tile for wide decode layers (2026-11-20)

Sweep of N at K = 4096 (cold L2, `block-tile-n=128`, `prefetch-distance=1`): the 128-deep K tile holds 80 resident blocks on the
RTX 4050, and for 81 to 96 column blocks (N = 10240 .. 12288) the 32-deep tile (all blocks resident, one wave) is 8-12 % faster
(N = 11008: 136 against 150 us); at 104 blocks and more the deep tile wins again. `_decode_opts` uses bk = 32 for
10240 <= N <= 12288, 128 for N >= 8192 otherwise, 64 below; the autotuner has bk = 32 candidates (candidates version 3).
1 x 4096 x 11008: 149.5 -> 135.7 us (0.93 of Marlin), 16 x 4096 x 11008: 149.2 -> 135.5 us (0.94).

## Split-K (`split-k=S`, experimental, off by default)

`quantforge-tiling{split-k=S}` first cuts the K reduction with `linalg::tileReductionUsingForall` (`numThreads = [0, 0, S]`): a
`scf.forall (z)` whose body is an ordinary `[M, K/S] x [K/S, N]` GEMM, writing into a partial-result tensor, and a `linalg.generic` that sums
the S partial results. The block tiling then runs inside it. `mergeSplitKForall` (TilingPass.cpp) folds the nested foralls into one
`scf.forall (m, n, z)` so that `quantforge-forall-to-gpu` sees a 3-D grid (`quantforge-gpu-mapping` gives it `[y, x, z]`), and changes three things:

* the partial buffer is `P[S, M, N]`, not `[M, N, S]`: the tile of a block must be contiguous, otherwise `convert-vector-to-gpu` does not recognise the accumulator
  load and the contraction stays a `vector.contract` that cannot be lowered;
* blocks read and write their own tile `P[z, m, n]` directly (a whole-slice view `P[z]` makes bufferization copy the full `[M, N]` slice per block, with a device `malloc`);
* the zero fill of `P` is dropped when the kernel overwrites its output (`split-k-init=false`, set by the pipeline for `accumulate=false`): a fill is vectorised into a
  host-side `vector<M x N x S x f32>` otherwise. The reduce `linalg.generic` is excluded from vectorisation (`quantforge.no_vectorize`) and becomes a second kernel.

Constraints: `K / S` must be a multiple of `block-tile-k` and of the group size. Result: correct (`test/Transforms/gpu_pipeline_split_k.mlir`, `test/E2E/test_split_k.py`)
but 14-23 % faster only for long-K, narrow-N layers at 32 rows, and slower for most shapes at 64 rows (second kernel, `S x M x N` extra floats, a device allocation per call that
would break CUDA-graph capture). A useful version would reduce inside the kernel (atomics / semaphore) and take the scratch as an argument; see the blog post for the numbers.

## INT8 GEMM (experimental)

`i8 x i8 -> i32` goes through the same pipeline as the W4A16 kernels (`linalg.matmul_transpose_b`, B as `[N, K]`: `ldmatrix` cannot transpose 8-bit data). What it needed:
`FoldExtSIIntoContract` in `quantforge-vectorization` (the vectorised matmul is `contract(extsi a, extsi b, acc)`; LLVM 18 folds only `extf`), the pipeline option `thread-tile-k` (32 selects
`mma.m16n8k32`; 16 would give `m16n8k16`, which is no faster than f16), `smem-pad=16` (a row of an int8 tile must stay 16-byte aligned for `ldmatrix`) and i32 support in `quantforge-overwrite-output`.
Schedule: `quantforge.e2e.INT8_GEMM_OPTS` (best of 129 exact combinations). `SharedMemoryPromotionPass::isTileSizedSubview` recognises operand tiles by shape (`[M-tile x K]`, `[K x N-tile]`); a transposed B is
`[N-tile x K]` and used to be found only when block N happened to equal block M / warp M, otherwise the *warp* tile was promoted into a buffer shared by all warps (wrong results for 48 of 180 combinations);
`[N-tile x K]` is accepted now, and `test/E2E/test_int8_gemm.py` covers the combinations that failed.

### `epilogue=int8-dequant`

The program is `matmul_transpose_b (i8 -> i32)` followed by a `linalg.generic` (`quantforge.no_tile`, `quantforge.no_vectorize`) computing `y = f16(c * sx[m] * sw[n] + bias[n])`; the function arguments are tagged
`quantforge.int8_row_scale / int8_col_scale / int8_bias / int8_out`. `quantforge-fuse-epilogue{kind=int8-dequant}` (after the overwrite pass) rewrites every `vector<2xi32>` accumulator store of the GEMM kernel: the pair is converted
(`sitofp`, two multiplies, bias, `truncf`) and stored to the f16 output at absolute coordinates with ONE 4-byte `st.global.b32` (inline asm; the generic lowering of a `vector<2xf16>` store gave two 2-byte stores). The elementwise
kernel (every other `gpu.launch` that touches the int32 buffer), the dead subviews and the buffer's `gpu.alloc` / `gpu.dealloc` are erased: **the device allocation matters**, a first version left it and every call did a `cudaMalloc` of
`M x N x 4` bytes (device time 1.1 ms, wall time 2.1 ms; ncu said the kernel was fine, the profiler's device time versus the wall time per launch said where the second millisecond was).

### W4A8 (`W4A8GemmDq`): a plain `linalg.generic` as the weight producer

INT4 weights with INT8 activations need no new pass: the program is a `linalg.generic` (`quantforge.no_tile`, `quantforge.no_vectorize`) that builds the int8 weight `Bt[n, k] = (nibble(Bp[n, k floordiv 2], k mod 2) - zp[n, k floordiv 128]) * gs[n, k floordiv 128]`
(all `i8` arithmetic; the packed byte, scale and zero point are read with `tensor.extract` at `k divui 2` / `k divui 128`, **not** through affine maps with `floordiv`: those expand to signed floor-division sequences that `quantforge-simplify-index-math` cannot fold, so the byte loads stayed scalar `ld.global.u8` (80 per tile instead of 16 `ld.global.u32`; -7 % time)) followed by the `matmul_transpose_b` and the `int8-dequant` generic above. After bufferisation the producer is a standalone elementwise kernel writing an `[N x K]` int8 buffer, and the existing
`quantforge-fuse-producer-into-loads` pass (it is not specific to `qf.dequant`: any store whose backward slice is pure and reads only inputs) recomputes it at the loads that stage the B tile and deletes the kernel and the buffer,
so the 4-bit weights are the only weight traffic and the whole GEMM + dequantisation is one kernel (`w4a8_gemm_dq`, 1 launch in the profiler). The accumulation is int32 over all of K: the group scale is an *integer* and is folded
into the int8 weight before the `mma`, the only float scale is per output channel and lives in the epilogue (QServe's progressive quantisation). `quantforge.quant.w4a8_quantize` picks (gs, zp) per group by minimum squared error among the
pairs for which `(q - zp) * gs` stays inside int8, because the kernel's 8-bit arithmetic wraps. Cost: the dequantisation (and, per row block, its repetition) is ALU work in the main loop, 50-58 TOPS against 59-72 for the int8 kernel (ncu: 3.7x the instructions of the int8 kernel, issue-bound; the 8-bit arithmetic is still done in 16-bit registers and re-packed with `bfi`: a SWAR form on the loaded 32-bit word, as `quantforge-vectorize-dequant` does for the f16 path, is the next step and would be worth about 3 % of whole-model prefill).

### Packed W4A8 dequantisation (`quantforge-vectorize-dequant`, `pack_k8` layout)

The W4A8 nibbles are stored so that no byte permutation is needed to unpack them: in each block of 8 consecutive k, byte j holds element j (low nibble) and element j + 4 (high), so `w & 0x0F0F0F0F` of a 32-bit word is four *consecutive* elements and
`(w >> 4) & 0x0F0F0F0F` the next four (`quant.pack_k8`). The producer is written so that the index arithmetic simplifies completely (`divui` / `remui` only, the group index taken from `(k / 16) * 16`, or `/ 8` for the 8-wide f16 items of the decode kernel,
so that the (scale, zero point) loads of one work item are ONE value; a signed `floordiv` or `k & 4` defeats `quantforge-simplify-index-math`). `quantforge-vectorize-dequant` then recognises the 16 scalar trees `(nibble - z) * s` of a `vector<16xi8>` store and emits 32-bit word arithmetic:
`word * s` (a nibble times s <= 16 is at most 240: no carry between bytes) and the zero point subtracted per byte with `((x | H) - (y & ~H)) ^ ((x ^ ~y) & H)`, H = 0x80808080: about 10 instructions per four elements instead of ~30, bit-identical (mod 256). The f16 decode variant
(`vector<8xf16>`, `mulf(sitofp(nibble - z), s)`) uses the magic-number form of the W4A16 path (`1152 + q`, `sub.f16x2`, `mul.f16x2`). FileCheck: `test/Transforms/vectorize_dequant_w4a8.mlir`.
A scalar-load extension of `quantforge-prefetch-weight-loads` (the scale / zero point `memref.load`s) was tried for the decode kernel and gave nothing: reverted.
