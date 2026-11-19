# QuantForge

An MLIR-based generator of GPU kernels for **weight-only quantized (W4A16) matrix
multiplication** — the INT4-weights × FP16-activations GEMM that dominates LLM
inference. QuantForge lowers

```mlir
%w  = qf.unpack  %packed : tensor<KxN/2xi8> -> tensor<KxNxi8>      // 2 INT4 per byte
%wf = qf.dequant %w, %scale, %zp : ... -> tensor<KxNxf16>
%c  = linalg.matmul ins(%act, %wf) outs(%acc)
```

to a single PTX kernel that reads the **packed** weights from HBM, dequantizes
them while staging tiles into shared memory, and feeds Tensor Cores
(`mma.sync.m16n8k16` + `ldmatrix`), with `cp.async` + double buffering.

> **Status: research prototype (v0.0.x).** W4A16 and W8A16 GEMM with per-tensor
> **or group-wise** scale / zero point (AWQ / GPTQ checkpoints can be repacked),
> a skinny-M **decode** schedule, an autotuner; static tile-aligned shapes in the
> compiler, NVIDIA sm_80+. See [Limitations](#limitations).

## Performance snapshot

RTX 4050 Laptop (sm_89, 20 SMs, ~192 GB/s), f32 accumulate, group size 128 for the
Marlin comparison. Reproduce with `benchmarks/compare_marlin.py` and
`python/quantforge/benchmark.py`.

**vs. Marlin** (vLLM 0.25.1's hand-written W4A16 kernel; same INT4 weights and scales;
device kernel time from the CUPTI trace, L2 flushed before every call so the weights
come from DRAM; kernels timed in interleaved rounds after a GPU warm-up (laptop clocks drift ±30 % otherwise); QuantForge autotuned per shape;
[`benchmarks/marlin_rtx4050_sm89.json`](benchmarks/marlin_rtx4050_sm89.json)):

| M × K × N | Marlin (µs) | QuantForge (µs) | cuBLAS fp16 (µs, 4× the weight bytes) | QuantForge / Marlin speed |
|---|--:|--:|--:|--:|
| 1 × 4096 × 4096 | 49.4 | 38.0 | 180.8 | 1.30× |
| 4 × 4096 × 4096 | 49.4 | 38.0 | 181.7 | 1.30× |
| 16 × 4096 × 4096 | 50.0 | 50.0 | 183.0 | 1.00× |
| 1 × 4096 × 11008 | 126.7 | 139.2 | 482.3 | 0.91× |
| 1 × 11008 × 4096 | 127.0 | 135.1 | 483.3 | 0.94× |
| 16 × 4096 × 11008 | 127.4 | 148.9 | 498.6 | 0.86× |
| 512 × 4096 × 4096 | 387.9 | 456.7 | 371.1 | 0.85× |
| 2048 × 4096 × 4096 | 1514.6 | 1764.6 | 1427.1 | 0.86× |

QuantForge is 2.5–4.8× faster than an fp16 cuBLAS GEMM at decode (it moves a quarter of
the bytes) and 0.85–1.30× of Marlin: **the "≥ 90 % of Marlin" goal is reached for decode
(M ≤ 16) at 4096 × 4096 (1.0–1.3×) and for 1 × 4096 × 11008 / 1 × 11008 × 4096 (0.91 / 0.94×); not yet
for 16 × 4096 × 11008 (0.86×) and the prefill GEMMs (0.85–0.86×)**. Activations are re-touched after the
L2 flush (they are L2-resident in a real model, written by the previous layer). At
prefill sizes the compute-bound kernel is ~15 % slower than cuBLAS fp16 with
per-tensor parameters and ~30 % slower with group-wise ones.

Without group-wise parameters (per-tensor scale), TFLOP/s, 4096³: 36.9 (cuBLAS fp16
43.6); 1 × 4096 × 4096 decode: 45 µs warm-L2 (4× faster than cuBLAS). Numbers from a
laptop GPU are only meaningful relative to the same machine and clock state.

## Build

Supported toolchain: **LLVM/MLIR 18.1.x** (the MLIR C++ API changes every release;
CMake refuses other majors unless `-DQUANTFORGE_ALLOW_UNSUPPORTED_LLVM=ON`).

```bash
# Ubuntu 24.04
sudo apt install cmake ninja-build llvm-18-dev libmlir-18-dev mlir-18-tools
pip install lit                    # for `check-quantforge` (any python env on PATH)

cmake --preset default             # RelWithDebInfo, apt LLVM 18, ./build
cmake --build --preset default
```

For another LLVM 18 install pass `-DMLIR_DIR=<prefix>/lib/cmake/mlir -DLLVM_DIR=<prefix>/lib/cmake/llvm`.

## Test

```bash
cmake --build build --target check-quantforge     # 37 FileCheck tests, CPU only
ctest --test-dir build -L lit                     # same, via CTest

# GPU end-to-end (needs an NVIDIA GPU, CUDA toolkit in /usr/local/cuda or $CUDA_HOME,
# python with torch + pytest). Skipped automatically when unavailable.
QUANTFORGE_BUILD_DIR=$PWD/build python -m pytest -v
scripts/ci_gpu_preflight.sh                       # checks a (CI) machine: toolchain, CUDA, torch, RAM
scripts/sanitize.sh                               # compute-sanitizer memcheck + racecheck (22 pipeline variants)
python python/quantforge/benchmark.py --cold --group-size 128 --shape 1x4096x4096 \
    "decode=true" --compare benchmarks/baseline_decode_rtx4050_sm89.json   # perf regression (--cold: DRAM weights)
scripts/check.sh --gpu --sanitize                 # everything
```

## Use

```bash
# qf IR -> LLVM dialect with embedded PTX
build/tools/quantforge-opt in.mlir --quantforge-gpu-pipeline='chip=sm_80' -o lowered.mlir
```

```python
import sys; sys.path.insert(0, "python")
from quantforge.e2e import W4A16Gemm, pack_int4
from quantforge import quant, autotune

# per-tensor scale / zero point, any shape (N even)
kern = W4A16Gemm.build(M=100, K=70, N=90, pad=True)
out = kern.run(act_f16, pack_int4(q_uint4), scale_f16, zero_point_i8)   # C += A @ dequant(W)

# group-wise (G rows of K share a scale/zero point): scale f16[K/G, N], zero point i8[K/G, N]
kern = W4A16Gemm.build(M=1, K=4096, N=4096, group_size=128, decode=True, pad=True)
out = kern.run(act, packed, scale, zero)                  # bias=/residual= are fused as accumulator init

# checkpoints: AWQ / GPTQ int32 tensors -> QuantForge layout
packed, scale, zero = quant.from_awq(qweight, qzeros, scales)
packed, scale, zero = quant.from_gptq(qweight, qzeros, scales, g_idx)

# INT8 weights (W8A16): packed_w is a plain int8 [K, N] tensor
kern = W4A16Gemm.build(M=128, K=4096, N=4096, bits=8, group_size=128)

# fused gated MLP: one GEMM for gate+up, silu(x@Wg) * (x@Wu) written directly
w_gu = quant.interleave_gate_up((packed_g, scale_g, zero_g), (packed_u, scale_u, zero_u))
kern = W4A16Gemm.build(M=1, K=4096, N=2 * 11008, group_size=128, decode=True, pad=True, epilogue="swiglu")
hidden = kern.run(act, *w_gu)                              # [1, 11008]

# fused activation on the accumulator: relu | silu | gelu (f32 accumulator)
kern = W4A16Gemm.build(M=1, K=4096, N=11008, group_size=128, decode=True, pad=True, epilogue="silu")

# best tile configuration per (GPU, shape), measured once and cached in ~/.cache/quantforge
kern = autotune.autotune(M=1, K=4096, N=11008, group_size=128)
```

### As a PyTorch op

```python
import quantforge.torch_ops                       # registers torch.ops.quantforge.w4a16_gemm
y = quantforge.torch_ops.w4a16_linear(x, packed, scale, zero_point, group_size=128, bias=b)
# x: f16 [..., K]  ->  y: f16 [..., N] (f32 accumulation); also works under torch.compile
```

The op is a `torch.library.custom_op` (fake implementation included), is covered by `opcheck`,
`torch.compile(fullgraph=True)` and CUDA-graph capture/replay (`test/E2E/test_torch_op.py`).
The op is registered by a small C++ extension (`runtime/torch_runtime.cpp`, built on first import in
about 10 s and cached by torch; `QUANTFORGE_NO_CPP=1` or a failed build falls back to the Python
implementation) that keeps the whole call in C++ and calls back into Python only to compile a new
shape or for the uncommon cases (K / N padding, per-tensor parameters, unaligned tensors).
Kernels are compiled for a *row bucket* (≤ 16, 64, 128, 256, 384, 512, … 4096, then multiples of 1024; the
activations are zero-padded to the bucket) on first use, so 24 different prompt lengths need 9 kernels, and the
compiled libraries are kept in `~/.cache/quantforge/kernels` (key = IR + options + tool binaries + shim source;
`QUANTFORGE_KERNEL_CACHE` relocates it, `QUANTFORGE_NO_KERNEL_CACHE=1` disables it): a restarted server
compiles nothing (the test suite takes 4 s instead of 46 s on a warm cache). Kernels are compiled per shape and cached; **warm each shape up once before
capturing a graph** (module loading is not allowed during capture). `set_autotune(True)` tunes
the tile options per shape instead of using the default decode schedule.
Measured 4096 × 4096, weights L2-resident (8 MB < 24 MB L2): op ≈ 52–61 µs per call (eager or
graph) against Marlin's 19–25 µs — with the L2 warm the QuantForge kernel (~43 µs) is
latency-bound, Marlin is not. With the weights streamed from DRAM (a real model has far more weight
than L2) the kernel is faster than Marlin (table above); the extra small kernels around it
(output init, f16 conversion, M < 16 padding) cost a few µs.

The generated kernels issue 16-byte vector loads on the packed weights, scales and zero
points: those tensors must be 16-byte aligned (cudaMalloc / PyTorch allocations are; the
harness raises on interior views).

The whole lowering is described in [`python/quantforge/PIPELINE.md`](python/quantforge/PIPELINE.md)
(stages, every option, measured effect of each optimization).

## Repository layout

| path | content |
|------|---------|
| `include/QuantForge`, `lib/Dialect` | the `qf` dialect (`qf.unpack`, `qf.dequant` with per-tensor / group-wise parameters) |
| `lib/Transforms` | passes + `--quantforge-gpu-pipeline` (`GPUPipeline.cpp`) |
| `tools/quantforge-opt` | `mlir-opt`-style driver (all upstream passes + QuantForge's) |
| `python/quantforge` | `e2e.py` (compile + run harness), `quant.py` (group-wise quantization, AWQ/GPTQ repack), `autotune.py`, `benchmark.py`, `smoke.py`, CUDA runtime shim |
| `test/Transforms`, `test/Dialect` | lit/FileCheck tests |
| `test/E2E` | GPU tests: compile → run → compare with PyTorch |
| `benchmarks/` | recorded baselines (JSON, with environment) and `compare_marlin.py` |
| `scripts/` | `check.sh`, `sanitize.sh` |

## End-to-end on a real model

`benchmarks/qwen_end_to_end.py` and `benchmarks/qwen_decode_graph.py` run **Qwen2.5-0.5B-Instruct**
with its official GPTQ-Int4 checkpoint, every decoder Linear replaced by the QuantForge kernel (the
`gate_proj` + `up_proj` fused into one GEMM with the `swiglu` epilogue, `q/k/v_proj` fused into one
GEMM with `quant.concat_columns`), on the RTX 4050 laptop GPU:

| wikitext-2 test, 8 × 512 tokens | perplexity |
|---|--:|
| fp16 original | 19.93 |
| GPTQ weights dequantised to dense fp16 (reference) | 23.195 |
| **QuantForge** | **23.194** |
| QuantForge + `lm_head` RTN 4-bit (`--quant-head`, not in the checkpoint) | 25.36 |

The QuantForge model reproduces the dequantised reference (logit |Δ|max 0.10 on a logit std of 2.9,
top-1 agreement 99.2 %, the fused gate/up keeps `silu * up` in f32 where the reference rounds to f16
in between; 48 of 48 greedy tokens identical). Batch-1 decode, whole step captured in a CUDA graph (static KV cache):

| | tokens/s |
|---|--:|
| fp16 | 129 |
| QuantForge (GPTQ checkpoint) | 196 (176 with q/k/v as three GEMMs) |
| QuantForge + RTN 4-bit `lm_head` | 239 (perplexity +2.2: not recommended) |

GPU time of one decode step before the q/k/v and gate/up fusions: the 168 W4A16 kernels took 2.6 ms
against 4.4 ms for the same layers in cuBLAS fp16 (now 120 kernels); what stays is the fp16 `lm_head` (272 MB, ~1.7 ms), attention / norms (~1.3 ms)
and the small kernels around each call (activation padding, f32 → f16).
Without the graph (HF `generate`, Python launching every kernel) the per-call overhead decides:
the Python op gives 54 tokens/s, the C++ dispatcher hot path (see "As a PyTorch op") ~79 (71 before
the fused gate/up, q/k/v and the overwrite kernel removed launches), fp16 ~100 — the layers
of a 0.5 B model are only 10–15 µs of GPU time each and every call still launches four small
kernels around the GEMM (activation pad, f32 → f16 conversion, plus the GEMM; the accumulator
zero-init is gone since kernels can overwrite their output).

### Other models (from a Hugging Face cache)

`benchmarks/hf_quantize_and_run.py` takes any fp16 Llama / Qwen2 / OLMo-2 / Phi-3 style model, quantises
every decoder Linear to INT4 (round-to-nearest, asymmetric, group 128 — a plain quantiser: use GPTQ / AWQ
checkpoints for quality), runs it on the kernels (q/k/v and gate/up fused) and compares it with the
original fp16 model and with the same INT4 weights dequantised to dense fp16. wikitext-2 test, 8 × 512
tokens, RTX 4050 laptop (6 GB):

| model | perplexity fp16 | INT4 dense reference | **QuantForge** | greedy tokens = reference | decode tokens/s, CUDA graph, fp16 → QuantForge |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-1.5B-Instruct | 13.48 | 15.296 | **15.297** | 48/48 | 52 → 90 |
| Llama-3.2-1B-Instruct | 20.15 | 24.399 | **24.403** | 48/48 | 67 → 124 |
| OLMo-2-0425-1B-Instruct | 19.32 | 21.294 | **21.294** | 48/48 | 63 → 124 |
| Qwen2.5-3B-Instruct | does not fit | – | 14.284 | – | – → 56 |
| Llama-3.2-3B-Instruct | does not fit | – | 18.544 | – | – → 55 |
| Phi-3.5-mini-instruct (3.8 B) | does not fit | – | 9.618 | – | – (HF generate: 54) |

A 7 B model fits the 6 GB card too, from a bitsandbytes NF4 checkpoint (`benchmarks/hf_bnb_to_quantforge.py`
dequantises each Linear and re-quantises it to uniform INT4, layer by layer in place, since NF4's
non-uniform code cannot be fed to an INT4 GEMM): `Mistral-7B-Instruct-v0.3` has wikitext-2 perplexity
8.366 under bitsandbytes and 8.506 under QuantForge (the extra 0.14 is the second quantisation), 4.2 GB of
GPU memory (4.8 GB peak), coherent generation. Its batch-1 decode matches bitsandbytes' own NF4 kernel:
37 vs 38 tokens/s with HF `generate` (CUDA graph: 36). One decode step is 29 ms of GPU time: the
fused gate/up GEMM (4096 → 28672) 13.8 ms, o_proj + down_proj 7.6 ms, q/k/v 2.7 ms, the fp16 `lm_head` 2.1 ms
and 1400 small non-GEMM kernels 2.9 ms; at 180 GB/s the weights alone would take 22 ms. Profiling that step
found that the K tile of the default decode schedule matters for wide layers (128 beats 64 by 18 % at
N = 14336 and 28 % at N = 28672, loses by 10–15 % at N <= 6144), now chosen by the output width
(32 → 36 tokens/s); an occasional 1.4 ms outlier in a 0.4 ms kernel is another process sharing the GPU.

The kernels reproduce the dense INT4 reference to 0.004 perplexity (logits top-1 agreement 99.6–100 %); the
gap to fp16 is the round-to-nearest quantiser's. The 3 B and 3.8 B models only run quantised on this
card: 3.1 / 3.2 / 2.6 GB of GPU memory at peak (their fp16 weights are 5.6 – 7.3 GB). Phi-3.5's CUDA-graph
capture fails inside Hugging Face's LongRoPE code (a data-dependent `if` on a GPU tensor), not in the
kernels; its numbers above are from `generate`. Models with a fused `qkv_proj` / `gate_up_proj` (Phi-3)
are handled, models whose `in_features` is not a multiple of 128 are rejected.

## Limitations

* **Speed:** 0.85–1.30× Marlin on the RTX 4050 (see above); prefill GEMMs and
  16 × 4096 × 11008 are at 0.85–0.86×. Not done: stream-K / persistent scheduling (the
  decode grid is bigger than the resident-block count, see `PIPELINE.md`), Marlin-style
  offline weight repacking.
* No split-K (1 × 11008 × 4096 leaves only 64 blocks), no stream-K; the decode schedule
  uses one 16-row tile (M ≤ 16; larger M uses the 128-row schedule).
* W4A16, W8A16 and FP8 (E4M3) weights with f16 activations (the torch op picks the kernel from the
  weight: packed INT4 `[K, N/2]`, plain int8 `[K, N]`, or a `torch.float8_e4m3fn` `[K, N]` from
  `quant.quantize_fp8`; 8-bit weights are group-wise only and have no fused swiglu, FP8 ignores the zero
  point). No NF4, no activation quantization (FP8 *activations* / FP8 tensor cores are not used). Act-order GPTQ (`desc_act=True`)
  works through `quant.from_gptq_act_order` (rows permuted offline, `perm=` in `w4a16_linear`
  gathers the activations: one extra small kernel per call).
  Zero points are stored unpacked (`int8 [K/G, N]`); AWQ / GPTQ checkpoints are repacked
  offline in Python (`quantforge.quant`). Besides round-trip tests against independent packers,
  real 4-bit checkpoints (Qwen2.5-0.5B-Instruct AWQ and GPTQ-Int4, group size 128, layers of
  shape 896 → 4864, 4864 → 896, 896 → 128, 896 → 896) are run through the kernels and compared
  with vLLM's AWQ dequantisation and an f32 reference built from the GPTQ file format
  (`test/E2E/test_real_checkpoints.py`: downloads a few MB with HTTP Range requests, skips offline).
  Real checkpoints are covered for `desc_act=False` GPTQ and `zero_point=True` AWQ (GEMM version);
  act-order is tested against a synthetic g_idx only.
* Epilogue: bias and residual are fused by initializing the accumulator
  (`run(bias=, residual=)`); `relu` / `silu` / `gelu` (tanh form) are applied to the f32
  accumulator before the store (`build(epilogue="silu")`, `activation=` in the torch op), i.e.
  `act(bias + A @ W)`. A gated MLP is one GEMM: `quant.interleave_gate_up` interleaves the gate and
  up weights by column and `epilogue="swiglu"` stores `silu(x @ Wg) * (x @ Wu)` (`activation="swiglu"`
  in the op, output half as wide, no bias). `accumulate=False` makes any kernel write `C = A @ W`
  without reading C. Not fused: gelu-gated variants, residual after the activation, quantised output.
* Tile-aligned static shapes in the compiler; other shapes work through zero-padding
  in the Python harness. The degenerate decode case of a single block (N = 64) is not
  fused into one kernel (correct, but two launches).
* NVIDIA sm_80+ only (PTX is JIT-compiled by the driver); no Hopper `wgmma`/TMA.
* The scalar FMA fallback (`tensor-core=false`) is correct but ~10× slower; LLVM 18's
  NVPTX backend miscompiles one variant of it (documented in `PIPELINE.md`).
* Kernels are PTX loaded through a small CUDA driver shim (`runtime/mgpu_cuda_runtime.c`):
  the JIT-compiled module is cached and launches go on PyTorch's current stream,
  asynchronously like any other op (host cost ~45–50 µs per call from Python for a
  4096 × 4096 decode GEMM, kernel 38 µs; Marlin's launch is ~20–25 µs wall). The torch op's hot path is C++
  (no interpreter per call); the Python `W4A16Gemm.run` still costs ~45 µs. The benchmark tables report device kernel time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 with LLVM Exceptions (the same license as LLVM/MLIR) — see [LICENSE](LICENSE).
