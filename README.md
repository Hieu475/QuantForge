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
| 1 × 4096 × 4096 | 49.6 | 50.0 | 180.9 | 0.99× |
| 4 × 4096 × 4096 | 49.4 | 49.8 | 181.5 | 0.99× |
| 16 × 4096 × 4096 | 50.0 | 49.8 | 182.3 | 1.01× |
| 1 × 4096 × 11008 | 126.8 | 135.7 | 482.4 | 0.93× |
| 1 × 11008 × 4096 | 127.2 | 131.1 | 484.0 | 0.97× |
| 16 × 4096 × 11008 | 127.7 | 135.5 | 499.9 | 0.94× |
| 32 × 4096 × 4096 | 50.9 | 58.8 | 181.0 | 0.87× |
| 32 × 4096 × 11008 | 130.3 | 149.3 | 497.3 | 0.87× |
| 64 × 4096 × 4096 | 58.3 | 87.4 | 181.4 | 0.67× |
| 64 × 4096 × 11008 | 139.9 | 200.0 | 506.9 | 0.70× |
| 128 × 4096 × 4096 | 105.1 | 152.2 | 188.7 | 0.69× |
| 256 × 4096 × 4096 | 201.0 | 263.0 | 208.2 | 0.76× |
| 512 × 4096 × 4096 | 391.5 | 461.6 | 373.4 | 0.85× |
| 2048 × 4096 × 4096 | 1532.1 | 1775.8 | 1453.0 | 0.86× |

### Against other kernels (gemlite, bitsandbytes)

Same method as above (`benchmarks/compare_baselines.py`, [`benchmarks/compare_baselines_rtx4050_sm89.json`](benchmarks/compare_baselines_rtx4050_sm89.json)),
kernel time in µs, the fastest of the three INT4 kernels in bold. gemlite is a Triton W4A16 kernel (0.6.0, its own autotuner,
same INT4 weights and scales); bitsandbytes uses its NF4 code book on the same fp16 matrix (a different quantiser, listed for speed
only); cuBLAS reads the pre-dequantised fp16 weights (4× the bytes).

| M × K × N | QuantForge | Marlin | gemlite (Triton) | bitsandbytes NF4 | cuBLAS fp16 |
|---|--:|--:|--:|--:|--:|
| 1 × 4096 × 4096 | 50.0 | 49.3 | **49.3** | 55.9 | 180.8 |
| 4 × 4096 × 4096 | 49.8 | **49.3** | 100.0 | 155.1 | 181.7 |
| 16 × 4096 × 4096 | **49.8** | 50.0 | 61.4 | 156.8 | 182.6 |
| 32 × 4096 × 4096 | 58.8 | **50.9** | 71.0 | 160.0 | 180.9 |
| 64 × 4096 × 4096 | 87.4 | **57.8** | 89.8 | 194.3 | 181.5 |
| 128 × 4096 × 4096 | 152.2 | **104.4** | 159.2 | 241.0 | 188.3 |
| 256 × 4096 × 4096 | 263.0 | **199.6** | 300.3 | 457.6 | 207.6 |
| 512 × 4096 × 4096 | 461.6 | **388.9** | 529.9 | 600.9 | 371.1 |
| 2048 × 4096 × 4096 | 1775.8 | **1527.5** | 2002.7 | 1711.1 | 1457.4 |
| 1 × 4096 × 11008 | 135.7 | **126.8** | 129.3 | 139.6 | 482.4 |
| 1 × 11008 × 4096 | 131.1 | **127.1** | 129.2 | 146.4 | 484.2 |

QuantForge is on par with gemlite at M = 1 (49.9 / 49.3 µs), ahead of it for M = 4 … 2048 (gemlite needs 100 µs at M = 4 and
61 µs at M = 16 where QuantForge needs 50), and 5 % behind it on the wide decode shape 1 × 4096 × 11008 (135.7 against 129.4 µs, after the wave-aware K tile described in `PIPELINE.md`; it was 149.5 µs before). Marlin is
the fastest kernel from M = 32 up. The Triton kernel's strength is its autotuned decode path; its prefill kernel is ~15 % slower than
QuantForge's.

### End to end against vLLM

vLLM 0.25.1 (offline engine, CUDA graphs, one request, 256 new tokens, best of 3; `benchmarks/vllm_e2e.py`) on Qwen2.5-0.5B-Instruct:
fp16 **165 tokens/s**, the official GPTQ-Int4 checkpoint with Marlin kernels **310 tokens/s** (1.88× faster). QuantForge inside a plain
Hugging Face model with a static cache and a CUDA graph: fp16 129, INT4 (+ INT8 head) 209 tokens/s (1.62×), 226 with
`torch.compile` (1.75×). In absolute terms vLLM is faster, because its model implementation is fully fused (norms, rotary, residuals,
sampling) while the Hugging Face graph launches ~1400 small kernels per step; for the GEMMs themselves the table above shows
QuantForge on par with Marlin at batch 1. Running the QuantForge kernels *inside* vLLM (a plugin) would be the fair end-to-end
comparison and is not done.

QuantForge is 3.2–3.7× faster than an fp16 cuBLAS GEMM at decode (it moves a quarter of the bytes) and
0.67–1.01× of Marlin. **The "≥ 90 % of Marlin" goal is reached for decode (M ≤ 16): 4096 × 4096 (0.99–1.01×), 1 × 11008 × 4096 (0.97×), 1 × 4096 × 11008 (0.93×) and 16 × 4096 × 11008 (0.94×); it is not reached for M = 32 … 256 (0.67–0.87×;
M = 64 / 128 are the worst: the kernel is limited by the Tensor Core / `ldmatrix` path at ~50 % tensor-pipe activity, not
by DRAM, while Marlin stays memory-bound up to M = 64) and for the prefill GEMMs (0.85–0.86×)**. Activations are
re-touched after the L2 flush (they are L2-resident in a real model, written by the previous layer). At prefill sizes the
compute-bound kernel is ~25 % slower than cuBLAS fp16 with group-wise parameters.

> **Correction (2026-11-20).** Earlier versions of this README reported 38.0 µs / 1.30× Marlin for M = 1 and 4 at
> 4096 × 4096 (and 0.91 / 0.94× for the wide decode shapes). That figure could not be reproduced: re-running the same
> harness on the very commit that recorded it gives 50.2 µs (0.99×). With the current method the kernel is on the DRAM
> limit: `ncu` (`dram__bytes_read`, caches flushed) shows 9.2 MB read from DRAM for the 8.65 MB of weights and scales, i.e.
> ~184 GB/s at 50 µs, and a 1 GB flush instead of the 256 MB one changes neither kernel's time (50.3 µs both).
> The table above was regenerated in one run (`benchmarks/compare_marlin.py --autotune`, interleaved rounds);
> `benchmarks/l2_residency_check.py` reproduces the flush and DRAM-byte checks.

Without group-wise parameters (per-tensor scale), TFLOP/s, 4096³: 36.9 (cuBLAS fp16
43.6); 1 × 4096 × 4096 decode: ~49 µs with the weights L2-resident (the kernel is latency-bound there, 3.7× faster than cuBLAS). Numbers from a
laptop GPU are only meaningful relative to the same machine and clock state.

## Install the Python package

```bash
# 1. build the C++ tool once (see Build below), 2. install the Python part, 3. check
pip install ./python[hf]                      # needs a CUDA build of torch; [hf] adds transformers / safetensors / pandas
export QUANTFORGE_BUILD_DIR=$PWD/build        # the CMake build directory containing tools/quantforge-opt
quantforge-doctor                             # GPU, nvcc / ptxas, quantforge-opt, and one small kernel against a reference
```

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
scripts/sanitize.sh                               # compute-sanitizer memcheck + racecheck (27 pipeline variants, 54 runs)
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

### Quantise a whole model

```python
import torch, quantforge.nn as qnn
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16)
stats = qnn.quantize_model(model, fmt="int4", group_size=128)   # int4 | int8 | fp8, RTN, layer by layer on the GPU
qnn.quantize_model(model, "int8", only=("lm_head",), fuse=False)  # optional: a lossless INT8 head (+2-16 % decode)
out = model.generate(ids.cuda(), max_new_tokens=64)             # the Linears are QuantLinear modules now
```

`quantize_model` replaces every `nn.Linear` (the `lm_head` stays dense unless asked) with
`quantforge.nn.QuantLinear`, fuses q / k / v into one GEMM and a SiLU-gated MLP into gate + up (swiglu
epilogue) for Llama / Qwen / Mistral style blocks, and works under `torch.compile` and CUDA graphs.
Layers whose `in_features` is not a multiple of the group size fall back to 64 / 32 / 16 (else stay dense).
`QuantLinear.from_quantized(packed, scale, zero, ...)` wraps GPTQ / AWQ tensors converted with `quantforge.quant`.

Real AWQ / GPTQ checkpoints go through `load_quantized_checkpoint` (4-bit, group-wise; `desc_act` act-order GPTQ is
handled with a row permutation, `zeros_offset=0` for `gptq_v2`):

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16)   # the architecture
qnn.load_quantized_checkpoint(model, "Qwen2.5-0.5B-Instruct-GPTQ-Int4/model.safetensors", method="gptq")
```

The embeddings and norms are copied from the checkpoint as well (AWQ folds its activation scales into the norms, so
the dense base model's norms are *not* the right ones). Checked against a dense twin (same dequantised weights in
`nn.Linear`) with `benchmarks/hf_load_checkpoint.py`, wikitext-2 16 x 512: Qwen2.5-0.5B GPTQ-Int4 ppl 26.336
vs 26.335, 59/59 greedy tokens identical, 182 tok/s CUDA-graph decode; AWQ ppl 26.689 vs 26.689, 59/59, 184 tok/s;
**TinyLlama-1.1B-Chat GPTQ with `desc_act=True`** (TheBloke; all 154 layers act-ordered) ppl 17.259 vs 17.259, 60/60 greedy tokens;
**a real FP8 checkpoint** (`RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic`, `method="fp8"`, per-channel scales expanded to group 128)
ppl 22.543 vs 22.543 for its twin and 22.441 for the fp16 base model (greedy decoding diverges after 28 of 59 tokens: near-ties
flip under the different fp16 rounding of the twin's matmul; the logits agree to 99.4 % top-1).
Example with perplexity and decode speed: `benchmarks/hf_quantize_model.py` (RTN INT4, group 128, INT8 head). Perplexity on the
**whole** Wikitext-2 test split (the Phi row: first 8 windows only, marked *); decode speed is the mean of 7 interleaved CUDA-graph
runs in one session on the 6 GB RTX 4050 (`benchmarks/decode_repeat.py`, spread in the table below):

| model | fp16 | INT4 + INT8 head |
|---|---|---|
| Qwen2.5-0.5B-Instruct | ppl 18.79, 120 tok/s | 22.91, 197 tok/s |
| Qwen2.5-1.5B-Instruct | 12.66, 48 tok/s (3.1 GB) | 14.30, 91 tok/s (1.5 GB) |
| Qwen2.5-3B-Instruct | does not fit in 6 GB | 13.13, ~61 tok/s (2.5 GB; speed from an earlier session) |
| Llama-3.2-1B-Instruct | 17.91, 63 tok/s (2.5 GB) | 22.28, 132 tok/s (1.3 GB) |
| OLMo-2-0425-1B-Instruct | 17.24, 57 tok/s (3.0 GB) | 19.02, 129 tok/s (1.2 GB) |
| Phi-3.5-mini-instruct (3.8B) | does not fit in 6 GB | 9.62 *, 2.2 GB (gate_up_proj fused into swiglu; CUDA-graph capture fails inside HF's LongRoPE) |

`--compile` (`torch.compile` of the quantised model before the CUDA-graph capture, which fuses HF's norm / rotary /
elementwise kernels) adds ~10 % on the 0.5B model (206 -> 226 tok/s, 86 of 96 greedy tokens identical: fp16 rounding
differs), 4 % on Qwen2.5-1.5B (97 -> 101) and 3 % on Llama-3.2-1B (142 -> 146); the share of those kernels shrinks as
the GEMMs grow.

(RTN INT4 costs 2-4 perplexity points on these models; quantise with GPTQ / AWQ and wrap the result with
`QuantLinear.from_quantized` for a better quality.) Layers are quantised in row chunks, so the 150k-row
`lm_head` of a 3B model needs no fp32 copy of itself on the GPU.

### Better 4-bit quality: GPTQ in the library

```python
import quantforge.gptq as qgptq
calib = qgptq.calibration_ids(tokenizer, train_text, nsamples=128, seqlen=512)    # [128, 512] token ids
qgptq.quantize_model_gptq(model, calib, group_size=128)                          # INT4, replaces RTN
qnn.quantize_model(model, "int8", only=("lm_head",), fuse=False)                  # lossless INT8 head
```

Sequential GPTQ (Hessian of every Linear input from the already quantised blocks, column-wise error compensation
through the Cholesky factor of H^-1, scales rounded to f16 before the error is propagated, calibration token norms capped at
10x the median, see below). 49 s for Qwen2.5-0.5B, ~150 s for 1.5B, 300 s for 3B on the RTX 4050. Wikitext-2 perplexity on the
**whole test split** (564-584 windows of 512 tokens; calibration: 128 x 512 tokens from the first 20 000 rows of the *train* split;
INT8 head; `benchmarks/hf_gptq.py --nseq 0 --fp16`):

| model | fp16 | RTN INT4 | GPTQ INT4 | share of the RTN loss recovered |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 18.79 | 22.91 | **20.45** | 60 % |
| Qwen2.5-1.5B-Instruct | 12.66 | 14.30 | **13.38** | 56 % |
| Llama-3.2-1B-Instruct | 17.91 | 22.28 | **20.21** | 47 % |
| OLMo-2-0425-1B-Instruct | 17.24 | 19.02 | **18.13** | 50 % |
| Qwen2.5-3B-Instruct (no fp16 reference: does not fit) | - | 13.13 | **12.06** | - |

The speed of the kernels does not change (same layout).

**A pitfall that the first version of this table hid.** On 8 test windows GPTQ looked better than RTN everywhere; on the
whole test split it was *worse than RTN on Llama-3.2-1B* (24.08 against 22.28) and recovered only 24 % on Qwen2.5-1.5B.
`benchmarks/ppl_per_window.py` showed GPTQ better than RTN in 83 % of the windows but with a few windows completely broken
(loss 9-10 per token instead of ~3), and a layer-by-layer comparison traced it to `layers.1.mlp.down_proj`: the layer that
creates the "massive activation" of the attention-sink token (its input has a token 675x the median norm in Llama-3.2,
790x in Qwen2.5-1.5B). That token dominates the Hessian and the error compensation then damages other tokens. Capping every
calibration token's norm at a multiple of the median before it enters the Hessian (`token_clip`) fixes it. A cap of 30 fixed the
whole-split perplexity (Llama 20.12) but a slow regression test on the windows that had been broken (`test/E2E/test_gptq_real_model.py`)
showed one still broken (window 70: loss 5.9 against 3.0 for fp16); a per-window analysis of all 564 windows found no window worse than
round-to-nearest at a cap of 10 (30: two windows, one by 2.8 nats; 5: none, but 0.2 ppl worse), so **10 is the default**
(`token_clip=10`; cost: ~0.07 ppl against 30). Quantising just the massive-activation layer with RTN also fixes Llama, but *hurts*
Qwen-0.5B, so a fixed list of layers does not work. Options that did not help consistently, hence off by default: `clip=True`
(per-group range shrinking), `act_order=True` (the layers lose the q/k/v fusion), larger damping, more calibration windows.

**How robust is it?** (`benchmarks/hf_gptq.py --seed/--train`, `benchmarks/arc_eval.py`; whole Wikitext-2 test split, cap 10)

| check | Qwen2.5-0.5B | Llama-3.2-1B | Qwen2.5-1.5B |
|---|--:|--:|--:|
| RTN INT4 | 22.91 | 22.28 | 14.30 |
| GPTQ, Wikitext-2 calibration, 3 seeds | 20.45 / 20.46 / 20.48 (sd 0.01) | 20.21 / 20.22 / 20.25 (sd 0.02) | 13.38 (one seed) |
| GPTQ, calibration on a different corpus (cnn_dailymail articles) | 20.70 | 20.43 | 13.65 |

The spread between calibration seeds is 0.01-0.02 (it was 0.08-0.15 with a cap of 30: the cap also stabilises the result); calibrating on
another domain costs 0.2-0.3 points on Wikitext-2 and still beats RTN by 2 points or more (the numbers in the first row are *in-domain*).
Zero-shot ARC accuracy of the GPTQ models (lm-evaluation-harness scoring, `acc` / `acc_norm`, %; standard error 1.0 pp on ARC-Easy, n = 2376,
1.4 pp on ARC-Challenge, n = 1172):

| model | variant | ARC-Easy | ARC-Challenge |
|---|---|--:|--:|
| Qwen2.5-0.5B | fp16 / RTN / GPTQ | 65.6 / 62.5 / 63.3 · 59.2 / 59.3 / 60.5 | 30.7 / 31.7 / 29.4 · 33.8 / 32.3 / 33.0 |
| Llama-3.2-1B | fp16 / RTN / GPTQ | 69.0 / 64.3 / 66.4 · 63.8 / 60.4 / 62.5 | 34.6 / 33.5 / 33.2 · 37.9 / 35.3 / 36.9 |
| OLMo-2-1B | fp16 / RTN / GPTQ | 71.8 / 71.2 / 70.4 · 71.0 / 68.3 / 69.5 | 41.0 / 39.8 / 39.3 · 42.0 / 40.7 / 42.2 |
| Qwen2.5-1.5B | fp16 / RTN / GPTQ | 76.8 / 65.6 / **73.1** · 76.0 / 62.3 / **72.0** | 43.5 / 37.0 / **40.5** · 46.7 / 39.7 / **44.0** |

(each cell: `acc` triple, then `acc_norm` triple.) Perplexity understated the damage of round-to-nearest on Qwen2.5-1.5B: it loses 11 points on ARC-Easy
(76.8 -> 65.6) and GPTQ gets back 7.5 of them (73.1; 75.0 with a cap of 30, within about 2 standard errors); for the other three models the differences
between the variants are mostly within the standard error, i.e. these small models lose little on ARC at INT4 either way.

**Run-to-run spread** (`benchmarks/decode_repeat.py`, `benchmarks/kernel_repeat.py`). Batch-1 CUDA-graph decode, 7 repetitions with fp16 and INT4
(+ INT8 head) interleaved, mean +- standard deviation, one session:

| model | fp16 (tok/s) | INT4 + INT8 head (tok/s) | speed-up |
|---|--:|--:|--:|
| Qwen2.5-0.5B | 120.2 +- 0.6 | 197.1 +- 1.6 | 1.64x +- 0.02 |
| Qwen2.5-1.5B | 47.9 +- 2.1 | 91.3 +- 5.4 | 1.91x +- 0.07 |
| Llama-3.2-1B | 62.5 +- 3.3 | 132.0 +- 7.3 | 2.11x +- 0.12 |
| OLMo-2-1B | 57.2 +- 2.6 | 128.9 +- 7.7 | 2.26x +- 0.16 |

Absolute speeds drift by 5-10 % within a session and by up to 15 % between sessions (the GPU is shared with a desktop; the single numbers
quoted elsewhere in this README come from the faster sessions), the *speed-up* is stable to about 5 %. Kernel time against Marlin, 5 independent
invocations of `compare_marlin.py --autotune`: 1 x 4096 x 4096 ratio 0.990 +- 0.000 (Marlin 49.7 +- 0.1 us, QuantForge 50.2 +- 0.1), 1 x 4096 x 11008
0.930 +- 0.000, 64 x 4096 x 4096 0.676 +- 0.005, 2048 x 4096 x 4096 0.842 +- 0.011 (the long prefill kernels drift by ~1.5 % with the GPU temperature,
both kernels together).

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

### Inside vLLM (plugin)

```bash
pip install -e ./python --no-deps       # registers the plugin through the `vllm.general_plugins` entry point
export QUANTFORGE_BUILD_DIR=$PWD/build
python -c "from vllm import LLM; LLM('Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', quantization='quantforge', dtype='float16')"
```

`quantization="quantforge"` loads a 4-bit GPTQ checkpoint exactly as vLLM does (tensor parallelism, fused qkv / gate_up layers, `desc_act`) and converts it once into
the QuantForge layout; every linear layer then runs `torch.ops.quantforge.w4a16_gemm` (works under vLLM's `torch.compile` and CUDA graphs). Same GPTQ checkpoint,
same engine, throughput in tokens/s summed over the concurrent requests (RTX 4050, best of 3, two independent runs agreed to 1-2 %;
`benchmarks/vllm_batch_sweep.py`):

| model | requests | Marlin (vLLM) | QuantForge | QuantForge / Marlin |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B GPTQ-Int4 | 1 | 298 | 231 | 0.78 |
| | 4 | 1153 | 893 | 0.77 |
| | 16 | 4188 | 3403 | 0.81 |
| | 64 | 10418 | 7747 | 0.74 |
| Qwen2.5-1.5B GPTQ-Int4 | 1 | 138 | 99 | 0.72 |
| | 16 | 2012 | 1542 | 0.77 |
| | 64 | 4929 | 3528 | 0.72 |

Output agrees with Marlin (`benchmarks/vllm_compare_outputs.py`: 4 of 5 prompts identical over 64 greedy tokens, the fifth for 34 tokens; mean |log-probability
difference| 0.002, max 0.017). The kernel-level table (M = 1, K x N) showed where the rest went: qkv 1536 x 2048 0.94x, gate/up 1536 x 17920 0.93x, but `o_proj`
1536 x 1536 **0.73x**, `down_proj` 8960 x 1536 **0.62x** and 4864 x 896 **0.55x**: a long K with a narrow N gives only 12-24 blocks for 20 SMs. No tile / prefetch
option fixes it (swept), so the plugin **folds K** (`python/quantforge/kfold.py`, `QUANTFORGE_KFOLD=0` to disable): K is cut into S slices that are laid side by side as
columns of one wider GEMM (and as S rows of the activation). The tensor-core tile already has 16 rows, so while `M * S <= 16` the S-fold redundant arithmetic is free, the weights
are still read once, by S times as many blocks with a K loop S times shorter. No compiler change; costs a second copy of the weights of the folded layers (only layers with
N <= 2048). Kernel level, M = 1 (us, whole op in a CUDA graph): 4864 x 896 37.7 -> 23.3, 8960 x 1536 68.5 -> 43.0, 1536 x 1536 14.4 -> 12.6. Inside vLLM, same checkpoint
(tokens/s summed over the requests, interleaved A/B, two runs each, identical across runs):

| model | requests | Marlin (tok/s) | QuantForge (tok/s) | ratio: first version | + K-fold | + schedules (now) |
|---|--:|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B | 1 | 295 | 258 | 0.78 | 0.85 | 0.87 |
|  | 4 | 1129 | 995 | 0.77 | 0.86 | 0.88 |
|  | 16 | 4221 | 3650 | 0.81 | 0.84 | 0.86 |
|  | 64 | 11306 | 9100 | 0.74 | 0.70 | 0.80 |
| Qwen2.5-1.5B | 1 | 137 | 115 | 0.72 | 0.85 | 0.84 |
|  | 4 | 531 | 427 | n/a | 0.74 | 0.80 |
|  | 16 | 2008 | 1646 | 0.77 | 0.80 | 0.82 |
|  | 64 | 5614 | 4090 | 0.72 | 0.68 | 0.73 |

Fold helps for 1-4 requests (+9 % / +15 %) and does nothing from 16 requests up (more than `16 // S` rows: the plain kernel runs). The last column adds the width-aware default schedules: the fixed tile options of the 17 to 64-row kernels and of the narrow-output decode kernels were tuned on 4096 x 4096 and were 1.5 to 3 times too slow for the small shapes of these models (autotune picks 32 / 64-column blocks and a look-ahead there; I turned its winners into rules, `benchmarks/sweep_skinny.py`). Together with the fold, 1 to 16 requests reach 0.80 to 0.88 of Marlin and 64 requests 0.73 to 0.80 (+16 % and +10 % over the fold alone). Ratios come from one run of Marlin and QuantForge back to back; Marlin itself moves by about 4 % between runs with the laptop's clocks.

### FP8 W8A8 on the Ada FP8 tensor cores (a negative result, for now)

`quantize_model(model, "fp8_w8a8")` replaces the Linears with `FP8W8A8Linear`: FP8 E4M3 weights *and* activations (per-output-channel weight scale, dynamic per-token
activation scale) through `torch._scaled_mm` (cuBLASLt) on the FP8 tensor cores of sm_89, which are 1.8-2.6x faster than the fp16 ones for a bare GEMM on this card
(73 against 28 TFLOP/s at M = 512). It is **not** generated by the QuantForge compiler (LLVM 18 cannot emit the FP8 `mma`). Quality is good
(Qwen2.5-0.5B wikitext-2 perplexity 19.10 against 18.79 for fp16 and 22.91 for INT4 RTN; Llama-3.2-1B 18.13 against 17.91), but the **speed is not**: whole-model
prefill of Qwen2.5-0.5B takes 117 ms for 1 x 2048 tokens against 108 ms in fp16 (0.92x; per-tensor scales: 103 ms, but 326 against 214 ms for 8 x 512), because the quantise /
rescale passes around the GEMM cost as much as the GEMM saves at these layer sizes (`benchmarks/hf_fp8_w8a8.py`). Per layer the speed ranges from 0.35x to 1.5x of fp16
depending on the shape. A speed-up would need the activation quantisation and the scale epilogue fused into the GEMM (a compiler job). Use it for the 2x smaller weights
and the quality, not for speed.

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

* **Speed:** 0.85–1.01× Marlin for M ≤ 16 and prefill (M ≥ 512) on the RTX 4050 (decode 4096² on par, the wide decode shapes 0.93–0.97×), but only **0.67–0.87× at
  M = 32 … 256** (see the table above). The gap is not DRAM, tile choice or dequantisation arithmetic (ablated:
  1–3 %); a hand-written CUDA prototype that keeps B in registers like Marlin reaches 75.8 µs at M = 64 (compiler
  87, Marlin 59), see `PIPELINE.md` -- closing it needs tuning of a hand-written kernel, which is not done.
  Also not done: stream-K / persistent scheduling and split-K (the decode grid can exceed the resident-block
  count, `PIPELINE.md`).
* Schedules by rows: M ≤ 16 one 16-row decode tile, ≤ 32 / ≤ 64 skinny 32 / 64-row tiles, larger M 64 / 128-row
  tiles (the torch op buckets M and picks them; `autotune` searches them).
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
  Real checkpoints are covered for `desc_act=False` and `desc_act=True` GPTQ, `zero_point=True` AWQ (GEMM version) and
  FP8-dynamic (weights only: 0.55 % of its bf16 scales are subnormal in f16).
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
  4096 × 4096 decode GEMM, kernel ~50 µs; Marlin's launch is ~20–25 µs wall). The torch op's hot path is C++
  (no interpreter per call); the Python `W4A16Gemm.run` still costs ~45 µs. The benchmark tables report device kernel time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 with LLVM Exceptions (the same license as LLVM/MLIR) — see [LICENSE](LICENSE).
