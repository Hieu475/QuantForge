---
title: QuantForge inside vLLM: a fair end-to-end comparison, and where the 20 to 30 percent went
date: 2026-11-20
order: 6
summary: A vLLM quantization plugin that runs the generated kernels in a real serving engine, the throughput against Marlin at 1 to 64 concurrent requests, and how a per-layer diagnosis led to K-folding and shape-aware default schedules (70-80 percent of Marlin to 73-88 percent).
---

The kernel benchmarks of the [overview](01-overview.html) compare GEMMs in isolation, and the end-to-end numbers compare a Hugging Face model with a different one. Neither answers the practical question: **if I swap Marlin for the generated kernels in the same engine, with the same checkpoint, how much throughput do I lose or gain?** A vLLM plugin makes that experiment possible.

## The plugin

vLLM lets a package register a quantization method. The plugin (`quantforge.vllm_plugin`) is small because it reuses vLLM's GPTQ machinery:

* `QuantForgeConfig` subclasses vLLM's GPTQ config and answers to `quantization="quantforge"`. It accepts the same checkpoints (symmetric or asymmetric, group 32/64/128, `desc_act`);
* `QuantForgeLinearMethod` lets vLLM create and load the GPTQ tensors exactly as it does for Marlin (tensor parallelism, fused q/k/v and gate/up layers work unchanged), then **converts them once after loading** into the QuantForge layout and calls `torch.ops.quantforge.w4a16_gemm` in `apply`;
* the op already works under `torch.compile` and CUDA graphs, which vLLM uses for decode; the kernels for every row bucket are compiled in vLLM's warm-up passes, before graph capture;
* it registers itself through the `vllm.general_plugins` entry point (`pip install -e ./python`), because vLLM's engine runs in a separate process and pickles the config to it: the config classes must live at module level (my first version defined them inside a function and failed with a pickling error).

```python
from vllm import LLM
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", quantization="quantforge", dtype="float16")
```

## Same checkpoint, same engine: the result

Throughput is the number of generated tokens per second summed over the concurrent requests (RTX 4050, best of 3, two independent runs agreed to 1 to 2 percent):

| model | concurrent requests | Marlin (vLLM) | QuantForge | QuantForge / Marlin |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B GPTQ-Int4 | 1 | 298 | 231 | 0.78 |
| | 4 | 1153 | 893 | 0.77 |
| | 16 | 4188 | 3403 | 0.81 |
| | 64 | 10418 | 7747 | 0.74 |
| Qwen2.5-1.5B GPTQ-Int4 | 1 | 138 | 99 | 0.72 |
| | 16 | 2012 | 1542 | 0.77 |
| | 64 | 4929 | 3528 | 0.72 |

The outputs agree: on five prompts, four generate the same 64 greedy tokens with both kernels and the fifth agrees for 34 tokens before a near-tie flips (fp16 rounding differs); the mean absolute difference of the log-probability of the generated tokens is 0.002 (maximum 0.017). So **the plugin is correct, and inside a real serving engine the first version of the plugin reaches 70 to 80 percent of Marlin's throughput** (below: what closed part of the gap), not the 100 percent that the batch-1 kernel table for 4096 x 4096 layers might suggest.

## Where the missing 20 to 30 percent goes

My first guess was overhead: the op launches three kernels per GEMM (copy the activations into a 16-row padded buffer, the GEMM, convert the f32 result to f16), Marlin one. A profile of one M = 1 call confirms the extra kernels (about 2 microseconds of device time) but also that this cannot be the main cause, because the gap does not shrink for the larger model. The per-layer kernel comparison does (M = 1, microseconds, same method as the kernel tables):

| layer of Qwen2.5-1.5B / 0.5B | K x N | Marlin | QuantForge | ratio |
|---|---|--:|--:|--:|
| qkv | 1536 x 2048 | 11.7 | 12.4 | 0.94 |
| gate / up (merged) | 1536 x 17920 | 78.5 | 84.3 | 0.93 |
| gate / up (merged, 0.5B) | 896 x 9728 | 27.6 | 28.4 | 0.97 |
| o_proj | 1536 x 1536 | 9.7 | 13.4 | **0.73** |
| **down_proj** | 8960 x 1536 | 40.8 | 66.2 | **0.62** |
| **down_proj (0.5B)** | 4864 x 896 | 15.2 | 27.7 | **0.55** |

Wide layers are fine. The layers with a **long K and a narrow N**, which in every transformer is the `down_proj` of the MLP, are 40 to 45 percent slower than Marlin. With 128 or 64 output columns per block, N = 1536 gives 12 to 24 blocks for 20 SMs: not enough warps in flight to stream the weights at full speed. Marlin splits the K dimension across blocks; the generated kernels do not.

This overturns a conclusion I wrote earlier ("split-K is not the lever"): it was measured at 1 x 11008 x 4096, where N = 4096 already gives 64 blocks. The shapes that matter for decode in a serving engine are different, and the engine comparison found them where the isolated benchmark did not. The textbook fix is split-K in the kernel, which in this compiler means touching six passes (tiling, mapping, shared-memory promotion, prefetch, accumulator hoisting, store). Before that I tried what could be done without the compiler.

## Fixing it without touching the compiler: K-folding

Three things that did not work: sweeping the tile and prefetch options (nothing below 67 microseconds for the 8960 x 1536 layer, 22 percent above Marlin's 41), a multi-stream split-K (several concurrent launches into separate accumulators, summed afterwards: 72 to 56 microseconds for that layer, no gain for 4864 x 896), and a deeper pipeline.

What worked is a re-layout. Cut K into S slices and lay them side by side as the *columns* of one wider GEMM, with the S slices of the activation as S *rows*:

```
W' = [W_0 | W_1 | ... | W_{S-1}]   shape [K/S, S*N]        A' = x reshaped to [M*S, K/S]  (free)
C' = A' @ W'   ->   y[m] = sum_s C'[m*S + s, s*N:(s+1)*N]    (S diagonal blocks)
```

The kernel computes every (slice, slice) pair, S times more arithmetic than needed, but a tensor-core tile has 16 rows anyway, so as long as `M * S <= 16` the extra arithmetic is free: the weights are still read exactly once, by S times as many blocks and with a K loop S times shorter. The factor is chosen per layer shape by measuring the real op once (cached): S = 2 for the 4864 x 896 layer, 7 for 8960 x 1536. Layers with N >= 4096 already have enough blocks and are left alone. The cost is a second copy of the weights of the folded layers (the plain layout serves the calls with more rows), which only matters for small models because only N <= 2048 is folded.

Kernel level, one row, whole op inside a CUDA graph (microseconds): 4864 x 896 37.7 to 23.3, 8960 x 1536 68.5 to 43.0, 1536 x 1536 14.4 to 12.6. In vLLM, same checkpoint, interleaved runs that agreed to 1 percent:

| model | requests | Marlin (tok/s) | QuantForge (tok/s) | ratio: first version | + K-fold | + shape-aware schedules | + narrow blocks (now) |
|---|--:|--:|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B | 1 | 289 | 277 | 0.78 | 0.85 | 0.87 | 0.96 |
|  | 4 | 1112 | 1050 | 0.77 | 0.86 | 0.88 | 0.94 |
|  | 16 | 4066 | 3796 | 0.81 | 0.84 | 0.86 | 0.93 |
|  | 32 | 7289 | 6533 | n/a | n/a | n/a | 0.90 |
|  | 64 | 11076 | 9590 | 0.74 | 0.70 | 0.80 | 0.87 |
| Qwen2.5-1.5B | 1 | 141 | 122 | 0.72 | 0.85 | 0.84 | 0.87 |
|  | 4 | 546 | 446 | n/a | 0.74 | 0.80 | 0.82 |
|  | 16 | 2095 | 1702 | 0.77 | 0.80 | 0.82 | 0.81 |
|  | 32 | 3681 | 2896 | n/a | n/a | n/a | 0.79 |
|  | 64 | 5796 | 4300 | 0.72 | 0.68 | 0.73 | 0.74 |

Folding gained 9 percent (0.5B) and 15 percent (1.5B) for one to four requests when it was introduced (about 4 percent now that the plain schedules are better) and nothing from 16 requests up, where `M * S` exceeds 16 and the plain kernel runs. The last two columns add, first, the width-aware default schedules (the tile options of the 17 to 64-row kernels and of the narrow-output decode kernels were tuned on 4096 x 4096 and were 1.5 to 3 times too slow for the small shapes of these models; autotune picks 32 / 64-column blocks and a look-ahead there, and I turned its winners into rules, `benchmarks/sweep_skinny.py`), then a second sweep that found 32 and 64-column blocks 11 to 52 % faster than 128-column ones for N <= 2048 at 32 and 64 rows (for wide outputs no candidate of 74 beat the defaults by more than 4 %, `benchmarks/sweep_skinny_blocks.py`) and a fused epilogue kernel for the folded layers (3 kernels per call instead of 5 to 7: 20 to 35 % faster per layer, but invisible in whole-model time, about 1 to 2 %, because the GPU is busy 97 % of the time anyway). Together: the 0.5B model reaches 0.87 to 0.96 of Marlin at 1 to 64 requests, the 1.5B model 0.74 to 0.87; the absolute speed rose by 4 to 7 % over the previous column. Ratios come from one run of Marlin and QuantForge back to back; Marlin itself moves by about 4 % between runs with the laptop's clocks.

### A measurement trap

My first A/B showed no difference at all (241 against 240 tokens per second, four interleaved runs each), and a profile of the same decode with the compile cache switched off showed 4.16 against 3.81 milliseconds per token. The reason: vLLM caches its compiled graphs on disk under a hash of its configuration that knows nothing about a plugin, so a graph compiled earlier, with the plain path, was silently reused, and the fold path never ran. The same mechanism could have produced wrong results (a cached graph with a fold factor baked in as a constant, used with weights folded differently). The plugin now makes every layer call the same op and reads the factor from the tensor shapes at run time, so a stale graph is merely slow, never wrong; after upgrading the plugin, clear `~/.cache/vllm/torch_compile_cache` once.

## Split-K in the compiler: it works, and it does not pay

The `ncu` comparison pointed at work balance (Marlin: 20 persistent blocks), and the textbook answer is split-K. I implemented it as a pipeline option (`--quantforge-gpu-pipeline='... split-k=S'`, off by default), and the implementation turned out to need fewer changes than I feared, because MLIR already has `linalg::tileReductionUsingForall`:

1. the tiling pass cuts the K reduction into S chunks with that helper: a `scf.forall (z)` whose body is an ordinary smaller-K GEMM, writing into a partial-result tensor, and a `linalg.generic` that adds the S partial results;
2. the existing block tiling runs inside it, producing a forall nested in a forall; a small rewrite merges the two into one `scf.forall (m, n, z)` so that the existing launch lowering gets a 3-D grid;
3. the rest of the pipeline needed three fixes, each found by reading the failing IR: the zero-fill of the partial buffer was vectorised into a host-side `vector<16x256x4xf32>` (dropped in overwrite mode, where every block writes its whole tile); the partial buffer was laid out `[M, N, S]`, which makes a block's tile non-contiguous and stops the Tensor Core lowering from recognising it (it is `[S, M, N]` now); and the reduce had to be excluded from vectorisation so that it becomes its own small kernel.

It is **correct**: the result matches the unsplit kernel and the dense reference (relative error 2e-4 in every configuration, three test shapes in the suite, 38 of 38 FileCheck tests unchanged). The scale and zero-point indexing survives a K loop that starts at `z * Kc` because the dequantisation is fused through `extract_slice` offsets, not through loop counters. **It is not worth turning on**, measured as device time of all its kernels (RTX 4050, the op's default schedule):

| shape (rows x K x N) | unsplit | split-K | K-folding (16 rows only) |
|---|--:|--:|--:|
| 16 x 8960 x 1536 | 70.3 us | 60.0 (S=5) | 46.0 |
| 16 x 4864 x 896 | 28.3 | 26.6 (S=2) | 18.2 |
| 32 x 8960 x 1536 | 86.5 | 66.2 (S=5) | |
| 32 x 4864 x 896 | 36.9 | 28.7 (S=2) | |
| 64 x 4864 x 896 | 50.1 | 43.2 (S=2) | |
| 64 x 8960 x 1536 | 101.5 | 110.7 (S=5) | |
| 64 x 4096 x 4096 | 94.0 | 113.8 (S=4) | |
| 64 x 11008 x 4096 | 240.1 | 260.1 (S=4) | |
| 64 x 896 x 9728 | 51.3 | 165.6 (S=7) | |

It helps the long-K, narrow-N layers by 14 to 23 percent at 32 rows, loses to K-folding at 16 rows, and makes most layers slower at 64 rows. The reasons are the ones Marlin avoids by reducing inside the kernel: a second kernel for the reduce, S times M x N extra floats written and read back, and a buffer allocated per call, which would also break CUDA-graph capture (the scratch would have to become a function argument). Fixing those (an in-kernel reduction with atomics or a semaphore, scratch passed in) is the real work of a Marlin-style persistent kernel, and the best case is the 14 to 23 percent on the layers that make up a quarter of the GEMM time, so roughly 3 to 5 percent of a decode step. I stopped there: the code stays in the compiler as an experimental, tested, default-off option, and the diagnosis of the gap to Marlin at 64 rows is the useful result.

## Other checkpoint formats

The plugin also reads AWQ 4-bit and GPTQ 8-bit checkpoints (Qwen2.5-0.5B, the same engine, tokens/s summed over the requests, two runs agreed to 1 %; each against its own Marlin method):

| format | requests | Marlin | QuantForge | QuantForge / Marlin | output vs Marlin |
|---|--:|--:|--:|--:|---|
| AWQ 4-bit (`awq_marlin`) | 1 | 307 | 270 | 0.88 | 287 of 320 greedy tokens identical, mean abs. log-prob diff 0.0020 |
| | 4 | 1168 | 1040 | 0.89 | |
| | 16 | 4357 | 3811 | 0.87 | |
| | 64 | 11715 | 9509 | 0.81 | |
| GPTQ 8-bit (`gptq_marlin`) | 1 | 239 | 222 | 0.93 | 320 of 320 identical, mean diff 0.0018 |
| | 4 | 919 | 856 | 0.93 | |
| | 16 | 3482 | 3235 | 0.93 | |
| | 64 | 10157 | 8600 | 0.85 | |

The 8-bit GPTQ path needed one conversion detail: GPTQ stores unsigned bytes, the kernels take int8, so the weights and the zero points are both shifted by 128, which leaves `q - zero` unchanged. AWQ differs from GPTQ in its packing (along the output dimension, with a permuted nibble order), which the converter undoes.

Other models (same engine, GPTQ 4-bit; tokens/s summed over the requests, two runs agreed to 1 %):

| model | requests | Marlin | QuantForge | QuantForge / Marlin | output vs Marlin |
|---|--:|--:|--:|--:|---|
| TinyLlama-1.1B (Llama architecture, `desc_act=true`: real act-order) | 1 | 232 | 209 | 0.90 | 320 of 320 greedy tokens identical, mean abs. log-prob diff 0.0012 |
| | 4 | 904 | 804 | 0.89 | |
| | 16 | 3288 | 3003 | 0.91 | |
| | 64 | 8604 | 6547 | 0.76 | |
| Qwen2.5-3B | 1 | 81.5 | 71 | 0.87 | 320 of 320 identical, mean diff 0.0019 |
| | 4 | 320 | 280 | 0.88 | |
| | 16 | 1215 | 1073 | 0.88 | |

K-folding changes nothing for the 3B model (71 tokens/s with and without it): its narrow layers (2048 outputs) did not gain enough to be folded. **Memory:** a first run of the 3B model failed to start
on the 6 GB card (vLLM reported -0.32 GiB for the KV cache): the op allocated an f32 accumulator for all the rows of a call, and vLLM's profiling pass sends 8192 tokens, i.e. 720 MB for the
22016-wide gate/up layer. The op now processes long inputs in chunks of 2048 rows written straight into the f16 result (temporary memory bounded, KV cache +0.1 GiB against Marlin's +0.24).

## Update: where the decode gap to Marlin was hiding (2026-11-22)

The remaining decode gap (0.85 to 0.9 of Marlin) did not look like a kernel problem: per layer the kernels were within 0.88 to 1.03 of Marlin's, except `down_proj` before K-folding. An `nsys` summary of the same 128-token decode in vLLM showed what was different: **33 313 `float16_copy` kernels** in the QuantForge run, 4 microseconds each, that Marlin does not launch. Counting kernels per op call settled it: a plain decode call was a copy that pads the activations to 16 rows, the GEMM into an f32 buffer, and a conversion to f16; with a bias (the q, k and v of Qwen) two more small kernels to expand it into the accumulator, five in all, against Marlin's one. For the 1152-wide qkv layer at one row that is 10.7 microseconds against 6.4 for Marlin, with 5.6 of them the GEMM.

Two steps. First, one existing fused kernel (the epilogue of the K-fold, with S = 1) adds the bias and writes f16, which takes the bias layers from five kernels to three. Then the proper fix in the compiler: `epilogue=f16-store`, the machinery of Update 3's `int8-dequant` (rewrite the accumulator stores of the GEMM into converted, biased, 4-byte f16 stores and delete the elementwise kernel and the buffer) taken to an f32 accumulator. A plain GEMM is then two kernels, the padding copy and the GEMM: qkv at one row **10.7 to 6.5 microseconds**, in line with Marlin's 6.4; the gate / up layer 23.1 to 21.0 (Marlin 27.3). It also removes the f32 buffer from prefill. The first change made a test fail on purpose: it demanded bit-equality between starting the accumulator from the bias and adding it afterwards, two orders of f32 summation that agree to one f16 ulp; I relaxed it to a tolerance for that case and kept exact equality for the activations.

vLLM 0.25.1, same checkpoints, decode tokens per second at 1 / 4 / 16 / 64 requests (QuantForge against Marlin, interleaved in one session): Qwen2.5-0.5B 286 / 1091 / 3974 / 10191 against 303 / 1168 / 4333 / 11665, **0.94 / 0.93 / 0.92 / 0.87**; Qwen2.5-1.5B 128 / 461 / 1771 / 4711 against 140 / 548 / 2070 / 5814, **0.91 / 0.84 / 0.86 / 0.81** (it was 0.74 to 0.87). The prefill of the plugin went from 20.6k to 28.0k tokens per second on the 0.5B (Marlin 33.7k). The W4A8 method inherits all of it in its two-layout mode: decode 285 / 3988 at 1 / 16 requests on the 0.5B (0.94 / 0.93 of Marlin) with prefill 1.09x Marlin; its one-copy mode, which uses its own decode kernel, also uses the new epilogue (264 / 3952) and stays at 0.87 / 0.92.


## Limitations of the plugin

GPTQ 4-bit and 8-bit and AWQ 4-bit with zero points; no MoE layers; AWQ without zero points is refused. First start of a model compiles one kernel per layer shape and row bucket (about 0.3 seconds each, cached on disk afterwards). Only one GPU architecture (sm_89) and two small models were tested.
