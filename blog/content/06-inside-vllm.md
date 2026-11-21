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

Folding gains 9 percent (0.5B) and 15 percent (1.5B) for one to four requests and nothing from 16 requests up, where `M * S` exceeds 16 and the plain kernel runs. The last column adds the width-aware default schedules: the fixed tile options of the 17 to 64-row kernels and of the narrow-output decode kernels were tuned on 4096 x 4096 and were 1.5 to 3 times too slow for the small shapes of these models (autotune picks 32 / 64-column blocks and a look-ahead there; I turned its winners into rules, `benchmarks/sweep_skinny.py`). Together with the fold, 1 to 16 requests reach 0.80 to 0.88 of Marlin and 64 requests 0.73 to 0.80 (+16 % and +10 % over the fold alone). Ratios come from one run of Marlin and QuantForge back to back; Marlin itself moves by about 4 % between runs with the laptop's clocks.

### A measurement trap

My first A/B showed no difference at all (241 against 240 tokens per second, four interleaved runs each), and a profile of the same decode with the compile cache switched off showed 4.16 against 3.81 milliseconds per token. The reason: vLLM caches its compiled graphs on disk under a hash of its configuration that knows nothing about a plugin, so a graph compiled earlier, with the plain path, was silently reused, and the fold path never ran. The same mechanism could have produced wrong results (a cached graph with a fold factor baked in as a constant, used with weights folded differently). The plugin now makes every layer call the same op and reads the factor from the tensor shapes at run time, so a stale graph is merely slow, never wrong; after upgrading the plugin, clear `~/.cache/vllm/torch_compile_cache` once.

## Limitations of the plugin

4-bit GPTQ only: no AWQ, no 8-bit, no MoE layers. First start of a model compiles one kernel per layer shape and row bucket (about 0.3 seconds each, cached on disk afterwards). Only one GPU architecture (sm_89) and two small models were tested.
