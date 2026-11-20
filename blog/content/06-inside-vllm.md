---
title: QuantForge inside vLLM: a fair end-to-end comparison, and where the 20 to 30 percent go
date: 2026-11-20
order: 6
summary: A vLLM quantization plugin that runs the generated kernels in a real serving engine, the throughput against Marlin at 1 to 64 concurrent requests, and a per-layer diagnosis that points at one kernel weakness (long K, narrow N).
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

The outputs agree: on five prompts, four generate the same 64 greedy tokens with both kernels and the fifth agrees for 34 tokens before a near-tie flips (fp16 rounding differs); the mean absolute difference of the log-probability of the generated tokens is 0.002 (maximum 0.014). So **the plugin is correct, and inside a real serving engine QuantForge reaches 70 to 80 percent of Marlin's throughput**, not the 100 percent that the batch-1 kernel table for 4096 x 4096 layers might suggest.

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

This overturns a conclusion I wrote earlier ("split-K is not the lever"): it was measured at 1 x 11008 x 4096, where N = 4096 already gives 64 blocks. The shapes that matter for decode in a serving engine are different, and the engine comparison found them where the isolated benchmark did not. The next kernel task is a split-K schedule for these layers (K-range per `blockIdx.z`, accumulation with atomics or a second reduction), plus fusing the pad and the f16 convert into the kernel.

## Limitations of the plugin

4-bit GPTQ only: no AWQ, no 8-bit, no MoE layers. First start of a model compiles one kernel per layer shape and row bucket (about 0.3 seconds each, cached on disk afterwards). Only one GPU architecture (sm_89) and two small models were tested.
