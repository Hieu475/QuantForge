---
title: Faster than Marlin only on the INT8 tensor cores: what W4A16 could not do and what W4A8 took
date: 2026-11-23
order: 8
summary: The exact-numerics W4A16 path stays at 0.8 to 0.94 of Marlin from 32 requests up, whatever I tried; an INT4-weight, INT8-activation kernel with measured schedules passes Marlin from about 128 concurrent requests. The negative results, and three benchmarking traps that cost days.
---

The [vLLM post](06-inside-vllm.html) ended with a gap: at 32 to 256 concurrent requests the generated W4A16 kernel is slower than Marlin. This post is what happened to that gap since: a list of things that did not close it, one thing that did (by changing the arithmetic, not the schedule), and the mistakes in the measurements on the way.

## What did not close the W4A16 gap

Ratios against Marlin in vLLM, Qwen2.5-1.5B, same weights, interleaved runs: **0.94, 0.94, 0.89, 0.84, 0.80, 0.86, 0.85** at 1, 16, 32, 64, 128, 192 and 256 concurrent requests. The profiler explains the shape:

| 128 x 4096 x 4096, cold weights (`ncu`) | Marlin | generated W4A16 |
|---|--:|--:|
| duration | 115 us | 154 us |
| registers per thread | 255 | 124 |
| resident blocks per SM | 1 (persistent) | about 3 |
| DRAM throughput | 50 % | 36 % |

Both kernels are latency-bound; Marlin keeps more bytes in flight with fewer, fatter blocks and loads its B fragments from global memory straight into registers (the weights are pre-shuffled for it). Things I tried that did not move the gap: a sweep of block, warp and K tiles per shape; split-K with atomics at 128 rows (165 us against 155 us: more blocks do not help, so it is not wave quantisation); a single kernel for the row padding instead of three small launches (inside the noise); the memory fraction handed to vLLM (the ratio is the same at 0.7 and 0.78, so it is not a KV-cache effect). What is left is the weight format, which means a different layout and a different dequantisation path, and I have not done it.

## Changing the arithmetic: W4A8

The INT8 tensor cores do twice the work of the fp16 ones and the INT4 weights stay the only weight traffic. The kernel (`W4A8GemmDq`) expands the nibbles to int8 while the tile is staged into shared memory, accumulates in int32 over the whole of K and applies one float scale per output channel in the epilogue. The format is QServe's progressive quantisation: INT8 per output channel, then INT4 per group of 128 with an *integer* scale and zero point, chosen so that `(q - z) * s` stays inside int8 and the kernel's 8-bit arithmetic cannot wrap. Three things made it fast rather than merely correct:

* **K-major scales with a padding row.** The 8 output channels a thread dequantises have 8 adjacent scale bytes in `[K/128, N]` order, one vector load instead of 8 bytes 32 apart: 3 to 5 %. The look-ahead loads of the pipeline read the scale of the group after the last one, so the buffer has one extra row of zeros (the first version read past the end of the tensor).
* **A planner that measures.** The schedule rule had been tuned on layers with N <= 2048; at 64 x 4096 x 4096 it ran 115 us where another schedule runs 65 us. `int8_plan` times a handful of candidates per (rows, K, N) once, caches the winner, and the op only looks it up. A kernel pack carries the decisions.
* **A native activation-quantisation kernel.** Quantising the activations per token was a `torch.compile`d pair of kernels plus two padding copies; one CUDA kernel does the amax, the scale, the rounding and the padding. It had to be bit-identical to the torch expression it replaced, and it was not at first: torch divides by a scalar as a multiplication by the float reciprocal, and rounds half to even.

{{chart:w4a8_vs_marlin}}

| concurrent requests | 1 | 16 | 32 | 64 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|
| W4A16 (exact Marlin numerics) | 0.94 | 0.94 | 0.89 | 0.84 | 0.80 | 0.86 | 0.85 |
| W4A8, Hadamard-rotated (default) | 0.92 | 0.93 | 0.97 | 0.98 | 1.01 | 1.11 | 1.16 |
| W4A8, no rotation | 0.95 | 0.96 | 0.99 | 0.99 | 1.04 | 1.13 | 1.18 |

The honest reading: W4A8 is faster than Marlin from about 128 concurrent requests (and from 64 rows at the level of one GEMM), and below that it is 2 to 8 % slower, because up to 64 rows it uses the same W4A16 kernels as before. It also is not the same arithmetic: Marlin's checkpoints are W4A16, this path quantises the activations to INT8. [The next post](09-hadamard-rotation.html) is about what that costs in quality, and how the rotation turns the cost into a gain.

## Three measurement traps

* **Flush the L2 by reading, not by writing.** My sweep flushed with `zero_()`; the dirty lines written back during the timed kernel added 30 to 50 us to *every* time, which changed which schedule won. The same artifact inflated the first MoE numbers.
* **vLLM changes torch's default dtype while it loads.** The planner created its random test tensors with `torch.rand`, which was f16 inside vLLM and f32 everywhere else; the kernel read an f16 buffer as f32 and failed with an illegal address *only inside vLLM*. `compute-sanitizer python script.py` on the engine load found it in minutes after days of staring; the fix is to pass `dtype=` everywhere.
* **A throughput comparison in vLLM is a KV-cache comparison.** Any extra buffer an op allocates shrinks the KV cache, vLLM preempts, and the ratio moves by 30 %. [The MoE post](10-moe-grouped-gemm.html) has the worst case.
