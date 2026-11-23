---
title: Mixture-of-experts as one kernel launch, a prefill gap nobody had measured, and a 0.66x that was an artifact
date: 2026-11-23
order: 10
summary: Grouped W4A16 GEMMs through two compiler passes (the expert of each row block read from a device array, no host synchronisation), the first end-to-end MoE numbers, a prefill that ran at 0.58 of Marlin MoE until two fixes, and a diagnosis that chased a gap that was really the KV cache.
---

A mixture-of-experts layer is many small GEMMs with the same shape and different weights. Launching one kernel per expert wastes the GPU and synchronises the host, and a CUDA graph needs static shapes; the way out is to sort the rows by expert, pad them to the row tile and run **one** kernel whose `grid.z` is a list of row-block problems.

## The compiler side

Two passes (`ExpertBatchPass.cpp`, pipeline options `expert-problems` and `expert-roles`) turn the ordinary dense kernel into a grouped one: `grid.z` becomes the problem index; the kernel gets one extra argument, an `int32` array `meta` holding the expert of every problem; a negative entry is padding and the block returns at once; every weight, scale and zero-point argument advances by `meta[z]` slabs and the activation and output arguments by `z` row slabs. All shapes are static and nothing waits for the host. Around the two grouped GEMMs (gate/up with the SwiGLU epilogue, then down) three small CUDA kernels route, gather and combine; the vLLM plugin (`QuantForgeMoEMethod`) takes vLLM's own weight loader and router and replaces only the expert computation.

## What was measured, and a number that was wrong

OLMoE-1B-7B (64 experts, top-8), the decode step against vLLM's Marlin MoE:

{{chart:moe_ratio}}

| concurrent requests | 1 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|
| Marlin MoE, tokens/s | 197 | 814 | 1556 | 2536 |
| QuantForge, tokens/s | 189 | 771 | 1475 | 2354 |
| ratio | 0.96 | 0.95 | 0.95 | 0.93 |

The first version of this table said **0.66** at 32 requests, and I spent days on it: `nsys` traces, a roofline, replaying real routing, schedule sweeps that all plateaued. It was an artifact. The first op wrote the gate/up result as an f32 buffer; that buffer came out of the KV cache, vLLM ran out of room and preempted, and the throughput of the *run* dropped by a third while every kernel was as fast as before. Once the op no longer needed the buffer the same measurement gave 0.93. The lesson is in the [W4A8 post](08-w4a8-beats-marlin.html) too: before diagnosing a gap in vLLM throughput, compare the reported KV cache size and the preemptions of both runs.

## Prefill: 0.58 to 0.88

Everything above is decode. Nobody had measured prefill (16 prompts of 1275 tokens, 1024-token steps): vLLM's Marlin MoE did 12117 / 11809 tokens/s and this op 6873 / 6944, **0.58**. Two causes, both fixed:

* The op processed large batches in chunks of 256 tokens to bound the f32 gate/up buffer, and every chunk reads all 64 experts' weights again (about 200 MB a layer). The SwiGLU epilogue now stores `silu(gate) * up` straight into its own f16 output (`quantforge.swiglu_out`, a second function argument the fusion pass fills): no f32 buffer, no conversion kernel, so the chunk grew to 512 and the weights are read half as often.
* The grouped kernels reused the dense row-count schedules, which put a 64-row block inside a 128-row problem (the weights were read twice) with the wrong warp tiles. A sweep over block, warp and K tiles with the read-flush found one family (K tile 64, one stage, look-ahead 1, 128-column blocks) 17 to 27 % faster than the dense schedule; the row tile is 16, 32 or 64 by the average number of rows per expert (128 never won).

Prefill is now 10616 / 10518 tokens/s, **0.88** of Marlin MoE, and the generated text is identical to Marlin MoE's on four prompts (first-token log-probabilities within 0.001).

## A path I measured and did not build

An INT8 grouped kernel (the W4A8 kernel through the same pass) is 5 to 6 % faster than the grouped W4A16 one at 256 tokens (both near the 192 GB/s of the memory) and 14 to 15 % faster at 512. But it needs a quantisation kernel before each grouped GEMM and an f16 gate/up buffer plus a SwiGLU-and-requantise kernel between them, about 0.15 to 0.2 ms of traffic per layer, which eats most of the 0.34 ms gain: prefill would go from 0.88 to about 0.92, not past 1, for a new online-quantisation loader in vLLM. Not built.

Limits: one model run end to end (OLMoE), 4-bit silu-gated experts, no expert parallelism, shared experts untested; Mixtral, Qwen-MoE and Qwen3-MoE shapes compile into the kernel pack but were not run.
