---
title: QuantForge: generating INT4 / INT8 / FP8 GEMM kernels with MLIR
date: 2026-11-20
order: 1
summary: What the project is, how the compiler pipeline is organised, and the full set of results: kernel time against Marlin, end-to-end decode speed on real models, and correctness evidence.
---

## Why weight-only quantisation, and why a compiler

During autoregressive decoding an LLM reads **every weight once per generated token** and does very little arithmetic with it: at batch size 1 each weight takes part in a single multiply-add. The step is limited by memory bandwidth, not by compute. Storing weights as 4-bit integers instead of 16-bit floats therefore makes a decode step up to four times cheaper in bytes, as long as the dequantisation is cheap enough to hide behind the memory traffic. This is the idea behind GPTQ, AWQ and kernels such as Marlin.

Hand-written kernels like Marlin reach the memory bandwidth limit, but they are written for specific shapes and bit-widths. The question behind QuantForge is: **how close can a compiler get, when it generates the kernel from a high-level description and a set of composable passes, without per-shape hand tuning?** The pipeline starts from a tensor-level description of

```
C = A @ dequant(unpack(W_packed))        # A: fp16 activations, dequant(q) = (q - zero) * scale per group of 128 rows
```

and ends at PTX that is launched from PyTorch.

## The pipeline

The input is a function in the `qf` dialect:

```mlir
func.func @w4a16_gemm(%act: tensor<MxKxf16>, %wp: tensor<Kx(N/2)xi8>,
                      %scale: tensor<(K/G)xNxf16>, %zp: tensor<(K/G)xNxi8>,
                      %c: tensor<MxNxf32>) -> tensor<MxNxf32> {
  %w  = qf.unpack %wp : tensor<Kx(N/2)xi8> -> tensor<KxNxi8>      // two INT4 per byte
  %wf = qf.dequant %w, %scale, %zp : ... -> tensor<KxNxf16>        // (q - zero) * scale, group-wise
  %r  = linalg.matmul ins(%act, %wf) outs(%c)
  return %r
}
```

`quantforge-opt --quantforge-gpu-pipeline` lowers it in four stages. The names below are the actual pass names:

| stage | what happens | passes |
|---|---|---|
| tensor level | the dequantisation becomes element-wise `linalg.generic`, unpack and dequant are fused | `fuse-unpack-dequant`, `lower-dequant-to-arith` |
| tiling and mapping | block / warp tiling, mapping to the GPU grid, optional L2-friendly tile swizzle | `tiling`, `forall-to-gpu` (`block-swizzle`), `thread-distribute` |
| memory hierarchy | A tile through shared memory with a multi-stage `cp.async` pipeline; **the dequantisation is recomputed while the weight tile is staged**, so only the packed INT4 bytes ever travel from DRAM | `smem-promotion`, `fuse-producer-into-loads` |
| instruction level | vectorised 16-byte loads, packed `f16x2` dequantisation, accumulators in registers, `mma.sync.m16n8k16`, weight-load look-ahead, fused epilogues | `simplify-index-math`, `vectorize-global-loads`, `vectorize-dequant`, `hoist-vector-transfers`, `convert-vector-to-gpu`, `prefetch-weight-loads`, `fuse-epilogue`, `overwrite-output` |

The result is a single kernel per GEMM. The dequantised f16 weights are never materialised in memory; that is what makes the INT4 kernels 3.2 to 3.7 times faster than a cuBLAS fp16 GEMM at decode.

### Runtime

The generated PTX is wrapped in a shared library and loaded through a small CUDA driver shim. On top of it:

* a PyTorch custom op `torch.ops.quantforge.w4a16_gemm` registered in the dispatcher from C++ (about 45 microseconds of host time per call from Python, instead of 210 to 270 with the first, pure-Python launcher), working in eager mode, under `torch.compile` and inside CUDA graphs;
* a persistent on-disk cache of compiled kernels and **row buckets** (M up to 16, 32, 64, 128, 256, ... are served by one kernel each) so that a server seeing prompt lengths from 1 to 4096 compiles about a dozen kernels per layer shape instead of one per length;
* an autotuner that measures tile options once per shape and caches the winner.

Four kernel schedules are used, chosen from the number of activation rows M:

| M | schedule |
|---|---|
| 1 to 16 | *decode*: one 16-row Tensor Core tile per block, N spread over blocks, weights read exactly once |
| 17 to 32 / 33 to 64 | *skinny*: 32- or 64-row block tiles |
| 65 to 256 | 64- or 128-row tiles |
| above 256 | *prefill*: 128 x 128 tiles, a 3-stage pipeline for large weights |

## Results: kernels against Marlin

All kernel numbers are **device time from the CUPTI trace**, best of many runs, with the weights read from DRAM: the L2 cache is flushed before every call by *reading* a 256 MB buffer (writing it leaves dirty lines that slow the next kernel by 20 to 50 percent), and the activations are touched again afterwards because in a real model the previous layer has just written them. Marlin is vLLM 0.25.1's `marlin_gemm` on the same INT4 values and scales. Group size 128.

{{chart:marlin_ratio}}

| M x K x N | Marlin (us) | QuantForge (us) | cuBLAS fp16 (us) | QuantForge / Marlin |
|---|--:|--:|--:|--:|
| 1 x 4096 x 4096 | 49.6 | 50.0 | 180.9 | 0.99x |
| 4 x 4096 x 4096 | 49.4 | 49.8 | 181.5 | 0.99x |
| 16 x 4096 x 4096 | 50.0 | 49.8 | 182.3 | 1.01x |
| 1 x 4096 x 11008 | 126.8 | 135.7 | 482.4 | 0.93x |
| 1 x 11008 x 4096 | 127.2 | 131.1 | 484.0 | 0.97x |
| 16 x 4096 x 11008 | 127.7 | 135.5 | 499.9 | 0.94x |
| 32 x 4096 x 4096 | 50.9 | 58.8 | 181.0 | 0.87x |
| 32 x 4096 x 11008 | 130.3 | 149.3 | 497.3 | 0.87x |
| 64 x 4096 x 4096 | 58.3 | 87.4 | 181.4 | 0.67x |
| 64 x 4096 x 11008 | 139.9 | 200.0 | 506.9 | 0.70x |
| 128 x 4096 x 4096 | 105.1 | 152.2 | 188.7 | 0.69x |
| 256 x 4096 x 4096 | 201.0 | 263.0 | 208.2 | 0.76x |
| 512 x 4096 x 4096 | 391.5 | 461.6 | 373.4 | 0.85x |
| 2048 x 4096 x 4096 | 1532.1 | 1775.8 | 1453.0 | 0.86x |

{{chart:kernel_us}}

What the table says:

* **Decode (M up to 16)**: at 4096 x 4096 the generated kernel is **on par with Marlin** (0.99 to 1.01x, about 50 microseconds, which is the DRAM limit: `ncu` measures 9.2 MB read for 8.65 MB of weights and scales, about 184 GB/s); on the wide 4096 x 11008 shapes it reaches 93 to 94 percent (it was 85 percent until a wave-aware K tile was added, see [the decode post](02-decode-instruction-bound.html)), on 1 x 11008 x 4096 97 percent. (An earlier version of this post reported 1.30x at 4096 x 4096; that figure did not survive re-measurement, see [the decode post](02-decode-instruction-bound.html).)
* **Prefill (M of 512 and more)**: 85 to 86 percent of Marlin, about 20 to 25 percent slower than cuBLAS fp16, which reads four times the bytes but has no dequantisation to do: the kernel is compute-bound there and the dequantisation competes with the Tensor Cores for issue slots.
* **M between 32 and 256** is the weak spot, 13 to 33 percent behind. [A separate post](04-mid-batch-postmortem.html) investigates it in detail.

### Against other kernels

Is Marlin the only fair yardstick? The same method (L2 flushed, device time, interleaved rounds) was applied to **gemlite**, a Triton W4A16 kernel with its own autotuner (version 0.6.0, the same INT4 weights and scales), to **bitsandbytes NF4** (its own 4-bit code book on the same fp16 matrix, so a different quantiser: speed only) and to cuBLAS fp16. Kernel time in microseconds, the fastest INT4 kernel in bold:

| M x K x N | QuantForge | Marlin | gemlite (Triton) | bitsandbytes NF4 | cuBLAS fp16 |
|---|--:|--:|--:|--:|--:|
| 1 x 4096 x 4096 | 50.0 | 49.3 | **49.3** | 55.9 | 180.8 |
| 4 x 4096 x 4096 | 49.8 | **49.3** | 100.0 | 155.1 | 181.7 |
| 16 x 4096 x 4096 | **49.8** | 50.0 | 61.4 | 156.8 | 182.6 |
| 32 x 4096 x 4096 | 58.8 | **50.9** | 71.0 | 160.0 | 180.9 |
| 64 x 4096 x 4096 | 87.4 | **57.8** | 89.8 | 194.3 | 181.5 |
| 128 x 4096 x 4096 | 152.2 | **104.4** | 159.2 | 241.0 | 188.3 |
| 256 x 4096 x 4096 | 263.0 | **199.6** | 300.3 | 457.6 | 207.6 |
| 512 x 4096 x 4096 | 461.6 | **388.9** | 529.9 | 600.9 | 371.1 |
| 2048 x 4096 x 4096 | 1775.8 | **1527.5** | 2002.7 | 1711.1 | 1457.4 |
| 1 x 4096 x 11008 | 135.7 | **126.8** | 129.3 | 139.6 | 482.4 |
| 1 x 11008 x 4096 | 131.1 | **127.1** | 129.2 | 146.4 | 484.2 |

Reading it:

* **QuantForge is on par with gemlite at batch 1** (49.9 against 49.3 microseconds) and **clearly ahead from batch 4 up**: gemlite needs 100 microseconds at M = 4 and 61 at M = 16, where QuantForge needs 50. Its prefill kernel is about 15 percent slower than QuantForge's.
* gemlite is still **5 percent faster on the wide decode shape** 1 x 4096 x 11008 (129 against 136 microseconds, after the improvement; it was 150): the weak point of QuantForge's decode kernel is the grid / wave behaviour on wide layers, not the 4096 x 4096 case.
* Marlin is the fastest kernel from M = 32 up, as before; bitsandbytes' NF4 path is slower everywhere except the wide decode shapes, where it is within 8 to 15 percent.
* All three INT4 kernels hit the same ~50 microsecond DRAM limit at 4096 x 4096 batch 1, which is the correct outcome for a memory-bound kernel.

## Results: whole models

The kernels are only useful if a real model runs correctly and faster. `quantforge.nn.quantize_model` replaces the `nn.Linear` layers of any Hugging Face causal LM, fuses the q/k/v projections into one GEMM and the gate/up projections of a SiLU-gated MLP into one GEMM with a fused `silu(gate) * up` epilogue. The table is batch-1 decode with a CUDA graph, perplexity on the whole Wikitext-2 test split (the Phi row: first 8 windows only), simple round-to-nearest INT4 (group 128) with an INT8 head.

{{chart:e2e_speed}}

| model | fp16 ppl | fp16 tok/s | INT4 ppl | INT4 tok/s | weights fp16 -> INT4 |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 18.79 | 120 | 22.91 | 197 | 0.72 -> 0.19 GB |
| Qwen2.5-1.5B-Instruct | 12.66 | 48 | 14.30 | 91 | 2.62 -> 0.69 GB |
| Qwen2.5-3B-Instruct | does not fit in 6 GB | | 13.13 | ~61 | 5.55 -> 1.45 GB |
| Llama-3.2-1B-Instruct | 17.91 | 63 | 22.28 | 132 | 1.95 -> 0.51 GB |
| OLMo-2-0425-1B-Instruct | 17.24 | 57 | 19.02 | 129 | 2.15 -> 0.56 GB |
| Phi-3.5-mini-instruct (3.8B) | does not fit | | 9.62 (8 windows) | not measured | 7.25 -> 1.90 GB |
| Mistral-7B-Instruct-v0.3 (from an NF4 checkpoint) | | | 8.51 (bitsandbytes NF4: 8.37) | 35.3 (bitsandbytes: 36.0) | 4.65 GB peak |

A 3B and a 7B model run on a 6 GB card that cannot even hold them in fp16. The perplexity cost of round-to-nearest 4-bit is real (1 to 4 points); [the GPTQ post](05-gptq-and-checkpoints.html) shows how much of it a smarter quantiser recovers. The speed-up shrinks as the model grows because the lm_head, attention and the roughly 1400 small kernels of the Hugging Face graph take a larger share of each step.

### Run-to-run spread

The single numbers above are the best of their session, so here is how much they move. Batch-1 decode with a CUDA graph, 7 repetitions with fp16 and INT4 (plus an INT8 head) interleaved so that clock drift hits both equally, mean and standard deviation of one session:

| model | fp16 (tokens/s) | INT4 + INT8 head (tokens/s) | speed-up |
|---|--:|--:|--:|
| Qwen2.5-0.5B | 120.2 +- 0.6 | 197.1 +- 1.6 | 1.64x +- 0.02 |
| Qwen2.5-1.5B | 47.9 +- 2.1 | 91.3 +- 5.4 | 1.91x +- 0.07 |
| Llama-3.2-1B | 62.5 +- 3.3 | 132.0 +- 7.3 | 2.11x +- 0.12 |
| OLMo-2-1B | 57.2 +- 2.6 | 128.9 +- 7.7 | 2.26x +- 0.16 |

Absolute speeds move by 5 to 10 percent inside a session and by up to 15 percent between sessions (the GPU also drives a desktop), so the table further up, whose numbers come from the faster sessions, should be read as "about 200" and not "209". The **speed-up** against fp16 is far more stable (a few percent). The kernel times against Marlin were repeated in 5 independent invocations: 1 x 4096 x 4096 gives a ratio of 0.990 +- 0.000 (Marlin 49.7 +- 0.1 microseconds, QuantForge 50.2 +- 0.1), 1 x 4096 x 11008 gives 0.930 +- 0.000, 64 x 4096 x 4096 gives 0.676 +- 0.005 and 2048 x 4096 x 4096 gives 0.842 +- 0.011; only the long prefill kernels drift (about 1.5 percent, with the GPU temperature, both kernels together). The ratios in the kernel table are therefore reproducible to the digits shown.

### End to end against vLLM

vLLM 0.25.1 (offline engine, CUDA graphs, one request, 256 new tokens, best of 3) on Qwen2.5-0.5B-Instruct: fp16 **165 tokens/s**, the official GPTQ-Int4 checkpoint with Marlin kernels **310 tokens/s** (1.88x faster than its own fp16). QuantForge inside a plain Hugging Face model with a static cache and a CUDA graph: fp16 129, INT4 with an INT8 head 209 tokens/s (1.62x), 226 with `torch.compile` (1.75x).

**In absolute terms vLLM is clearly faster**, and the reason is not the GEMM kernels (the table above shows QuantForge on par with Marlin at batch 1) but the rest of the model: vLLM's implementation fuses norms, rotary embeddings, residual adds and sampling, while the Hugging Face graph launches about 1400 small kernels per decode step. For a 0.5B model those dominate. The fair end-to-end comparison would run the QuantForge kernels *inside* vLLM (a quantisation plugin); that is future work.

## Correctness evidence

Speed numbers are worthless if the kernel is subtly wrong, so the project leans heavily on checking:

* every end-to-end run is compared against a **dense twin**: the same model with the INT4 weights dequantised into ordinary fp16 `nn.Linear` layers. Perplexity agrees to within 0.002 and greedy decoding produces identical tokens (48/48 or 59/59) on every model tried;
* real AWQ and GPTQ-Int4 checkpoints of Qwen2.5-0.5B run through the library match their dense twins: GPTQ 26.336 against 26.335 perplexity, AWQ 26.689 against 26.689;
* `compute-sanitizer` (memcheck and racecheck) is clean for 54 pipeline variants, including the prefetch pipelines, fused epilogues and every schedule above;
* 37 FileCheck tests guard the structure of the generated code (for example "no scalar int-to-float conversion per weight", the regression that once cost 40 percent).

## Using it

```python
import torch, quantforge.nn as qnn
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16)
qnn.quantize_model(model, fmt="int4", group_size=128)          # int4 | int8 | fp8
qnn.quantize_model(model, "int8", only=("lm_head",), fuse=False)  # a lossless INT8 head
out = model.generate(ids.cuda(), max_new_tokens=64)            # runs on the generated kernels
```

## Limitations, stated plainly

* All measurements come from **one laptop GPU** with drifting clocks; there is no data-centre GPU result yet.
* Baselines: Marlin, gemlite (Triton), bitsandbytes and cuBLAS at the kernel level, and vLLM end to end on one model. Machete and torchao are not compared, and QuantForge has not been run *inside* vLLM.
* Kernels are PTX built with LLVM 18's NVPTX backend: no Hopper `wgmma` or TMA, no FP8 *activations*.
* M between 32 and 256 and prefill stay behind Marlin; the next posts explain what was tried.
