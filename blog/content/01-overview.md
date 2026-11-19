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

The result is a single kernel per GEMM. The dequantised f16 weights are never materialised in memory; that is what makes the INT4 kernels 2.5 to 4.8 times faster than a cuBLAS fp16 GEMM at decode.

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
| 1 x 4096 x 4096 | 49.4 | 38.0 | 180.8 | 1.30x |
| 4 x 4096 x 4096 | 49.4 | 38.0 | 181.7 | 1.30x |
| 16 x 4096 x 4096 | 50.0 | 50.0 | 183.0 | 1.00x |
| 1 x 4096 x 11008 | 126.7 | 139.2 | 482.3 | 0.91x |
| 1 x 11008 x 4096 | 127.0 | 135.1 | 483.3 | 0.94x |
| 16 x 4096 x 11008 | 127.4 | 148.9 | 498.6 | 0.86x |
| 32 x 4096 x 4096 | 50.8 | 57.8 | 181.0 | 0.88x |
| 32 x 4096 x 11008 | 129.8 | 148.8 | 497.1 | 0.87x |
| 64 x 4096 x 4096 | 58.6 | 86.9 | 181.4 | 0.67x |
| 64 x 4096 x 11008 | 139.4 | 197.9 | 507.0 | 0.70x |
| 128 x 4096 x 4096 | 104.1 | 150.8 | 188.7 | 0.69x |
| 256 x 4096 x 4096 | 198.7 | 261.1 | 207.2 | 0.76x |
| 512 x 4096 x 4096 | 387.9 | 456.7 | 371.1 | 0.85x |
| 2048 x 4096 x 4096 | 1514.6 | 1764.6 | 1427.1 | 0.86x |

{{chart:kernel_us}}

What the table says:

* **Decode (M up to 16)**: at 4096 x 4096 the generated kernel is at or above Marlin (the 1.30x implies more than the measured DRAM read ceiling, so part of it is probably L2 residency that the flush did not remove; the wide shapes are the cleaner comparison, see [the decode post](02-decode-instruction-bound.html)); for the wide 4096 x 11008 shapes it reaches 86 to 94 percent. The weights stream from DRAM at 150 to 175 GB/s on a card whose measured read ceiling is about 183 GB/s.
* **Prefill (M of 512 and more)**: 85 to 86 percent of Marlin, about 20 to 25 percent slower than cuBLAS fp16, which reads four times the bytes but has no dequantisation to do: the kernel is compute-bound there and the dequantisation competes with the Tensor Cores for issue slots.
* **M between 32 and 256** is the weak spot, 12 to 33 percent behind. [A separate post](04-mid-batch-postmortem.html) investigates it in detail.

## Results: whole models

The kernels are only useful if a real model runs correctly and faster. `quantforge.nn.quantize_model` replaces the `nn.Linear` layers of any Hugging Face causal LM, fuses the q/k/v projections into one GEMM and the gate/up projections of a SiLU-gated MLP into one GEMM with a fused `silu(gate) * up` epilogue. The table is batch-1 decode with a CUDA graph, perplexity on 8 x 512 tokens of the Wikitext-2 test split, simple round-to-nearest INT4 (group 128) with an INT8 head.

{{chart:e2e_speed}}

| model | fp16 ppl | fp16 tok/s | INT4 ppl | INT4 tok/s | weights fp16 -> INT4 |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 19.93 | 129 | 24.08 | 209 | 0.72 -> 0.19 GB |
| Qwen2.5-1.5B-Instruct | 13.48 | 49 | 15.31 | 97 | 2.62 -> 0.69 GB |
| Qwen2.5-3B-Instruct | does not fit in 6 GB | | 14.29 | 61 | 5.55 -> 1.45 GB |
| Llama-3.2-1B-Instruct | 20.15 | 63 | 24.42 | 142 | 1.95 -> 0.51 GB |
| OLMo-2-0425-1B-Instruct | 19.32 | 60 | 21.31 | 138 | 2.15 -> 0.56 GB |
| Phi-3.5-mini-instruct (3.8B) | does not fit | | 9.62 | not measured | 7.25 -> 1.90 GB |
| Mistral-7B-Instruct-v0.3 (from an NF4 checkpoint) | | | 8.51 (bitsandbytes NF4: 8.37) | 35.3 (bitsandbytes: 36.0) | 4.65 GB peak |

A 3B and a 7B model run on a 6 GB card that cannot even hold them in fp16. The perplexity cost of round-to-nearest 4-bit is real (1 to 4 points); [the GPTQ post](05-gptq-and-checkpoints.html) shows how much of it a smarter quantiser recovers. The speed-up shrinks as the model grows because the lm_head, attention and the roughly 1400 small kernels of the Hugging Face graph take a larger share of each step.

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
* The only baselines are Marlin and cuBLAS. Machete, Triton kernels such as gemlite, torchao and end-to-end vLLM are not compared.
* Kernels are PTX built with LLVM 18's NVPTX backend: no Hopper `wgmma` or TMA, no FP8 *activations*.
* M between 32 and 256 and prefill stay behind Marlin; the next posts explain what was tried.
