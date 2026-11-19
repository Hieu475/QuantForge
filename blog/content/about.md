---
title: About
---
This blog documents **QuantForge**, an open-source MLIR-based kernel compiler for weight-only quantised LLM inference (W4A16, W8A16 and FP8-weight GEMMs on NVIDIA sm_80+ GPUs), and the experiments around it.

## What is in the project

* A `qf` MLIR dialect (`qf.unpack`, `qf.dequant`) and about twenty passes that lower a quantised matmul to PTX: tiling, shared-memory promotion with `cp.async` pipelines, fusion of the dequantisation into the weight loads, vectorised loads, packed `f16x2` dequantisation, register-resident accumulators, `mma.sync` Tensor Core lowering, weight prefetch, fused epilogues.
* A PyTorch custom op (`quantforge::w4a16_gemm`) with a C++ dispatcher hot path, works under `torch.compile` and CUDA graphs.
* `quantforge.nn` / `quantforge.gptq`: quantise any Hugging Face causal LM (RTN or GPTQ) or load real AWQ / GPTQ checkpoints.
* 37 FileCheck tests, about 70 GPU end-to-end tests, and a `compute-sanitizer` (memcheck + racecheck) script that runs 54 pipeline variants.

## How to read the numbers

All measurements come from **one laptop GPU** (RTX 4050, sm_89, 6 GB) that is also driving a desktop. Clock speeds drift by up to 30 percent between sessions, so ratios measured in the same run (for example against Marlin) are far more trustworthy than absolute times. There are no results yet on data-centre GPUs (A100, H100); that is the biggest open limitation.

## Status

Research prototype. It is useful for experiments and for understanding what a compiler can and cannot do for this class of kernel; it is not a replacement for production inference engines.
