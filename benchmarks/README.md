# Benchmarks and experiments

Every script is run from the repository root, with `QUANTFORGE_BUILD_DIR=<build dir with quantforge-opt>` where it compiles kernels (the docstring at the top of each
file gives the exact command and its options). The paths are referenced from the README, the CHANGELOG, the blog and CI, which is why the folder is flat.
Numbers measured with them are in the README ("Performance snapshot", "Headline results") and the CHANGELOG, with every negative result.

Method that matters for all kernel timings: cold weights (flush the L2 before each call), interleaved rounds after a warm-up (the laptop GPU's clocks drift 3-30 %),
device time from CUDA events / CUPTI, not wall clock. `l2_residency_check.py` verifies the cold-cache method; `kernel_repeat.py` and `decode_repeat.py` give the run-to-run spread.

## Kernels (one GEMM at a time)

| script | what it answers |
|---|---|
| `compare_marlin.py` | QuantForge vs Marlin vs cuBLAS fp16 per shape (`--autotune`); results in `marlin_rtx4050_sm89.json` |
| `compare_baselines.py` | the same against gemlite (Triton), bitsandbytes NF4; `compare_baselines_rtx4050_sm89.json` |
| `baseline_*_rtx4050_sm89.json` | regression baselines of the kernel timings (`python/quantforge/benchmark.py`) |
| `sweep_skinny.py`, `sweep_skinny_blocks.py`, `sweep_skinny64.py` | tile sweeps of the 32- and 64-row schedules (several are negative results, see the CHANGELOG) |
| `kernel_repeat.py`, `l2_residency_check.py` | spread of the ratio against Marlin; is the traffic really DRAM |
| `mma_peak.cu` | peak `mma.sync` throughput of the GPU, registers only (the 44 TFLOP/s the compiler loop is compared with) |
| `int8_gemm_probe.py`, `int8_dequant_probe.py`, `w4a8_gemm.py`, `w4a8_decode_layout.py` | INT8 / W4A8 kernels against cuBLASLt and the W4A16 kernels |
| `moe_op_bench.py`, `moe_sweep_schedule.py` | the grouped mixture-of-experts op, and a sweep of the pipeline options of one grouped GEMM |
| `proto/bir_w4a16.cu`, `proto/bir_abl.cu` | hand-written CUDA prototype with B in registers (Marlin-style) and its ablations: the bound for what the compiler could reach (`python/quantforge/PIPELINE.md`) |

## Quality (perplexity, accuracy)

| script | what it answers |
|---|---|
| `qwen_end_to_end.py`, `qwen_decode_graph.py` | Qwen2.5-0.5B end to end (perplexity, CUDA-graph decode) |
| `hf_quantize_and_run.py`, `hf_quantize_model.py`, `hf_load_checkpoint.py`, `hf_bnb_to_quantforge.py` | quantise / load a Hugging Face model (Llama, Qwen2, OLMo-2, Phi-3; AWQ, GPTQ, bitsandbytes NF4) and run it |
| `hf_gptq.py`, `gptq_layer_error.py`, `ppl_per_window.py`, `arc_eval.py` | GPTQ against round-to-nearest: perplexity, per-layer error, per-window loss, ARC accuracy |
| `hf_fp8_w8a8.py`, `w4a8_gptq.py` | FP8 W8A8 and W4A8 quality and speed |
| `decode_repeat.py` | batch-1 decode speed with its spread |

## Checkpoints

| script | writes |
|---|---|
| `make_gptq_checkpoint.py` | an ordinary GPTQ checkpoint made with this repository's GPTQ (vLLM loads it) |
| `make_w4a8_checkpoint.py` | a W4A8 checkpoint (GPTQ quality) for `quantization="quantforge_w4a8"` |
| `make_olmoe_router_fp16.py` | a loadable copy of the OLMoE GPTQ checkpoint (its router is quantised, vLLM cannot load that) |

## vLLM (needs vLLM; use `quantization="quantforge"` against vLLM's own kernels)

| script | what it answers |
|---|---|
| `vllm_batch_sweep.py` | throughput against the number of concurrent requests (`QF_GPU_UTIL` sets the memory fraction: the KV cache can limit a run) |
| `vllm_e2e.py`, `vllm_prefill.py` | batch-1 decode and prefill throughput |
| `vllm_compare_outputs.py`, `vllm_check_gptq_checkpoint.py` | is the generated text the same as Marlin's / Hugging Face's |
| `vllm_moe_decode.py` | pure decode step time of a MoE checkpoint (prefill cancels by differencing two lengths) |

A lesson from these runs: a throughput comparison is only fair when both sides have the same KV-cache capacity. Extra buffers of one side shrink its cache, vLLM then preempts and recomputes,
and the number measures that instead of the kernels (it cost us a wrong 0.28x for the MoE path once, and a wrong "W4A8 is slow at 192-256 requests" before).
