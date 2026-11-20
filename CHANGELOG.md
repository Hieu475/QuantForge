# Changelog

## Unreleased

* vLLM plugin (`quantization="quantforge"`, entry point `vllm.general_plugins`): GPTQ 4-bit checkpoints on the QuantForge kernels; 70-80 % of Marlin's
  throughput on Qwen2.5-0.5B / 1.5B. `FP8W8A8Linear` / `quantize_model(fmt="fp8_w8a8")` (cuBLASLt FP8 tensor cores; quality good, no whole-model speed-up).

* GPTQ: calibration token norms are capped at 10x the median in the Hessian (`token_clip=10`), which removes a failure on models
  with a massive-activation layer (Llama-3.2-1B: GPTQ was worse than RTN on the whole Wikitext-2 test split, 24.08 against 22.28;
  now 20.21). A slow regression test (`test/E2E/test_gptq_real_model.py`, `QUANTFORGE_SLOW_TESTS=1`) guards the broken windows.
* Decode: K tile 32 for 10240 <= N <= 12288 (1 x 4096 x 11008: 149.5 -> 135.7 us, 0.93x Marlin); autotune candidates version 3.
* Benchmarks: `arc_eval.py`, `decode_repeat.py`, `kernel_repeat.py`, `ppl_per_window.py`; the comparison JSON is now
  `compare_baselines_rtx4050_sm89.json`.

## 0.2.0 (2026-11-20)

* `quantforge.nn`: `QuantLinear`, `quantize_model` (INT4 / INT8 / FP8, RTN), fused q/k/v and SiLU-gated MLP
  (also Phi-3's stacked `gate_up_proj`), `load_quantized_checkpoint` for AWQ / GPTQ files (incl. act-order).
* `quantforge.gptq`: sequential GPTQ producing the kernel layouts.
* FP8 (E4M3) weights (`qf.dequant {fp8}`, packed f16x2 dequantisation), W8A16 in the torch op.
* Prefill 3-stage schedule for large weights, 32- / 64-row skinny schedules, autotune cache versioning.
* `quantforge-doctor`, `pyproject.toml` (installable), benchmarks: `compare_baselines.py` (gemlite, bitsandbytes),
  `l2_residency_check.py`, `hf_gptq.py`, `hf_load_checkpoint.py`, `vllm_e2e.py`.
* **Correction:** the decode speed against Marlin at 4096 x 4096 is 0.99-1.01x, not the 1.30x reported before (the
  figure did not reproduce); the Marlin table was regenerated in one run.

## 0.1.0

* First end-to-end W4A16 pipeline (qf IR -> PTX), torch op, AWQ / GPTQ repacking, autotuner.
