# Changelog

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
