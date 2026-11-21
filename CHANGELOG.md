# Changelog

## Unreleased

* W4A8: new nibble layout (`quant.pack_k8`: byte j of an 8-block holds elements j and j + 4) and a packed dequantisation pattern in `quantforge-vectorize-dequant` (32-bit word multiply by the scale + SWAR per-byte subtract of the zero point for the int8 prefill tile; `f16x2` magic numbers for the f16 decode tile; FileCheck `vectorize_dequant_w4a8.mlir`). Kernel 1.30-1.48x cuBLAS fp16 (was 1.19-1.40x), 55-63 TOPS; whole-model prefill vs fp16 with compiled norms 1.23x / 1.26x / 1.34x (was 1.16x / 1.22x / 1.28x). vLLM compact mode: prefill 36948 / 13273 / 6678 tokens/s on Qwen2.5-0.5B / 1.5B / 3B (1.12x / 1.15x / 1.15x of Marlin), decode 0.80-0.90x of Marlin. Breaking: W4A8 tensors made with the previous layout are not readable.

* vLLM W4A8: the fp16 weights of the layers to quantise are created on the CPU and moved to the GPU one module at a time by vLLM's `device_loading_context` while they are quantised, so a model larger than the GPU in fp16 loads: Qwen2.5-3B on the 5.6 GB card, compact mode, 2.01 GiB, prefill 6308 tokens/s (Marlin 5782), decode 61 / 935 at 1 / 16 requests (Marlin 81 / 1201). `QUANTFORGE_W4A8_LOAD_ON_CPU=0` turns it off.

* vLLM W4A8: `QUANTFORGE_W4A8_DECODE=compact` keeps one copy of the nibbles (`e2e.W4A8DecodeGemm`: the decode reads the W4A8 layout through a fused producer, 1.1-1.7x slower than the W4A16 decode kernel): weights 1.13 GiB instead of 2.04 GiB (Qwen2.5-1.5B), decode 10-17 % lower at 1 request, 2-10 % at 16, prefill unchanged. `w4a8_hybrid` got a `scale_nk` argument. `benchmarks/w4a8_decode_layout.py`.

* vLLM: `quantization="quantforge_w4a8"` (`QuantForgeW4A8Config` / `QuantForgeW4A8Method`, `quantforge.w4a8_op`, op `quantforge::w4a8_hybrid`): an ordinary fp16 checkpoint is quantised to W4A8 online; above 16 rows the W4A8 kernel, below it the W4A16 decode kernels, from the same weight (weights kept in both layouts). vLLM 0.25.1, Qwen2.5-0.5B: prefill 33.3k tokens/s (fp16 31.8k, Marlin 32.9k, quantforge W4A16 20.6k), decode 257 / 3583 at 1 / 16 requests (Marlin 297 / 4230); 1.5B prefill 12.0k vs 11.5k Marlin. `benchmarks/vllm_prefill.py`, `test/E2E/test_w4a8_op.py`.

* Measured on Qwen2.5-3B (fp16 does not fit the 5.6 GB card): W4A8 prefill 356 ms / 2.07 GB, INT4 W4A16 498 ms / 2.25 GB, INT8 W8A8 310 ms / 3.42 GB; perplexity RTN / GPTQ W4A8 13.47 / 12.54, W4A16 13.13 / 12.06, INT8 12.15. `benchmarks/hf_fp8_w8a8.py` variant `int4_fphead`.

* INT8 W8A8 and W4A8: the RMSNorm in front of the shared q/k/v and gate/up layers quantises its own output (`_rmsnorm_quantize`, installed by `_fuse_int8(fuse_norm=True)`, matched on the identity of the tensor the norm returned): 5-10 % faster whole-model prefill. Against fp16 with compiled RMSNorms: INT8 1.28x / 1.34x / 1.43x, W4A8 1.16x / 1.22x / 1.28x (Qwen2.5-0.5B / Llama-3.2-1B / Qwen2.5-1.5B); against eager fp16 1.37x / 1.46x / 1.49x and 1.24x / 1.33x / 1.34x. Perplexity unchanged.

* INT8 W8A8 and W4A8: `silu(gate) * up` and the per-token INT8 quantisation of the `down_proj` input are one compiled function (`_silu_mul_quantize`, installed on the MLP by `_fuse_int8`; `fuse_activation=False` turns it off): whole-model prefill 8-14 % faster than without it, INT8 1.22x / 1.28x / 1.36x of fp16, W4A8 1.21x / 1.16x / 1.24x (Qwen2.5-0.5B / Llama-3.2-1B / Qwen2.5-1.5B), perplexity unchanged (+0.002..0.012).

* W4A8 kernel: the weight producer reads the packed byte / scale / zero point with `tensor.extract` and unsigned index arithmetic instead of affine `floordiv` maps, so the loads vectorise (`ld.global.u32`): kernel 1.19-1.40x cuBLAS fp16 (was 1.11-1.26x), 50-58 TOPS; whole-model prefill 1.03x / 1.07x / 1.10x of fp16 (was 1.01x / 1.07x / 1.04x).

* GPTQ for W4A8: `quantforge.gptq.gptq_quantize_weight_w4a8`, `quantize_model_gptq(..., fmt="w4a8")`, `quant.w4a8_group_params` (shared by RTN and GPTQ): perplexity 23.7 -> 21.1 / 22.1 -> 19.9 / 14.7 -> 13.7 (Qwen2.5-0.5B / Llama-3.2-1B / Qwen2.5-1.5B), within 0.6 of INT4 W4A16 GPTQ. `benchmarks/w4a8_gptq.py`.

* W4A8: `quantize_model(model, "w4a8")` / `W4A8Linear` / `quantforge.e2e.W4A8GemmDq` / `quantforge.quant.w4a8_quantize`: INT4 weights (progressive quantisation: INT8 per channel, INT4 per group of 128 with an integer scale and zero point) and INT8 activations in ONE kernel,
  the int8 weight built by a `linalg.generic` producer fused into the tile loads (no compiler change). Kernel 1.11-1.26x cuBLAS fp16, 1.37-1.87x QF W4A16; whole-model prefill 1.01x / 1.07x / 1.04x of fp16 (Qwen2.5-0.5B / Llama-3.2-1B / Qwen2.5-1.5B) at INT4 quality, 2-2.6x less weight memory
  than fp16. `benchmarks/w4a8_gemm.py`, `benchmarks/hf_fp8_w8a8.py` variant `w4a8`; `INT8W8A8Linear` gained `concat` / `_build_kernel` / `_launch` hooks (shared-input fusion works for both).

* `epilogue=int8-dequant` (`quantforge-fuse-epilogue`) and `quantforge.e2e.Int8GemmDq`: INT8 GEMM + `* sx[m] * sw[n] + bias[n]` -> f16 in ONE kernel with no int32 intermediate (1.29-1.81x cuBLAS fp16 for GEMM + dequantisation).
  `INT8W8A8Linear` uses it by default (`backend="quantforge"`, `quantize_model(..., int8_backend=)`): whole-model prefill 1.10x / 1.18x / 1.20x of fp16 for Qwen2.5-0.5B / Llama-3.2-1B / Qwen2.5-1.5B (cuBLASLt route: 0.91-1.08x). q/k/v and gate/up share one activation quantisation and one GEMM (`_Int8Shared`, 1-3 %); `int8_gemm_swizzle`: block rastering only for weights larger than the L2.

* Experimental compiler-generated INT8 GEMM (`quantforge.e2e.Int8Gemm`, `benchmarks/int8_gemm_probe.py`): `mma.m16n8k32.s8` through the existing pipeline, bit-exact for 129 schedules, 59-65 TOPS on large shapes, 1.1-1.7x cuBLAS fp16.
  Compiler: `arith.extsi` operands are folded into `vector.contract` (vectorization pass), new `thread-tile-k` pipeline option, `quantforge-overwrite-output` handles i32 accumulators; `quantforge-smem-promotion` recognises a transposed-B tile `[N-tile x K]` (48 of 180 block/warp combinations of the int8 GEMM gave wrong results before).

* `INT8W8A8Linear` / `quantize_model(fmt="int8_w8a8")` (cuBLASLt `_int_mm`) and `benchmarks/mma_peak.cu`: INT8 `mma` is 2.00x fp16 in hardware (88.2 vs 44.1 TOPS), 80-83 TOPS through cuBLASLt with a column-major B, but 0.90-1.07x of fp16
  for whole-model prefill; perplexity +0.33..0.41. Corrected the FP8 vs fp16 GEMM ratio to 1.9-2.1x (the fp16 baseline had been measured before the GPU reached its sustained clocks).

* Experimental `split-k=S` pipeline option (`quantforge-tiling{split-k=S}`, `quantforge-gpu-pipeline`): the K reduction cut over a third grid axis with partial results and a reduce kernel. Correct (FileCheck + numerical tests)
  but off by default: faster only for long-K, narrow-N layers at 32 rows (14-23 %), slower for most shapes at 64 rows, behind K-folding at 16 rows. Results and diagnosis (`ncu`: Marlin uses 20 persistent blocks) in the blog and README.

* 32 / 64-row schedules: 32 / 64-column blocks for N <= 2048 (11-52 % faster at kernel level); `benchmarks/sweep_skinny_blocks.py`. K-fold: the epilogue (sum of the diagonal blocks + bias + f16) is one CUDA kernel
  in the C++ runtime (`kfold_reduce.cu`, op `w4a16_gemm_kfold`), 3 kernels per folded call instead of 5-7. vLLM, QuantForge / Marlin: Qwen2.5-0.5B 0.87-0.96, 1.5B 0.74-0.87 at 1-64 requests.
* A failed build of the C++ runtime now says why in its warning (it used to be cut to the compiler command).

* `docker/Dockerfile`: image with the compiler, torch and the package (optionally vLLM); `quantforge-doctor` and the torch op verified inside it with `--gpus all`.
* vLLM plugin works on vLLM 0.24.0, 0.25.1 and 0.34.0 (fixes for 0.34: `use_batch_invariant`, the new `is_layer_skipped` keyword); the slow generation test now fails, instead of skipping, when only the plugin fails.
* vLLM plugin: needs vLLM >= 0.24 (the `auto_gptq` / `auto_awq` classes it extends are in 0.24-0.34.0 and absent before); with an older or missing vLLM it now warns once
  instead of registering nothing silently.

## 0.3.0 (2026-11-21)

* The op processes inputs of more than 2048 rows in chunks (bounded f32 temporary: Qwen2.5-3B could not start under vLLM on a 6 GB card because of the 8192-token profiling pass). `QUANTFORGE_KFOLD_BUDGET_MB` caps the extra memory of K-folding.
  Checked in vLLM: TinyLlama-1.1B act-order GPTQ (0.76-0.91 of Marlin), Qwen2.5-3B (0.87-0.88).

* vLLM plugin reads GPTQ 8-bit (`quant.from_gptq8`, act-order included) and AWQ 4-bit checkpoints; Qwen2.5-0.5B throughput vs the matching Marlin method: AWQ 0.81-0.89, GPTQ-Int8 0.85-0.93.

* Width-aware default schedules: `_decode_opts` (32 / 64-column blocks for N <= 2048) and `_skinny_opts` (17..64 rows, shape dependent); `torch_ops.default_opts`
  is shared with `benchmarks/compare_marlin.py` (its non-autotuned mode used an empty schedule for M > 16, not the op's). vLLM, QuantForge / Marlin: 0.80-0.88 at 1-16 requests, 0.73-0.80 at 64.

* `quantforge.kfold`: K-folding for long-K, narrow-N decode layers (`QuantLinear.enable_kfold()`, `nn.enable_kfold(model)`, `kfold=` in `quantize_model` / `load_quantized_checkpoint`,
  on by default in the vLLM plugin): +9 % (Qwen2.5-0.5B) / +15 % (1.5B) vLLM throughput at 1-4 requests, ~0.85x of Marlin there.

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
