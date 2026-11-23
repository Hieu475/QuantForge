---
title: A Hadamard rotation makes W4A8 more accurate than W4A16, and one layer of Llama-3.2-3B breaks it
date: 2026-11-23
order: 9
summary: Rotating the input of every linear layer (and its weight) by a block Hadamard matrix removes almost all of the INT8-activation loss and pays for itself in accuracy. What it did on seven models, the layer where it fails, the per-layer decision that fixes it, and the ideas that did not work.
---

INT8 per-token activation quantisation is the price of the faster [W4A8 path](08-w4a8-beats-marlin.html), and it is paid by outlier channels: one large channel sets the token's scale and the rest of the row loses its resolution. The cure known from QuaRot and SpinQuant is to multiply the input by an orthonormal matrix `B` and the weight by the same, `(xB)(WB)^T = xW^T`, so that the layer is unchanged in exact arithmetic but both operands are flatter when they are quantised.

## The version that is cheap enough

A full rotation of a 4096-wide layer is a 4096 x 4096 product. A **block-diagonal Walsh-Hadamard matrix**, one 128 x 128 block per group of 128 inputs, is a butterfly: 7 stages of adds and subtracts per block, no multiplications. `WB` is computed once when the weights are quantised (at load, or inside GPTQ with the Hessian rotated to `B^T H B`). On the activation side one warp transforms a block with shuffles, and above 64 rows the transform is *inside the INT8 quantisation kernel*: the rotated row waits in shared memory for its scale, so the cost is no extra pass over the activations (the fused kernel is bit-identical to rotating and then quantising, which is tested). Measured cost: 2 to 4 % of the decode step at 1 to 64 requests, 1 to 2 % above.

## What it buys

Wikitext-2 perplexity through vLLM (the kernels that really run; online round-to-nearest weights, INT8 activations):

{{chart:rot_ppl}}

| model | fp16 | W4A8, no rotation | W4A8, rotated (default) |
|---|--:|--:|--:|
| Qwen2.5-1.5B | 11.932 | 13.949 | **13.376** |
| Qwen2.5-0.5B | 18.10 | 23.148 | 21.395 |
| OLMo-2-0425-1B | 17.188 | 20.843 | 18.987 |
| Llama-3.2-1B | 18.353 | 22.930 | **21.539** |
| Phi-3.5-mini | does not fit | 9.472 | **8.718** |
| Qwen2.5-3B | does not fit | 12.767 | **11.830** |
| Llama-3.2-3B | does not fit | 16.640 | 16.658 |

With GPTQ weights (Qwen2.5-1.5B, calibrated on wikitext-2 train, so the numbers are flattering in absolute terms): W4A16 12.605, W4A8 12.864, **W4A8 rotated 12.331**. The loss of INT8 activations, +0.42 and +0.63 perplexity on the two Qwen sizes in a fake-quantised simulation, falls to +0.01 and +0.04 on top of the rotated 4-bit weights; and a rotated W4A8 checkpoint is more accurate than the unrotated W4A16 one. Halving the group of the integer scales to 64 (`quantforge_w4a8_g64`) gains another 0.21 online and 0.08 with GPTQ for 2 to 3 % speed.

## The layer that breaks it

Rotating *everything* made Llama-3.2-3B worse: 16.640 became **17.752**. Rotating only layers with a K of 8192 gave 17.71, rotating only the others 16.74, so it was the `down_proj` layers; a per-layer error table on real activations (CPU forward, the real kernels on the GPU) pointed at one:

| layer 1 `down_proj` of Llama-3.2-3B | relative output error |
|---|--:|
| plain W4A8 | 0.054 |
| rotated | 0.080 |

Its input has a token with **530 times the median norm**, the attention sink. A rotation spreads that single huge channel over the whole block of 128, and the one token whose value the model depends on is now quantised with a row-wide error. Nothing in the weights flags such a layer (column norms are ordinary), so there is no way to know without activations. Two fixes:

* **GPTQ has calibration data**, so it decides per layer: for every group of linears that share an input it compares the output error of the rotated and the plain quantisation on a few tokens of every window, including the one of largest norm, and lists the layers that stay plain in `config.json` (`unrotated_layers`; for Llama-3.2-3B, `layers.1.mlp.down_proj` and `layers.23.self_attn.o_proj`). GPTQ W4A8 of the 3B: 16.537 plain, 16.377 decided per layer.
* **Online quantisation has none**, so it follows a rule: the `down_proj` of the first four decoder layers stays unrotated. The table above is the evidence for it, seven models, and it is a rule, not a proof: on a model family I have not run, `quantforge_w4a8_plain` is the baseline to compare with.

## Things that did not work

* **Choosing a rotation per layer** among cheap candidates (identity, random sign flips, a channel permutation) by the output error on calibration tokens: 13.612 against 13.606 for the fixed Hadamard. What is left of the gap to fp16 is the 4-bit weights, not the basis.
* **FP8 (E4M3) activations instead of INT8** on the Ada tensor cores: not built. With the rotation the INT8 activations cost only +0.01 to +0.04 perplexity, the FP8 and INT8 rates are the same on this chip, and LLVM 18's NVVM dialect has no FP8 mma.
* **3-bit weights** with the rotation and a clip search: GPTQ 17.2 plain, 14.6 rotated, 13.6 rotated with clipping, against 13.1 / 12.5 / 12.4 for 4 bits on the same calibration. The 3-bit model matches the public Marlin 4-bit GPTQ checkpoint, but against 4 bits with the same calibration it costs 0.5 to 1.2 perplexity for 25 % fewer bytes, and the kernels have no 3-bit unpack (two bit planes read by `tensor.extract` would be slower than the 4-bit decode kernel). 2 bits is unusable.
* **A clip search on the channel scale** of the real format: -0.08, +0.08 and -0.26 on three models, so it stays off.
* **fp8 KV cache** (vLLM's own): it doubles the KV capacity (90k to 171k tokens on the 6 GB card) and breaks Qwen2.5 with the default scale (12.33 to 24.96; Marlin 13.64 to 28.73), which is vLLM's attention domain.

## Two bugs worth the paragraph

vLLM's compile cache is keyed by the quantisation *name*, not by environment variables. My first rotation switch was an environment variable; the second run reused the first run's compiled graph, and the perplexity came out at 10^7. The fix is that a setting that changes the traced graph is a different quantisation name (`quantforge_w4a8_plain`, `quantforge_w4a8_g64`). And a custom op's fake implementation must return the strides of the real output: `empty_like` of a non-contiguous input does not, and the result is silently wrong under `torch.compile`.
