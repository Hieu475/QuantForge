---
title: Better 4-bit quality with GPTQ, loading real AWQ / GPTQ checkpoints, and a few traps
date: 2026-11-20
order: 5
summary: Round-to-nearest INT4 costs 1 to 4 perplexity points. A GPTQ implementation in the library recovers 17 to 65 percent of that, real AWQ and GPTQ checkpoints are loaded and verified, and the things that did not help are listed.
---

Fast kernels are only half of a quantisation story: the other half is how much quality the 4-bit weights lose. The library therefore has two ways to obtain weights, a one-line round-to-nearest quantiser and a calibrated one, and it can load checkpoints produced by other tools.

## Round-to-nearest: the baseline

`quantforge.nn.quantize_model(model, "int4", group_size=128)` quantises every `nn.Linear` independently: each group of 128 input rows of a column maps its minimum and maximum to the 16 levels. It takes 3 seconds for a 0.5B model, needs no data, and loses a visible amount of quality (1 to 4 perplexity points on the models below). The head (`lm_head`) is kept in fp16 by default; quantising it to INT8 is lossless (perplexity 24.08 against 24.09 for Qwen2.5-0.5B) and 16 percent faster, so the numbers in this post use an INT8 head everywhere.

## GPTQ in the library

`quantforge.gptq.quantize_model_gptq` implements sequential GPTQ (Frantar et al.) producing exactly the layouts the kernels consume:

1. the decoder layers are run on 128 calibration windows of 512 tokens; the hidden states entering the first layer are captured with a stand-in module that records its inputs and aborts the forward pass;
2. layer by layer, a forward pre-hook on every Linear accumulates the Hessian of its input, `H = 2 X^T X / n`, while the layer processes the calibration activations;
3. each Linear is quantised column by column; the error of every rounded column, scaled by the diagonal of the inverse Cholesky factor of `H`, is pushed onto the columns not yet quantised;
4. the group scale and zero point are computed from the *updated* weights at the start of each group, with the scale rounded to f16 **before** the error is propagated so the optimisation sees the weights the kernel will actually use;
5. the quantised layer is run again on the calibration set; its outputs are the inputs of the next layer (so later layers see the quantisation noise of earlier ones).

Calibration takes 43 seconds for Qwen2.5-0.5B and 125 seconds for 1.5B on the 6 GB laptop GPU. Memory is bounded by one layer, its Hessians (the largest, `down_proj`, is `intermediate x intermediate` in f32) and the stored activations.

{{chart:gptq_ppl}}

| model | fp16 | RTN INT4 | GPTQ INT4 | share of the RTN loss recovered |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 19.93 | 24.08 | **21.38** | 65 % |
| Qwen2.5-1.5B-Instruct | 13.48 | 15.31 | **15.00** | 17 % |
| Llama-3.2-1B-Instruct | 20.15 | 24.42 | **23.39** | 24 % |
| OLMo-2-0425-1B-Instruct | 19.32 | 21.31 | **20.62** | 34 % |

Calibration uses windows of the Wikitext-2 *train* split; the perplexity is measured on 8 x 512 tokens of the *test* split. Decode speed does not change (the layout is the same): 226 tokens/s against 219 for Qwen2.5-0.5B in the same session, which is within the noise of this machine.

**An honest reading**: GPTQ always helps, but the gain is very uneven (65 percent of the loss on the smallest model, 17 percent on 1.5B). I do not have a clean explanation; models whose RTN loss is dominated by a few outlier channels benefit most. With an eight-window test set of 4096 tokens the second decimal of these perplexities is also noise-sensitive. A proper study would use the full test set and several seeds for the calibration windows.

### Things that did not help (so they are off by default)

| variant | Qwen 0.5B | Qwen 1.5B | Llama-3.2 1B | OLMo-2 1B |
|---|--:|--:|--:|--:|
| GPTQ, default (damping 0.01, group 128) | 21.38 | 15.00 | 23.39 | 20.62 |
| + per-group range clipping (`clip=True`) | 21.66 | **14.40** | 30.87 | **19.77** |
| act-order (`act_order=True`) | 21.74 | | | |
| damping 0.1 / 0.03 | 21.54 / 21.61 | | | |
| 256 calibration windows | 21.44 | | | |
| group size 64 | **21.26** | | | |

Range clipping (shrink the min/max of every group row when that lowers its weight-space rounding error) is the interesting one: it gives the best result on two models and the worst on Llama-3.2-1B, where perplexity jumps from 23.4 to 30.9. The criterion it optimises, the error of the weights, is not the error of the layer output that GPTQ optimises, and the two disagree on some layers. It stays an option. Act-order also costs speed: a layer with an input permutation cannot take part in the q/k/v fusion, so decode falls from 226 to 169 tokens/s on Qwen2.5-0.5B.

## Real AWQ and GPTQ checkpoints

Most 4-bit models on the Hub are AWQ or GPTQ files. `quantforge.nn.load_quantized_checkpoint(model, "model.safetensors", method="gptq" | "awq")` converts the int32-packed tensors into the kernel layout (`quant.from_gptq`, `quant.from_awq`), supports GPTQ `desc_act` checkpoints by permuting the rows offline (the activation permutation is applied by one small gather at run time) and the `gptq_v2` zero-point convention.

Verification against a dense twin (the same dequantised weights in ordinary `nn.Linear` layers), Qwen2.5-0.5B-Instruct, 16 x 512 test tokens:

| checkpoint | perplexity, dense twin | perplexity, QuantForge | greedy decode | decode speed |
|---|--:|--:|--:|--:|
| GPTQ-Int4 (official Qwen) | 26.335 | 26.336 | 59 / 59 tokens identical | 182 tok/s |
| AWQ (official Qwen) | 26.689 | 26.689 | 59 / 59 tokens identical | 184 tok/s |

### The trap that cost an hour: AWQ changes the norms

The first AWQ run printed a perplexity of **83,014** for the dense twin while the QuantForge model, built the same way, printed 26.7. A single layer matched to the last bit, so the kernel was fine. The cause: AWQ scales the input channels of a linear layer by a learned factor `s` to protect salient weights, and folds the inverse factor into the *preceding* operation, usually the RMSNorm weight. The norm weights stored in an AWQ checkpoint are therefore not the original model's. Loading only the quantised linear layers on top of the base model's norms gives garbage; the loader now copies every non-linear tensor (embeddings, norms) that exists in both the checkpoint and the model, and the README says so. GPTQ does not do this, which is why GPTQ worked on the first try.

## Reproducing

```bash
# calibrated 4-bit, with the comparison against round-to-nearest
python benchmarks/hf_gptq.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --train wikitext2-train.parquet --text wikitext2-test.parquet

# a real checkpoint against its dense twin
python benchmarks/hf_load_checkpoint.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --ckpt Qwen2.5-0.5B-Instruct-GPTQ-Int4/model.safetensors --method gptq --text wikitext2-test.parquet
```

Everything above ran on one RTX 4050 Laptop GPU with 6 GB of memory; the 3B model and Phi-3.5 (3.8B) are loaded on the CPU and quantised layer by layer on the GPU, and a 7B model runs from a 4-bit bitsandbytes checkpoint in 4.65 GB.
