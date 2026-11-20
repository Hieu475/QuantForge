---
title: Better 4-bit quality with GPTQ, loading real AWQ / GPTQ checkpoints, and a few traps
date: 2026-11-20
order: 5
summary: Round-to-nearest INT4 costs 2 to 4 perplexity points. A GPTQ implementation recovers 47 to 60 percent of that, after a debugging story about massive activations that made it worse than RTN on Llama. Real AWQ, GPTQ (including act-order) and FP8 checkpoints are loaded and verified.
---

Fast kernels are only half of a quantisation story: the other half is how much quality the 4-bit weights lose. The library therefore has two ways to obtain weights, a one-line round-to-nearest quantiser and a calibrated one, and it can load checkpoints produced by other tools.

## Round-to-nearest: the baseline

`quantforge.nn.quantize_model(model, "int4", group_size=128)` quantises every `nn.Linear` independently: each group of 128 input rows of a column maps its minimum and maximum to the 16 levels. It takes 3 seconds for a 0.5B model, needs no data, and loses a visible amount of quality (1 to 4 perplexity points on the models below). The head (`lm_head`) is kept in fp16 by default; quantising it to INT8 is lossless (perplexity 24.08 against 24.09 for Qwen2.5-0.5B) and 16 percent faster, so the numbers in this post use an INT8 head everywhere.

## GPTQ in the library

`quantforge.gptq.quantize_model_gptq` implements sequential GPTQ (Frantar et al.) producing exactly the layouts the kernels consume:

1. the decoder layers are run on 128 calibration windows of 512 tokens; the hidden states entering the first layer are captured with a stand-in module that records its inputs and aborts the forward pass;
2. layer by layer, a forward pre-hook on every Linear accumulates the Hessian of its input, `H = 2 X^T X / n`, while the layer processes the calibration activations (with every token's norm capped at 10 times the median, see the pitfall below);
3. each Linear is quantised column by column; the error of every rounded column, scaled by the diagonal of the inverse Cholesky factor of `H`, is pushed onto the columns not yet quantised;
4. the group scale and zero point are computed from the *updated* weights at the start of each group, with the scale rounded to f16 **before** the error is propagated so the optimisation sees the weights the kernel will actually use;
5. the quantised layer is run again on the calibration set; its outputs are the inputs of the next layer (so later layers see the quantisation noise of earlier ones).

Calibration takes 49 seconds for Qwen2.5-0.5B and about 150 seconds for 1.5B on the 6 GB laptop GPU. Memory is bounded by one layer, its Hessians (the largest, `down_proj`, is `intermediate x intermediate` in f32) and the stored activations.

{{chart:gptq_ppl}}

| model | fp16 | RTN INT4 | GPTQ INT4 | share of the RTN loss recovered |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 18.79 | 22.91 | **20.45** | 60 % |
| Qwen2.5-1.5B-Instruct | 12.66 | 14.30 | **13.38** | 56 % |
| Llama-3.2-1B-Instruct | 17.91 | 22.28 | **20.21** | 47 % |
| OLMo-2-0425-1B-Instruct | 17.24 | 19.02 | **18.13** | 50 % |
| Qwen2.5-3B-Instruct (no fp16 reference: does not fit) | - | 13.13 | **12.06** | - |

These are perplexities on the **whole** Wikitext-2 test split (564 to 584 windows of 512 tokens); calibration uses 128 windows drawn from the first 20 000 rows of the *train* split. Decode speed does not change (the layout is the same).

### A pitfall: the first version of this table was wrong, and so was the second

My first measurement used only the first 8 test windows (4096 tokens), and on those GPTQ beat round-to-nearest on every model. Re-measuring on the whole test split showed that on **Llama-3.2-1B GPTQ was worse than RTN (24.08 against 22.28)** and on Qwen2.5-1.5B it recovered only 24 percent of the loss. Eight windows are simply too few for a method whose failures are rare. The investigation, in four steps:

1. **Per-window losses** (`benchmarks/ppl_per_window.py`): GPTQ was better than RTN in 83 percent of the windows (median gap -0.11 nats), but in a few windows it was completely broken, with a loss of 9 to 10 per token instead of about 3. A handful of such windows dominates the average.
2. **Layer by layer** (hidden states against the fp16 model on a good and a broken window): the error was small until layer 1 and then exploded at *one token position* (position 1 in the broken window) and spread from there to every later token.
3. **Which layer?** Quantising only `layers.1.mlp.down_proj` with RTN (all the others with GPTQ) brought Llama from 24.08 to 20.26. That layer creates the *massive activation* of the attention-sink token: its input contains one token whose norm is **675 times the median** (790 times in Qwen2.5-1.5B). In the same layer GPTQ's mean error over the calibration tokens was 2.5 to 5 times lower than RTN's, as designed, so a check of the average error cannot see the problem.
4. **Why**: that single token dominates the Hessian `H = 2 X^T X / n`, the error compensation is tuned to it, and the other tokens, and sink tokens in other positions, pay for it.

A fixed list of layers is not a fix: forcing RTN on the massive-activation layers helps Llama (20.26) and Qwen-1.5B (13.47) but hurts Qwen-0.5B (20.83 against 20.20). What works for all three is a **cap on the token norm in the Hessian**: before a calibration token enters `H`, its norm is limited to a multiple of the median token norm. The massive token still takes part, but with a bounded weight.

**The second version was wrong too.** I first chose a cap of 30, which fixed the whole-split perplexity (Llama 24.08 to 20.12). A slow regression test that I then wrote, on the five windows that had been broken, showed that **one of them was still broken** (window 70: loss 5.9 against 3.0 for fp16), although the perplexity of the whole split could not tell. A per-window analysis of all 564 windows settled the choice:

| cap | whole-split ppl | windows where GPTQ is worse than RTN | worst window against RTN |
|---|--:|--:|--:|
| 30 | 20.123 | 2 (1 %) | +2.76 nats |
| **10** | 20.192 | **0** | none |
| 5 | 20.328 | 0 | none |

The default is therefore **10** (`token_clip=10`): it costs about 0.07 perplexity points against 30 and removes the rare catastrophic windows. Lesson: for a method whose failures are rare, an average is the wrong acceptance criterion; the test that matters looks at the worst case.

### Things that did not help (so they are off by default)

* per-group range clipping (`clip=True`): helps OLMo and Qwen-1.5B, ruins Llama-3.2-1B (30.9 on the first 8 windows against 23.4 without it): the weight-space rounding error it minimises is not the layer-output error GPTQ minimises;
* act-order (`act_order=True`): a layer with an input permutation cannot take part in the q/k/v fusion, so decode falls from 226 to 169 tokens/s on Qwen2.5-0.5B, and the perplexity was not better;
* larger damping: before the token cap, 0.1 turned Llama into a model with perplexity 58 (the cap makes the result much less sensitive to it);
* more calibration windows (256) or longer ones (2048 tokens): no consistent gain.

### How robust is it?

Three checks beyond a single perplexity number (whole Wikitext-2 test split, GPTQ with the defaults):

| check | Qwen2.5-0.5B | Llama-3.2-1B | Qwen2.5-1.5B |
|---|--:|--:|--:|
| RTN INT4 | 22.91 | 22.28 | 14.30 |
| GPTQ, 3 calibration seeds | 20.45 / 20.46 / 20.48 (sd 0.01) | 20.21 / 20.22 / 20.25 (sd 0.02) | 13.38 (one seed) |
| GPTQ, calibrated on cnn_dailymail articles instead of Wikitext | 20.70 | 20.43 | 13.65 |

* **Seeds:** the spread between random calibration windows is 0.01 to 0.02 perplexity points, far below the gap to RTN. It was 0.08 to 0.15 with a cap of 30: bounding the influence of the massive token also stabilises the result.
* **Another domain:** calibrating on news articles and testing on Wikipedia costs 0.2 to 0.3 points and still beats RTN by two points or more. The Wikitext-on-Wikitext numbers in the main table are *in-domain*, which flatters GPTQ slightly.
* **A bigger model:** Qwen2.5-3B (no fp16 reference: it does not fit on this card): RTN 13.13, GPTQ **12.06**, 300 seconds of calibration.

**A task instead of a likelihood.** Zero-shot ARC accuracy of the GPTQ models next to fp16 and RTN (lm-evaluation-harness scoring: `acc` picks the choice with the highest log-likelihood, `acc_norm` the highest per character), in percent. Standard error: 1.0 points on ARC-Easy (2376 questions), 1.4 on ARC-Challenge (1172):

| model | variant | ARC-Easy acc | ARC-Easy acc_norm | ARC-Challenge acc | ARC-Challenge acc_norm |
|---|---|--:|--:|--:|--:|
| Qwen2.5-0.5B | fp16 | 65.6 | 59.2 | 30.7 | 33.8 |
| | RTN | 62.5 | 59.3 | 31.7 | 32.3 |
| | GPTQ | 63.3 | 60.5 | 29.4 | 33.0 |
| Llama-3.2-1B | fp16 | 69.0 | 63.8 | 34.6 | 37.9 |
| | RTN | 64.3 | 60.4 | 33.5 | 35.3 |
| | GPTQ | 66.4 | 62.5 | 33.2 | 36.9 |
| OLMo-2-1B | fp16 | 71.8 | 71.0 | 41.0 | 42.0 |
| | RTN | 71.2 | 68.3 | 39.8 | 40.7 |
| | GPTQ | 70.4 | 69.5 | 39.3 | 42.2 |
| Qwen2.5-1.5B | fp16 | 76.8 | 76.0 | 43.5 | 46.7 |
| | RTN | 65.6 | 62.3 | 37.0 | 39.7 |
| | GPTQ | **73.1** | **72.0** | **40.5** | **44.0** |

The surprise is **Qwen2.5-1.5B**: round-to-nearest loses 11 points on ARC-Easy (76.8 to 65.6) and 6.5 on ARC-Challenge, much more than its modest perplexity increase (12.66 to 14.30) suggests, and GPTQ gets back 7.5 and 3.5 points of them. (With the earlier cap of 30 it got back 9.4 and 5.1: the differences are about two standard errors, so I would not call the cap of 10 worse on this model, but the data do not exclude it.) For the other three models the differences between fp16, RTN and GPTQ are mostly within the standard error: at INT4 these small models lose little on ARC either way. A perplexity table alone can hide a real quality problem, and it can also overstate one.

## Real AWQ and GPTQ checkpoints

Most 4-bit models on the Hub are AWQ or GPTQ files. `quantforge.nn.load_quantized_checkpoint(model, "model.safetensors", method="gptq" | "awq")` converts the int32-packed tensors into the kernel layout (`quant.from_gptq`, `quant.from_awq`), supports GPTQ `desc_act` checkpoints by permuting the rows offline (the activation permutation is applied by one small gather at run time) and the `gptq_v2` zero-point convention.

Verification against a dense twin (the same dequantised weights in ordinary `nn.Linear` layers), Qwen2.5-0.5B-Instruct, 16 x 512 test tokens:

| checkpoint | perplexity, dense twin | perplexity, QuantForge | greedy decode | decode speed |
|---|--:|--:|--:|--:|
| GPTQ-Int4 (official Qwen) | 26.335 | 26.336 | 59 / 59 tokens identical | 182 tok/s |
| AWQ (official Qwen) | 26.689 | 26.689 | 59 / 59 tokens identical | 184 tok/s |
| **TinyLlama-1.1B GPTQ with `desc_act=True`** (TheBloke; all 154 layers act-ordered) | 17.259 | 17.259 | 60 / 60 tokens identical | 124 tok/s (act-order layers are not fused) |
| **FP8-dynamic** (RedHatAI, per-channel scales), Qwen2.5-0.5B | 22.543 | 22.543 (fp16 base model: 22.441) | 28 / 59 identical, see below | 165 tok/s |

The FP8 file only supplies the *weights* (the activations stay fp16); its per-channel scales are expanded to group 128 and 0.55 percent of them are subnormal in f16. The greedy decodes diverge after 28 tokens because two near-tied tokens flip under the slightly different fp16 rounding of the twin's matmul; the logits agree to 99.4 percent top-1 and the perplexities are identical to three decimals, so this is a tie-break, not an error. The act-order checkpoint is the first *real* test of the row-permutation path (it had only been tested with synthetic files).

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
