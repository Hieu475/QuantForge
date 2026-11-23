"""Perplexity of a model served by vLLM (the kernels vLLM really runs), on wikitext-2 test, in non-overlapping windows.

    python benchmarks/vllm_ppl.py <model> <quantization|none> [--windows 40] [--len 512] [--text test.parquet]
    # fp16:     python benchmarks/vllm_ppl.py Qwen/Qwen2.5-1.5B-Instruct none
    # Marlin:   python benchmarks/vllm_ppl.py Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 gptq_marlin
    # ours:     python benchmarks/vllm_ppl.py Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 quantforge
    # W4A8:     python benchmarks/vllm_ppl.py Qwen/Qwen2.5-1.5B-Instruct quantforge_w4a8

The text is the test split of Salesforce/wikitext (wikitext-2-raw-v1) joined with newlines, tokenised once, cut into windows of `--len` tokens; the
loss of every token but the first of a window comes from vLLM's `prompt_logprobs`. Same windows for every model of the same tokenizer, so the numbers compare.
"""
import argparse
import math
import os
import warnings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("quant")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--text")
    ap.add_argument("--util", type=float, default=0.7)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    path = a.text or hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(path)["text"].tolist())
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(text).input_ids
    wins = [ids[i * a.len:(i + 1) * a.len] for i in range(min(a.windows, len(ids) // a.len))]
    llm = LLM(a.model, quantization=None if a.quant == "none" else a.quant, dtype="float16", gpu_memory_utilization=a.util,
              max_model_len=a.len + 8, enable_prefix_caching=False, max_num_seqs=8, enforce_eager=bool(os.environ.get("EAGER")), kv_cache_dtype=os.environ.get("QF_KV", "auto"))
    outs = llm.generate([{"prompt_token_ids": w} for w in wins], SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
    nll, n = 0.0, 0
    for w, o in zip(wins, outs):
        for t, lp in zip(w[1:], o.prompt_logprobs[1:]):
            nll -= lp[t].logprob
            n += 1
    print(f"PPL {a.model} {a.quant}: {math.exp(nll / n):.3f}  ({n} tokens in {len(wins)} windows of {a.len})")


if __name__ == "__main__":
    main()
