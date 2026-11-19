"""Quantise any Hugging Face causal LM with the library API (`quantforge.nn.quantize_model`) and
measure perplexity and decode speed against the fp16 model.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/hf_quantize_model.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --text <wikitext-2 test parquet> [--fmt int4|int8|fp8] [--group-size 128] [--head int8|int4|fp8|none]

The quantisation is round-to-nearest (see benchmarks/hf_quantize_and_run.py for the comparison with a
dense dequantised reference and per-format tables). Needs torch, transformers, pandas/pyarrow, a CUDA GPU.
"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantforge.nn as qnn  # noqa: E402
from qwen_decode_graph import run as graph_run  # noqa: E402
from qwen_end_to_end import perplexity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--fmt", choices=qnn.FORMATS, default="int4")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--head", choices=("none",) + qnn.FORMATS, default="none",
                    help="also quantise the lm_head with this format (group size from --group-size)")
    ap.add_argument("--no-fp16", action="store_true", help="skip the fp16 baseline (does not fit the GPU)")
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ns = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    ids = tok("\n\n".join(pd.read_parquet(ns.text)["text"]), return_tensors="pt").input_ids[0][: ns.nseq * ns.seqlen]
    prompt = "Explain in one paragraph why the sky is blue."

    def load():
        return AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()

    res = {}
    if not ns.no_fp16:
        m = load().cuda()
        res["fp16"] = (perplexity(m, ids, ns.seqlen), graph_run(m, tok, prompt, 48)[0], torch.cuda.memory_allocated())
        del m; gc.collect(); torch.cuda.empty_cache()

    m = load()
    t = time.time()
    st = qnn.quantize_model(m, ns.fmt, ns.group_size)
    if ns.head != "none":
        qnn.quantize_model(m, ns.head, ns.group_size, only=("lm_head",), fuse=False)
    print(f"quantised {st['quantized']} layers to {ns.fmt} in {time.time() - t:.1f} s "
          f"(weights {st['bytes_before'] / 1e9:.2f} -> {st['bytes_after'] / 1e9:.2f} GB; fused qkv {st['fused_qkv']}, "
          f"mlp {st['fused_mlp']}; skipped {len(st['skipped'])})")
    res[f"quantforge {ns.fmt}"] = (perplexity(m, ids, ns.seqlen), graph_run(m, tok, prompt, 48)[0],
                                   torch.cuda.memory_allocated())
    for k, (ppl, tps, mem) in res.items():
        print(f"  {k:16s} ppl {ppl:7.3f}   CUDA-graph decode {tps:6.1f} tok/s   GPU {mem / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
