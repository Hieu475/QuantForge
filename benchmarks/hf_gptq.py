"""GPTQ (quantforge.gptq) versus round-to-nearest on a Hugging Face causal LM: perplexity and decode speed.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/hf_gptq.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --train <wikitext-2 train parquet> --text <wikitext-2 test parquet> [--act-order] [--group-size 128]

The calibration windows come from the *train* split, the perplexity is measured on the test split. The head is
quantised to INT8 (lossless) in both variants so that only the decoder weights differ.
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
import quantforge.gptq as qgptq  # noqa: E402
import quantforge.nn as qnn  # noqa: E402
from qwen_decode_graph import run as graph_run  # noqa: E402
from qwen_end_to_end import perplexity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--clip", action="store_true", help="per-row range shrinking search at every group")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--no-rtn", action="store_true")
    ns = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    test = tok("\n\n".join(pd.read_parquet(ns.text)["text"]), return_tensors="pt").input_ids[0][: ns.nseq * ns.seqlen]
    calib = qgptq.calibration_ids(tok, "\n\n".join(pd.read_parquet(ns.train)["text"]), ns.nsamples, ns.calib_len)
    prompt = "Explain in one paragraph why the sky is blue."

    def load():
        return AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()

    rows = []

    def evaluate(label, m, seconds):
        rows.append((label, perplexity(m, test, ns.seqlen), graph_run(m, tok, prompt, 48)[0], seconds,
                     torch.cuda.memory_allocated() / 1e9))

    if not ns.no_rtn:
        m = load()
        t = time.time()
        qnn.quantize_model(m, "int4", ns.group_size)
        qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
        evaluate("RTN", m, time.time() - t)
        del m; gc.collect(); torch.cuda.empty_cache()

    m = load()
    t = time.time()
    st = qgptq.quantize_model_gptq(m, calib, ns.group_size, ns.damp, ns.act_order, clip=ns.clip)
    qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
    evaluate("GPTQ" + (" act-order" if ns.act_order else "") + (" clip" if ns.clip else ""), m, time.time() - t)
    print(f"({st['quantized']} layers, fused qkv {st['fused_qkv']}, mlp {st['fused_mlp']}, act-order layers {st['act_order']}; "
          f"calibration {ns.nsamples} x {ns.calib_len} tokens, group {ns.group_size})")
    for label, ppl, tps, sec, mem in rows:
        print(f"  {label:16s} ppl {ppl:7.3f}   CUDA-graph decode {tps:6.1f} tok/s   quantisation {sec:6.0f} s   GPU {mem:.2f} GB")


if __name__ == "__main__":
    main()
