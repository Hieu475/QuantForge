"""Perplexity of the W4A8 format with round-to-nearest and with GPTQ, against INT4 W4A16 (RTN and GPTQ) and fp16.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/w4a8_gptq.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --train <wikitext-2 train parquet> --text <wikitext-2 test parquet> [--variants fp16,int4,int4_gptq,w4a8,w4a8_gptq]

The head stays fp16 in every variant; the whole test split, windows of 512 tokens; calibration windows from the train split.
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
from qwen_end_to_end import perplexity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--variants", default="fp16,int4,int4_gptq,w4a8,w4a8_gptq")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--rtn", nargs="*", default=[], help="layers kept at round-to-nearest in the GPTQ variants (e.g. 1.mlp.down_proj)")
    ns = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    test = tok("\n\n".join(pd.read_parquet(ns.text)["text"]), return_tensors="pt").input_ids[0]
    calib = qgptq.calibration_ids(tok, "\n\n".join(pd.read_parquet(ns.train)["text"].iloc[:20000]), ns.nsamples, ns.calib_len, 0)
    for v in ns.variants.split(","):
        m = AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()
        t = time.time()
        if v == "fp16":
            m = m.cuda()
        elif v == "int4":
            qnn.quantize_model(m, "int4", 128)
        elif v == "w4a8":
            qnn.quantize_model(m, "w4a8")
        elif v in ("int4_gptq", "w4a8_gptq"):
            qgptq.quantize_model_gptq(m, calib, 128, rtn=tuple(ns.rtn), fmt=v.split("_")[0], verbose=False)
        else:
            raise SystemExit(f"unknown variant {v}")
        sec = time.time() - t
        print(f"{ns.model} {v:10s} ppl {perplexity(m, test, 512):7.3f}   quantisation {sec:5.0f} s   GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
