"""Quantise a Hugging Face causal LM with the GPTQ of this repository and write it as an ordinary GPTQ checkpoint that vLLM loads.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/make_gptq_checkpoint.py <hf model id or dir> <out dir> --train <wikitext-2 train parquet>
    vllm serve <out dir> --quantization gptq_marlin      # or --quantization quantforge

The calibration text comes from the ``text`` column of the parquet (128 windows of 512 tokens). ``--group-size`` 128 by default.
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--train", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=512)
    ns = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import quantforge.gptq as qg
    from quantforge.gptq_checkpoint import save_gptq

    tok = AutoTokenizer.from_pretrained(ns.model)
    model = AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    text = "\n\n".join(pd.read_parquet(ns.train)["text"].iloc[:20000])
    calib = qg.calibration_ids(tok, text, ns.nsamples, ns.calib_len, 0)
    stats = qg.quantize_model_gptq(model, calib, ns.group_size, fuse=False, verbose=False)
    print("quantised", stats)
    save_gptq(model, ns.out, tok)
    print("saved", ns.out)


if __name__ == "__main__":
    main()
