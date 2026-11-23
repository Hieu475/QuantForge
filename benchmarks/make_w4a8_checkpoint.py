"""Quantise a Hugging Face model to W4A8 (GPTQ or round-to-nearest) and save it as a checkpoint for vLLM.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/make_w4a8_checkpoint.py Qwen/Qwen2.5-3B-Instruct out_dir \
        --train <wikitext-2 train parquet>            # GPTQ (calibration: 128 windows of 512 tokens from the train split)
    ... --rtn                                         # round-to-nearest, no calibration data
    ... --rotate                                      # quantise in the block-Hadamard basis (GPTQ: its Hessians too); the checkpoint is marked "rotated": true and the layers
                                                      # rotate their input at run time (see the CHANGELOG, research/rotation_ppl.py)

    python -c "from vllm import LLM; LLM('out_dir', quantization='quantforge_w4a8', dtype='float16')"

The model is read on the CPU and quantised one layer at a time, so a model larger than the GPU in fp16 is fine (Qwen2.5-3B: ~4 minutes).
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.gptq as qgptq  # noqa: E402
import quantforge.nn as qnn  # noqa: E402
import quantforge.w4a8_checkpoint as ck  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--train", help="parquet with a `text` column (GPTQ calibration)")
    ap.add_argument("--rtn", action="store_true")
    ap.add_argument("--group", type=int, default=128, help="group size of the integer scales along K (128, or 64: more accurate, 2-3 %% slower)")
    ap.add_argument("--rotate", action="store_true", help="block-Hadamard rotation of weights and activations (better quality, ~2 %% slower)")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=512)
    ns = ap.parse_args()
    if not ns.rtn and not ns.train:
        ap.error("--train is needed for GPTQ (or pass --rtn)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ns.model)
    m = AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    if ns.rtn:
        qnn.quantize_model(m, "w4a8", ns.group if ns.group != 128 else 128, fuse=False)
        if ns.rotate:
            raise SystemExit("--rotate with --rtn: use quantization='quantforge_w4a8' on the fp16 model (online, rotated by default)")
    else:
        import pandas as pd

        calib = qgptq.calibration_ids(tok, "\n\n".join(pd.read_parquet(ns.train)["text"].iloc[:20000]), ns.nsamples, ns.calib_len, 0)
        qgptq.quantize_model_gptq(m, calib, ns.group, fmt="w4a8", fuse=False, verbose=False, rotate=ns.rotate)
    print("saved", ck.save_w4a8(m, ns.out, tok))


if __name__ == "__main__":
    main()
