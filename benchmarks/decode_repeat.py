"""Batch-1 CUDA-graph decode speed with its run-to-run spread: fp16 against INT4 (+ INT8 head), repeated.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/decode_repeat.py --model Qwen/Qwen2.5-0.5B-Instruct --repeats 7

Each repetition captures a fresh CUDA graph and replays 96 decode steps; fp16 and INT4 repetitions are interleaved
so that clock drift of the laptop GPU hits both equally. Prints mean +- standard deviation (and min / max) of tokens/s
and of the INT4 / fp16 speed-up ratio of adjacent repetitions.
"""
import argparse
import gc
import statistics
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantforge.nn as qnn  # noqa: E402
from qwen_decode_graph import run as graph_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--fp16", action="store_true", help="also measure fp16 (the model must fit the GPU twice over: fp16 + INT4)")
    ns = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    prompt = "Explain in one paragraph why the sky is blue."
    load = lambda: AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()  # noqa: E731
    m4 = load(); qnn.quantize_model(m4, "int4", 128); qnn.quantize_model(m4, "int8", only=("lm_head",), fuse=False)
    m16 = load().cuda() if ns.fp16 else None
    r4, r16 = [], []
    for _ in range(ns.repeats):
        if m16 is not None:
            r16.append(graph_run(m16, tok, prompt, 96)[0])
        r4.append(graph_run(m4, tok, prompt, 96)[0])
    f = lambda xs: f"{statistics.mean(xs):6.1f} +- {statistics.stdev(xs):4.1f} (min {min(xs):.1f}, max {max(xs):.1f})"  # noqa: E731
    print(f"{ns.model}: INT4 + INT8 head {f(r4)} tok/s over {ns.repeats} runs")
    if r16:
        ratio = [a / b for a, b in zip(r4, r16)]
        print(f"{ns.model}: fp16 {f(r16)} tok/s;  speed-up {statistics.mean(ratio):.2f}x +- {statistics.stdev(ratio):.2f}")


if __name__ == "__main__":
    main()
