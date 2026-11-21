"""FP8 W8A8 (FP8 tensor cores through cuBLASLt) against fp16 and INT4 W4A16: prefill speed and perplexity.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/hf_fp8_w8a8.py --model Qwen/Qwen2.5-0.5B-Instruct --text <wikitext-2 test parquet>

Prefill = one forward pass over 2048 tokens (batch 1) or 8 x 512 tokens, best of 10, `torch.cuda.synchronize` timing (the
passes are compute-bound, so launch overhead does not matter). Perplexity: the whole test split. The head stays fp16 in the
fp16 and W8A8 variants and is INT8 in the INT4 variant.
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
from qwen_end_to_end import perplexity  # noqa: E402


@torch.no_grad()
def prefill_ms(m, shape, vocab):
    x = torch.randint(0, vocab, shape, device="cuda")
    for _ in range(3):
        m(x)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(10):
        t = time.time(); m(x); torch.cuda.synchronize(); best = min(best, time.time() - t)
    return best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--variants", default="fp16,int4,fp8_w8a8,int8_w8a8_cublaslt,int8_w8a8")
    ns = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    ids = tok("\n\n".join(pd.read_parquet(ns.text)["text"]), return_tensors="pt").input_ids[0]
    for v in ns.variants.split(","):
        m = AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()
        if v == "fp16":
            m = m.cuda()
        elif v == "int4":
            qnn.quantize_model(m, "int4", 128); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
        elif v == "int8_w8a8":            # INT8 weights + per-token INT8 activations, GEMM + dequant epilogue compiled by QuantForge
            qnn.quantize_model(m, "int8_w8a8", fuse=True)          # q/k/v and gate/up share one quantisation + one GEMM
        elif v == "w4a8":                 # INT4 weights (progressive quantisation) + per-token INT8 activations, one compiled kernel
            qnn.quantize_model(m, "w4a8", fuse=True)
        elif v in ("int8_w8a8_nonorm", "w4a8_nonorm"):    # the fused MLP activation, but the norms do not quantise their output
            qnn.quantize_model(m, v.split("_nonorm")[0], fuse=False); qnn._fuse_int8(m, fuse_norm=False)
        elif v == "fp16_cnorm":           # fp16 with every RMSNorm compiled (what the norm fusion gives without any quantisation)
            m = m.cuda()
            for mod in m.modules():
                if type(mod).__name__.lower().endswith("rmsnorm") and hasattr(mod, "variance_epsilon"):
                    mod.forward = torch.compile(mod.forward, dynamic=True)
        elif v == "int8_w8a8_noact":      # as int8_w8a8 without the fused silu * up + quantisation
            qnn.quantize_model(m, "int8_w8a8", fuse=False); qnn._fuse_int8(m, fuse_activation=False)
        elif v == "w4a8_noact":
            qnn.quantize_model(m, "w4a8", fuse=False); qnn._fuse_int8(m, fuse_activation=False)
        elif v == "int8_w8a8_nofuse":
            qnn.quantize_model(m, "int8_w8a8", fuse=False)
        elif v == "int8_w8a8_cublaslt":   # the same through cuBLASLt (torch._int_mm) with the epilogue as separate torch ops
            qnn.quantize_model(m, "int8_w8a8", fuse=False, int8_backend="cublaslt")
        elif v == "fp8_w8a8_tensor":       # one scale per tensor on both sides
            qnn.quantize_model(m, "fp8_w8a8", fuse=False, w8a8_per_token=False)
        else:                              # fp8_w8a8: per-token activation scale + per-channel weight scale (the default)
            qnn.quantize_model(m, "fp8_w8a8", fuse=False)
        vocab = m.config.vocab_size
        t1, t2 = prefill_ms(m, (1, 2048), vocab), prefill_ms(m, (8, 512), vocab)
        ppl = perplexity(m, ids, 512)
        print(f"{ns.model} {v:9s} prefill 1x2048: {t1:7.1f} ms ({2048 / t1 * 1e3:6.0f} tok/s)   8x512: {t2:7.1f} ms ({4096 / t2 * 1e3:6.0f} tok/s)   ppl {ppl:.3f}   GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
