"""Per-window loss distribution of RTN versus GPTQ models (diagnoses a perplexity that disagrees between subsets).

    python benchmarks/ppl_per_window.py --model unsloth/Llama-3.2-1B-Instruct --train <train parquet> --text <test parquet>

Prints the perplexity on the first 8 / 64 windows and on all windows, the share of windows where GPTQ is worse than
RTN, and the worst windows (a handful of windows with a huge loss -- fp16 overflow in a massive-activation channel,
say -- would dominate the perplexity of the whole split).
"""
import argparse
import gc
import math
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.gptq as qgptq  # noqa: E402
import quantforge.nn as qnn  # noqa: E402


@torch.no_grad()
def window_losses(model, ids, seqlen):
    out = []
    for i in range(ids.numel() // seqlen):
        x = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).cuda()
        logits = model(x).logits.float()
        out.append(nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="mean").item())
    return torch.tensor(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--group-size", type=int, default=128)
    ns = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    ids = tok("\n\n".join(pd.read_parquet(ns.text)["text"]), return_tensors="pt").input_ids[0]
    calib = qgptq.calibration_ids(tok, "\n\n".join(pd.read_parquet(ns.train)["text"]), 128, 512)
    load = lambda: AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()  # noqa: E731
    res = {}
    m = load().cuda(); res["fp16"] = window_losses(m, ids, ns.seqlen); del m; gc.collect(); torch.cuda.empty_cache()
    m = load(); qnn.quantize_model(m, "int4", ns.group_size); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
    res["RTN"] = window_losses(m, ids, ns.seqlen); del m; gc.collect(); torch.cuda.empty_cache()
    m = load(); qgptq.quantize_model_gptq(m, calib, ns.group_size, verbose=False); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
    res["GPTQ"] = window_losses(m, ids, ns.seqlen)
    n = len(res["fp16"])
    for name, l in res.items():
        print(f"{name:5s} ppl first 8: {math.exp(l[:8].mean()):7.3f}  first 64: {math.exp(l[:64].mean()):7.3f}  all {n}: {math.exp(l.mean()):7.3f}"
              f"   max window loss {l.max():.2f}  nan/inf windows {int((~torch.isfinite(l)).sum())}")
    d = res["GPTQ"] - res["RTN"]
    print(f"GPTQ worse than RTN in {(d > 0).float().mean() * 100:.0f} % of windows; mean loss gap {d.mean():+.4f}; "
          f"median {d.median():+.4f}; top-5 gaps {[round(x, 2) for x in d.topk(5).values.tolist()]} at windows {d.topk(5).indices.tolist()}")
    print("worst GPTQ windows (loss, fp16 loss, RTN loss):",
          [(round(res['GPTQ'][i].item(), 2), round(res['fp16'][i].item(), 2), round(res['RTN'][i].item(), 2)) for i in res["GPTQ"].topk(4).indices.tolist()])


if __name__ == "__main__":
    main()
