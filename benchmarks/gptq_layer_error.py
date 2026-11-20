"""Per-layer output error of RTN versus GPTQ on the calibration activations (a sanity check of quantforge.gptq).

    python benchmarks/gptq_layer_error.py --model unsloth/Llama-3.2-1B-Instruct --train <wikitext-2 train parquet>

GPTQ minimises ||X (W - Q)^T||^2 over the calibration inputs X, so on those inputs its error must be lower than
RTN's for every layer; if it is not, the implementation is wrong. (Whether that lowers the model's perplexity on
other text is a separate question: see hf_gptq.py.)
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.gptq as qg  # noqa: E402
from quantforge import quant  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 1, 7, 15])
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=128)
    ns = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    model = AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    calib = qg.calibration_ids(tok, "\n\n".join(pd.read_parquet(ns.train)["text"]), ns.nsamples, 512).cuda()
    layers = qg._find_layers(model)
    for name, child in model.named_children():
        pass
    model.cuda()
    # input of every Linear of the selected layers, from the *dense* model
    acts = {}
    hooks = []
    for li in ns.layers:
        for n, m in layers[li].named_modules():
            if isinstance(m, nn.Linear):
                acts[(li, n)] = []
                hooks.append(m.register_forward_pre_hook(lambda _, a, key=(li, n): acts[key].append(a[0].reshape(-1, a[0].shape[-1]).half())))
    for i in range(calib.shape[0]):
        model(calib[i:i + 1])
    for h in hooks:
        h.remove()
    print(f"{'layer':32s} {'RTN mse':>12s} {'GPTQ mse':>12s} {'GPTQ / RTN':>11s}")
    for (li, n), xs in acts.items():
        X = torch.cat(xs).float()
        W = layers[li].get_submodule(n).weight.data.float()                         # [out, in]
        H = 2 * X.t() @ X / X.shape[0]
        q, s, z, _ = qg.gptq_quantize_weight(W, H, ns.group_size)
        deq_g = ((q.float() - z.float().repeat_interleave(ns.group_size, 0)) * s.float().repeat_interleave(ns.group_size, 0))
        qr, sr, zr = quant.quantize_groupwise(W.t().contiguous(), ns.group_size)
        deq_r = ((qr.float() - zr.float().repeat_interleave(ns.group_size, 0)) * sr.float().repeat_interleave(ns.group_size, 0))
        ref = X @ W.t()
        e_g = ((X @ deq_g - ref) ** 2).mean().item()
        e_r = ((X @ deq_r - ref) ** 2).mean().item()
        print(f"layer {li:2d} {n:25s} {e_r:12.5f} {e_g:12.5f} {e_g / e_r:11.3f}")


if __name__ == "__main__":
    main()
