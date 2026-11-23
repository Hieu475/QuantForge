"""Per-layer choice of the rotation: block-Hadamard with random sign flips / a channel permutation, picked by the layer-output error on calibration activations.

    python research/rotation_search.py Qwen/Qwen2.5-1.5B-Instruct [--cands 8] [--windows 40]

Fake quantisation as in rotation_ppl.py (W4 asymmetric g128 RTN + per-token INT8 activations). Candidate t: x -> ((x[:, perm]) * sign) H_blockdiag; the candidate list is
{identity, 1 .. cands-1 random sign vectors, magnitude-interleaved permutation}; the winner minimises || X W^T - Q(X T) Q(W T)^T ||_F over 1024 calibration tokens.
Prints perplexity of: fixed Hadamard, searched, and (for scale) the average error of the candidates.
"""
import argparse, math, warnings
import torch
from rotation_ppl import hadamard, quant_w, quant_a
warnings.filterwarnings("ignore")


def transform(x, perm, sign, H, block):
    if perm is not None:
        x = x[..., perm]
    if sign is not None:
        x = x * sign
    sh = x.shape
    return (x.float().reshape(-1, sh[-1] // block, block) @ H).reshape(sh)


class QL(torch.nn.Module):
    def __init__(self, lin, perm, sign, H, block, a8=True):
        super().__init__()
        self.perm, self.sign, self.H, self.block, self.a8 = perm, sign, H, block, a8
        self.weight = quant_w(transform(lin.weight.data, perm, sign, H, block), 128, 4).to(lin.weight.dtype)
        self.bias = lin.bias

    def forward(self, x):
        xr = transform(x, self.perm, self.sign, self.H, self.block).to(x.dtype)
        if self.a8:
            xr = quant_a(xr)
        return torch.nn.functional.linear(xr, self.weight, self.bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--block", type=int, default=128); ap.add_argument("--cands", type=int, default=8)
    ap.add_argument("--windows", type=int, default=40); ap.add_argument("--len", type=int, default=512)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    test = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"].tolist()
    train = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"))["text"].tolist()
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok("\n\n".join(test)).input_ids
    wins = [ids[i * a.len:(i + 1) * a.len] for i in range(min(a.windows, len(ids) // a.len))]
    cal = tok("\n\n".join(train[:3000])).input_ids[:4 * a.len]
    cal = torch.tensor(cal).reshape(4, a.len).cuda()
    H = hadamard(a.block).cuda()

    def ppl(m):
        nll, n = 0.0, 0
        with torch.no_grad():
            for w in wins:
                t = torch.tensor([w], device="cuda")
                lg = m(t).logits[0, :-1].float()
                nll += torch.nn.functional.cross_entropy(lg, t[0, 1:], reduction="sum").item(); n += len(w) - 1
        return math.exp(nll / n)

    for mode in ("fixed", "search"):
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
        cap = {}
        hooks = []
        if mode == "search":
            for li, layer in enumerate(m.model.layers):
                for parent, pn in ((layer.self_attn, "attn"), (layer.mlp, "mlp")):
                    for name, ch in parent.named_children():
                        if isinstance(ch, torch.nn.Linear):
                            def hk(mod, inp, out, key=(li, pn, name)):
                                x = inp[0].reshape(-1, inp[0].shape[-1])
                                cap.setdefault(key, []).append(x[torch.randperm(x.shape[0], device=x.device)[:256]].detach())
                            hooks.append(ch.register_forward_hook(hk))
            with torch.no_grad():
                for i in range(cal.shape[0]):
                    m(cal[i:i + 1])
            for h in hooks:
                h.remove()
        g = torch.Generator(device="cuda").manual_seed(0)
        errs_fixed, errs_best = [], []
        for li, layer in enumerate(m.model.layers):
            for parent, pn in ((layer.self_attn, "attn"), (layer.mlp, "mlp")):
                for name, ch in list(parent.named_children()):
                    if not isinstance(ch, torch.nn.Linear):
                        continue
                    K = ch.in_features
                    if mode == "fixed":
                        setattr(parent, name, QL(ch, None, None, H, a.block).cuda())
                        continue
                    X = torch.cat(cap[(li, pn, name)])
                    ref = torch.nn.functional.linear(X.float(), ch.weight.data.float(), None if ch.bias is None else ch.bias.data.float())
                    cands = [(None, None)]
                    for _ in range(a.cands - 2):
                        cands.append((None, (torch.randint(0, 2, (K,), device="cuda", generator=g).float() * 2 - 1).half()))
                    mag = X.float().abs().amax(0)
                    order = mag.argsort(descending=True)
                    nb = K // a.block
                    # interleave by magnitude: rank r goes to block r % nb
                    perm = order.reshape(a.block, nb).t().reshape(-1)
                    cands.append((perm, None))
                    best, best_e = None, float("inf")
                    for ci, (p_, s_) in enumerate(cands):
                        q = QL(ch, p_, s_, H, a.block).cuda()
                        with torch.no_grad():
                            e = ((q(X.half()).float() - ref).norm() / ref.norm()).item()
                        if ci == 0:
                            errs_fixed.append(e)
                        if e < best_e:
                            best, best_e = q, e
                        else:
                            del q
                    errs_best.append(best_e)
                    setattr(parent, name, best)
        torch.cuda.empty_cache()
        extra = f"   mean layer error fixed {sum(errs_fixed)/len(errs_fixed):.4f} -> searched {sum(errs_best)/len(errs_best):.4f}" if errs_best else ""
        print(f"PPL {mode:7s} {ppl(m):.3f}{extra}", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
