"""Does a block-Hadamard rotation of the input dimension make W4A8 (and W4A16) round-to-nearest quantisation lose less?

    python research/rotation_ppl.py Qwen/Qwen2.5-1.5B-Instruct [--block 128] [--windows 40]

Fake quantisation in torch (no kernels): every Linear of the decoder layers gets W (asymmetric 4-bit, group 128, round to nearest) and, optionally, per-token symmetric INT8 on its
input. With ``rot`` both are first rotated by the same orthonormal block-diagonal Hadamard H (x -> xH, W -> WH, the product is unchanged in exact arithmetic).
Perplexity on wikitext-2 test, non-overlapping windows (same text as benchmarks/vllm_ppl.py).
"""
import argparse, math, warnings
import torch
warnings.filterwarnings("ignore")


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def quant_w(W, G=128, bits=4):                     # [N, K] -> fake-quantised [N, K], asymmetric, group along K
    N, K = W.shape
    w = W.float().reshape(N, K // G, G)
    lo, hi = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    q = 2 ** bits - 1
    s = ((hi - lo) / q).clamp_min(1e-8)
    z = (-lo / s).round()
    return (((w / s).round() + z).clamp(0, q) - z).mul(s).reshape(N, K)


def quant_a(x):                                    # per-token symmetric int8
    xf = x.float()
    s = (xf.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
    return (xf / s).round().clamp(-127, 127).mul(s).to(x.dtype)


class QLinear(torch.nn.Module):
    def __init__(self, lin, rot, a8, block, wbits):
        super().__init__()
        W = lin.weight.data.float()
        self.H = None
        if rot:
            self.H = hadamard(block).to(W.device)
            N, K = W.shape
            W = (W.reshape(N, K // block, block) @ self.H).reshape(N, K)
        self.weight = quant_w(W, 128, wbits).to(lin.weight.dtype) if wbits else W.to(lin.weight.dtype)
        self.bias, self.a8, self.block = lin.bias, a8, block

    def forward(self, x):
        if self.H is not None:
            sh = x.shape
            x = (x.float().reshape(-1, sh[-1] // self.block, self.block) @ self.H).reshape(sh).to(x.dtype)
        if self.a8:
            x = quant_a(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--block", type=int, default=128); ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--len", type=int, default=512); ap.add_argument("--variants", default="fp16,w4,w4a8,rot-w4,rot-w4a8,rot-a8")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(path)["text"].tolist())
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(text).input_ids
    wins = [ids[i * a.len:(i + 1) * a.len] for i in range(min(a.windows, len(ids) // a.len))]
    for v in a.variants.split(","):
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
        if v != "fp16":
            rot, a8 = v.startswith("rot"), v.endswith("a8")
            wbits = 0 if v.endswith("-a8") and v.startswith("rot-a8") and False else 4
            if v == "rot-a8":
                wbits = 0
            for layer in m.model.layers:
                for parent in (layer.self_attn, layer.mlp):
                    for name, child in list(parent.named_children()):
                        if isinstance(child, torch.nn.Linear):
                            setattr(parent, name, QLinear(child, rot, a8, a.block, wbits).cuda())
            torch.cuda.empty_cache()
        nll, n = 0.0, 0
        with torch.no_grad():
            for w in wins:
                t = torch.tensor([w], device="cuda")
                lg = m(t).logits[0, :-1].float()
                nll += torch.nn.functional.cross_entropy(lg, t[0, 1:], reduction="sum").item(); n += len(w) - 1
        print(f"PPL {v:10s} {math.exp(nll / n):.3f}", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
