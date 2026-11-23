"""Sequential GPTQ at 4 / 3 bits with and without the block-Hadamard rotation, fake-quantised (dequantised float weights), INT8 activations optional.

    python research/gptq_bits.py Qwen/Qwen2.5-1.5B-Instruct --bits 3 [--rotate] [--a8] [--clip] [--nsamples 64]

Layer by layer: the Hessians of the layer's linears come from the calibration windows run through the model whose earlier layers are already quantised (as `quantize_model_gptq`).
"""
import argparse, math, sys, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantforge import gptq as qg, w4a8_op  # noqa: E402
from rotation_ppl import quant_a  # noqa: E402


class FakeQ(torch.nn.Module):
    def __init__(self, weight, bias, rot, a8):
        super().__init__()
        self.weight, self.bias, self.rot, self.a8 = weight, bias, rot, a8

    def forward(self, x):
        if self.rot:
            x = w4a8_op._rotate_impl(x.reshape(-1, x.shape[-1]).contiguous()).reshape(x.shape)
        if self.a8:
            x = quant_a(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--bits", type=int, default=4); ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--a8", action="store_true"); ap.add_argument("--clip", action="store_true"); ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--windows", type=int, default=40); ap.add_argument("--len", type=int, default=512); ap.add_argument("--token-clip", type=float, default=10.0)
    a = ap.parse_args()
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tr = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset"))["text"].iloc[:20000]
    te = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"]
    calib = qg.calibration_ids(tok, "\n\n".join(tr), a.nsamples, a.len, 0).cuda()
    ids = tok("\n\n".join(te)).input_ids
    wins = [ids[i * a.len:(i + 1) * a.len] for i in range(min(a.windows, len(ids) // a.len))]
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    G = 128
    for li, layer in enumerate(m.model.layers):
        targets = {(p, n): c for p in (layer.self_attn, layer.mlp) for n, c in p.named_children() if isinstance(c, torch.nn.Linear)}
        hess = {k: torch.zeros(c.in_features, c.in_features, device="cuda") for k, c in targets.items()}
        ntok = calib.shape[0] * calib.shape[1]

        def mk(k):
            def hook(_, args):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                nrm = x.norm(dim=1)
                if a.token_clip > 0:
                    cap = a.token_clip * nrm.median().clamp(min=1e-6)
                    x = x * (cap / nrm.clamp(min=1e-6)).clamp(max=1.0).unsqueeze(1)
                hess[k].addmm_(x.t(), x, alpha=2.0 / ntok)
            return hook
        hs = [c.register_forward_pre_hook(mk(k)) for k, c in targets.items()]
        with torch.no_grad():
            for i in range(calib.shape[0]):
                m(calib[i:i + 1])
        for h in hs:
            h.remove()
        for (parent, name), c in targets.items():
            W = c.weight.data.float()
            H = hess[(parent, name)]
            if a.rotate:
                W, H = w4a8_op.rotate_weight(W), qg.rotate_hessian(H)
            q, scale, zero, _ = qg.gptq_quantize_weight(W.half() if not a.rotate else W, H, G, 128, 0.01, False, a.bits, a.clip)
            K = q.shape[0]
            wq = ((q.float() - zero.float().repeat_interleave(G, 0)) * scale.float().repeat_interleave(G, 0)).t().contiguous()
            setattr(parent, name, FakeQ(wq.half(), c.bias, a.rotate, a.a8))
        del hess
        torch.cuda.empty_cache()
        print(f"layer {li} done", flush=True) if li % 7 == 0 else None
    nll, n = 0.0, 0
    with torch.no_grad():
        for w in wins:
            t = torch.tensor([w], device="cuda")
            lg = m(t).logits[0, :-1].float()
            nll += torch.nn.functional.cross_entropy(lg, t[0, 1:], reduction="sum").item(); n += len(w) - 1
    print(f"PPL gptq bits={a.bits} rotate={a.rotate} a8={a.a8} clip={a.clip}: {math.exp(nll / n):.3f}", flush=True)


if __name__ == "__main__":
    main()
