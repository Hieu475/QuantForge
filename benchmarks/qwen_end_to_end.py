"""
End-to-end check on a real model: Qwen2.5-0.5B-Instruct with its GPTQ-Int4 checkpoint.

Three variants of the same network are compared on wikitext-2 (perplexity, logits) and greedy
generation:
  fp16      the original fp16 weights                    (quality baseline)
  ref       the GPTQ weights dequantised to dense fp16   (what the quantised model should compute)
  quantforge every decoder Linear replaced by the QuantForge W4A16 kernel (gate_proj + up_proj fused into
            one GEMM with the swiglu epilogue; q / k / v of the attention as one GEMM); embeddings / norms / lm_head stay fp16

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/qwen_end_to_end.py \
        --ckpt <.../gptq/model.safetensors> --text <wikitext-2 test parquet> [--nseq 16 --seqlen 512]

Needs torch, transformers, safetensors, pandas/pyarrow and a CUDA GPU. Checkpoint:
huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 (desc_act=False, group size 128).
"""
import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge import quant  # noqa: E402
import quantforge.torch_ops as qt  # noqa: E402

G = 128
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def gptq_dense_weight(qw, qz, sc, K, N):
    """f32 [K, N] weights from the GPTQ file layout (own unpacking, independent of quantforge.quant)."""
    shifts = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q = ((qw.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N).float()
    z = ((qz.unsqueeze(-1) >> shifts) & 0xF).reshape(qz.shape[0], N).float() + 1
    return (q - z.repeat_interleave(G, 0)) * sc.float().repeat_interleave(G, 0)


class QFLinear(nn.Module):
    def __init__(self, qw, qz, sc, gi, bias, activation="none"):
        super().__init__()
        self.packed, self.scale, self.zero = quant.from_gptq(qw, qz, sc, gi)
        self.bias, self.activation = bias, activation

    def forward(self, x):
        return qt.w4a16_linear(x, self.packed, self.scale, self.zero, G, self.bias, self.activation)


class QFLinearRTN(nn.Module):
    """A dense [N, K] fp16 weight quantised on the fly (round-to-nearest, asymmetric, group 128)."""
    def __init__(self, weight):
        super().__init__()
        q, self.scale, self.zero = quant.quantize_groupwise(weight.t().contiguous(), G)
        self.packed = quant.pack_int4(q)
        self.bias = None

    def forward(self, x):
        return qt.w4a16_linear(x, self.packed, self.scale, self.zero, G, None)


class QFQKV(nn.Module):
    """q / k / v of one attention layer as ONE GEMM (weights and biases concatenated along N).
    The HF attention calls q_proj, k_proj, v_proj on the same tensor: the first call runs the fused
    GEMM and the other two take their slice of the result (keyed on the identity of the input)."""
    def __init__(self, q, k, v):
        super().__init__()
        self.packed, self.scale, self.zero = quant.concat_columns(
            *((m.packed, m.scale, m.zero) for m in (q, k, v)))
        self.sizes = [m.scale.shape[1] for m in (q, k, v)]
        biases = [m.bias if m.bias is not None else torch.zeros(n, dtype=torch.float16, device=q.packed.device)
                  for m, n in zip((q, k, v), self.sizes)]
        self.bias = torch.cat(biases)
        self._x, self._outs = None, None

    def parts(self, x):
        if self._x is not x:
            self._outs = qt.w4a16_linear(x, self.packed, self.scale, self.zero, G, self.bias).split(self.sizes, -1)
            self._x = x
        return self._outs


class QFQKVPart(nn.Module):
    def __init__(self, fused, index):
        super().__init__()
        self.fused, self.index = fused, index

    def forward(self, x):
        return self.fused.parts(x)[self.index]


class QFMLP(nn.Module):
    """down(silu(gate(x)) * up(x)). INT4: ONE fused GEMM (gate / up weights interleaved by column,
    silu * up in the epilogue) followed by down_proj. INT8 / FP8 weights have no fused-swiglu kernel:
    gate (with a fused silu epilogue) and up are two GEMMs."""
    def __init__(self, gate, up, down):
        super().__init__()
        self.fused = gate.packed.shape[1] * 2 == gate.scale.shape[1]            # INT4
        if self.fused:
            self.packed, self.scale, self.zero = quant.interleave_gate_up(
                (gate.packed, gate.scale, gate.zero), (up.packed, up.scale, up.zero))
        else:
            self.gate, self.up = gate, up
        self.down_proj = down

    def forward(self, x):
        if self.fused:
            return self.down_proj(qt.w4a16_linear(x, self.packed, self.scale, self.zero, G, None, "swiglu"))
        g = qt.w4a16_linear(x, self.gate.packed, self.gate.scale, self.gate.zero, G, None, "silu")
        return self.down_proj(g * self.up(x))


def load_model(base, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16).to(device).eval()


def build_variants(base, ckpt, device, quant_head=False, fuse_qkv=True):
    from safetensors.torch import load_file
    sd = load_file(ckpt, device=device)
    fp16 = load_model(base, device)
    ref = load_model(base, device)
    qf = load_model(base, device)
    for L, (lr, lq) in enumerate(zip(ref.model.layers, qf.model.layers)):
        mods = {}
        for name in PROJ:
            pre = f"model.layers.{L}.{name}"
            qw, qz, sc, gi = (sd[f"{pre}.{k}"] for k in ("qweight", "qzeros", "scales", "g_idx"))
            K, N = qw.shape[0] * 8, qw.shape[1]
            bias = sd.get(f"{pre}.bias")
            parent_r = lr.get_submodule(name.rsplit(".", 1)[0])
            parent_q = lq.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
            # reference: dense dequantised weight in the nn.Linear
            lin = getattr(parent_r, leaf)
            lin.weight.data = gptq_dense_weight(qw, qz, sc, K, N).t().contiguous().half()
            if bias is not None:
                lin.bias.data = bias.half()
            mods[name] = QFLinear(qw, qz, sc, gi, bias.half() if bias is not None else None,
                                  "none")
            if not name.startswith("mlp."):
                setattr(parent_q, leaf, mods[name])
        lq.mlp = QFMLP(mods["mlp.gate_proj"], mods["mlp.up_proj"], mods["mlp.down_proj"])
        if fuse_qkv:
            fused = QFQKV(mods["self_attn.q_proj"], mods["self_attn.k_proj"], mods["self_attn.v_proj"])
            for i, leaf in enumerate(("q_proj", "k_proj", "v_proj")):
                setattr(lq.self_attn, leaf, QFQKVPart(fused, i))
    for m in (fp16, ref, qf):  # embeddings / norms (and the tied lm_head) from the checkpoint
        m.model.embed_tokens.weight.data = sd["model.embed_tokens.weight"].half()
        m.model.norm.weight.data = sd["model.norm.weight"].half()
    if quant_head:  # the tied lm_head (151936 x 896 fp16 = 272 MB) is the largest remaining read
        qf.lm_head = QFLinearRTN(qf.lm_head.weight.data)
    # the fp16 baseline keeps the ORIGINAL weights (reload to undo the embedding override above)
    fp16 = load_model(base, device)
    return fp16, ref, qf


@torch.no_grad()
def perplexity(model, ids, seqlen):
    nll, n = 0.0, 0
    for i in range(ids.numel() // seqlen):
        x = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).cuda()
        logits = model(x).logits.float()
        loss = nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum")
        nll += loss.item(); n += seqlen - 1
    return math.exp(nll / n)


@torch.no_grad()
def greedy(model, tok, prompt, new_tokens):
    x = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = model.generate(x, max_new_tokens=new_tokens, do_sample=False)
    return out[0, x.shape[1]:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True, help="wikitext-2 test parquet (column `text`)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--nseq", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--no-fuse-qkv", action="store_true", help="keep q / k / v as three GEMMs")
    ap.add_argument("--quant-head", action="store_true",
                    help="also quantise the (tied) lm_head to 4 bit with RTN (not part of the checkpoint)")
    ns = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.base)
    text = "\n\n".join(pd.read_parquet(ns.text)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][: ns.nseq * ns.seqlen]
    print(f"{ids.numel()} tokens = {ids.numel() // ns.seqlen} x {ns.seqlen}")

    fp16, ref, qf = build_variants(ns.base, ns.ckpt, "cuda", ns.quant_head, not ns.no_fuse_qkv)
    res = {}
    for name, m in (("fp16", fp16), ("ref (GPTQ dequantised)", ref), ("quantforge", qf)):
        t = time.time()
        res[name] = perplexity(m, ids, ns.seqlen)
        print(f"  perplexity {name:24s} {res[name]:8.3f}   ({time.time() - t:5.1f} s)", flush=True)

    x = ids[: ns.seqlen].unsqueeze(0).cuda()
    with torch.no_grad():
        lr, lq = ref(x).logits.float(), qf(x).logits.float()
    print(f"  logits quantforge vs ref: max|d| = {(lr - lq).abs().max().item():.4f}  "
          f"mean|d| = {(lr - lq).abs().mean().item():.5f}  (logit std {lr.std().item():.2f})  "
          f"top-1 agreement {(lr.argmax(-1) == lq.argmax(-1)).float().mean().item() * 100:.2f} %")

    prompt = "Explain in one paragraph why the sky is blue."
    tr, tq = greedy(ref, tok, prompt, ns.new_tokens), greedy(qf, tok, prompt, ns.new_tokens)
    same = int((tr == tq).sum().item())
    print(f"  greedy decode: {same}/{tr.numel()} tokens identical to the dequantised reference")
    print("  quantforge:", tok.decode(tq, skip_special_tokens=True).replace("\n", " ")[:300])

    for name, m in (("fp16", fp16), ("quantforge", qf)):
        greedy(m, tok, prompt, 8)
        torch.cuda.synchronize(); t = time.time()
        out = greedy(m, tok, prompt, ns.new_tokens)
        torch.cuda.synchronize()
        print(f"  decode speed {name:12s} {out.numel() / (time.time() - t):6.1f} tokens/s (HF generate, batch 1)")


if __name__ == "__main__":
    main()
