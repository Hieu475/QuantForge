"""
Take a Hugging Face causal LM in fp16 (Llama / Qwen2 / OLMo-2 / Phi-3 style), quantise every decoder
Linear to INT4 (round-to-nearest, asymmetric, group size 128), run it on the QuantForge kernels
(q/k/v fused into one GEMM, gate/up fused with the swiglu epilogue) and compare it with

  fp16   the original model                                    (quality / speed baseline)
  ref    the same INT4 weights dequantised to dense fp16       (what the kernels must reproduce)
  qf     QuantForge

on wikitext-2 (perplexity, logits) plus greedy generation and batch-1 decode speed. RTN is a plain
quantiser: the perplexity gap to fp16 is the quantiser's, the gap between ref and qf is the kernels'.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/hf_quantize_and_run.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --text <wikitext-2 test parquet>

The variants run one after the other on the GPU (they would not fit together on a 6 GB card).
Embeddings, norms and lm_head stay fp16. Needs torch, transformers, pandas/pyarrow, a CUDA GPU.
"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantforge import quant  # noqa: E402
import quantforge.torch_ops as qt  # noqa: E402
from qwen_end_to_end import QFMLP, QFQKV, QFQKVPart, perplexity  # noqa: E402

G = 128


class QLinear(nn.Module):
    """One quantised Linear: packed INT4 weight + group-wise scale / zero point (+ bias)."""
    def __init__(self, packed, scale, zero, bias, group_size=G):
        super().__init__()
        self.packed, self.scale, self.zero, self.bias = packed, scale, zero, bias
        self.group_size = group_size

    def forward(self, x):
        return qt.w4a16_linear(x, self.packed, self.scale, self.zero, self.group_size, self.bias)


def quantize_head(lm_head, group_size, bits=4, dense=True):
    """RTN INT4 / INT8 of the (possibly tied) lm_head with its own group size -> (QLinear, dense
    dequantised fp16 [V, K] weight or None). INT8 (symmetric, per group) is practically lossless; INT4
    with a smaller group costs a few % more bytes but keeps the logits closer to fp16. Quantised in
    chunks of rows (groups run along K) so a 7B head never needs a full fp32 copy on a 6 GB GPU."""
    w = lm_head.weight.data.t()                                                # [K, V] view
    K, step = w.shape[0], max(group_size, (1024 // group_size) * group_size)
    qs, ss, zs, deqs = [], [], [], []
    for r in range(0, K, step):
        wc = w[r:r + step].to("cuda", torch.float32).contiguous()
        if bits == 8:
            q, scale, zero = quant.quantize_groupwise_int8(wc, group_size)
        else:
            q, scale, zero = quant.quantize_groupwise(wc, group_size)
        qs.append(q if bits == 8 else quant.pack_int4(q)); ss.append(scale); zs.append(zero)
        if dense:
            deqs.append(((q.float() - zero.float().repeat_interleave(group_size, 0))
                         * scale.float().repeat_interleave(group_size, 0)).half())
        del wc
    mod = QLinear(torch.cat(qs), torch.cat(ss), torch.cat(zs), None, group_size)
    return mod, (torch.cat(deqs).t().contiguous() if dense else None)


def quantize_linear(lin):
    """(packed, scale, zero, bias) of an nn.Linear, quantised on the GPU, results kept on the CPU."""
    state = getattr(lin.weight, "quant_state", None)
    if state is not None:                      # bitsandbytes NF4 / FP4 weight: dequantise first
        import bitsandbytes as bnb
        w = bnb.functional.dequantize_4bit(lin.weight.data, state).to("cuda", torch.float32).t().contiguous()
    else:
        w = lin.weight.data.to("cuda", torch.float32).t().contiguous()       # [K, N]
    if w.shape[0] % G:
        raise ValueError(f"in_features {w.shape[0]} is not a multiple of the group size {G}")
    q, scale, zero = quant.quantize_groupwise(w, G)
    out = (quant.pack_int4(q).cpu(), scale.cpu(), zero.cpu(),
           None if lin.bias is None else lin.bias.data.half().cpu())
    del w, q
    return out


def dequantize(packed, scale, zero):
    """Dense fp16 [N, K] weight of a quantised Linear (what the ref variant multiplies with)."""
    q = quant.unpack_int4(packed).float()
    w = (q - zero.float().repeat_interleave(G, 0)) * scale.float().repeat_interleave(G, 0)
    return w.half().t().contiguous()


def split_columns(triple, sizes):
    """Split a (packed, scale, zero) triple along N into consecutive column blocks."""
    packed, scale, zero = triple
    q = quant.unpack_int4(packed)
    out, a = [], 0
    for n in sizes:
        out.append((quant.pack_int4(q[:, a:a + n]).contiguous(), scale[:, a:a + n].contiguous(),
                    zero[:, a:a + n].contiguous()))
        a += n
    return out


def layer_linears(layer):
    """{name: (parent module, attribute)} of the decoder layer's Linears."""
    found = {}
    for parent_name in ("self_attn", "mlp"):
        parent = getattr(layer, parent_name)
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "gate_proj", "up_proj",
                     "gate_up_proj", "down_proj"):
            if isinstance(getattr(parent, attr, None), nn.Linear):
                found[f"{parent_name}.{attr}"] = (parent, attr)
    return found


def quantize_model(model):
    """{layer index: {name: (packed, scale, zero, bias)}} for every decoder Linear."""
    qp = {}
    for i, layer in enumerate(model.model.layers):
        qp[i] = {name: quantize_linear(getattr(parent, attr))
                 for name, (parent, attr) in layer_linears(layer).items()}
    return qp


def apply_ref(model, qp):
    """Dense fp16 weights = dequantised INT4 (what the kernels have to reproduce)."""
    for i, layer in enumerate(model.model.layers):
        for name, (parent, attr) in layer_linears(layer).items():
            packed, scale, zero, _ = qp[i][name]
            getattr(parent, attr).weight.data = dequantize(packed.cuda(), scale.cuda(), zero.cuda())


def apply_qf(model, qp, fuse_qkv=True):
    for i, layer in enumerate(model.model.layers):
        layer.cuda()                                   # norms; the fp16 Linears are replaced below
        cuda = {n: tuple(None if t is None else t.cuda() for t in v) for n, v in qp[i].items()}
        mods = {n: QLinear(*v) for n, v in cuda.items()}
        attn, mlp = layer.self_attn, layer.mlp
        if "self_attn.qkv_proj" in mods:                                   # Phi-3: already fused
            attn.qkv_proj = mods["self_attn.qkv_proj"]
        elif fuse_qkv:
            fused = QFQKV(mods["self_attn.q_proj"], mods["self_attn.k_proj"], mods["self_attn.v_proj"])
            for j, leaf in enumerate(("q_proj", "k_proj", "v_proj")):
                setattr(attn, leaf, QFQKVPart(fused, j))
        else:
            for leaf in ("q_proj", "k_proj", "v_proj"):
                setattr(attn, leaf, mods[f"self_attn.{leaf}"])
        attn.o_proj = mods["self_attn.o_proj"]
        if "mlp.gate_up_proj" in mods:                                     # Phi-3: [gate | up] stacked
            packed, scale, zero, _ = cuda["mlp.gate_up_proj"]
            half = scale.shape[1] // 2
            (gp, gs, gz), (up_, us, uz) = split_columns((packed, scale, zero), [half, half])
            gate, up = QLinear(gp, gs, gz, None), QLinear(up_, us, uz, None)
        else:
            gate, up = mods["mlp.gate_proj"], mods["mlp.up_proj"]
        layer.mlp = QFMLP(gate, up, mods["mlp.down_proj"])
        del cuda, mods, gate, up
        gc.collect()          # the replaced modules sit in reference cycles: free their weights NOW,
        torch.cuda.empty_cache()   # not at some later collection (a 7B NF4 model leaves no VRAM to spare)


def load(name, dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, low_cpu_mem_usage=True).eval()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def generate(model, tok, prompt, n):
    x = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = model.generate(x, max_new_tokens=n, do_sample=False)
    return out[0, x.shape[1]:]


def hf_speed(model, tok, prompt, n):
    generate(model, tok, prompt, 8)
    torch.cuda.synchronize()
    t = time.time()
    out = generate(model, tok, prompt, n)
    torch.cuda.synchronize()
    return out.numel() / (time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or path of an fp16 causal LM")
    ap.add_argument("--text", required=True, help="wikitext-2 test parquet (column `text`)")
    ap.add_argument("--quant-head", type=int, default=0, metavar="GROUP",
                    help="also RTN-quantise the lm_head to INT4 with this group size (0 = keep fp16)")
    ap.add_argument("--head-bits", type=int, choices=(4, 8), default=4,
                    help="bit width of the quantised lm_head (with --quant-head)")
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--no-fp16", action="store_true", help="skip the fp16 baseline (model too big for the GPU)")
    ap.add_argument("--no-ref", action="store_true", help="skip the dense dequantised reference")
    ap.add_argument("--no-graph", action="store_true", help="skip the CUDA-graph decode speed")
    ns = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    text = "\n\n".join(pd.read_parquet(ns.text)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][: ns.nseq * ns.seqlen]
    prompt = "Explain in one paragraph why the sky is blue."
    x0 = ids[: ns.seqlen].unsqueeze(0).cuda()
    print(f"model {ns.model}: {ids.numel()} wikitext tokens = {ids.numel() // ns.seqlen} x {ns.seqlen}")
    try:
        from qwen_decode_graph import run as graph_run
    except Exception:  # noqa: BLE001
        graph_run = None

    def graph_speed(m):
        if ns.no_graph or graph_run is None:
            return None
        try:
            return graph_run(m, tok, prompt, 48)[0]
        except Exception as e:  # noqa: BLE001
            print(f"    (CUDA-graph decode unavailable: {type(e).__name__}: {str(e)[:80]})")
            return None

    # --- quantise once; the fp16 model is dropped right after (host RAM), results stay on the CPU
    def quantise():
        cpu = load(ns.model)
        nparams = sum(p.numel() for p in cpu.parameters())
        fp16_gb = sum(p.numel() for l in cpu.model.layers for p in l.parameters()) * 2 / 1e9
        t = time.time()
        qp = quantize_model(cpu)
        wbytes = sum(v[0].numel() + 2 * v[1].numel() + v[2].numel() for d in qp.values() for v in d.values())
        print(f"  {nparams / 1e9:.2f} B parameters, {len(qp)} layers quantised to INT4 in {time.time() - t:.1f} s "
              f"(decoder weights {fp16_gb:.2f} GB fp16 -> {wbytes / 1e9:.2f} GB)")
        return qp

    qp = quantise()
    cleanup()
    results = {}

    def stage_fp16():
        m = load(ns.model).to("cuda")
        r = {"ppl": perplexity(m, ids, ns.seqlen), "speed": (hf_speed(m, tok, prompt, ns.new_tokens), graph_speed(m))}
        print(f"  fp16        perplexity {r['ppl']:8.3f}", flush=True)
        return r

    def stage_ref():
        m = load(ns.model).to("cuda")
        apply_ref(m, qp)
        if ns.quant_head:      # untie from the embedding first: only the head is quantised
            m.lm_head.weight = nn.Parameter(quantize_head(m.lm_head, ns.quant_head, ns.head_bits)[1], requires_grad=False)
        r = {"ppl": perplexity(m, ids, ns.seqlen)}
        with torch.no_grad():
            r["logits"] = m(x0).logits.float().cpu()
        r["tokens"] = generate(m, tok, prompt, ns.new_tokens).cpu()
        print(f"  ref (INT4 dequantised) perplexity {r['ppl']:8.3f}", flush=True)
        return r

    def stage_qf():
        m = load(ns.model)
        for name, child in m.model.named_children():   # everything but the decoder layers (those are
            if name != "layers":                       # moved one at a time while they are quantised)
                child.cuda()
        m.lm_head.cuda()
        apply_qf(m, qp)
        if ns.quant_head:
            m.lm_head = quantize_head(m.lm_head, ns.quant_head, ns.head_bits, dense=False)[0]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = {"ppl": perplexity(m, ids, ns.seqlen)}
        print(f"  quantforge  perplexity {r['ppl']:8.3f}   ({time.time() - t:.1f} s, "
              f"GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak)", flush=True)
        with torch.no_grad():
            r["logits"] = m(x0).logits.float().cpu()
        r["tokens"] = generate(m, tok, prompt, ns.new_tokens).cpu()
        r["speed"] = (hf_speed(m, tok, prompt, ns.new_tokens), graph_speed(m))
        return r

    if not ns.no_fp16:
        results["fp16"] = stage_fp16()
        cleanup()
    if not ns.no_ref:
        results["ref"] = stage_ref()
        cleanup()
    results["quantforge"] = stage_qf()

    qf, ref = results["quantforge"], results.get("ref")
    if ref is not None:
        d = (qf["logits"] - ref["logits"]).abs()
        print(f"  logits quantforge vs ref: max|d| = {d.max().item():.4f}  mean|d| = {d.mean().item():.5f}  "
              f"(logit std {ref['logits'].std().item():.2f})  top-1 agreement "
              f"{(qf['logits'].argmax(-1) == ref['logits'].argmax(-1)).float().mean().item() * 100:.2f} %")
        n = min(len(qf["tokens"]), len(ref["tokens"]))
        print(f"  greedy decode: {int((qf['tokens'][:n] == ref['tokens'][:n]).sum())}/{n} tokens identical "
              f"to the dequantised reference")
    print("  quantforge says:", tok.decode(qf["tokens"], skip_special_tokens=True).replace("\n", " ")[:260])
    print("  decode speed (tokens/s), batch 1:        HF generate    CUDA graph")
    for k, r in results.items():
        if "speed" in r:
            a, b = r["speed"]
            print(f"    {k:12s}                             {a:8.1f}    {b if b is None else round(b, 1)!s:>10}")


if __name__ == "__main__":
    main()
