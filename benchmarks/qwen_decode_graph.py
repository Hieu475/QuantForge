"""
Decode speed of Qwen2.5-0.5B-Instruct (GPTQ-Int4) with the whole decode step captured in a CUDA graph:
fp16 weights vs the QuantForge W4A16 kernels. A graph removes the Python launch overhead (the
kernels are tiny at this model size), so what remains is GPU time.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/qwen_decode_graph.py --ckpt <.../model.safetensors>
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_end_to_end import build_variants  # noqa: E402


@torch.no_grad()
def run(model, tok, prompt, steps, max_len=256, compile=False):
    from transformers import StaticCache

    if compile:       # inductor fuses the norms / rotary / elementwise kernels of the HF graph (+~10 %)
        model = torch.compile(model)

    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    T = ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    out = model(ids, past_key_values=cache, cache_position=torch.arange(T, device="cuda"), use_cache=True)
    tok_in = out.logits[:, -1].argmax(-1, keepdim=True)
    pos = torch.tensor([T], device="cuda")

    def step():
        o = model(tok_in, past_key_values=cache, cache_position=pos, position_ids=pos.unsqueeze(0),
                  use_cache=True)
        return o.logits[:, -1].argmax(-1, keepdim=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):          # warm-up: compiles / loads every kernel, outside the graph
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        nxt = step()
    toks = [tok_in.clone()]
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps):
        g.replay()
        tok_in.copy_(nxt)
        pos += 1
        toks.append(nxt.clone())
    torch.cuda.synchronize()
    dt = time.time() - t
    return steps / dt, torch.cat(toks, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--quant-head", action="store_true", help="also quantise the tied lm_head (RTN)")
    ap.add_argument("--no-fuse-qkv", action="store_true", help="keep q / k / v as three GEMMs")
    ns = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.base)
    fp16, ref, qf = build_variants(ns.base, ns.ckpt, "cuda", ns.quant_head, not ns.no_fuse_qkv)
    prompt = "Explain in one paragraph why the sky is blue."
    res = {}
    for name, m in (("fp16", fp16), ("quantforge", qf)):
        tps, toks = run(m, tok, prompt, ns.steps)
        res[name] = toks
        print(f"  {name:11s} {tps:7.1f} tokens/s  (CUDA graph, batch 1)")
        print("   ", tok.decode(toks, skip_special_tokens=True).replace("\n", " ")[:160])


if __name__ == "__main__":
    main()
