"""Run a real AWQ / GPTQ checkpoint on QuantForge through `quantforge.nn.load_quantized_checkpoint`
and check it against a dense twin (the same weights dequantised to fp16 in a plain nn.Linear model).

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/hf_load_checkpoint.py --base Qwen/Qwen2.5-0.5B-Instruct \
        --ckpt <model.safetensors of Qwen2.5-0.5B-Instruct-GPTQ-Int4 / -AWQ> --method gptq|awq \
        --text <wikitext-2 test parquet>

Reports perplexity of both, the logit difference, greedy-token agreement and the CUDA-graph decode speed.
"""
import argparse
import copy
import gc
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantforge.nn as qnn  # noqa: E402
from qwen_decode_graph import run as graph_run  # noqa: E402
from qwen_end_to_end import perplexity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF id of the dense architecture / embeddings")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method", choices=("gptq", "awq", "fp8"), required=True)
    ap.add_argument("--from-config", action="store_true",
                    help="build the architecture from the config of --base (random init) instead of loading dense weights: "
                         "for repositories that only hold the quantised weights; all tensors then come from the checkpoint")
    ap.add_argument("--text", required=True)
    ap.add_argument("--zeros-offset", type=int, default=1, help="1 for gptq, 0 for gptq_v2")
    ap.add_argument("--nseq", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ns = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.base)
    ids = tok("\n\n".join(pd.read_parquet(ns.text)["text"]), return_tensors="pt").input_ids[0][: ns.nseq * ns.seqlen]
    prompt = "Explain in one paragraph why the sky is blue."

    if ns.from_config:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(ns.base)
        if hasattr(cfg, "quantization_config"):
            del cfg.quantization_config
        dense = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16).eval()
    else:
        dense = AutoModelForCausalLM.from_pretrained(ns.base, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    twin = copy.deepcopy(dense)
    st = qnn.load_quantized_checkpoint(twin, ns.ckpt, ns.method, fuse=False, zeros_offset=ns.zeros_offset)
    # dense twin: every QuantLinear becomes a plain nn.Linear holding the dequantised weight (the embeddings
    # and norms come from the checkpoint as well: AWQ folds its activation scales into the norms)
    for name, mod in list(twin.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, qnn.QuantLinear):
                lin = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                                      dtype=torch.float16, device="cuda")
                lin.weight.data = child.dequantized_weight()
                if child.bias is not None:
                    lin.bias.data = child.bias.clone()
                setattr(mod, cname, lin)
    ref = twin.cuda()
    qf = dense
    st = qnn.load_quantized_checkpoint(qf, ns.ckpt, ns.method, zeros_offset=ns.zeros_offset)
    gc.collect(); torch.cuda.empty_cache()
    print(f"loaded {st['quantized']} {ns.method} layers (act-order: {st['act_order']}, fused qkv {st['fused_qkv']}, "
          f"mlp {st['fused_mlp']})")
    ppl_ref, ppl_qf = perplexity(ref, ids, ns.seqlen), perplexity(qf, ids, ns.seqlen)
    x = ids[: ns.seqlen].unsqueeze(0).cuda()
    with torch.no_grad():
        lr, lq = ref(x).logits.float(), qf(x).logits.float()
    print(f"  perplexity  dequantised twin {ppl_ref:.3f}   quantforge {ppl_qf:.3f}")
    print(f"  logits: max|d| {(lr - lq).abs().max().item():.4f}  top-1 agreement "
          f"{(lr.argmax(-1) == lq.argmax(-1)).float().mean().item() * 100:.2f} %")
    gen = lambda m: m.generate(tok(prompt, return_tensors="pt").input_ids.cuda(), max_new_tokens=48, do_sample=False)[0]  # noqa: E731
    a, b = gen(ref), gen(qf)
    print(f"  greedy decode: {int((a == b).sum())}/{a.numel()} tokens identical to the dequantised twin")
    print("  quantforge says:", tok.decode(b, skip_special_tokens=True).replace("\n", " ")[-160:])
    del ref; gc.collect(); torch.cuda.empty_cache()
    print(f"  CUDA-graph decode: {graph_run(qf, tok, prompt, 48)[0]:.1f} tok/s")


if __name__ == "__main__":
    main()
