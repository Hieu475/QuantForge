"""
Run a bitsandbytes 4-bit (NF4) Hugging Face checkpoint on the QuantForge kernels.

bitsandbytes' NF4 is a non-uniform 16-level code, which an INT4 GEMM with a (scale, zero point) per group
cannot hold, so each Linear is dequantised and re-quantised to INT4 (round-to-nearest, group 128): the
result carries both quantisation errors. The model is converted layer by layer in place (a 7B model in
NF4 is ~4 GB, it would not fit twice on a 6 GB GPU). Compared with the bitsandbytes model itself:

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/hf_bnb_to_quantforge.py \
        --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --text <wikitext-2 test parquet>

Needs bitsandbytes and a CUDA GPU with enough memory for the NF4 model.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_quantize_and_run as H  # noqa: E402
from qwen_end_to_end import perplexity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--quant-head", type=int, default=0, metavar="GROUP",
                    help="also RTN-quantise the lm_head with this group size (0 = keep fp16)")
    ap.add_argument("--head-bits", type=int, choices=(4, 8), default=8)
    ns = ap.parse_args()

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    text = "\n\n".join(pd.read_parquet(ns.text)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][: ns.nseq * ns.seqlen]
    prompt = "Explain in one paragraph why the sky is blue."
    print(f"model {ns.model}: {ids.numel()} wikitext tokens = {ids.numel() // ns.seqlen} x {ns.seqlen}")

    m = AutoModelForCausalLM.from_pretrained(ns.model, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    print(f"  loaded with bitsandbytes: {torch.cuda.memory_allocated() / 1e9:.2f} GB on the GPU")
    ppl_bnb = perplexity(m, ids, ns.seqlen)
    print(f"  bitsandbytes NF4   perplexity {ppl_bnb:8.3f}", flush=True)
    toks_bnb = H.generate(m, tok, prompt, ns.new_tokens).cpu()
    speed_bnb = H.hf_speed(m, tok, prompt, ns.new_tokens)

    t = time.time()
    qp = H.quantize_model(m)                      # dequantise NF4 -> INT4 RTN, results on the CPU
    H.apply_qf(m, qp)                             # replaces the Linears layer by layer
    for p in m.parameters():                      # the kernels take float16 activations: embeddings,
        if p.dtype == torch.bfloat16:             # norms, lm_head (buffers, e.g. rotary, stay as they are)
            p.data = p.data.half()
    m.config.torch_dtype = torch.float16
    if ns.quant_head:
        old = m.lm_head
        m.lm_head = H.quantize_head(old, ns.quant_head, ns.head_bits, dense=False)[0]
        if old.weight.data_ptr() != m.get_input_embeddings().weight.data_ptr():
            old.weight.data = torch.empty(0, dtype=old.weight.dtype, device=old.weight.device)  # something
            # (accelerate's hooks) keeps the module alive: release its 268 MB explicitly
        del old
        H.cleanup()
    H.cleanup()
    print(f"  converted to INT4 + QuantForge in {time.time() - t:.1f} s: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
          f"on the GPU", flush=True)
    torch.cuda.reset_peak_memory_stats()
    ppl_qf = perplexity(m, ids, ns.seqlen)
    print(f"  quantforge INT4    perplexity {ppl_qf:8.3f}   (GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")
    toks_qf = H.generate(m, tok, prompt, ns.new_tokens).cpu()
    n = min(len(toks_bnb), len(toks_qf))
    print(f"  greedy decode: {int((toks_bnb[:n] == toks_qf[:n]).sum())}/{n} tokens identical to the bitsandbytes model")
    print("  bitsandbytes says:", tok.decode(toks_bnb, skip_special_tokens=True).replace("\n", " ")[:200])
    print("  quantforge says:  ", tok.decode(toks_qf, skip_special_tokens=True).replace("\n", " ")[:200])
    speed_qf = H.hf_speed(m, tok, prompt, ns.new_tokens)
    print(f"  decode speed (HF generate, batch 1): bitsandbytes {speed_bnb:.1f} tokens/s, quantforge {speed_qf:.1f} tokens/s")
    try:
        from qwen_decode_graph import run as graph_run
        print(f"  decode speed (CUDA graph): quantforge {graph_run(m, tok, prompt, 48)[0]:.1f} tokens/s")
    except Exception as e:  # noqa: BLE001
        print(f"  (CUDA-graph decode unavailable: {type(e).__name__}: {str(e)[:100]})")


if __name__ == "__main__":
    main()
