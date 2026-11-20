"""Zero-shot ARC (Easy / Challenge) accuracy of fp16, round-to-nearest and GPTQ models (a task beyond perplexity).

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/arc_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --arc-easy arc_easy_test.parquet --arc-challenge arc_challenge_test.parquet --train wikitext2_train.parquet

Scoring as in lm-evaluation-harness: context "Question: <q>\nAnswer:", continuation " <choice>"; `acc` picks the choice
with the highest log-likelihood, `acc_norm` the highest log-likelihood per character of the choice. Parquet files:
allenai/ai2_arc, ARC-Easy / ARC-Challenge, test split.
"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.gptq as qgptq  # noqa: E402
import quantforge.nn as qnn  # noqa: E402


@torch.no_grad()
def evaluate(model, tok, df, batch=16):
    items = []                                                  # (question index, choice index, ctx ids, cont ids)
    for qi, row in enumerate(df.itertuples()):
        ctx = f"Question: {row.question}\nAnswer:"
        ctx_ids = tok(ctx).input_ids
        for ci, text in enumerate(row.choices["text"]):
            whole = tok(ctx + " " + text).input_ids
            items.append((qi, ci, ctx_ids, whole[len(ctx_ids):], len(text)))
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]) + len(items[i][3]))
    ll = [0.0] * len(items)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        seqs = [items[i][2] + items[i][3] for i in idx]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros(len(idx), L, dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
        logits = model(ids.cuda(), attention_mask=mask.cuda()).logits          # fp16 [B, L, V]
        for r, i in enumerate(idx):
            n_ctx, n_cont = len(items[i][2]), len(items[i][3])
            tgt = ids[r, n_ctx:n_ctx + n_cont].cuda()
            logp = F.log_softmax(logits[r, n_ctx - 1:n_ctx - 1 + n_cont].float(), dim=-1)    # only the scored positions
            ll[i] = logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        del logits
    correct = correct_norm = 0
    labels = [list(r.choices["label"]).index(r.answerKey) for r in df.itertuples()]
    by_q = {}
    for i, it in enumerate(items):
        by_q.setdefault(it[0], []).append((it[1], ll[i], ll[i] / max(it[4], 1)))
    for qi, cs in by_q.items():
        correct += max(cs, key=lambda c: c[1])[0] == labels[qi]
        correct_norm += max(cs, key=lambda c: c[2])[0] == labels[qi]
    return correct / len(df), correct_norm / len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arc-easy", required=True)
    ap.add_argument("--arc-challenge", required=True)
    ap.add_argument("--train", required=True, help="calibration parquet (column text) for GPTQ")
    ap.add_argument("--variants", default="fp16,rtn,gptq")
    ap.add_argument("--limit", type=int, default=0)
    ns = ap.parse_args()
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ns.model)
    sets = {"arc_easy": pd.read_parquet(ns.arc_easy), "arc_challenge": pd.read_parquet(ns.arc_challenge)}
    if ns.limit:
        sets = {k: v.iloc[:ns.limit] for k, v in sets.items()}
    load = lambda: AutoModelForCausalLM.from_pretrained(ns.model, dtype=torch.float16, low_cpu_mem_usage=True).eval()  # noqa: E731
    for variant in ns.variants.split(","):
        m = load()
        if variant == "fp16":
            m = m.cuda()
        elif variant == "rtn":
            qnn.quantize_model(m, "int4", 128); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
        elif variant == "gptq":
            calib = qgptq.calibration_ids(tok, "\n\n".join(pd.read_parquet(ns.train)["text"]), 128, 512)
            qgptq.quantize_model_gptq(m, calib, 128, verbose=False); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
        res = {k: evaluate(m, tok, v) for k, v in sets.items()}
        print(f"{ns.model} {variant:5s} " + "  ".join(f"{k}: acc {a * 100:.1f} acc_norm {b * 100:.1f}" for k, (a, b) in res.items()), flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
