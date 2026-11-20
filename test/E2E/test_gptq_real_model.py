"""Slow regression test on a real model (skipped unless enabled): GPTQ must not be worse than RTN on windows where it once broke.

    QUANTFORGE_SLOW_TESTS=1 QUANTFORGE_WIKITEXT_TRAIN=<train.parquet> QUANTFORGE_WIKITEXT_TEST=<test.parquet> \
        python -m pytest test/E2E/test_gptq_real_model.py -q

Llama-3.2-1B-Instruct (it must be in the Hugging Face cache): without the token-norm cap GPTQ gave a loss of 8-10 per token on windows
70, 478, 546, 235 and 136 of the Wikitext-2 test split (fp16: ~3), which made its whole-split perplexity worse than round-to-nearest
(24.08 against 22.28). Takes about 3 minutes on an RTX 4050.
"""
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402

MODEL = "unsloth/Llama-3.2-1B-Instruct"
BROKEN_WINDOWS = [70, 478, 546, 235, 136]


@pytest.mark.skipif(os.environ.get("QUANTFORGE_SLOW_TESTS") != "1", reason="slow test: set QUANTFORGE_SLOW_TESTS=1")
def test_gptq_is_not_worse_than_rtn_on_the_windows_where_it_once_broke():
    _require_gpu()
    train, test = os.environ.get("QUANTFORGE_WIKITEXT_TRAIN"), os.environ.get("QUANTFORGE_WIKITEXT_TEST")
    if not (train and test):
        pytest.skip("set QUANTFORGE_WIKITEXT_TRAIN / QUANTFORGE_WIKITEXT_TEST to the Wikitext-2 parquet files")
    pd = pytest.importorskip("pandas")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import quantforge.gptq as qg
    import quantforge.nn as qnn

    try:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        load = lambda: AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, local_files_only=True,  # noqa: E731
                                                            low_cpu_mem_usage=True).eval()
    except OSError:
        pytest.skip(f"{MODEL} is not in the Hugging Face cache")
    ids = tok("\n\n".join(pd.read_parquet(test)["text"]), return_tensors="pt").input_ids[0]
    wins = BROKEN_WINDOWS + list(range(10))
    calib = qg.calibration_ids(tok, "\n\n".join(pd.read_parquet(train)["text"]), 128, 512)

    @torch.no_grad()
    def losses(m):
        out = []
        for w in wins:
            x = ids[w * 512:(w + 1) * 512].unsqueeze(0).cuda()
            out.append(nn.functional.cross_entropy(m(x).logits[0, :-1].float(), x[0, 1:]).item())
        return torch.tensor(out)

    m = load(); qnn.quantize_model(m, "int4", 128); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
    rtn = losses(m); del m; gc.collect(); torch.cuda.empty_cache()
    m = load(); qg.quantize_model_gptq(m, calib, 128, verbose=False); qnn.quantize_model(m, "int8", only=("lm_head",), fuse=False)
    gptq = losses(m)
    assert gptq[:len(BROKEN_WINDOWS)].max() < 5.0, gptq[:len(BROKEN_WINDOWS)]          # was 8.7 - 10.6 without the cap
    assert gptq.mean() <= rtn.mean() + 0.05, (gptq.mean().item(), rtn.mean().item())
