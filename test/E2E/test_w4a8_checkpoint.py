"""W4A8 checkpoints: ``save_w4a8`` writes the quantised tensors, vLLM (``quantization="quantforge_w4a8"``) loads them and computes the same
logits as the Hugging Face model they came from."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402


def _tiny_model(fuse=False):
    transformers = pytest.importorskip("transformers")
    import quantforge.nn as qnn

    cfg = transformers.LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=512, max_position_embeddings=128)
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg).half().eval()
    qnn.quantize_model(m, "w4a8", fuse=fuse)
    return m


def test_save_w4a8_writes_the_quantised_tensors_and_the_config(tmp_path):
    _require_gpu()
    import quantforge.w4a8_checkpoint as ck
    from safetensors.torch import load_file

    m = _tiny_model()
    ck.save_w4a8(m, tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "quantforge_w4a8"
    sd = load_file(str(tmp_path / "model.safetensors"))
    q = "model.layers.0.self_attn.q_proj"
    assert sd[q + ".weight"].dtype == torch.int8 and sd[q + ".weight"].shape == (256, 128)          # [N, K / 2]
    assert sd[q + ".group_scale"].shape == (256, 2) and sd[q + ".group_zero"].shape == (256, 2)
    assert sd[q + ".w_scale"].dtype == torch.float32 and sd[q + ".w_scale"].shape == (256,)
    assert sd["model.embed_tokens.weight"].dtype == torch.float16                                  # the rest stays dense


def test_save_w4a8_refuses_models_vllm_could_not_load(tmp_path):
    _require_gpu()
    import quantforge.w4a8_checkpoint as ck

    with pytest.raises(ValueError, match="fuse=False"):
        ck.save_w4a8(_tiny_model(fuse=True), tmp_path)
    transformers = pytest.importorskip("transformers")
    dense = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, vocab_size=64)).half()
    with pytest.raises(ValueError, match="dense"):
        ck.save_w4a8(dense, tmp_path)


def test_vllm_config_knows_checkpoints():
    pytest.importorskip("vllm")
    import quantforge.vllm_plugin as vp

    cfg = vp.QuantForgeW4A8Config
    assert cfg.from_config({"quant_method": "quantforge_w4a8"}).checkpoint is True
    assert cfg.from_config({}).checkpoint is False and cfg().checkpoint is False


@pytest.mark.skipif(os.environ.get("QUANTFORGE_SLOW_TESTS") != "1", reason="slow test: set QUANTFORGE_SLOW_TESTS=1")
@pytest.mark.parametrize("mode", ["w4a16", "compact"])
def test_vllm_loads_a_w4a8_checkpoint_and_matches_the_huggingface_logits(tmp_path, mode):
    _require_gpu()
    pytest.importorskip("vllm")
    import subprocess

    import quantforge.w4a8_checkpoint as ck

    m = _tiny_model()
    ck.save_w4a8(m, tmp_path)
    ids = [5, 17, 99, 3, 250, 8, 41, 7, 63, 12, 300, 9, 77, 21, 140, 33, 2, 6, 88, 120]            # 20 tokens: the W4A8 kernel path of the layers
    with torch.no_grad():
        lp = torch.log_softmax(m.cuda()(torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
    top = lp.topk(3)
    code = (
        "import json, sys\nfrom vllm import LLM, SamplingParams\n"
        f"llm = LLM('{tmp_path}', quantization='quantforge_w4a8', dtype='float16', gpu_memory_utilization=0.4, max_model_len=128, "
        "skip_tokenizer_init=True, enforce_eager=True)\n"
        f"r = llm.generate([{{'prompt_token_ids': {ids}}}], SamplingParams(temperature=0.0, max_tokens=1, logprobs=3))\n"
        "print('LP:' + json.dumps({str(t): v.logprob for t, v in r[0].outputs[0].logprobs[0].items()}))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "QUANTFORGE_W4A8_DECODE": mode})
    assert "LP:" in r.stdout, r.stderr[-2000:]
    got = json.loads(r.stdout.split("LP:")[1].splitlines()[0])
    assert str(top.indices[0].item()) in got
    for t, v in zip(top.indices.tolist(), top.values.tolist()):
        if str(t) in got:
            assert abs(got[str(t)] - v) < 0.05, (t, got[str(t)], v)
