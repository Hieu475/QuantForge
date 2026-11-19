"""quantforge.gptq: the weight-level algorithm beats round-to-nearest on the calibration distribution, and a
tiny model goes through the whole sequential pipeline."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402


def _layer_problem(seed=0, rows=256, cols=512, tokens=4096):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mix = torch.randn(cols, cols, device="cuda", generator=g) / cols ** 0.5     # correlated inputs
    x = torch.randn(tokens, cols, device="cuda", generator=g) @ mix
    x[:, :8] *= 6.0                                                              # a few outlier channels
    w = torch.randn(rows, cols, device="cuda", generator=g) * 0.05
    return w, x, 2.0 * x.t() @ x / tokens


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_beats_rtn_on_the_layer_output_error(act_order):
    _require_gpu()
    from quantforge import quant
    import quantforge.gptq as qg

    w, x, H = _layer_problem()
    q, scale, zero, perm = qg.gptq_quantize_weight(w, H, 128, act_order=act_order)
    assert q.shape == (512, 256) and scale.shape == (4, 256) and zero.shape == (4, 256)
    assert q.min() >= 0 and q.max() <= 15 and (perm is not None) == act_order
    deq = (q.float() - zero.float().repeat_interleave(128, 0)) * scale.float().repeat_interleave(128, 0)   # [in, out]
    xin = x[:, perm] if perm is not None else x
    err_gptq = ((xin @ deq - x @ w.t()) ** 2).mean().item()
    qr, sr, zr = quant.quantize_groupwise(w.t().contiguous(), 128)
    deq_r = (qr.float() - zr.float().repeat_interleave(128, 0)) * sr.float().repeat_interleave(128, 0)
    err_rtn = ((x @ deq_r - x @ w.t()) ** 2).mean().item()
    assert err_gptq < 0.8 * err_rtn, (err_gptq, err_rtn)                        # clearly better, not just equal


def test_gptq_on_a_tiny_llama_end_to_end():
    transformers = pytest.importorskip("transformers")
    _require_gpu()
    import quantforge.nn as qnn
    import quantforge.gptq as qg

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = transformers.LlamaForCausalLM(cfg).half().eval()
    ids = torch.randint(0, 512, (16, 64))
    dense = copy.deepcopy(model).cuda()
    st = qg.quantize_model_gptq(model, ids, group_size=64, verbose=False)
    assert st["quantized"] == 2 * 7 and st["fused_qkv"] == 2 and st["fused_mlp"] == 2
    assert all(next(model.parameters()).is_cuda for _ in (0,)) and not any(
        isinstance(m, torch.nn.Linear) and "layers" in n for n, m in model.named_modules())
    x = torch.randint(0, 512, (2, 32), device="cuda")
    with torch.no_grad():
        d, q = dense(x).logits.float(), model(x).logits.float()
    assert torch.isfinite(q).all()
    assert (q - d).abs().mean().item() < 0.2 * d.abs().mean().item()            # close to the fp16 model
    # act-order variant works too (permuted layers stay unfused)
    model2 = copy.deepcopy(dense).cpu()
    st2 = qg.quantize_model_gptq(model2, ids, group_size=64, act_order=True, verbose=False)
    assert st2["act_order"] == 14 and st2["fused_qkv"] == 0
    with torch.no_grad():
        assert torch.isfinite(model2(x).logits).all()
