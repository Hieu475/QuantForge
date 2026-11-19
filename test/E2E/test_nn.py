"""quantforge.nn: quantise the Linear layers of a (tiny, random) Hugging Face Llama / Qwen2 model."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402


def _tiny(kind="llama"):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    common = dict(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    if kind == "llama":
        cfg = transformers.LlamaConfig(**common)
        return transformers.LlamaForCausalLM(cfg).half().eval()
    cfg = transformers.Qwen2Config(**common)                    # q / k / v have biases
    return transformers.Qwen2ForCausalLM(cfg).half().eval()


def _dense_reference(dense, quantised):
    """A copy of the dense model whose Linear weights are the dequantised ones of the quantised model."""
    ref = copy.deepcopy(dense)
    for name, mod in quantised.named_modules():
        if hasattr(mod, "dequantized_weight"):
            lin = ref.get_submodule(name)
            lin.weight.data = mod.dequantized_weight().to(lin.weight.device)
    return ref


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
@pytest.mark.parametrize("fmt", ["int4", "int8", "fp8"])
def test_quantize_model_matches_dequantised_reference(kind, fmt):
    _require_gpu()
    import quantforge.nn as qnn

    model = _tiny(kind)
    ids = torch.randint(0, 512, (2, 24), device="cuda")
    unfused = copy.deepcopy(model)
    st = qnn.quantize_model(unfused, fmt, group_size=64, fuse=False)
    assert st["quantized"] == 2 * 7 and st["fused_qkv"] == 0 and "lm_head" not in str(st["skipped"])
    assert isinstance(unfused.lm_head, nn.Linear)                       # the head stays dense by default
    ref = _dense_reference(model, unfused).cuda()
    fused = copy.deepcopy(model)
    st = qnn.quantize_model(fused, fmt, group_size=64, fuse=True)
    assert st["fused_qkv"] == 2 and st["fused_mlp"] == 2
    assert st["bytes_after"] < st["bytes_before"] * (0.6 if fmt != "int4" else 0.35)
    with torch.no_grad():
        r, u, f = (m(ids).logits.float() for m in (ref, unfused, fused))
    scale = r.abs().max().item()
    assert (u - r).abs().max().item() < 5e-3 * scale + 5e-3, "quantised model must match its dense twin"
    assert (f - u).abs().max().item() < 5e-3 * scale + 5e-3, "fusion must not change the result"
    assert (f.argmax(-1) == r.argmax(-1)).float().mean() > 0.95
    # a second call (new input tensor) must not reuse the fused QKV result of the first
    ids2 = torch.randint(0, 512, (2, 24), device="cuda")
    with torch.no_grad():
        f2, r2 = fused(ids2).logits.float(), ref(ids2).logits.float()
    assert (f2 - r2).abs().max().item() < 5e-3 * r2.abs().max().item() + 5e-3


def test_quantize_head_only_and_group_fallback_and_generate():
    _require_gpu()
    import quantforge.nn as qnn

    model = _tiny("llama")
    ids = torch.randint(0, 512, (1, 16), device="cuda")
    model = model.cuda()
    with torch.no_grad():
        base = model(ids).logits.float()
    st = qnn.quantize_model(model, "int8", group_size=128, only=("lm_head",))
    assert st["quantized"] == 1 and isinstance(model.lm_head, qnn.QuantLinear)
    assert model.lm_head.group_size == 128 and model.lm_head.fmt == "int8"
    with torch.no_grad():
        q = model(ids).logits.float()
    assert (q - base).abs().max().item() < 0.02 * base.abs().max().item()        # INT8 head: near lossless
    out = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert out.shape == (1, 20)
    # in_features 96 is not a multiple of 128: the group size falls back to 32
    lin = nn.Linear(96, 64).half()
    assert qnn.QuantLinear.from_linear(lin, "int4", 128).group_size == 32
    with pytest.raises(ValueError):
        qnn.QuantLinear.from_linear(nn.Linear(97, 64).half(), "int4", 128)


def test_quantized_model_under_torch_compile_and_chunked_quantisation():
    _require_gpu()
    import quantforge.nn as qnn

    model = _tiny("llama")
    qnn.quantize_model(model, "int4", 64)
    cm = torch.compile(model)
    with torch.no_grad():
        for _ in range(2):                                   # fresh inputs: the fused QKV must not be stale
            x = torch.randint(0, 512, (2, 24), device="cuda")
            assert (cm(x).logits.float() - model(x).logits.float()).abs().max().item() < 5e-3
    # a layer with many rows is quantised in chunks: identical to the one-shot quantisation
    lin = nn.Linear(2048, 4096, bias=False).half()
    a = qnn.QuantLinear.from_linear(lin, "int4", 128)
    from quantforge import quant
    q, s, z = quant.quantize_groupwise(lin.weight.data.t().float().cuda().contiguous(), 128)
    assert torch.equal(a.packed, quant.pack_int4(q)) and torch.equal(a.scale, s) and torch.equal(a.zero, z)
