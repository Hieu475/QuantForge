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


def _fake_checkpoint(model, method, act_order, G=64, seed=1):
    """Quantise every decoder Linear with RTN, write the result in GPTQ / AWQ checkpoint layout (with a
    scattered g_idx for the layers selected by act_order) and return (tensors, dense reference model)."""
    from quantforge import quant

    g = torch.Generator().manual_seed(seed)
    tensors, ref = {}, copy.deepcopy(model)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name == "lm_head":
            continue
        w = mod.weight.data.t().float().contiguous()                 # [K, N]
        K, N = w.shape
        q, scale, zero = quant.quantize_groupwise(w, G)
        deq = (q.float() - zero.float().repeat_interleave(G, 0)) * scale.float().repeat_interleave(G, 0)
        if method == "awq":
            qw, qz, sc = quant.to_awq(q, scale, zero)
        else:
            if act_order(name):
                perm = torch.randperm(K, generator=g)                # row perm[i] of the file holds row i
                q_file, deq_file = torch.empty_like(q), torch.empty_like(deq)
                q_file[perm], deq_file[perm] = q, deq
                gi = torch.empty(K, dtype=torch.int32)
                gi[perm] = (torch.arange(K) // G).to(torch.int32)
                tensors[f"{name}.g_idx"] = gi
                q, deq = q_file, deq_file
            else:
                tensors[f"{name}.g_idx"] = (torch.arange(K) // G).to(torch.int32)
            qw, qz, sc = quant.to_gptq(q, scale, zero)
        tensors.update({f"{name}.qweight": qw, f"{name}.qzeros": qz, f"{name}.scales": sc})
        if mod.bias is not None:
            tensors[f"{name}.bias"] = mod.bias.data.clone()
        ref.get_submodule(name).weight.data = deq.half().t().contiguous()
    return tensors, ref


@pytest.mark.parametrize("method,act_order", [("gptq", False), ("gptq", True), ("awq", False)])
def test_load_gptq_awq_checkpoint(method, act_order):
    _require_gpu()
    import quantforge.nn as qnn

    model = _tiny("qwen2")                                           # with q / k / v biases
    tensors, ref = _fake_checkpoint(model, method, lambda n: act_order and "down_proj" in n or
                                    act_order and "q_proj" in n)
    st = qnn.load_quantized_checkpoint(model, tensors, method)
    assert st["quantized"] == 2 * 7
    assert st["act_order"] == (4 if act_order else 0)               # q_proj and down_proj of 2 layers
    # the q / k / v of a block with an act-order q_proj stay unfused (different input permutations)
    assert st["fused_qkv"] == (0 if act_order else 2)
    ids = torch.randint(0, 512, (2, 24), device="cuda")
    with torch.no_grad():
        r, f = ref.cuda()(ids).logits.float(), model(ids).logits.float()
    assert (f - r).abs().max().item() < 5e-3 * r.abs().max().item() + 5e-3
    if act_order:
        lin = model.model.layers[0].mlp
        assert any(isinstance(m, qnn.QuantLinear) and m.perm is not None for m in model.modules())
    with pytest.raises(ValueError):
        qnn.load_quantized_checkpoint(_tiny("qwen2"), {"nothing": torch.zeros(1)}, method)


@pytest.mark.parametrize("fmt", ["int4", "fp8"])
def test_phi3_style_stacked_gate_up_is_fused(fmt):
    transformers = pytest.importorskip("transformers")
    _require_gpu()
    import quantforge.nn as qnn

    torch.manual_seed(0)
    cfg = transformers.Phi3Config(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                                  original_max_position_embeddings=128, pad_token_id=0)
    model = transformers.Phi3ForCausalLM(cfg).half().eval()
    ids = torch.randint(0, 512, (2, 24), device="cuda")
    unfused = copy.deepcopy(model)
    qnn.quantize_model(unfused, fmt, 64, fuse=False)
    fused = copy.deepcopy(model)
    st = qnn.quantize_model(fused, fmt, 64, fuse=True)
    assert st["fused_mlp"] == 2 and not hasattr(fused.model.layers[0].mlp, "gate_up_proj")
    with torch.no_grad():
        u, f = unfused(ids).logits.float(), fused(ids).logits.float()
    assert (f - u).abs().max().item() < 5e-3 * u.abs().max().item() + 5e-3


def test_load_fp8_per_channel_checkpoint():
    """compressed-tensors style FP8 file: `<layer>.weight` float8_e4m3fn [out, in] + `<layer>.weight_scale` [out, 1]."""
    _require_gpu()
    import quantforge.nn as qnn

    model = _tiny("qwen2")
    tensors, ref = {}, copy.deepcopy(model)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name == "lm_head":
            continue
        w = mod.weight.data.float()                                           # [out, in]
        scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-8)   # per output channel
        w8 = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"] = w8, scale.to(torch.bfloat16)
        if mod.bias is not None:
            tensors[f"{name}.bias"] = mod.bias.data.clone()
        ref.get_submodule(name).weight.data = (w8.float() * scale.to(torch.bfloat16).float()).half()
    st = qnn.load_quantized_checkpoint(model, tensors, "fp8")
    assert st["quantized"] == 2 * 7 and st["fused_qkv"] == 2
    ids = torch.randint(0, 512, (2, 24), device="cuda")
    with torch.no_grad():
        r, f = ref.cuda()(ids).logits.float(), model(ids).logits.float()
    assert (f - r).abs().max().item() < 5e-3 * r.abs().max().item() + 5e-3
