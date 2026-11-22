"""`quantforge.gptq_checkpoint`: the tensors written for a quantised layer read back, through the GPTQ reader of the vLLM plugin, as the same weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from quantforge import quant  # noqa: E402
from quantforge.gptq_checkpoint import gptq_tensors, save_gptq  # noqa: E402
from quantforge.nn import QuantLinear  # noqa: E402


def _layer(K=256, N=192, G=64, bias=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, s, z = quant.quantize_groupwise(torch.randn(K, N, generator=g) * 0.05, G)
    return QuantLinear(quant.pack_int4(q), s, z, torch.randn(N, generator=g).half() if bias else None, G, "int4"), q, s, z


@pytest.mark.parametrize("bias", [False, True])
def test_written_tensors_read_back_as_the_same_weights(bias):
    layer, q, s, z = _layer(bias=bias)
    t = gptq_tensors(layer)
    assert t["qweight"].dtype == torch.int32 and t["qweight"].shape == (256 // 8, 192)
    assert t["qzeros"].shape == (256 // 64, 192 // 8) and t["g_idx"].shape == (256,)
    packed, scale, zero = quant.from_gptq(t["qweight"], t["qzeros"], t["scales"])
    assert torch.equal(packed, layer.packed) and torch.equal(zero, layer.zero) and torch.equal(scale, layer.scale)
    assert torch.equal(t["g_idx"], (torch.arange(256) // 64).to(torch.int32))
    assert ("bias" in t) == bias


def test_a_model_with_a_dense_decoder_layer_or_fused_parts_is_refused(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(ValueError):
        save_gptq(model, tmp_path)                       # no QuantLinear at all / a dense decoder Linear


def test_act_order_layers_are_refused():
    layer, *_ = _layer()
    layer.perm = torch.arange(256).flip(0)
    with pytest.raises(NotImplementedError):
        gptq_tensors(layer)
