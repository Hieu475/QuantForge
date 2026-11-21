"""``torch.ops.quantforge.w4a8_hybrid``: W4A8 above 16 rows, the W4A16 decode kernels below, from the same weight; and the vLLM method
``quantforge_w4a8`` (online quantisation of a plain checkpoint)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402


def _layer(N=256, K=512, seed=0):
    from quantforge import kfold, quant, w4a8_op

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.05
    packed, gs, zp, sw = quant.w4a8_quantize(w)
    p16, s16, z16 = w4a8_op.from_w4a8(packed, gs, zp, sw)
    folded = kfold.no_fold(w.device)
    return w, (packed, gs, zp, sw), (p16, s16, z16), folded, g


def test_the_w4a16_layout_dequantises_to_the_same_weight():
    _require_gpu()
    from quantforge import quant

    w, (packed, gs, zp, sw), (p16, s16, z16), _, _ = _layer()
    ref = quant.w4a8_dequantize(packed, gs, zp, sw)                                   # [N, K]
    q = quant.unpack_int4(p16).float()                                                # [K, N]
    deq16 = ((q - z16.float().repeat_interleave(128, 0)) * s16.float().repeat_interleave(128, 0)).t()
    assert (deq16 - ref).abs().max() <= 1e-3 * ref.abs().max()                        # only gs * sw is rounded to f16


@pytest.mark.parametrize("rows", [1, 8, 16, 17, 40, 200])
@pytest.mark.parametrize("bias", [False, True])
def test_hybrid_op_matches_the_dequantised_reference_on_both_paths(rows, bias):
    _require_gpu()
    from quantforge import quant

    w, (packed, gs, zp, sw), (p16, s16, z16), (pf, sf, zf), g = _layer()
    b = torch.randn(256, device="cuda", dtype=torch.float16, generator=g) if bias else None
    x = torch.randn(rows, 512, device="cuda", dtype=torch.float16, generator=g)
    y = torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw, b, p16, s16, z16, pf, sf, zf, 128)
    assert y.shape == (rows, 256) and y.dtype == torch.float16
    ref = x.float() @ quant.w4a8_dequantize(packed, gs, zp, sw).t()
    if bias:
        ref = ref + b.float()
    tol = 0.03 if rows > 16 else 0.01                                                  # the activation rounding only exists above 16 rows
    assert (y.float() - ref).norm() / ref.norm() < tol, rows
    y3 = torch.ops.quantforge.w4a8_hybrid(x.reshape(1, rows, 512), packed, gs, zp, sw, b, p16, s16, z16, pf, sf, zf, 128)
    assert y3.shape == (1, rows, 256)


def test_hybrid_op_traces_under_torch_compile():
    _require_gpu()
    w, (packed, gs, zp, sw), (p16, s16, z16), (pf, sf, zf), g = _layer()

    @torch.compile(fullgraph=True, dynamic=True)
    def f(x):
        return torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw, None, p16, s16, z16, pf, sf, zf, 128) + 1

    for rows in (4, 64):
        x = torch.randn(rows, 512, device="cuda", dtype=torch.float16, generator=g)
        eager = torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw, None, p16, s16, z16, pf, sf, zf, 128) + 1
        assert (f(x).float() - eager.float()).abs().max() < 2e-2


def test_vllm_w4a8_method_is_registered():
    pytest.importorskip("vllm")
    import quantforge.vllm_plugin as vp
    from vllm.model_executor.layers.quantization import get_quantization_config

    vp.register()
    cfg = get_quantization_config("quantforge_w4a8")
    assert cfg is vp.QuantForgeW4A8Config and cfg.get_name() == "quantforge_w4a8" and cfg.get_config_filenames() == []
    assert isinstance(cfg.from_config({}), cfg)


@pytest.mark.skipif(os.environ.get("QUANTFORGE_SLOW_TESTS") != "1", reason="slow test: set QUANTFORGE_SLOW_TESTS=1")
def test_vllm_generates_sensible_text_with_online_w4a8():
    pytest.importorskip("vllm")
    import subprocess

    code = (
        "from vllm import LLM, SamplingParams\n"
        "llm = LLM('Qwen/Qwen2.5-0.5B-Instruct', quantization='quantforge_w4a8', dtype='float16', gpu_memory_utilization=0.5, max_model_len=512)\n"
        "ps = ['What is the capital of France?'] + ['Count: ' + ' '.join(str(i) for i in range(60))] * 20\n"      # one short and 20 long prompts: both paths
        "out = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=24))\n"
        "print('TEXT:' + out[0].outputs[0].text.replace(chr(10), ' '))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "TEXT:" in r.stdout, r.stderr[-1500:]
    assert "Paris" in r.stdout.split("TEXT:")[1]
