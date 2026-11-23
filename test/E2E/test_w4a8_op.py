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


def _args(compact):
    """(w, W4A8 tensors, the decode tensors the op takes, generator): compact = no W4A16 copy, the decode reads the W4A8 layout."""
    w, (packed, gs, zp, sw), (p16, s16, z16), (pf, sf, zf), g = _layer()
    if compact:
        p16, s16, z16 = pf, sf, zf                                                      # empty placeholders
        snk = (gs.float() * sw[:, None]).half()
    else:
        snk = torch.empty(0, device="cuda", dtype=torch.float16)
    return (packed, gs, zp, sw), (p16, s16, z16, pf, sf, zf), snk, g


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 16, 17, 40, 200])
@pytest.mark.parametrize("bias", [False, True])
def test_hybrid_op_matches_the_dequantised_reference_on_both_paths(rows, bias, compact):
    _require_gpu()
    from quantforge import quant

    (packed, gs, zp, sw), dec, snk, g = _args(compact)
    b = torch.randn(256, device="cuda", dtype=torch.float16, generator=g) if bias else None
    x = torch.randn(rows, 512, device="cuda", dtype=torch.float16, generator=g)
    y = torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw, b, *dec, 128, snk)
    assert y.shape == (rows, 256) and y.dtype == torch.float16
    ref = x.float() @ quant.w4a8_dequantize(packed, gs, zp, sw).t()
    if bias:
        ref = ref + b.float()
    tol = 0.03 if rows > 16 else 0.01                                                  # the activation rounding only exists above 16 rows
    assert (y.float() - ref).norm() / ref.norm() < tol, rows
    y3 = torch.ops.quantforge.w4a8_hybrid(x.reshape(1, rows, 512), packed, gs, zp, sw, b, *dec, 128, snk)
    assert y3.shape == (1, rows, 256)


@pytest.mark.parametrize("compact", [False, True])
def test_hybrid_op_traces_under_torch_compile(compact):
    _require_gpu()
    (packed, gs, zp, sw), dec, snk, g = _args(compact)

    @torch.compile(fullgraph=True, dynamic=True)
    def f(x):
        return torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw, None, *dec, 128, snk) + 1

    for rows in (4, 64):
        x = torch.randn(rows, 512, device="cuda", dtype=torch.float16, generator=g)
        eager = torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw, None, *dec, 128, snk) + 1
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
@pytest.mark.parametrize("mode", ["w4a16", "compact"])
def test_vllm_generates_sensible_text_with_online_w4a8(mode):
    pytest.importorskip("vllm")
    import subprocess

    code = (
        "from vllm import LLM, SamplingParams\n"
        "llm = LLM('Qwen/Qwen2.5-0.5B-Instruct', quantization='quantforge_w4a8', dtype='float16', gpu_memory_utilization=0.5, max_model_len=512)\n"
        "ps = ['What is the capital of France?'] + ['Count: ' + ' '.join(str(i) for i in range(60))] * 20\n"      # one short and 20 long prompts: both paths
        "out = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=24))\n"
        "print('TEXT:' + out[0].outputs[0].text.replace(chr(10), ' '))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "QUANTFORGE_W4A8_DECODE": mode})
    assert "TEXT:" in r.stdout, r.stderr[-1500:]
    assert "Paris" in r.stdout.split("TEXT:")[1]


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_compact_decode_reads_only_its_own_rows(rows):
    """The one-copy decode kernel for exactly 1 / 2 / 4 / 8 rows takes the activations as they are: NaN in the neighbouring rows must not reach the output."""
    _require_gpu()
    from quantforge import quant

    (packed, gs, zp, sw), dec, snk, g = _args(True)
    x = torch.randn(rows, 512, device="cuda", dtype=torch.float16, generator=g)
    buf = torch.full((rows + 16, 512), float("nan"), device="cuda", dtype=torch.float16)
    buf[:rows] = x
    b = torch.randn(256, device="cuda", dtype=torch.float16, generator=g)
    y = torch.ops.quantforge.w4a8_hybrid(buf[:rows], packed, gs, zp, sw, b, *dec, 128, snk)
    ref = x.float() @ quant.w4a8_dequantize(packed, gs, zp, sw).t() + b.float()
    assert torch.isfinite(y).all() and (y.float() - ref).norm() / ref.norm() < 0.01


def test_int8_kernel_with_k_major_scales_matches_the_n_major_one():
    """The compact layer keeps the integer scales / zero points as [K/128, N] too (scale16 / zero16 slots): same result as the [N, K/128] kernel."""
    _require_gpu()
    from quantforge import quant, w4a8_op

    K, N, M = 512, 256, 96
    g = torch.Generator(device="cuda").manual_seed(11)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.03
    p8, gs, zp, sw = quant.w4a8_quantize(w)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
    e = torch.empty(0, device="cuda", dtype=torch.float16)
    scale_nk = (gs.float() * sw[:, None]).half()
    n_major = w4a8_op._impl(x, p8, gs, zp, sw, None, e, e, e, e, e, e, 128, scale_nk)
    k_major = w4a8_op._impl(x, p8, gs, zp, sw, None, e, quant.kmajor_scales(gs), quant.kmajor_scales(zp), e, e, e, 128, scale_nk)
    assert any(key[-1] for key in w4a8_op._KERNELS if isinstance(key[-1], bool) and len(key) == 5)
    assert torch.equal(n_major, k_major)
    ref = x.float() @ quant.w4a8_dequantize(p8, gs, zp, sw).float().t()
    assert (k_major.float() - ref).abs().max() < 0.05 * ref.abs().max()


def test_int8_plan_candidates_are_valid_and_plans_are_looked_up(tmp_path, monkeypatch):
    """The INT8 planner times schedules that divide the shape, keeps the winner in the autotune cache and `_int8_opts` then returns it (K-major layer only)."""
    _require_gpu()
    from quantforge import autotune, int8_plan, w4a8_op

    monkeypatch.setenv("QUANTFORGE_TUNE_CACHE", str(tmp_path / "tune.json"))
    autotune._CACHE = None if hasattr(autotune, "_CACHE") else None
    K, N = 512, 256
    for rows in int8_plan.ROWS:
        for o in int8_plan.candidates(rows, K, N):
            assert f"block-tile-m={rows}" not in o or rows % int(o.split("block-tile-m=")[1].split()[0]) == 0
    rule = w4a8_op._rule_opts(64, K, N)
    choice = int8_plan.plan_int8(64, K, N, rule)
    assert choice is None or choice in int8_plan.candidates(64, K, N)
    assert w4a8_op._int8_opts(64, K, N, kmajor=False) == rule             # the n-major layer keeps the rule
    assert w4a8_op._int8_opts(64, K, N, kmajor=True) == (choice or rule)
