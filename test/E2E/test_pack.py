"""
Kernel packs (`quantforge.pack`): a machine without the toolchain runs the kernels a pack holds, and a shape the pack lacks
falls back to a dense torch path instead of failing.
"""
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _group_inputs, _require_gpu  # noqa: E402
from quantforge import e2e  # noqa: E402
from quantforge.e2e import KernelUnavailable, W4A16Gemm, dequant_reference, pack_int4  # noqa: E402


def test_pack_key_does_not_depend_on_the_toolchain_and_changes_with_everything_else():
    a = e2e.pack_key("ir", "sm_80", "chip=sm_80 x=1")
    assert a == e2e.pack_key("ir", "sm_80", "chip=sm_80 x=1")
    assert len({a, e2e.pack_key("ir2", "sm_80", "chip=sm_80 x=1"), e2e.pack_key("ir", "sm_89", "chip=sm_80 x=1"),
                e2e.pack_key("ir", "sm_80", "chip=sm_80 x=2")}) == 4


def test_recorded_kernel_is_served_from_the_pack_without_the_toolchain(tmp_path, monkeypatch):
    _require_gpu()
    opts = "block-tile-m=16 block-tile-n=64 warp-tile-m=16 warp-tile-n=16 block-tile-k=64"
    M, K, N, G = 16, 256, 192, 64
    monkeypatch.setenv("QUANTFORGE_PACK_RECORD", str(tmp_path))
    first = W4A16Gemm.build(M, K, N, group_size=G, pad=False, pipeline_opts=opts, accumulate=False)
    recorded = list((tmp_path / "kernels").glob("*.so"))
    assert len(recorded) == 1

    monkeypatch.delenv("QUANTFORGE_PACK_RECORD")
    monkeypatch.setenv("QUANTFORGE_NO_JIT", "1")                 # no toolchain from here on
    monkeypatch.setenv("QUANTFORGE_PACK", str(tmp_path))
    monkeypatch.setenv("QUANTFORGE_KERNEL_CACHE", str(tmp_path / "no-cache"))
    second = W4A16Gemm.build(M, K, N, group_size=G, pad=False, pipeline_opts=opts, accumulate=False)
    assert second.compiled.cached and second.compiled.lib_path == recorded[0]

    act, q, scale, zp = _group_inputs(M, K, N, G, 0)
    ref = act.float() @ dequant_reference(q, scale, zp).float()
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    second.launch(act, pack_int4(q), scale, zp, out)
    assert (out - ref).abs().max().item() < 1e-2 * ref.abs().max().item()
    # a different shape is not in the pack: a clear error, not an obscure FileNotFoundError
    with pytest.raises(KernelUnavailable):
        W4A16Gemm.build(M, K, N + 64, group_size=G, pad=False, pipeline_opts=opts, accumulate=False)


@pytest.mark.parametrize("activation", ["none", "relu", "silu", "gelu", "swiglu"])
def test_dense_fallback_matches_the_kernel(activation):
    _require_gpu()
    import quantforge.torch_ops as qt

    M, K, N, G = 5, 256, 128, 64
    act, q, scale, zp = _group_inputs(M, K, N, G, 1)
    packed = pack_int4(q)
    bias = None if activation == "swiglu" else torch.randn(N, device="cuda", dtype=torch.float16)
    want = qt.w4a16_linear(act, packed, scale, zp, G, bias, activation)
    got = qt._dense_impl(act, packed, scale, zp, G, bias, activation)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert (got.float() - want.float()).abs().max().item() < 2e-2 * want.float().abs().max().item() + 1e-2, activation


def test_op_serves_a_shape_that_is_not_in_any_pack_with_the_dense_fallback(tmp_path, monkeypatch):
    _require_gpu()
    import quantforge.torch_ops as qt

    monkeypatch.setenv("QUANTFORGE_NO_JIT", "1")
    monkeypatch.setenv("QUANTFORGE_PACK", str(tmp_path))
    monkeypatch.setenv("QUANTFORGE_KERNEL_CACHE", str(tmp_path / "no-cache"))
    M, K, N, G = 3, 448, 576, 64                                 # a shape no other test uses
    act, q, scale, zp = _group_inputs(M, K, N, G, 2)
    ref = act.float() @ dequant_reference(q, scale, zp).float()
    before = qt.DENSE_FALLBACKS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # the one-time "slow dense fallback" warning
        out = qt.w4a16_linear(act, pack_int4(q), scale, zp, G)
    assert qt.DENSE_FALLBACKS > before
    assert (out.float() - ref).abs().max().item() < 2e-2 * ref.abs().max().item() + 1e-2


@pytest.mark.skipif(os.environ.get("QUANTFORGE_SLOW_TESTS") != "1", reason="slow test: set QUANTFORGE_SLOW_TESTS=1")
def test_build_ext_and_check_roundtrip_on_a_tiny_model(tmp_path):
    """`pack build --w4a8` (plans in one process, kernels recorded in a fresh one), `pack ext`, then `pack check --w4a8` with the JIT off: no dense fallback, no W4A8 fallback.

    Regression: kernels the planners build while recording is paused stay in the process caches, so a single-pass recording missed them."""
    _require_gpu()
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_type": "llama", "hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 4,
                               "intermediate_size": 1536, "head_dim": 64}))
    env = dict(os.environ, QUANTFORGE_TUNE_CACHE=str(tmp_path / "tune.json"), QUANTFORGE_KERNEL_CACHE=str(tmp_path / "kcache"))
    run = lambda *a: subprocess.run([sys.executable, "-m", "quantforge.pack", *a], env=env, capture_output=True, text=True)  # noqa: E731
    out = tmp_path / "pack"
    r = run("build", "--out", str(out), "--model", str(cfg), "--w4a8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(list((out / "kernels").glob("*.so"))) > 20 and (out / "plans.json").exists()
    r = run("ext", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    r = run("check", str(out), "--model", str(cfg), "--max-rows", "40", "--w4a8")
    assert r.returncode == 0, r.stdout + r.stderr


def test_w4a8_op_falls_back_when_its_kernel_is_unavailable(tmp_path, monkeypatch):
    """No W4A8 kernel in any pack and no toolchain: the hybrid op answers through the W4A16 layout (dual) or a dense torch matmul (compact), correctly."""
    _require_gpu()
    from quantforge import quant, w4a8_op

    monkeypatch.setenv("QUANTFORGE_NO_JIT", "1")
    monkeypatch.setenv("QUANTFORGE_PACK", str(tmp_path))
    monkeypatch.setenv("QUANTFORGE_KERNEL_CACHE", str(tmp_path / "no-cache"))
    K, N = 384, 640                                              # a shape no pack and no other test has
    g = torch.Generator(device="cuda").manual_seed(5)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.03
    p8, gs, zp, sw = quant.w4a8_quantize(w)
    p16, s16, z16 = w4a8_op.from_w4a8(p8, gs, zp, sw)
    ref_w = quant.w4a8_dequantize(p8, gs, zp, sw).half()
    empty = torch.empty(0, device="cuda", dtype=torch.float16)
    before = w4a8_op.FALLBACKS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for M, compact in ((200, False), (200, True), (5, True)):
            x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
            args = (empty, empty, empty, empty, empty, empty) if compact else (p16, s16, z16, empty, empty, empty)
            scale_nk = (gs.float() * sw[:, None]).half() if compact else empty
            y = w4a8_op._impl(x, p8, gs, zp, sw, None, *args, 128, scale_nk)
            ref = x.float() @ ref_w.float().t()
            assert (y.float() - ref).abs().max() < 0.03 * ref.abs().max(), (M, compact)
    assert w4a8_op.FALLBACKS >= before + 3


def test_moe_grouped_kernels_are_served_from_a_pack_without_the_toolchain(tmp_path, monkeypatch):
    """`moe.prepare` records the grouped kernels of a mixture-of-experts layer; with the JIT off the same op runs from the pack."""
    _require_gpu()
    from quantforge import moe
    from test_moe_grouped import _moe_weights

    E, H, I, k, G = 4, 256, 256, 2, 64
    w = _moe_weights(E, H, I, G, 3)
    monkeypatch.setenv("QUANTFORGE_PACK_RECORD", str(tmp_path))
    moe.prepare(E, H, I, k, G, max_tokens=8)
    assert len(list((tmp_path / "kernels").glob("*.so"))) >= 4
    monkeypatch.delenv("QUANTFORGE_PACK_RECORD")
    monkeypatch.setenv("QUANTFORGE_NO_JIT", "1")
    monkeypatch.setenv("QUANTFORGE_PACK", str(tmp_path))
    monkeypatch.setenv("QUANTFORGE_KERNEL_CACHE", str(tmp_path / "no-cache"))
    moe._kernels.clear()
    g = torch.Generator(device="cuda").manual_seed(4)
    x = (torch.randn(3, H, device="cuda", generator=g) * 0.5).half()
    tw, ids = torch.topk(torch.softmax(torch.randn(3, E, device="cuda", generator=g), -1), k, dim=-1)
    got = moe.moe_ffn_impl(x, ids, tw.float(), *w, G)
    ref = moe.moe_ffn_reference(x, ids, tw.float(), *w, G)
    assert (got.float() - ref.float()).abs().max().item() < 2e-2 * ref.float().abs().max().item() + 2e-3
    moe._kernels.clear()
