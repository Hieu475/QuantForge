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


def test_k_major_scales_match_the_n_major_kernels_decode_and_int8():
    """The compact layer keeps the integer scales / zero points as [K/128, N] too for the INT8 kernel (> 16 rows); the decode kernel (<= 16 rows) keeps reading the
    [N, K/128] ones (a K-major variant of it measured no faster): same results either way."""
    _require_gpu()
    from quantforge import quant, w4a8_op

    K, N = 512, 256
    g = torch.Generator(device="cuda").manual_seed(11)
    w = torch.randn(N, K, device="cuda", generator=g) * 0.03
    p8, gs, zp, sw = quant.w4a8_quantize(w)
    e = torch.empty(0, device="cuda", dtype=torch.float16)
    snk = (gs.float() * sw[:, None]).half()
    for M in (1, 5, 16, 96):
        x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
        bias = torch.randn(N, device="cuda", dtype=torch.float16, generator=g) if M == 5 else None
        n_major = w4a8_op._impl(x, p8, gs, zp, sw, bias, e, e, e, e, e, e, 128, snk)
        k_major = w4a8_op._impl(x, p8, gs, zp, sw, bias, e, quant.kmajor_scales(gs), quant.kmajor_scales(zp), e, e, e, 128, snk)
        assert torch.equal(n_major, k_major), M
        ref = x.float() @ quant.w4a8_dequantize(p8, gs, zp, sw).float().t() + (0 if bias is None else bias.float())
        assert (k_major.float() - ref).abs().max() < 0.05 * ref.abs().max() + 1e-2, M


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


@pytest.mark.parametrize("M,Mp,K", [(1, 1, 64), (5, 16, 128), (33, 64, 896), (64, 64, 1536), (100, 128, 4864)])
def test_native_activation_quantisation_is_bit_identical_to_the_eager_torch_expression(M, Mp, K):
    """`quantize_rows` (runtime/act_quant.cu) replaces `nn._int8_quantize_tokens` + the two padding copies: the int8 values and scales of the *eager* torch expression
    (IEEE division, round half to even), zero rows past M. (The torch.compile'd version multiplies by a reciprocal: it differs by one step on exact .5 ties and in the last bit of a scale.)"""
    _require_gpu()
    from quantforge import w4a8_op
    if not w4a8_op._native_quant():
        pytest.skip("the C++ runtime has no quantize_rows op")
    g = torch.Generator(device="cuda").manual_seed(M * 7 + K)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32, generator=g).mul(torch.rand(M, 1, device="cuda", generator=g) * 20).half()
    x[0, :4] = 0                                                   # a quarter-step tie and zeros
    if M > 2:
        x[1] = 0                                                   # an all-zero row: the scale floor 1e-8
    want_s = (x.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-8)
    want_q = (x.float() / want_s).round().clamp(-127, 127).to(torch.int8)
    xq, sx = torch.ops.quantforge.quantize_rows(x, Mp)
    assert xq.shape == (Mp, K) and sx.shape == (Mp,)
    assert torch.equal(xq[:M], want_q) and torch.equal(sx[:M], want_s.reshape(-1))
    assert not xq[M:].any() and not sx[M:].any()


def test_hadamard_rotation_kernel_matches_the_matrix_product_and_is_orthonormal():
    """`quantforge.hadamard128` (one warp per 128-block) == x @ H per block in f32 (rounded to f16); applying it twice gives x back."""
    _require_gpu()
    from quantforge import torch_ops, w4a8_op  # noqa: F401

    if not hasattr(torch.ops.quantforge, "hadamard128"):
        pytest.skip("the C++ runtime is not built")
    g = torch.Generator(device="cuda").manual_seed(0)
    for M, K in ((1, 128), (7, 384), (130, 1536)):
        x = (torch.randn(M, K, device="cuda", generator=g) * 3).half()
        y = w4a8_op.rotate(x)
        H = w4a8_op.hadamard_matrix(128, "cuda")
        ref = (x.float().reshape(M, K // 128, 128) @ H).reshape(M, K)
        assert (y.float() - ref).abs().max().item() < 2e-3 * ref.abs().max().item(), (M, K)
        back = w4a8_op.rotate(y)
        assert (back.float() - x.float()).abs().max().item() < 4e-3 * x.abs().max().item(), (M, K)


def test_fused_rotation_quantisation_is_bit_identical_to_rotate_then_quantize():
    """`quantize_rows_rot` (rotation + per-token INT8 in one kernel) == `quantize_rows(hadamard128(x))`: same int8 values, same scales."""
    _require_gpu()
    from quantforge import torch_ops, w4a8_op  # noqa: F401

    if not hasattr(torch.ops.quantforge, "quantize_rows_rot"):
        pytest.skip("the C++ runtime is not built")
    g = torch.Generator(device="cuda").manual_seed(1)
    for M, Mp, K in ((5, 8, 128), (70, 128, 1536), (130, 192, 8960)):
        x = (torch.randn(M, K, device="cuda", generator=g) * 2).half()
        x[0, 3] = 900.0                                                    # an outlier row
        xq, s = torch.ops.quantforge.quantize_rows_rot(x, Mp)
        rq, rs = torch.ops.quantforge.quantize_rows(w4a8_op.rotate(x), Mp)
        assert torch.equal(xq, rq) and torch.equal(s, rs), (M, K)


def test_hybrid_with_rotation_matches_rotating_outside_the_op():
    """`w4a8_hybrid(..., rotate=True)` on the weights rotated by `rotate_weight` == the op on `rotate(x)` with rotate=False, at decode, small-batch and INT8 row counts."""
    _require_gpu()
    from quantforge import kfold, quant, torch_ops, w4a8_op  # noqa: F401

    K, N = 1536, 1024
    g = torch.Generator(device="cuda").manual_seed(2)
    W = torch.randn(N, K, device="cuda", generator=g) * 0.05
    p8, gs, zp, sw = quant.w4a8_quantize(w4a8_op.rotate_weight(W))
    e = kfold.no_fold(W.device)
    args = (p8, gs, zp, sw.float(), None, e[0], quant.kmajor_scales(gs), quant.kmajor_scales(zp), e[0], e[1], e[2], 128, (gs.float() * sw.float()[:, None]).half())
    for M in (1, 8, 40, 100, 300):
        x = torch.randn(M, K, device="cuda", generator=g).half()
        a = torch.ops.quantforge.w4a8_hybrid(x, *args, True)
        b = torch.ops.quantforge.w4a8_hybrid(w4a8_op.rotate(x), *args, False)
        assert torch.equal(a, b), M
