"""
`torch.ops.quantforge.w4a16_gemm`: eager, torch.compile and CUDA-graph use of the generated
kernels, checked against the PyTorch reference.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _group_inputs, _require_gpu  # noqa: E402
from quantforge.e2e import dequant_reference, pack_int4  # noqa: E402


def _case(M, K, N, G, seed=0):
    act, q, scale, zp = _group_inputs(M, K, N, G, seed)
    ref = act.float() @ dequant_reference(q, scale, zp).float()
    return act, pack_int4(q), scale, zp, ref


def _close(out, ref):
    return (out.float() - ref).abs().max().item() < 2e-3 * ref.abs().max().item() + 1e-2  # f16 output


def test_eager_2d_3d_and_bias():
    _require_gpu()
    import quantforge.torch_ops as qt

    for (M, K, N, G) in [(1, 512, 256, 128), (16, 256, 128, 64), (40, 256, 192, 64)]:
        act, packed, scale, zp, ref = _case(M, K, N, G)
        out = qt.w4a16_linear(act, packed, scale, zp, G)
        assert out.shape == (M, N) and out.dtype == torch.float16
        assert _close(out, ref), (M, K, N, G)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        out_b = qt.w4a16_linear(act, packed, scale, zp, G, bias)
        assert _close(out_b, ref + bias.float()), "bias"
    # leading dims are flattened into M
    act, packed, scale, zp, ref = _case(6, 256, 128, 64)
    out3 = qt.w4a16_linear(act.reshape(2, 3, 256), packed, scale, zp, 64)
    assert out3.shape == (2, 3, 128) and _close(out3.reshape(6, 128), ref)


def test_per_tensor_parameters():
    _require_gpu()
    import quantforge.torch_ops as qt

    g = torch.Generator(device="cuda").manual_seed(1)
    act = (torch.rand(4, 256, device="cuda", generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (256, 128), device="cuda", dtype=torch.int8, generator=g)
    scale = torch.tensor(0.05, device="cuda", dtype=torch.float16)
    zp = torch.tensor(8, device="cuda", dtype=torch.int8)
    out = qt.w4a16_linear(act, pack_int4(q), scale, zp, 0)
    assert _close(out, act.float() @ dequant_reference(q, scale, zp).float())


def test_opcheck_schema_and_fake():
    _require_gpu()
    import quantforge.torch_ops as qt  # noqa: F401

    act, packed, scale, zp, _ = _case(8, 256, 128, 64)
    torch.library.opcheck(torch.ops.quantforge.w4a16_gemm, (act, packed, scale, zp, 64),
                          test_utils=["test_schema", "test_faketensor"])


def test_torch_compile_fullgraph():
    _require_gpu()
    import quantforge.torch_ops as qt

    act, packed, scale, zp, ref = _case(4, 512, 256, 128)

    def layer(x):
        return torch.relu(qt.w4a16_linear(x, packed, scale, zp, 128)) * 2

    compiled = torch.compile(layer, fullgraph=True, backend="aot_eager")
    out = compiled(act)
    assert _close(out, torch.relu(ref) * 2)


def test_cuda_graph_capture_and_replay():
    """Warm up (compile + JIT load), capture one call, then replay with new inputs."""
    _require_gpu()
    import quantforge.torch_ops as qt

    M, K, N, G = 1, 512, 256, 128
    act, packed, scale, zp, ref = _case(M, K, N, G)
    static_x = act.clone()
    qt.w4a16_linear(static_x, packed, scale, zp, G)          # warm-up outside the graph
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = qt.w4a16_linear(static_x, packed, scale, zp, G)
    for seed in (0, 1, 2):
        x2, _, _, _, _ = _case(M, K, N, G, seed + 10)
        static_x.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        want = x2.float() @ dequant_reference(*_unpack(packed), scale, zp).float()
        assert _close(static_out, want), seed


def test_activation_epilogue_in_op():
    _require_gpu()
    import torch.nn.functional as F
    import quantforge.torch_ops as qt

    act, packed, scale, zp, ref = _case(2, 256, 128, 64)
    bias = torch.randn(128, device="cuda", dtype=torch.float16)
    pre = ref + bias.float()
    for name, fn in (("relu", F.relu), ("silu", F.silu), ("gelu", lambda t: F.gelu(t, approximate="tanh"))):
        out = qt.w4a16_linear(act, packed, scale, zp, 64, bias, activation=name)
        assert _close(out, fn(pre)), name
    compiled = torch.compile(lambda x: qt.w4a16_linear(x, packed, scale, zp, 64, bias, "silu"),
                             fullgraph=True, backend="aot_eager")
    assert _close(compiled(act), F.silu(pre))


def test_cpp_runtime_is_used_and_matches_python_op():
    """The C++ dispatcher hot path is built and loaded (unless QUANTFORGE_NO_CPP is set) and
    gives exactly the result of the Python implementation, for padded M, bias and 3-D input."""
    _require_gpu()
    import os
    import quantforge.torch_ops as qt

    if os.environ.get("QUANTFORGE_NO_CPP"):
        assert not qt.CPP_RUNTIME
        return
    import shutil
    if not (shutil.which("ninja") and (shutil.which("g++") or shutil.which("c++"))):
        raise unittest.SkipTest("no ninja / C++ compiler: the extension cannot be built here")
    assert qt.CPP_RUNTIME, "the C++ runtime failed to build (see the warning)"
    for (M, K, N, G) in [(1, 512, 256, 128), (7, 256, 128, 64), (64, 256, 192, 64)]:
        act, packed, scale, zp, _ = _case(M, K, N, G)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        for act_name in ("none", "silu"):
            want = qt._python_impl(act, packed, scale, zp, G, bias, act_name)
            got = torch.ops.quantforge.w4a16_gemm(act, packed, scale, zp, G, bias, act_name)
            if act_name == "none":      # the C++ path adds the bias after the f32 accumulation (the Python one starts from it): 1 f16 ulp apart at most
                assert _close(got, want), (M, K, N, act_name)
            else:
                assert torch.equal(got, want), (M, K, N, act_name)
    x3 = _case(6, 256, 128, 64)[0].reshape(2, 3, 256)
    _, packed, scale, zp, _ = _case(6, 256, 128, 64)
    assert torch.ops.quantforge.w4a16_gemm(x3, packed, scale, zp, 64).shape == (2, 3, 128)
    # per-tensor parameters take the Python fallback from inside the C++ op
    g = torch.Generator(device="cuda").manual_seed(2)
    a = (torch.rand(3, 256, device="cuda", generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (256, 128), device="cuda", dtype=torch.int8, generator=g)
    s0, z0 = torch.tensor(0.05, device="cuda", dtype=torch.float16), torch.tensor(8, device="cuda", dtype=torch.int8)
    assert _close(torch.ops.quantforge.w4a16_gemm(a, pack_int4(q), s0, z0, 0),
                  a.float() @ dequant_reference(q, s0, z0).float())


def _gate_up(M, K, N, G):
    """x, interleaved (packed, scale, zp) and the f32 reference silu(x @ Wg) * (x @ Wu)."""
    import torch.nn.functional as F
    from quantforge import quant

    act, qg, sg, zg = _group_inputs(M, K, N, G, seed=1)
    _, qu, su, zu = _group_inputs(M, K, N, G, seed=2)
    packed, scale, zero = quant.interleave_gate_up((pack_int4(qg), sg, zg), (pack_int4(qu), su, zu))
    ref = F.silu(act.float() @ dequant_reference(qg, sg, zg).float()) * (
        act.float() @ dequant_reference(qu, su, zu).float())
    return act, packed, scale, zero, ref


def test_swiglu_op_eager_compile_graph():
    _require_gpu()
    import quantforge.torch_ops as qt

    for (M, K, N, G) in [(1, 512, 256, 128), (9, 256, 128, 64), (64, 256, 192, 64)]:
        act, packed, scale, zero, ref = _gate_up(M, K, N, G)
        out = qt.w4a16_linear(act, packed, scale, zero, G, None, "swiglu")
        assert out.shape == (M, N) and out.dtype == torch.float16, out.shape
        assert _close(out, ref), (M, K, N)
        if not os.environ.get("QUANTFORGE_NO_CPP"):
            assert torch.equal(out, qt._python_impl(act, packed, scale, zero, G, None, "swiglu"))
    act, packed, scale, zero, ref = _gate_up(2, 256, 128, 64)
    x3 = act.reshape(1, 2, 256)
    assert qt.w4a16_linear(x3, packed, scale, zero, 64, None, "swiglu").shape == (1, 2, 128)
    try:
        qt.w4a16_linear(act, packed, scale, zero, 64, torch.zeros(256, device="cuda", dtype=torch.float16),
                        "swiglu")
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("swiglu must reject a bias")
    compiled = torch.compile(lambda x: qt.w4a16_linear(x, packed, scale, zero, 64, None, "swiglu") + 1,
                             fullgraph=True, backend="aot_eager")
    assert _close(compiled(act) - 1, ref)
    static_x = act.clone()
    qt.w4a16_linear(static_x, packed, scale, zero, 64, None, "swiglu")      # warm-up
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = qt.w4a16_linear(static_x, packed, scale, zero, 64, None, "swiglu")
    static_x.copy_(act * 0.5)
    g.replay()
    torch.cuda.synchronize()
    eager = qt.w4a16_linear(static_x, packed, scale, zero, 64, None, "swiglu")
    assert static_out.shape == (2, 128) and torch.equal(static_out, eager)


def test_no_bias_uses_overwrite_kernel_and_matches_bias_path():
    """Without a bias the op runs the overwrite kernel (no zero-init launch); with a zero bias it
    runs the accumulating one: both must agree exactly."""
    _require_gpu()
    import quantforge.torch_ops as qt

    act, packed, scale, zp, ref = _case(3, 256, 128, 64)
    a = qt.w4a16_linear(act, packed, scale, zp, 64)
    b = qt.w4a16_linear(act, packed, scale, zp, 64, torch.zeros(128, device="cuda", dtype=torch.float16))
    assert torch.equal(a, b) and _close(a, ref)


def test_row_buckets_share_kernels_and_stay_correct():
    """Calls with different M run the kernel of their row bucket: no kernel (= no compile) per
    length, results unchanged by the zero padding."""
    _require_gpu()
    import time
    import quantforge.torch_ops as qt

    assert [qt.bucket_rows(m) for m in (1, 16, 17, 32, 33, 64, 65, 128, 129, 300, 4096, 4097, 5000)] == \
        [16, 16, 32, 32, 64, 64, 128, 128, 256, 384, 4096, 5120, 5120]
    K, N, G = 256, 192, 64
    _, packed, scale, zp, _ = _case(1, K, N, G)
    qt.clear_cache()
    for M in (1, 3, 7, 16):                                    # one decode kernel for all
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 1, list(qt._kernels)
    for M in (17, 32):                                         # the 32-row bucket
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 2
    for M in (33, 40, 64):                                     # the 64-row bucket
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 3
    t = time.time()
    for M in (65, 100, 128, 129, 200, 256):                    # 128- and 256-row buckets
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 5
    # a length that falls in an already compiled bucket must not compile anything
    act = _case(250, K, N, G)[0]
    t = time.time()
    qt.w4a16_linear(act, packed, scale, zp, G)
    torch.cuda.synchronize()
    assert time.time() - t < 0.5, "an already compiled bucket must not recompile"
    assert len(qt._kernels) == 5


def test_stale_padding_rows_do_not_leak_into_results():
    """The C++ op reuses a padded activation buffer: rows past M keep what earlier calls left
    there (here NaN / inf). A GEMM row depends only on its own activation row, so every result
    must still be exact, whatever the call order."""
    _require_gpu()
    import quantforge.torch_ops as qt

    K, N, G = 256, 128, 64
    _, packed, scale, zp, _ = _case(1, K, N, G)
    poison = torch.full((16, K), float("nan"), device="cuda", dtype=torch.float16)
    poison[::2] = float("inf")
    qt.w4a16_linear(poison, packed, scale, zp, G)            # fills the whole 16-row buffer
    from quantforge import quant
    W = dequant_reference(quant.unpack_int4(packed), scale, zp).float()     # the weights under test
    for M in (1, 5, 16, 2, 9, 1):                            # descending / mixed order
        act = _case(M, K, N, G, seed=M)[0]
        ref = act.float() @ W
        out = qt.w4a16_linear(act, packed, scale, zp, G)
        assert torch.isfinite(out.float()).all(), M
        assert _close(out, ref), M


def test_default_decode_schedule_depends_on_width_and_is_correct():
    """The default M <= 16 schedule: 128-wide K tile for wide outputs (N >= 8192), 64 for the middle, 32 / 64 column blocks
    for narrow outputs (N <= 2048); and the 17..64-row schedules depend on the shape. All must be right on a real call."""
    _require_gpu()
    import quantforge.torch_ops as qt

    assert "block-tile-k=128" in qt._decode_opts(28672) and "block-tile-k=128" in qt._decode_opts(8192)
    assert "block-tile-k=64" in qt._decode_opts(6144) and "block-tile-n=32" in qt._decode_opts(896)
    assert "block-tile-n=64" in qt._decode_opts(1536) and "block-tile-n=128" in qt._decode_opts(4096)
    assert "block-tile-n=128" in qt._decode_opts(1000)                    # not a multiple of 32: the standard block
    assert "block-tile-m=32" in qt._skinny_opts(64, 4864, 896) and "block-tile-m=64" in qt._skinny_opts(64, 4096, 4096)
    assert "block-tile-k=64" in qt._skinny_opts(32, 1536, 17920) and "block-tile-k=128" in qt._skinny_opts(32, 4096, 4096)
    assert "block-tile-k=32" in qt._decode_opts(11008) and "block-tile-k=128" in qt._decode_opts(14336)
    for (K, N, G) in [(256, 8192, 128), (256, 256, 128), (256, 10240, 128), (512, 896, 128), (512, 1536, 128), (256, 2048, 128)]:
        for M in (3, 24, 50):                                          # decode, 32-row and 64-row buckets
            act, packed, scale, zp, ref = _case(M, K, N, G)
            assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), (M, K, N)


def _unpack(packed):
    lo, hi = packed & 0xF, (packed >> 4) & 0xF
    q = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    return (q.to(torch.int8),)


if __name__ == "__main__":
    unittest.main()


def test_gptq_act_order_end_to_end():
    """desc_act=True: scattered g_idx -> row permutation offline, activation gather in the op."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from quantforge import quant

    M, K, N, G = 5, 512, 256, 128
    g = torch.Generator().manual_seed(3)
    q = torch.randint(0, 16, (K, N), dtype=torch.int8, generator=g)
    scale = (torch.rand(K // G, N, generator=g) * 0.05 + 0.01).half()
    zero = torch.randint(0, 16, (K // G, N), dtype=torch.int8, generator=g)
    g_idx = (torch.randperm(K, generator=g) % (K // G)).to(torch.int32)
    qw, qz, s = quant.to_gptq(q, scale, zero)
    deq = (q.float() - zero.float()[g_idx.long()]) * scale.float()[g_idx.long()]   # act-order reference
    x = torch.randn(M, K, generator=g).half().cuda()
    ref = x.float() @ deq.cuda()
    packed, sc, zp, perm = quant.from_gptq_act_order(qw.cuda(), qz.cuda(), s.cuda(), g_idx.cuda())
    out = qt.w4a16_linear(x, packed, sc, zp, G, perm=perm)
    assert _close(out, ref)


def test_bad_option_combination_raises_instead_of_aborting():
    import pytest
    from quantforge.e2e import W4A16Gemm

    with pytest.raises(ValueError, match="Tensor Core"):
        W4A16Gemm.build(16, 256, 128, group_size=64, accumulate=False, pipeline_opts="hoist-acc=false")


def test_int8_weights_through_the_op_eager_compile_graph():
    """W8A16: a plain int8 [K, N] weight selects the 8-bit kernel (C++ runtime and Python fallback)."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from quantforge import quant

    K, N, G = 512, 384, 128
    g = torch.Generator().manual_seed(5)
    w = torch.randn(K, N, generator=g) * 0.05
    q8, scale, zero = quant.quantize_groupwise_int8(w, G)
    q8, scale, zero = q8.cuda(), scale.cuda(), zero.cuda()
    assert qt.weight_kind(q8, scale, G) == 8
    deq = (q8.float() - zero.float().repeat_interleave(G, 0)) * scale.float().repeat_interleave(G, 0)
    for M in (1, 7, 40):
        x = torch.randn(M, K, generator=g).half().cuda()
        ref = x.float() @ deq
        out = qt.w4a16_linear(x, q8, scale, zero, G)
        assert out.shape == (M, N) and _close(out, ref), M
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        assert _close(qt.w4a16_linear(x, q8, scale, zero, G, bias), ref + bias.float()), "bias"
    # INT4 and INT8 weights of the same shape must not share a kernel
    q4, s4, z4 = quant.quantize_groupwise(w, G)
    x = torch.randn(3, K, generator=g).half().cuda()
    p4 = quant.pack_int4(q4).cuda()
    deq4 = (q4.float() - z4.float().repeat_interleave(G, 0)) * s4.float().repeat_interleave(G, 0)
    assert _close(qt.w4a16_linear(x, p4, s4.cuda(), z4.cuda(), G), x.float() @ deq4.cuda())
    assert _close(qt.w4a16_linear(x, q8, scale, zero, G), x.float() @ deq)
    f = torch.compile(lambda a: qt.w4a16_linear(a, q8, scale, zero, G), fullgraph=True)
    assert _close(f(x), x.float() @ deq)
    graph = torch.cuda.CUDAGraph()
    qt.w4a16_linear(x, q8, scale, zero, G)
    with torch.cuda.graph(graph):
        y = qt.w4a16_linear(x, q8, scale, zero, G)
    graph.replay(); torch.cuda.synchronize()
    assert _close(y, x.float() @ deq)
    import pytest
    with pytest.raises(Exception):
        qt.w4a16_linear(x, q8, scale, zero, G, activation="swiglu")


def test_prefill_default_schedule_for_large_weights_is_correct():
    _require_gpu()
    import quantforge.torch_ops as qt

    assert qt._prefill_opts(512, 4096, 11008) == "block-tile-k=32 smem-stages=3"
    assert qt._prefill_opts(512, 4096, 4096) == "" and qt._prefill_opts(256, 4096, 11008) == ""
    act, packed, scale, zp, ref = _case(512, 4096, 11008, 128)
    out = qt.w4a16_linear(act, packed, scale, zp, 128)
    assert _close(out, ref)


def test_fp8_weights_through_the_op_eager_compile_graph():
    """FP8 E4M3 weights (torch.float8_e4m3fn [K, N]) select the FP8 kernel."""
    _require_gpu()
    import pytest
    import quantforge.torch_ops as qt
    from quantforge import quant

    K, N, G = 512, 384, 128
    g = torch.Generator().manual_seed(7)
    w = torch.randn(K, N, generator=g) * 0.05
    w8, scale, zero = quant.quantize_fp8(w, G)
    # the quantiser is good: FP8 keeps ~2 significant digits (3 mantissa bits)
    err = ((w8.float() * scale.float().repeat_interleave(G, 0)) - w).abs().max()
    assert err < 0.07 * w.abs().max()
    w8, scale, zero = w8.cuda(), scale.cuda(), zero.cuda()
    assert qt.weight_kind(w8, scale, G) == 9
    deq = w8.float() * scale.float().repeat_interleave(G, 0)
    for M in (1, 7, 40, 512):
        x = torch.randn(M, K, generator=g).half().cuda()
        ref = x.float() @ deq
        out = qt.w4a16_linear(x, w8, scale, zero, G)
        assert out.shape == (M, N) and _close(out, ref), M
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    x = torch.randn(3, K, generator=g).half().cuda()
    assert _close(qt.w4a16_linear(x, w8, scale, zero, G, bias), x.float() @ deq + bias.float()), "bias"
    # FP8 and INT8 weights of the same shape must not share a kernel
    q8, s8, z8 = quant.quantize_groupwise_int8(w, G)
    deq8 = (q8.float() - z8.float().repeat_interleave(G, 0)) * s8.float().repeat_interleave(G, 0)
    assert _close(qt.w4a16_linear(x, q8.cuda(), s8.cuda(), z8.cuda(), G), x.float() @ deq8.cuda())
    assert _close(qt.w4a16_linear(x, w8, scale, zero, G), x.float() @ deq)
    f = torch.compile(lambda a: qt.w4a16_linear(a, w8, scale, zero, G), fullgraph=True)
    assert _close(f(x), x.float() @ deq)
    graph = torch.cuda.CUDAGraph()
    qt.w4a16_linear(x, w8, scale, zero, G)
    with torch.cuda.graph(graph):
        y = qt.w4a16_linear(x, w8, scale, zero, G)
    graph.replay(); torch.cuda.synchronize()
    assert _close(y, x.float() @ deq)
    with pytest.raises(Exception):
        qt.w4a16_linear(x, w8, scale, zero, G, activation="swiglu")


def test_gptq_8bit_conversion_matches_the_dense_reference_including_act_order():
    """8-bit GPTQ words -> int8 kernel layout (weights and zero points shifted by 128); the op equals x @ dequant(W)."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from quantforge import quant

    torch.manual_seed(0)
    K, N, G = 512, 384, 128
    q = torch.randint(0, 256, (K, N), dtype=torch.int64, device="cuda")
    zero = torch.randint(100, 156, (K // G, N), dtype=torch.int64, device="cuda")
    scale = (torch.rand(K // G, N, device="cuda") * 0.01 + 0.001).half()

    def words(v, axis):                                    # four bytes per int32 along `axis` (0: K, 1: N)
        v = v.reshape(-1, 4, v.shape[1]) if axis == 0 else v.reshape(v.shape[0], -1, 4).transpose(1, 2)
        w = sum(v[:, i].to(torch.int64) << (8 * i) for i in range(4)) if axis == 0 else \
            sum(v[:, i, :].to(torch.int64) << (8 * i) for i in range(4))
        return torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)

    qweight, qzeros = words(q, 0), words(zero - 1, 1)      # GPTQ stores zero - 1
    ref_w = (q.float() - zero.float().repeat_interleave(G, 0)) * scale.float().repeat_interleave(G, 0)
    p, sc, zz = quant.from_gptq8(qweight, qzeros, scale)
    assert p.dtype == torch.int8 and p.shape == (K, N) and zz.shape == (K // G, N)
    x = torch.randn(5, K, device="cuda", dtype=torch.float16)
    ref = x.float() @ ref_w
    assert (qt.w4a16_linear(x, p, sc, zz, G).float() - ref).abs().max() < 2e-3 * ref.abs().max()
    # act-order: scattered groups -> rows permuted, activations gathered
    g_idx = torch.randperm(K, device="cuda") // G
    p2, sc2, zz2, perm = quant.from_gptq8_act_order(qweight, qzeros, scale, g_idx)
    group_of_row = g_idx.long()
    ref_w2 = (q.float() - zero.float()[group_of_row]) * scale.float()[group_of_row]
    ref2 = x.float() @ ref_w2
    assert (qt.w4a16_linear(x, p2, sc2, zz2, G, perm=perm).float() - ref2).abs().max() < 2e-3 * ref2.abs().max()


def test_long_inputs_are_processed_in_chunks_with_a_bounded_temporary():
    """More than 2048 rows: the op works in chunks of 2048 rows (same result; the f32 accumulator is not M x N x 4 bytes)."""
    _require_gpu()
    import quantforge.torch_ops as qt

    M, K, N, G = 6000, 256, 4096, 128
    act, packed, scale, zp, ref = _case(M, K, N, G)
    torch.cuda.synchronize()
    out_bytes = M * N * 2
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = qt.w4a16_linear(act, packed, scale, zp, G)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - base
    assert y.shape == (M, N) and _close(y, ref)
    assert peak_extra < 0.8 * M * N * 6, (peak_extra, out_bytes)     # unchunked: f32 accumulator + its f16 cast = M*N*6 (147 MB here, chunked ~100)
    y3 = qt.w4a16_linear(act.reshape(2, 3000, K), packed, scale, zp, G)             # leading dims are kept
    assert y3.shape == (2, 3000, N) and _close(y3.reshape(M, N), ref)


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_decode_call_reads_only_its_own_rows_and_is_one_kernel(rows, bias):
    """M in {1, 2, 4, 8}: the f16-store kernel for exactly that many rows of a 16-row tile (cp.async zero-fills the rest, the stores are skipped): the
    activations may sit in a buffer whose neighbouring rows are NaN, and the call is a single kernel (no padded copy, no conversion)."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from torch.profiler import ProfilerActivity, profile

    if qt._cpp is None:
        pytest.skip("needs the C++ runtime")
    g = torch.Generator(device="cuda").manual_seed(rows)
    K, N = 512, 256
    w = torch.randn(K, N, device="cuda", generator=g) * 0.05
    from quantforge import quant
    q, s, z = quant.quantize_groupwise(w, 128)
    packed = quant.pack_int4(q)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=g)
    buf = torch.full((rows + 16, K), float("nan"), device="cuda", dtype=torch.float16)
    buf[:rows] = x
    b = torch.randn(N, device="cuda", dtype=torch.float16, generator=g) if bias else None
    deq = (q.float() - z.float().repeat_interleave(128, 0)) * s.float().repeat_interleave(128, 0)
    ref = x.float() @ deq + (b.float() if bias else 0)
    y = torch.ops.quantforge.w4a16_gemm(buf[:rows], packed, s, z, 128, b, "none")
    assert y.shape == (rows, N) and torch.isfinite(y).all() and _close(y, ref)
    for _ in range(3):
        torch.ops.quantforge.w4a16_gemm(buf[:rows], packed, s, z, 128, b, "none")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        torch.ops.quantforge.w4a16_gemm(buf[:rows], packed, s, z, 128, b, "none")
        torch.cuda.synchronize()
    assert len([e for e in p.events() if e.device_time > 0]) == 1


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_kfold_call_reads_only_its_own_rows_and_is_two_kernels(rows):
    """K-folded GEMM (1536 x 1536 folded 3 ways, M * 3 rows of a 16-row tile): cp.async zero-fills the rest of the activation tile, so the
    activations may sit in a buffer whose neighbouring rows are NaN, and the call is the GEMM plus the reduce kernel (no padded copy)."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from quantforge import kfold, quant
    from torch.profiler import ProfilerActivity, profile

    if qt._cpp is None or not getattr(qt._cpp, "HAS_KFOLD", False):
        pytest.skip("needs the C++ runtime with the K-fold op")
    g = torch.Generator(device="cuda").manual_seed(rows)
    K = N = 1536
    S = 3
    w = torch.randn(K, N, device="cuda", generator=g) * 0.05
    q, s, z = quant.quantize_groupwise(w, 128)
    packed = quant.pack_int4(q)
    pf, sf, zf = kfold.fold_k(packed, s, z, S)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=g)
    buf = torch.full((rows + 16, K), float("nan"), device="cuda", dtype=torch.float16)
    buf[:rows] = x
    deq = (q.float() - z.float().repeat_interleave(128, 0)) * s.float().repeat_interleave(128, 0)
    ref = x.float() @ deq
    call = lambda: torch.ops.quantforge.w4a16_gemm_kfold(buf[:rows], packed, s, z, pf, sf, zf, 128, None)  # noqa: E731
    y = call()
    assert y.shape == (rows, N) and torch.isfinite(y).all() and _close(y, ref)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        call()
        torch.cuda.synchronize()
    assert len([e for e in p.events() if e.device_time > 0]) == 2


@pytest.mark.parametrize("rows", [16, 3])
def test_atomic_split_k_kernel(rows):
    """`W4A16GemmAtomic`: the K reduction is split over blocks that add their partial results with red.global.add.f32 (one kernel, no partial
    buffer). The result is added onto whatever `out` held; rows past `valid_rows` are never touched."""
    _require_gpu()
    from quantforge import quant
    from quantforge.e2e import W4A16GemmAtomic

    K, N, G, S = 2048, 128, 128, 4
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = torch.randn(K, N, device="cuda", generator=g) * 0.05
    q, s, z = quant.quantize_groupwise(w, G)
    packed = quant.pack_int4(q)
    deq = (q.float() - z.float().repeat_interleave(G, 0)) * s.float().repeat_interleave(G, 0)
    opts = "decode=true block-tile-n=64 warp-tile-n=16 block-tile-k=128 smem-stages=1 prefetch-distance=1"
    k = W4A16GemmAtomic.build(16, K, N, G, S, pipeline_opts=opts, valid_rows=rows if rows < 16 else 0)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=g)
    init = torch.randn(16, N, device="cuda", generator=g)
    out = init.clone()
    k.launch(x, packed, s, z, out)
    torch.cuda.synchronize()
    ref = init.clone()
    ref[:rows] += x.float() @ deq
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(out[rows:], init[rows:])


@pytest.mark.parametrize("rows", [1, 3, 8, 16])
def test_atomic_op_matches_the_plain_path(rows):
    """The op takes the atomic split-K path for a long-K narrow layer (here 8960 x 1536, where it measures faster on the RTX 4050) at any 1..16
    rows, also when the activation rows sit in a buffer whose neighbours are NaN; the result matches the dequantised reference."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from quantforge import atomic_plan, quant

    if qt._cpp is None:
        pytest.skip("needs the C++ runtime")
    K, N, G = 8960, 1536, 128
    if atomic_plan.plan_atomic(K, N, G) is None:
        pytest.skip("the planner keeps the plain kernel for this shape on this GPU")
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = torch.randn(K, N, device="cuda", generator=g) * 0.02
    q, s, z = quant.quantize_groupwise(w, G)
    packed = quant.pack_int4(q)
    deq = (q.float() - z.float().repeat_interleave(G, 0)) * s.float().repeat_interleave(G, 0)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=g)
    buf = torch.full((rows + 16, K), float("nan"), device="cuda", dtype=torch.float16)
    buf[:rows] = x
    b = torch.randn(N, device="cuda", dtype=torch.float16, generator=g)
    y = torch.ops.quantforge.w4a16_gemm(buf[:rows], packed, s, z, G, b, "none")
    ref = x.float() @ deq + b.float()
    assert y.shape == (rows, N) and torch.isfinite(y).all()
    assert (y.float() - ref).abs().max() < 0.05 * ref.abs().max().clamp(min=1.0)


@pytest.mark.parametrize("rows", [32, 64])
def test_atomic_split_k_kernel_skinny_tiles(rows):
    """`W4A16GemmAtomic` with the 32- / 64-row skinny schedules: same contract as for 16 rows (added onto `out`)."""
    _require_gpu()
    from quantforge import atomic_plan, quant
    from quantforge.e2e import W4A16GemmAtomic

    K, N, G, S = 2048, 128, 128, 4
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = torch.randn(K, N, device="cuda", generator=g) * 0.05
    q, s, z = quant.quantize_groupwise(w, G)
    packed = quant.pack_int4(q)
    deq = (q.float() - z.float().repeat_interleave(G, 0)) * s.float().repeat_interleave(G, 0)
    k = W4A16GemmAtomic.build(rows, K, N, G, S, pipeline_opts=atomic_plan.tile_opts(64, 16, 64, rows))
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=g)
    init = torch.randn(rows, N, device="cuda", generator=g)
    out = init.clone()
    k.launch(x, packed, s, z, out)
    torch.cuda.synchronize()
    assert torch.allclose(out, init + x.float() @ deq, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("rows", [17, 24, 32, 33, 48, 64])
def test_atomic_op_serves_17_to_64_rows(rows):
    """17..64 rows take the 32- / 64-row atomic split-K kernel of the bucket when the planner measures it faster (8960 x 1536 on the RTX 4050);
    bias and the rows past the bucket's padding behave as for the plain kernel."""
    _require_gpu()
    import quantforge.torch_ops as qt
    from quantforge import atomic_plan, quant

    if qt._cpp is None:
        pytest.skip("needs the C++ runtime")
    K, N, G = 8960, 1536, 128
    bucket = qt.bucket_rows(rows)
    if atomic_plan.plan_atomic(K, N, G, rows=bucket) is None:
        pytest.skip("the planner keeps the plain kernel for this shape on this GPU")
    g = torch.Generator(device="cuda").manual_seed(rows)
    w = torch.randn(K, N, device="cuda", generator=g) * 0.02
    q, s, z = quant.quantize_groupwise(w, G)
    packed = quant.pack_int4(q)
    deq = (q.float() - z.float().repeat_interleave(G, 0)) * s.float().repeat_interleave(G, 0)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn(N, device="cuda", dtype=torch.float16, generator=g)
    y = torch.ops.quantforge.w4a16_gemm(x, packed, s, z, G, b, "none")
    ref = x.float() @ deq + b.float()
    assert y.shape == (rows, N) and torch.isfinite(y).all()
    assert (y.float() - ref).abs().max() < 0.05 * ref.abs().max().clamp(min=1.0)
