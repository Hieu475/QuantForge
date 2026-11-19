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

    assert [qt.bucket_rows(m) for m in (1, 16, 17, 64, 65, 128, 129, 300, 4096, 4097, 5000)] == \
        [16, 16, 64, 64, 128, 128, 256, 384, 4096, 5120, 5120]
    K, N, G = 256, 192, 64
    _, packed, scale, zp, _ = _case(1, K, N, G)
    qt.clear_cache()
    for M in (1, 3, 7, 16):                                    # one decode kernel for all
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 1, list(qt._kernels)
    for M in (17, 40, 64):                                     # the 64-row bucket
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 2
    t = time.time()
    for M in (65, 100, 128, 129, 200, 256):                    # 128- and 256-row buckets
        act, _, _, _, ref = _case(M, K, N, G)
        assert _close(qt.w4a16_linear(act, packed, scale, zp, G), ref), M
    assert len(qt._kernels) == 4
    # a length that falls in an already compiled bucket must not compile anything
    act = _case(250, K, N, G)[0]
    t = time.time()
    qt.w4a16_linear(act, packed, scale, zp, G)
    torch.cuda.synchronize()
    assert time.time() - t < 0.5, "an already compiled bucket must not recompile"
    assert len(qt._kernels) == 4


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


def _unpack(packed):
    lo, hi = packed & 0xF, (packed >> 4) & 0xF
    q = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    return (q.to(torch.int8),)


if __name__ == "__main__":
    unittest.main()
