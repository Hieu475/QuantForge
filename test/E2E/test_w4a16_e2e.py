"""
End-to-end test: qf IR -> --quantforge-gpu-pipeline -> PTX -> run on GPU ->
compare against a PyTorch reference.

Run:   QUANTFORGE_BUILD_DIR=<build> python test/E2E/test_w4a16_e2e.py
   or: QUANTFORGE_BUILD_DIR=<build> pytest test/E2E
Skips (exit 0 / pytest skip) when no CUDA device is available.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

from quantforge.e2e import (  # noqa: E402
    W4A16Gemm,
    compile_ir,
    pack_int4,
    ptxas_check,
    reference_gemm,
    w4a16_gemm_ir,
)

HAS_CUDA = torch.cuda.is_available()


def _require_tools():
    """Skip (pytest + the script runner understand unittest.SkipTest) when the
    compiler tool-chain is not available."""
    try:
        from quantforge.e2e import _find_llvm_tool, _find_quantforge_opt

        _find_quantforge_opt()
        _find_llvm_tool("mlir-translate")
        _find_llvm_tool("llc")
    except FileNotFoundError as e:
        raise unittest.SkipTest(str(e))
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda / "bin" / "ptxas").exists():
        raise unittest.SkipTest(f"ptxas not found under {cuda} (set CUDA_HOME)")


def _require_gpu():
    _require_tools()
    if not HAS_CUDA:
        raise unittest.SkipTest("no CUDA device")

# (M, K, N): single tile, multi-block grid, K not a power of two, wide N
SHAPES = [(128, 128, 128), (256, 128, 256), (128, 192, 384)]


def _inputs(M, K, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
    scale = torch.tensor(0.05, device="cuda", dtype=torch.float16)
    zp = torch.tensor(8, device="cuda", dtype=torch.int8)
    return act, q, scale, zp


def test_ptx_assembles():
    """The emitted PTX must be accepted by ptxas for the target architecture."""
    _require_tools()
    with tempfile.TemporaryDirectory() as d:
        # default pipeline: dequant fused into the GEMM -> ONE kernel, using
        # shared memory and Tensor Core mma.sync + ldmatrix
        c = compile_ir(w4a16_gemm_ir(128, 128, 128), d, chip="sm_80")
        assert list(c.ptx) == ["w4a16_gemm_kernel"], c.ptx.keys()
        text = c.ptx["w4a16_gemm_kernel"]
        assert ".target sm_80" in text and ".entry" in text
        assert ".shared" in text, "expected shared-memory staging"
        assert "mma.sync" in text and "ldmatrix" in text, "expected Tensor Core code"
        assert "cp.async" in text, "expected cp.async for the A tile"
        assert "st.shared.v4" in text, "expected 128-bit shared stores for the B tile"
        assert "div.u" not in text and "rem.u" not in text, "index math must be constant-folded"
        ptxas_check(c.ptx, "sm_80", d)

    with tempfile.TemporaryDirectory() as d:
        # without fusion the dequantization is its own kernel
        c = compile_ir(w4a16_gemm_ir(128, 128, 128), d, pipeline_opts="fuse-dequant=false")
        assert len(c.ptx) == 2, c.ptx.keys()
        ptxas_check(c.ptx, "sm_80", d)

    with tempfile.TemporaryDirectory() as d:
        # chip option reaches the PTX header
        c = compile_ir(w4a16_gemm_ir(128, 128, 128), d, chip="sm_89")
        assert ".target sm_89" in c.ptx["w4a16_gemm_kernel"]
        ptxas_check(c.ptx, "sm_89", d)


# pipeline option sets that must all produce the same numbers
CONFIGS = [
    "threads=false smem=false",          # one thread per block, global memory only
    "threads=true smem=false",           # warp/lane tiles spread over 128 threads
    "tensor-core=false",                 # + shared memory, FMA code, fused dequant
    "tensor-core=false fuse-dequant=false",
    "",                                  # default: + Tensor Core mma.sync
    "fuse-dequant=false",                # separate dequant kernel
    "block-tile-k=64",
    "smem-pad=0 hoist-acc=false",
    "async-copy=false",                  # register path for A as well
    "async-copy=false vector-copy=false",  # scalar copies
    "register-tile=false",               # accumulators through memory per mma
    "smem-stages=1",                     # single buffer (default is 2)
    "smem-stages=1 tensor-core=false",
    "smem-stages=2 tensor-core=false",   # double buffering without Tensor Cores
    "smem-stages=2 fuse-dequant=false",
    "smem-stages=2 block-tile-k=64",     # 2 stages would exceed 48 KiB -> falls back
    "warp-tile-m=32 warp-tile-n=32",     # 16 warps / block
    "warp-tile-m=128 warp-tile-n=32",
]


def _check(M, K, N, acc, opts):
    act, q, scale, zp = _inputs(M, K, N)
    kern = W4A16Gemm.build(M, K, N, acc=acc, pipeline_opts=opts)
    out = kern.run(act, pack_int4(q), scale, zp)
    ref = reference_gemm(act, q, scale, zp)
    err = (out.float() - ref).abs().max().item()
    # f32 accumulate: only f16 rounding of the operands is shared with the
    # reference, so the error is tiny. f16 accumulate drifts with K.
    tol = (1e-4 if acc == "f32" else 1e-3) * ref.abs().max().item() + (1e-3 if acc == "f32" else 1e-2)
    print(f"  {acc} {opts!r:32} M={M} K={K} N={N}: max|err|={err:.5f} (tol {tol:.5f})")
    assert err < tol, f"mismatch for {(M, K, N, acc, opts)}: {err} >= {tol}"


def test_numerics():
    _require_gpu()
    for (M, K, N) in SHAPES:
        _check(M, K, N, "f32", "")          # default pipeline
    for opts in CONFIGS:                     # every pipeline variant, f32 and f16
        for acc in ("f32", "f16"):
            _check(256, 128, 256, acc, opts)


def test_accumulates_into_c():
    """C is `outs`: the kernel must add to the existing contents."""
    _require_gpu()
    M = K = N = 128
    act, q, scale, zp = _inputs(M, K, N, seed=1)
    c0 = torch.full((M, N), 1.0, device="cuda", dtype=torch.float32)
    out = W4A16Gemm.build(M, K, N).run(act, pack_int4(q), scale, zp, out=c0.clone())
    ref = reference_gemm(act, q, scale, zp) + 1.0
    assert (out.float() - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3


def test_arbitrary_shapes_via_padding():
    """Shapes that are not tile multiples work with pad=True (exact padding)."""
    _require_gpu()
    for (M, K, N) in [(1, 33, 2), (100, 70, 90), (130, 200, 258)]:
        act, q, scale, zp = _inputs(M, K, N, seed=3)
        kern = W4A16Gemm.build(M, K, N, pad=True)
        out = kern.run(act, pack_int4(q), scale, zp)
        assert out.shape == (M, N)
        ref = reference_gemm(act, q, scale, zp)
        err = (out.float() - ref).abs().max().item()
        print(f"  pad M={M} K={K} N={N} -> {kern.Mp}x{kern.Kp}x{kern.Np}: max|err|={err:.5f}")
        assert err < 1e-4 * ref.abs().max().item() + 1e-3


def _group_inputs(M, K, N, G, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
    scale = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
    zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
    return act, q, scale, zp


def _check_group(M, K, N, G, acc="f32", opts="", pad=False):
    act, q, scale, zp = _group_inputs(M, K, N, G)
    kern = W4A16Gemm.build(M, K, N, acc=acc, pipeline_opts=opts, pad=pad, group_size=G)
    out = kern.run(act, pack_int4(q), scale, zp)
    ref = reference_gemm(act, q, scale, zp)
    err = (out.float() - ref).abs().max().item()
    tol = (1e-4 if acc == "f32" else 3e-3) * ref.abs().max().item() + (1e-3 if acc == "f32" else 1e-2)
    print(f"  G={G} {acc} {opts!r:28} M={M} K={K} N={N} pad={pad}: max|err|={err:.5f} (tol {tol:.5f})")
    assert err < tol, f"mismatch for {(M, K, N, G, acc, opts)}: {err} >= {tol}"


def test_groupwise_numerics():
    """Group-wise scale / zero point (group sizes below, equal to and above the K tile)."""
    _require_gpu()
    for G in (32, 64, 128, 256):
        _check_group(128, 256, 128, G)
    _check_group(256, 512, 256, 128)                     # multi-block grid
    for opts in ("tensor-core=false", "fuse-dequant=false", "smem-stages=1",
                 "block-tile-k=64", "async-copy=false vector-copy=false",
                 "threads=false smem=false"):
        _check_group(128, 256, 128, 64, opts=opts)
    _check_group(128, 256, 128, 64, acc="f16")


def test_groupwise_padding():
    _require_gpu()
    for (M, K, N, G) in [(1, 128, 2, 64), (100, 192, 90, 64), (130, 384, 258, 128)]:
        _check_group(M, K, N, G, pad=True)


def test_w8a16():
    """INT8 weights (no unpack): per-tensor and group-wise, plain and decode schedule."""
    _require_gpu()
    from quantforge.quant import quantize_groupwise_int8

    g = torch.Generator(device="cuda").manual_seed(5)
    for (M, K, N, G, kw) in [(128, 256, 128, None, {}), (128, 256, 256, 64, {}),
                             (3, 512, 192, 128, {"decode": True, "pad": True})]:
        act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
        if G is None:
            q = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8, generator=g)
            scale = torch.tensor(0.01, device="cuda", dtype=torch.float16)
            zp = torch.tensor(3, device="cuda", dtype=torch.int8)
        else:
            w = torch.randn(K, N, device="cuda", generator=g)
            q, scale, zp = quantize_groupwise_int8(w, G)
        kern = W4A16Gemm.build(M, K, N, bits=8, group_size=G, **kw)
        out = kern.run(act, q.contiguous(), scale, zp)
        ref = reference_gemm(act, q, scale, zp)
        err = (out.float() - ref).abs().max().item()
        print(f"  w8a16 M={M} K={K} N={N} G={G} {kw}: max|err|={err:.5f}")
        assert err < 1e-4 * ref.abs().max().item() + 1e-3


def test_decode_schedule():
    """decode=true: skinny-M schedule; M = 1..16 padded to one 16-row tile; one kernel."""
    _require_gpu()
    for (M, K, N, G) in [(1, 256, 128, None), (16, 512, 256, 128), (5, 384, 192, 64)]:
        _check_group(M, K, N, G, opts="decode=true", pad=True) if G else _check_decode(M, K, N)
    with tempfile.TemporaryDirectory() as d:
        c = compile_ir(w4a16_gemm_ir(16, 512, 256, pipeline_opts="decode=true"), d,
                       pipeline_opts="decode=true")
        assert list(c.ptx) == ["w4a16_gemm_kernel"], "dequant must stay fused in the decode schedule"


def test_prefetch_weight_loads_is_bit_exact():
    """prefetch-distance only moves the weight loads earlier (rotating registers): the
    result must be bit-identical to distance 0, for both K-loop schedules, including a
    K loop shorter than the distance (clamped look-ahead re-reads the last tile)."""
    _require_gpu()
    schedules = ["decode=true block-tile-k=128 smem-stages=1",
                 "decode=true block-tile-k=64 smem-stages=2 sink-dequant-stores=true"]
    for (M, K, N, G) in [(1, 1024, 192, 128), (16, 256, 128, 128), (3, 128, 128, 64)]:
        act, q, scale, zp = _group_inputs(M, K, N, G)
        ref = reference_gemm(act, q, scale, zp)
        for sched in schedules:
            if K % 64 or (("block-tile-k=128" in sched) and K % 128):
                continue
            base = None
            for P in (0, 1, 2, 3):
                kern = W4A16Gemm.build(M, K, N, pad=True, group_size=G,
                                       pipeline_opts=f"{sched} prefetch-distance={P}")
                out = kern.run(act, pack_int4(q), scale, zp)
                assert (out.float() - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3
                if base is None:
                    base = out.clone()
                assert torch.equal(out, base), f"{sched} P={P} M={M} K={K} N={N}"


def test_launches_on_current_stream_and_reuses_module():
    """The runtime shim caches the JIT-compiled module and launches on PyTorch's current
    stream: many calls on a side stream, ordered after in-stream work (zero_ / copies), must
    all produce the reference result."""
    _require_gpu()
    M, K, N, G = 3, 256, 128, 64   # padded M path: out / act are copied around the kernel
    act, q, scale, zp = _group_inputs(M, K, N, G)
    packed = pack_int4(q)
    ref = reference_gemm(act, q, scale, zp)
    kern = W4A16Gemm.build(M, K, N, pad=True, group_size=G, pipeline_opts="decode=true")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        for i in range(20):
            out.fill_(float(i))          # pending work on the side stream
            out.sub_(float(i))           # ... must complete before the kernel reads `out`
            kern.run(act, packed, scale, zp, out=out)
            assert (out - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3, i
    torch.cuda.synchronize()


def _torch_act(name, x):
    import torch.nn.functional as F
    return {"relu": F.relu, "silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh")}[name](x)


def test_fused_activation_epilogue():
    """epilogue=relu|silu|gelu: out = act(bias + A @ W), applied to the f32 accumulator before
    the store (decode and the 128x128 schedule, padded M, group-wise)."""
    _require_gpu()
    for act_name in ("relu", "silu", "gelu"):
        for (M, K, N, G, opts) in [(5, 256, 192, 64, "decode=true"), (128, 128, 128, 64, ""),
                                   (3, 256, 128, 128, "decode=true prefetch-distance=1 smem-stages=1")]:
            act, q, scale, zp = _group_inputs(M, K, N, G)
            bias = torch.randn(N, device="cuda") * 5
            kern = W4A16Gemm.build(M, K, N, group_size=G, pad=True, pipeline_opts=opts,
                                   epilogue=act_name)
            out = kern.run(act, pack_int4(q), scale, zp, bias=bias)
            want = _torch_act(act_name, reference_gemm(act, q, scale, zp) + bias)
            err = (out - want).abs().max().item()
            tol = 1e-4 * want.abs().max().item() + 2e-4   # ex2.approx: a few ulp
            assert err < tol, f"{act_name} {opts!r} M={M}: {err} >= {tol}"


def test_epilogue_validation():
    _require_gpu()
    for bad in [dict(epilogue="swish"), dict(epilogue="silu", acc="f16")]:
        try:
            W4A16Gemm.build(128, 128, 128, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_overwrite_output_never_reads_c():
    """accumulate=False: out = A @ W even when the buffer holds garbage (NaNs); no bias."""
    _require_gpu()
    for (M, K, N, G, opts) in [(5, 256, 192, 64, "decode=true"), (128, 128, 256, 64, ""),
                               (3, 256, 128, 128, "decode=true prefetch-distance=1 smem-stages=1")]:
        act, q, scale, zp = _group_inputs(M, K, N, G)
        kern = W4A16Gemm.build(M, K, N, group_size=G, pad=True, pipeline_opts=opts, accumulate=False)
        garbage = torch.full((M, N), float("nan"), device="cuda")
        out = kern.run(act, pack_int4(q), scale, zp, out=garbage)
        ref = reference_gemm(act, q, scale, zp)
        assert (out - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3, (M, opts)
        fresh = kern.run(act, pack_int4(q), scale, zp)            # allocated by the kernel wrapper
        assert torch.equal(fresh, out)
    try:
        kern.run(act, pack_int4(q), scale, zp, bias=torch.zeros(N, device="cuda"))
    except ValueError:
        pass
    else:
        raise AssertionError("bias must be rejected by an overwrite kernel")


def test_swiglu_epilogue_matches_torch():
    """epilogue=swiglu on interleaved gate/up weights: silu(x @ Wg) * (x @ Wu), [M, N]."""
    _require_gpu()
    import torch.nn.functional as F
    from quantforge import quant

    for (M, K, N, G, opts) in [(1, 256, 128, 64, "decode=true"), (5, 512, 192, 128, "decode=true"),
                               (16, 512, 256, 128, "decode=true block-tile-n=128 warp-tile-n=16 "
                                                   "block-tile-k=64 smem-stages=1 prefetch-distance=1"),
                               (128, 128, 128, 64, "")]:
        act, qg, sg, zg = _group_inputs(M, K, N, G, seed=1)
        _, qu, su, zu = _group_inputs(M, K, N, G, seed=2)
        packed, scale, zero = quant.interleave_gate_up((pack_int4(qg), sg, zg), (pack_int4(qu), su, zu))
        kern = W4A16Gemm.build(M, K, 2 * N, group_size=G, pad=True, pipeline_opts=opts, epilogue="swiglu")
        out = kern.run(act, packed, scale, zero)
        want = F.silu(reference_gemm(act, qg, sg, zg)) * reference_gemm(act, qu, su, zu)
        assert out.shape == (M, N)
        err = (out - want).abs().max().item()
        assert err < 2e-4 * want.abs().max().item() + 1e-3, (M, K, N, opts, err)


def test_swiglu_validation():
    _require_gpu()
    for bad in [dict(epilogue="swiglu", bits=8), dict(epilogue="swiglu", acc="f16")]:
        try:
            W4A16Gemm.build(128, 128, 128, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_persistent_kernel_cache(monkeypatch=None):
    """A second compile of the same IR / options is served from the persistent cache (no tool
    runs, same PTX, kernel still runs); different options or a disabled cache compile again."""
    _require_gpu()
    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as w1, \
            tempfile.TemporaryDirectory() as w2, tempfile.TemporaryDirectory() as w3:
        old = os.environ.get("QUANTFORGE_KERNEL_CACHE")
        old_off = os.environ.pop("QUANTFORGE_NO_KERNEL_CACHE", None)   # the test needs the cache on
        os.environ["QUANTFORGE_KERNEL_CACHE"] = cache
        try:
            ir = w4a16_gemm_ir(128, 128, 128)
            first = compile_ir(ir, w1)
            second = compile_ir(ir, w2)
            assert not first.cached and second.cached
            assert first.ptx == second.ptx and second.lib_path.parent.parent == Path(cache)
            assert not compile_ir(ir, w3, pipeline_opts="smem-stages=1").cached     # other options
            os.environ["QUANTFORGE_NO_KERNEL_CACHE"] = "1"
            assert not compile_ir(ir, w3).cached
            del os.environ["QUANTFORGE_NO_KERNEL_CACHE"]
            kern = W4A16Gemm.build(128, 128, 128)       # built from the cache entry
            assert kern.compiled.cached
            act, q, scale, zp = _inputs(128, 128, 128)
            out = kern.run(act, pack_int4(q), scale, zp)
            ref = reference_gemm(act, q, scale, zp)
            assert (out - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3
        finally:
            os.environ.pop("QUANTFORGE_NO_KERNEL_CACHE", None)
            if old_off is not None:
                os.environ["QUANTFORGE_NO_KERNEL_CACHE"] = old_off
            if old is None:
                os.environ.pop("QUANTFORGE_KERNEL_CACHE", None)
            else:
                os.environ["QUANTFORGE_KERNEL_CACHE"] = old


def test_concat_columns_fuses_projections():
    """q / k / v style fusion: one GEMM over the column-concatenated weights == three GEMMs."""
    _require_gpu()
    from quantforge import quant

    M, K, G = 5, 256, 64
    widths = (192, 64, 128)
    parts, refs, biases = [], [], []
    for i, n in enumerate(widths):
        _, q, scale, zp = _group_inputs(M, K, n, G, seed=10 + i)
        parts.append((pack_int4(q), scale, zp))
        refs.append((q, scale, zp))
        biases.append(torch.randn(n, device="cuda") * 3)
    g = torch.Generator(device="cuda").manual_seed(99)
    act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    packed, scale, zero = quant.concat_columns(*parts)
    assert packed.shape == (K, sum(widths) // 2) and scale.shape == (K // G, sum(widths))
    kern = W4A16Gemm.build(M, K, sum(widths), group_size=G, pad=True, pipeline_opts="decode=true")
    out = kern.run(act, packed, scale, zero, bias=torch.cat(biases))
    got = out.split(list(widths), dim=-1)
    for (q, sc, zp), b, o in zip(refs, biases, got):
        want = reference_gemm(act, q, sc, zp) + b
        assert (o - want).abs().max().item() < 1e-4 * want.abs().max().item() + 1e-3
    for bad in [(parts[0], (parts[1][0][:-1], parts[1][1], parts[1][2]))]:
        try:
            quant.concat_columns(*bad)
        except ValueError:
            continue
        raise AssertionError("mismatched K must be rejected")


def _check_decode(M, K, N):
    act, q, scale, zp = _inputs(M, K, N)
    out = W4A16Gemm.build(M, K, N, decode=True, pad=True).run(act, pack_int4(q), scale, zp)
    ref = reference_gemm(act, q, scale, zp)
    assert (out.float() - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3


def test_bias_and_residual_epilogue():
    _require_gpu()
    M, K, N, G = 128, 256, 128, 64
    act, q, scale, zp = _group_inputs(M, K, N, G)
    g = torch.Generator(device="cuda").manual_seed(9)
    bias = torch.randn(N, device="cuda", generator=g).half()
    resid = torch.randn(M, N, device="cuda", generator=g).half()
    kern = W4A16Gemm.build(M, K, N, group_size=G)
    out = kern.run(act, pack_int4(q), scale, zp, bias=bias, residual=resid)
    ref = reference_gemm(act, q, scale, zp) + bias.float() + resid.float()
    assert (out.float() - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3
    try:
        kern.run(act, pack_int4(q), scale, zp, out=out, bias=bias)
    except ValueError:
        return
    raise AssertionError("bias with an explicit out must be rejected")


def test_autotune_caches_and_is_correct():
    _require_gpu()
    import json

    from quantforge import autotune as at

    with tempfile.TemporaryDirectory() as d:
        os.environ["QUANTFORGE_TUNE_CACHE"] = str(Path(d) / "tune.json")
        try:
            M, K, N, G = 5, 384, 192, 64
            kern = at.autotune(M, K, N, group_size=G, max_candidates=3)
            cache = json.loads((Path(d) / "tune.json").read_text())
            (key, entry), = cache.items()
            assert "M<=16" in key and f"K{K}|N{N}" in key and entry["tried"] >= 1, cache
            act, q, scale, zp = _group_inputs(M, K, N, G, seed=4)
            out = kern.run(act, pack_int4(q), scale, zp)
            ref = reference_gemm(act, q, scale, zp)
            assert (out.float() - ref).abs().max().item() < 1e-4 * ref.abs().max().item() + 1e-3
            # second call: served from the cache with the same options
            kern2 = at.autotune(M, K, N, group_size=G, max_candidates=3)
            assert json.loads((Path(d) / "tune.json").read_text()) == cache
            assert kern2.compiled.lowered_mlir.exists()
        finally:
            os.environ.pop("QUANTFORGE_TUNE_CACHE", None)


def test_rejects_unaligned_views():
    _require_gpu()
    M, K, N, G = 128, 256, 128, 64
    act, q, scale, zp = _group_inputs(M, K, N, G)
    kern = W4A16Gemm.build(M, K, N, group_size=G)
    flat = torch.zeros(scale.numel() + 8, dtype=torch.float16, device="cuda")
    bad = flat[1:1 + scale.numel()].view_as(scale)         # 2-byte offset: misaligned
    bad.copy_(scale)
    try:
        kern.run(act, pack_int4(q), bad, zp)
    except ValueError as e:
        assert "16-byte aligned" in str(e)
        return
    raise AssertionError("expected ValueError for a misaligned scale view")


def test_groupwise_rejects_bad_group_size():
    for G in (0, 48):
        try:
            w4a16_gemm_ir(128, 128, 128, group_size=G)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for group_size={G}")


def test_verifier_rejects_bad_scale_layout():
    """qf.dequant verifier: group extent must divide the input extent."""
    _require_tools()
    from quantforge.e2e import _find_quantforge_opt, _run

    bad = """func.func @f(%w: tensor<64x32xi8>, %s: tensor<5x32xf16>, %z: tensor<5x32xi8>) -> tensor<64x32xf16> {
      %r = qf.dequant %w, %s, %z : tensor<64x32xi8>, tensor<5x32xf16>, tensor<5x32xi8> -> tensor<64x32xf16>
      return %r : tensor<64x32xf16> }"""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "bad.mlir"
        f.write_text(bad)
        try:
            _run([_find_quantforge_opt(), str(f)])
        except RuntimeError as e:
            assert "block-wise layout" in str(e), e
            return
    raise AssertionError("verifier accepted a non-dividing group layout")


def test_shared_memory_budget_is_checked():
    """A tile configuration that exceeds the 48 KiB static smem limit is a
    clear compile-time error, not a late ptxas/driver failure."""
    _require_tools()
    with tempfile.TemporaryDirectory() as d:
        try:
            compile_ir(w4a16_gemm_ir(128, 256, 128), d, pipeline_opts="block-tile-k=128")
        except RuntimeError as e:
            assert "shared memory" in str(e), e
            return
    raise AssertionError("expected a shared-memory budget error")


def test_rejects_untileable_shape():
    try:
        w4a16_gemm_ir(100, 128, 128)
    except ValueError:
        return
    raise AssertionError("expected ValueError for M not multiple of 128")


if __name__ == "__main__":
    skipped = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name} ...")
            try:
                fn()
            except unittest.SkipTest as e:
                skipped += 1
                print(f"  SKIPPED: {e}")
    print("ALL E2E TESTS PASSED" + (f" ({skipped} skipped)" if skipped else ""))
