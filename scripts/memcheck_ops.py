"""Memory-safety check of the production ops: run them over many row counts and layer shapes under `compute-sanitizer` with exact allocations.

    PYTORCH_NO_CUDA_MEMORY_CACHING=1 QUANTFORGE_BUILD_DIR=<build> compute-sanitizer --tool memcheck python scripts/memcheck_ops.py w4a16   # or w4a8, moe

PyTorch's caching allocator hides small overruns inside a 2 MiB segment; with the cache off every tensor is its own allocation and any read or write past its end
is reported. The generated kernels issue look-ahead loads (prefetch-distance, pipelined stages): this is what shows whether one runs past the end of a tensor.
(The torch-profiler based planners cannot run under the sanitizer: `kfold.plan_ksplit` is skipped, the atomic planner uses CUDA events and runs.)
Modes: w4a16, w4a8, w4a8rot (Hadamard rotation), moe. Expected: ERROR SUMMARY: 0 errors for all of them (0.5.0: all four clean).
"""
import sys, torch, warnings
warnings.simplefilter("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.torch_ops as qt
from quantforge import kfold, quant, w4a8_op, int8_plan, moe
from quantforge.e2e import pack_int4
dev = "cuda"; g = torch.Generator(device=dev).manual_seed(0)
ROWS = (1, 2, 4, 8, 16, 17, 32, 33, 64, 65, 96, 128, 200, 256, 300)
which = sys.argv[1]
SH = ((896, 1152, True), (896, 896, False), (896, 9728, False), (4864, 896, False))
if which == "w4a16":
    for K, N, bias in SH:
        q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int8, generator=g)
        sc = (torch.rand(K // 128, N, device=dev, generator=g, dtype=torch.float32) * .02 + .005).half(); zp = torch.randint(0, 16, (K // 128, N), device=dev, dtype=torch.int8, generator=g)
        packed = pack_int4(q); b = torch.randn(N, device=dev, generator=g).half() if bias else None
        from quantforge import atomic_plan
        atomic_plan.preplan(K, N, 128)                                   # event timing only (the profiler cannot run under the sanitizer)
        for S_ in (1, 2):
            if S_ > 1 and K % 256: continue
            folded = kfold.fold_k(packed, sc, zp, S_) if S_ > 1 else kfold.no_fold(torch.device(dev))
            for M in ROWS:
                x = torch.randn(M, K, device=dev, generator=g, dtype=torch.float32).half()
                kfold.linear(x, packed, sc, zp, folded, 128, b, None)
        torch.cuda.synchronize(); print("w4a16", K, N, "ok", flush=True)
elif which == "w4a8":
    for K, N, bias in SH:
        w = torch.randn(N, K, device=dev, generator=g) * 0.03
        p8, gs, zp, sw = quant.w4a8_quantize(w)
        e = torch.empty(0, device=dev, dtype=torch.float16); snk = (gs.float() * sw[:, None]).half()
        b = torch.randn(N, device=dev, generator=g).half() if bias else None
        for M in ROWS:
            x = torch.randn(M, K, device=dev, generator=g, dtype=torch.float32).half()
            w4a8_op._impl(x, p8, gs, zp, sw, b, e, quant.kmajor_scales(gs), quant.kmajor_scales(zp), e, e, e, 128, snk)
        torch.cuda.synchronize(); print("w4a8", K, N, "ok", flush=True)
elif which == "w4a8rot":                               # the Hadamard rotation: fused into the INT8 quantisation above 64 rows, a kernel of its own below
    for K, N, bias in SH:
        w = torch.randn(N, K, device=dev, generator=g) * 0.03
        p8, gs, zp, sw = quant.w4a8_quantize(w4a8_op.rotate_weight(w))
        e = torch.empty(0, device=dev, dtype=torch.float16); snk = (gs.float() * sw[:, None]).half()
        b = torch.randn(N, device=dev, generator=g).half() if bias else None
        for M in ROWS:
            x = torch.randn(M, K, device=dev, generator=g, dtype=torch.float32).half()
            w4a8_op._impl(x, p8, gs, zp, sw, b, e, quant.kmajor_scales(gs), quant.kmajor_scales(zp), e, e, e, 128, snk, True)
            w4a8_op.rotate(x)
        torch.cuda.synchronize(); print("w4a8rot", K, N, "ok", flush=True)
else:
    E, H, I, G, k = 8, 256, 256, 64, 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test" / "E2E"))
    from test_moe_grouped import _moe_weights
    w = _moe_weights(E, H, I, G, 1)
    for T in (1, 3, 17, 100, 300):
        x = torch.randn(T, H, device=dev, generator=g, dtype=torch.float32).half()
        ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:k] for _ in range(T)]); tw = torch.softmax(torch.randn(T, k, device=dev, generator=g), -1)
        moe.moe_ffn_impl(x, ids, tw, *w, G)
    torch.cuda.synchronize(); print("moe ok", flush=True)
