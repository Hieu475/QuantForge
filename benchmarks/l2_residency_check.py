"""Is a kernel's weight traffic really served from DRAM? (checks the cold-cache benchmark method)

For one decode shape, times the QuantForge kernel and Marlin under
several L2 "flush" strategies, and optionally prints the DRAM bytes that `ncu` measures:

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/l2_residency_check.py [--shape 1x4096x4096]
    ncu --cache-control all --clock-control none -k regex:"a16_gemm|arlin" \
        --metrics dram__bytes_read.sum,gpu__time_duration.sum python benchmarks/l2_residency_check.py --once

A kernel that moves its bytes faster than the DRAM can deliver (peak 192 GB/s on the RTX 4050 laptop) must be
hitting L2 partly: its time then grows when the flush is made stronger.
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import W4A16Gemm, pack_int4  # noqa: E402

BIG = None


def make_flush(mode):
    global BIG
    if BIG is None:
        BIG = torch.zeros(1 << 30, dtype=torch.uint8, device="cuda")     # 1 GB
    if mode == "none":
        return lambda: None
    if mode == "read256":
        return lambda: BIG[: 256 << 20].max()
    if mode == "read1g":
        return lambda: BIG.max()
    if mode == "write1g":
        return lambda: BIG.fill_(1)
    if mode == "write+read":
        def f():
            BIG.fill_(1)
            BIG.max()
        return f
    raise KeyError(mode)


def timed(fn, names, flush, warm, iters=30, repeats=3):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                flush()
                for t in warm:
                    t.max()
                fn()
            torch.cuda.synchronize()
        ts = [e.device_time for e in prof.events() if e.device_time > 0 and any(n in e.name for n in names)]
        if ts:
            best = min(best, min(ts))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1x4096x4096")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--once", action="store_true", help="launch each kernel a few times and exit (for ncu)")
    ns = ap.parse_args()
    M, K, N = (int(x) for x in ns.shape.split("x"))
    G = ns.group_size

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
    from vllm.scalar_type import scalar_types

    torch.manual_seed(0)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16) / 4
    act = torch.rand(M, K, device="cuda", dtype=torch.float16) * 2 - 1
    _, marlin_q, marlin_s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, G, False)
    workspace = marlin_make_workspace_new(w.device)
    _, q_w, s, _, _ = gptq_quantize_weights(w, scalar_types.uint4b8, G, False)
    packed = pack_int4(q_w.reshape(K, N).to(torch.int8))
    s = s.reshape(K // G, N).to(torch.float16).contiguous()
    zp = torch.full_like(s, 8, dtype=torch.int8)
    from quantforge.autotune import autotune

    kern = autotune(M, K, N, group_size=G)          # the same autotuned kernel as compare_marlin.py
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    def qf():
        out.zero_()
        return kern.run(act, packed, s, zp, out=out)

    def marlin():
        return ops.marlin_gemm(act, None, marlin_q, None, marlin_s, None, None, None, g_idx, sort_idx, workspace,
                               scalar_types.uint4b8, M, N, K, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)

    if ns.once:
        flush = make_flush("read256")
        for fn in (qf, marlin):
            for _ in range(4):
                flush(); act.max(); fn()
            torch.cuda.synchronize()
        return
    nbytes = K * N // 2 + (K // G) * N * 2
    print(f"{M}x{K}x{N} g{G}: weights + scales = {nbytes / 1e6:.2f} MB (DRAM peak 192 GB/s -> {nbytes / 192e9 * 1e6:.1f} us)")
    print(f"{'flush':12s} {'QuantForge us':>14s} {'(GB/s)':>8s} {'Marlin us':>11s} {'(GB/s)':>8s}")
    for mode in ("none", "read256", "read1g", "write1g", "write+read"):
        flush = make_flush(mode)
        tq = timed(qf, ("a16_gemm",), flush, [act])
        tm = timed(marlin, ("arlin",), flush, [act])
        print(f"{mode:12s} {tq:14.1f} {nbytes / tq / 1e3:8.0f} {tm:11.1f} {nbytes / tm / 1e3:8.0f}")


if __name__ == "__main__":
    main()
