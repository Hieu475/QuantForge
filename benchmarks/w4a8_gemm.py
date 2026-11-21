"""W4A8 (INT4 weights expanded to int8 in the tile load, INT8 tensor cores) against INT8 W8A8, QuantForge's own W4A16 and cuBLAS fp16.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/w4a8_gemm.py --shape 2048x4096x4096 --shape 2048x4096x11008

Device time of the GEMM kernel(s) from the CUPTI trace, L2 flushed before every call, the variants interleaved over several rounds in
one session (each preceded by a warm-up that brings the clocks up: a laptop GPU drifts by ~10 % between sequential measurements).
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.torch_ops as ops  # noqa: E402
from quantforge.e2e import INT8_GEMM_OPTS, Int8GemmDq, W4A8GemmDq, W4A16Gemm, int8_gemm_swizzle, pack_int4  # noqa: E402
from quantforge.quant import w4a8_quantize  # noqa: E402


def warm(seconds=1.0):
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()


def best_us(fn, names, flush, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            flush.max()
            fn()
        torch.cuda.synchronize()
    ks = sorted((e for e in p.events() if e.device_time > 0 and any(n in e.name for n in names)), key=lambda e: e.time_range.start)
    per = max(len(ks) // iters, 1)
    return min(sum(k.device_time for k in ks[i:i + per]) for i in range(0, len(ks), per))


def run(M, K, N, rounds):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    a16 = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
    w = torch.randn(N, K, device=dev, generator=g) / 4
    xq = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev, generator=g)
    sx = torch.rand(M, device=dev, generator=g) * 0.01 + 0.001
    sw = torch.rand(N, device=dev, generator=g) * 0.01 + 0.001
    y = torch.empty(M, N, dtype=torch.float16, device=dev)
    flush = torch.zeros(256 * 2 ** 20, dtype=torch.uint8, device=dev)

    wh = w.half().t().contiguous()
    cublas = lambda: a16 @ wh                                                                    # noqa: E731
    k8 = Int8GemmDq.build(M, K, N, bias=False, pipeline_opts=f"{INT8_GEMM_OPTS} block-swizzle={int8_gemm_swizzle(K, N)}")
    w8 = (w / w.abs().amax(1, keepdim=True) * 127).round().to(torch.int8)
    w88 = lambda: k8.launch(xq, w8, sx, sw, None, y)                                             # noqa: E731
    k4 = W4A8GemmDq.build(M, K, N, bias=False)
    packed, gs, zp, sw4 = w4a8_quantize(w)
    w48 = lambda: k4.launch(xq, packed, gs, zp, sx, sw, None, y)                                 # noqa: E731
    G = 128
    scale = (w.reshape(N, K // G, G).abs().amax(2) / 7).t().contiguous().half()
    q = ((w.t().reshape(K // G, G, N) / scale[:, None, :].float()).round().clamp(-8, 7) + 8).reshape(K, N).to(torch.int8)
    zero = torch.full(scale.shape, 8, dtype=torch.int8, device=dev)
    kq = W4A16Gemm.build(M, K, N, group_size=G, pad=True, pipeline_opts=ops.default_opts(ops.bucket_rows(M), K, N), acc="f32")
    pk = pack_int4(q).contiguous()
    out = torch.zeros(M, N, device=dev, dtype=torch.float32)
    w416 = lambda: (out.zero_(), kq.run(a16, pk, scale, zero, out=out))                           # noqa: E731

    variants = {"cublas_fp16": (cublas, ["gemm", "cutlass", "sm80", "sm89", "ampere"]), "qf_w4a16": (w416, ["a16_gemm"]),
                "qf_int8_w8a8": (w88, ["i8_gemm"]), "qf_w4a8": (w48, ["w4a8_gemm"])}
    best = {k: float("inf") for k in variants}
    for _ in range(rounds):
        warm()
        for k, (f, names) in variants.items():
            best[k] = min(best[k], best_us(f, names, flush))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", required=True, metavar="MxKxN")
    ap.add_argument("--rounds", type=int, default=4)
    ns = ap.parse_args()
    print(f"{'shape':>16} " + " ".join(f"{k:>14}" for k in ("cublas_fp16", "qf_w4a16", "qf_int8_w8a8", "qf_w4a8")) + "   w4a8 vs fp16 / vs w4a16 / vs w8a8")
    for sh in ns.shape:
        M, K, N = (int(x) for x in sh.lower().split("x"))
        b = run(M, K, N, ns.rounds)
        cells = " ".join(f"{b[k]:8.0f}us {2 * M * N * K / b[k] / 1e6:4.0f}T" for k in ("cublas_fp16", "qf_w4a16", "qf_int8_w8a8", "qf_w4a8"))
        print(f"{sh:>16} {cells}   {b['cublas_fp16'] / b['qf_w4a8']:.2f}x / {b['qf_w4a16'] / b['qf_w4a8']:.2f}x / {b['qf_int8_w8a8'] / b['qf_w4a8']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
