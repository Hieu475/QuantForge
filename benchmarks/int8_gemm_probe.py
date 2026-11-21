"""Compiler-generated INT8 GEMM (``quantforge.e2e.Int8Gemm``) against cuBLASLt INT8 (``torch._int_mm``, column-major B) and cuBLAS fp16.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/int8_gemm_probe.py

Raw int32 result, no scales / epilogue / activation quantisation: it measures what the Tensor Core path of the compiler reaches. Every
result is checked bit-exact against cuBLASLt. Per shape both block-swizzle 0 and 16 are run and the faster is reported (swizzle helps
when the weight is larger than the L2). The GPU is pre-heated before each measurement (an unheated laptop GPU gives numbers 25 % low).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import INT8_GEMM_OPTS, Int8Gemm  # noqa: E402

SHAPES = [(2048, 4096, 4096), (4096, 4096, 4096), (512, 4096, 4096), (2048, 4096, 11008), (2048, 4864, 896), (2048, 896, 4864),
          (2048, 1536, 8960)]


def heat(sec=1.5):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    end = time.time() + sec
    while time.time() < end:
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()


def timeit(fn, n=8):
    fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(4):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n)
    return best


def main():
    print(f"{'shape':>18} | {'QuantForge int8':>22} | cuBLASLt int8 | cuBLAS fp16 | QF/fp16 | QF/cuBLASLt")
    for (M, K, N) in SHAPES:
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8, generator=g)
        bt = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8, generator=g)
        ref = torch._int_mm(a, bt.t())
        a16, b16 = torch.randn(M, K, device="cuda", dtype=torch.float16), torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = torch.empty(M, N, device="cuda", dtype=torch.int32)
        best = (1e9, None)
        for sw in (0, 16):
            k = Int8Gemm.build(M, K, N, pipeline_opts=f"{INT8_GEMM_OPTS} block-swizzle={sw}")
            k.launch(a, bt, c)
            torch.cuda.synchronize()
            assert torch.equal(c, ref), f"WRONG result for {(M, K, N)} swizzle {sw}"
            heat()
            t = timeit(lambda: k.launch(a, bt, c))
            if t < best[0]:
                best = (t, sw)
        heat()
        t_ci = timeit(lambda: torch._int_mm(a, bt.t()))
        heat()
        t_f16 = timeit(lambda: a16 @ b16)
        op = 2 * M * K * N / 1e9
        print(f"{M:5d}x{K:5d}x{N:5d} | {op / best[0]:6.1f} TOPS (swizzle {best[1]:2d}) | {op / t_ci:6.1f} TOPS | {op / t_f16:5.1f} TFLOP/s | "
              f"{t_f16 / best[0]:5.2f}x | {t_ci / best[0]:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
