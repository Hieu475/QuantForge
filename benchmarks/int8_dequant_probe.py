"""INT8 GEMM with the dequantisation fused into the store (``Int8GemmDq``) against the raw int32 GEMM (+ a torch epilogue) and cuBLAS fp16.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/int8_dequant_probe.py

Every fused result is checked equal to the f32 reference rounded once to f16. The GPU is pre-heated before each measurement.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import INT8_GEMM_OPTS, Int8Gemm, Int8GemmDq, int8_gemm_swizzle  # noqa: E402
def heat(sec=1.5):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16); end = time.time() + sec
    while time.time() < end:
        for _ in range(20): x @ x
        torch.cuda.synchronize()
def timeit(fn, n=8):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(4):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / n)
    return best
print(f"{'shape':>18} | fp16 ms | int32 GEMM ms | + torch epilogue ms | fused dq ms | fused vs fp16 | fused vs unfused")
for (M, K, N) in [(2048, 4096, 4096), (4096, 4096, 4096), (2048, 4096, 11008), (2048, 4864, 896), (2048, 896, 4864), (2048, 1536, 8960), (512, 4096, 4096)]:
    opts = f"{INT8_GEMM_OPTS} block-swizzle={int8_gemm_swizzle(K, N)}"
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8, generator=g)
    bt = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8, generator=g)
    sx = torch.rand(M, device="cuda", generator=g) * 0.02 + 0.001; sw = torch.rand(N, device="cuda", generator=g) * 0.02 + 0.001
    bias = torch.randn(N, device="cuda", generator=g)
    a16, b16 = torch.randn(M, K, device="cuda", dtype=torch.float16), torch.randn(K, N, device="cuda", dtype=torch.float16)
    raw = Int8Gemm.build(M, K, N, pipeline_opts=opts); fused = Int8GemmDq.build(M, K, N, pipeline_opts=opts)
    c = torch.empty(M, N, device="cuda", dtype=torch.int32); y = torch.empty(M, N, device="cuda", dtype=torch.float16)
    swh = sw.half(); sxh = sx
    fused.launch(a, bt, sx, sw, bias, y); torch.cuda.synchronize()
    ref = (torch._int_mm(a, bt.t()).float() * sx[:, None] * sw[None, :] + bias[None, :]).half()
    assert torch.equal(y, ref), f"fused result differs for {(M,K,N)}"
    def unfused():
        raw.launch(a, bt, c); return (c.float() * sx[:, None] * sw[None, :] + bias[None, :]).half()
    heat(); t16 = timeit(lambda: a16 @ b16)
    heat(); t_raw = timeit(lambda: raw.launch(a, bt, c))
    heat(); t_unf = timeit(unfused)
    heat(); t_fus = timeit(lambda: fused.launch(a, bt, sx, sw, bias, y))
    print(f"{M:5d}x{K:5d}x{N:5d} | {t16:7.3f} | {t_raw:13.3f} | {t_unf:19.3f} | {t_fus:11.3f} | {t16/t_fus:12.2f}x | {t_unf/t_fus:15.2f}x", flush=True)
