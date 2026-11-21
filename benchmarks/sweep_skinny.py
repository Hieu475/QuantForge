"""Sweep of the 32- and 64-row (skinny) schedules over twelve layer shapes (RTX 4050): which tile options win where.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/sweep_skinny.py
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import W4A16Gemm, pack_int4
from quantforge.autotune import device_time
import quantforge.torch_ops as ops
G = 128
base = "block-tile-n=128 smem-stages=1"
C = {
 32: {"cur": ops._SKINNY_OPTS[32],
      "cur+pf": ops._SKINNY_OPTS[32] + " prefetch-distance=1",
      "w16 k128": f"block-tile-m=32 warp-tile-m=16 warp-tile-n=16 {base} block-tile-k=128 prefetch-distance=1",
      "w16 k64": f"block-tile-m=32 warp-tile-m=16 warp-tile-n=16 {base} block-tile-k=64 prefetch-distance=1",
      "w32 k64": f"block-tile-m=32 warp-tile-m=16 warp-tile-n=32 {base} block-tile-k=64 prefetch-distance=1",
      "w64 k64": f"block-tile-m=32 warp-tile-m=16 warp-tile-n=64 {base} block-tile-k=64 prefetch-distance=1"},
 64: {"cur": ops._SKINNY_OPTS[64],
      "bm32 w16 k128": f"block-tile-m=32 warp-tile-m=16 warp-tile-n=16 {base} block-tile-k=128 prefetch-distance=1",
      "bm32 w32 k128": f"block-tile-m=32 warp-tile-m=16 warp-tile-n=32 {base} block-tile-k=128 prefetch-distance=1",
      "bm64 wm64 wn16 k64": f"block-tile-m=64 warp-tile-m=64 warp-tile-n=16 {base} block-tile-k=64 prefetch-distance=1",
      "bm64 wm32 wn32 k64": f"block-tile-m=64 warp-tile-m=32 warp-tile-n=32 {base} block-tile-k=64 prefetch-distance=1",
      "bm64 wm32 wn64 k64": f"block-tile-m=64 warp-tile-m=32 warp-tile-n=64 {base} block-tile-k=64 prefetch-distance=1"},
}
shapes = [(896,896),(896,1152),(1536,1536),(1536,2048),(896,9728),(1536,17920),(4864,896),(8960,1536),(4096,4096),(4096,11008),(11008,4096),(2048,2048)]
for M in (32, 64):
    names = list(C[M])
    print(f"M={M}  " + " | ".join(names))
    for (K, N) in shapes:
        g = torch.Generator(device="cuda").manual_seed(0)
        x = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
        w = pack_int4(torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g))
        sc = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
        zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
        out = torch.empty(M, N, device="cuda", dtype=torch.float32)
        row = []
        for n in names:
            try:
                k = W4A16Gemm.build(M, K, N, group_size=G, pad=False, pipeline_opts=C[M][n], accumulate=False)
                row.append(device_time(lambda: k.launch(x, w, sc, zp, out)) * 1e6)
            except Exception as e:
                row.append(float("nan"))
        best = min(range(len(row)), key=lambda i: row[i] if row[i] == row[i] else 1e9)
        print(f"  {K:5d}x{N:5d}: " + "  ".join(f"{t:7.1f}{'*' if i == best else ' '}" for i, t in enumerate(row)), flush=True)
