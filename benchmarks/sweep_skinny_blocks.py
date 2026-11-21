"""Second sweep of the 32- / 64-row schedules: 32 / 64 / 128 column blocks, 2-8 warps, K tile 64 / 128 (74 candidates for 64 rows),
against the default of `torch_ops.default_opts`, over eight layer shapes.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/sweep_skinny_blocks.py 64      # or 32
"""
import sys, itertools, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import W4A16Gemm, pack_int4
from quantforge.autotune import device_time
import quantforge.torch_ops as ops
G = 128
M = int(sys.argv[1])
shapes = [(4096,4096),(896,9728),(4096,11008),(1536,17920),(1536,1536),(11008,4096),(2048,2048),(896,896)]
def cands(M):
    out = []
    for bm in ((64, 32) if M == 64 else (32,)):
        for bn in (32, 64, 128):
            for wm in (16, 32, 64):
                for wn in (8, 16, 32, 64):
                    if wm > bm or wn > bn or bm % wm or bn % wn: continue
                    warps = (bm // wm) * (bn // wn)
                    if warps not in (2, 4, 8): continue
                    for bk in (64, 128):
                        out.append(f"block-tile-m={bm} warp-tile-m={wm} warp-tile-n={wn} block-tile-n={bn} block-tile-k={bk} smem-stages=1 prefetch-distance=1")
    return out
C = cands(M)
print(f"M={M}: {len(C)} candidates", flush=True)
for (K, N) in shapes:
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    w = pack_int4(torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g))
    sc = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
    zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
    out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    def t(opts):
        try:
            k = W4A16Gemm.build(M, K, N, group_size=G, pad=False, pipeline_opts=opts, accumulate=False)
            return device_time(lambda: k.launch(x, w, sc, zp, out), iters=10, repeats=2) * 1e6
        except Exception:
            return float("inf")
    cur = t(ops.default_opts(M, K, N))
    res = sorted((t(o), o) for o in C)
    b = res[0]
    short = b[1].replace("block-tile-", "b").replace("warp-tile-", "w").replace(" smem-stages=1 prefetch-distance=1", "")
    print(f"  {K:5d}x{N:5d}: default {cur:6.1f}  best {b[0]:6.1f} ({cur/b[0]:.2f}x)  {short}", flush=True)
