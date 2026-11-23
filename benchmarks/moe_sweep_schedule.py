"""Sweep the pipeline options of one grouped MoE GEMM at a token count (cold weights, uniform random routing); prints the six best next to the default.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/moe_sweep_schedule.py <tokens> <swiglu|f16>
"""
import sys
import itertools, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge import moe
from quantforge.e2e import pack_int4
E, H, I, G, k = 64, 2048, 1024, 128, 8
T = int(sys.argv[1]); which = sys.argv[2]            # swiglu | f16
dev = "cuda"; gen = torch.Generator(device=dev).manual_seed(0)
Tb = moe.bucket_tokens(T); mcap = moe._row_tile(Tb, k, E); P = moe._problems(Tb, k, E, mcap)
K, N = (H, 2 * I) if which == "swiglu" else (I, H)
packed = torch.randint(-128, 127, (E, K, N // 2), device=dev, dtype=torch.int8, generator=gen)
scale = (torch.rand(E, K // G, N, device=dev, generator=gen) * .02 + .005).half(); zero = torch.full((E, K // G, N), 8, device=dev, dtype=torch.int8)
ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:k] for _ in range(T)])
dest, token, dsort, meta, valid = moe.route(ids, E, mcap, P)
rows = torch.randn(P * mcap + 1, K, device=dev, generator=gen).half()
out = torch.empty(P * mcap, N, device=dev, dtype=torch.float32 if which == "swiglu" else torch.float16)
flush = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
active = int((meta >= 0).sum())
print(f"T={T} mcap={mcap} P={P} active problems={active} experts={len(set(meta[meta>=0].tolist()))} weights/expert={(K*N//2+K//G*N*3)/2**20:.2f} MiB")
def run(opts):
    try:
        kern = moe.GroupedGemm.build(mcap, K, N, E, P, G, which, opts)
    except Exception as e:
        return None
    for _ in range(3): kern.launch(rows, packed, scale, zero, out, meta)
    ts = []
    for _ in range(15):
        flush.zero_(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); kern.launch(rows, packed, scale, zero, out, meta); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return ts[len(ts) // 2]
base = moe._opts(mcap, K, N)
res = [(run(base), base + "  [default]")]
mid = f"block-tile-m={mcap} warp-tile-m={min(mcap,32)}"
cands = []
for bn, wn, bk, st, pd in itertools.product((32, 64, 128), (8, 16, 32), (64, 128, 256), (1, 2, 3), (0, 1)):
    if wn > bn or (st > 1 and pd): continue
    if mcap == 16: cands.append(f"decode=true block-tile-n={bn} warp-tile-n={wn} block-tile-k={bk} smem-stages={st} prefetch-distance={pd}")
    else: cands.append(f"block-tile-m={mcap} warp-tile-m={min(mcap,32)} block-tile-n={bn} warp-tile-n={max(wn,16)} block-tile-k={bk} smem-stages={st} prefetch-distance={pd}")
seen = set()
for c in cands:
    if c in seen: continue
    seen.add(c); t = run(c)
    if t: res.append((t, c))
res = [r for r in res if r[0]]; res.sort()
for t, o in res[:6]: print(f"{t:8.1f} us  {o}")
print("default:", [f"{t:.1f}" for t, o in res if "[default]" in o])
