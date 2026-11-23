"""Time `quantforge.moe.moe_ffn_impl` (routing + two grouped GEMMs + combine) with cold weights.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/moe_op_bench.py [E H I group top_k]      # default: 64 experts, H 2048, I 1024, group 128, top-8 (OLMoE)
"""
import sys
import time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge import moe, quant
from quantforge.e2e import pack_int4
E, H, I, G, k = [int(a) for a in sys.argv[1:6]] if len(sys.argv) > 5 else (64, 2048, 1024, 128, 8)
dev = "cuda"
gen = torch.Generator(device=dev).manual_seed(0)
def rand_q(K, N):
    q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int8, generator=gen)
    return pack_int4(q), (torch.rand(K // G, N, device=dev, generator=gen) * 0.02 + 0.005).half(), torch.randint(7, 9, (K // G, N), device=dev, dtype=torch.int8, generator=gen)
p13, s13, z13, p2, s2, z2 = [], [], [], [], [], []
for e in range(E):
    p, s, z = rand_q(H, 2 * I); p13.append(p); s13.append(s); z13.append(z)
    p, s, z = rand_q(I, H); p2.append(p); s2.append(s); z2.append(z)
st = lambda l: torch.stack(l).contiguous()
w = (st(p13), st(s13), st(z13), st(p2), st(s2), st(z2))
mb = sum(t.numel() * t.element_size() for t in w) / 2**20
print(f"E={E} H={H} I={I} k={k}: weights {mb:.0f} MiB per layer")
flush = torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)
def timeit(fn, n=20):
    for _ in range(3): fn()
    ts = []
    for _ in range(n):
        flush.zero_(); a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    ts.sort(); return ts[len(ts) // 2]
for T in (1, 4, 16, 64, 256, 1024):
    x = (torch.randn(T, H, device=dev, generator=gen) * 0.5).half()
    ids = torch.stack([torch.randperm(E, device=dev, generator=gen)[:k] for _ in range(T)])
    tw = torch.softmax(torch.randn(T, k, device=dev, generator=gen), -1)
    t = timeit(lambda: moe.moe_ffn_impl(x, ids, tw, *w, G))
    used = ids.unique().numel()
    byt = used * (mb / E) * 2**20
    print(f"T={T:5d}: {t:8.0f} us  ({used} experts used, {byt / 1e6:.1f} MB -> {byt / (t * 1e-6) / 1e9:.0f} GB/s effective)")
