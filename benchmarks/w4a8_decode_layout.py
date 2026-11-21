"""Decode kernel that reads the W4A8 layout (nibbles along K) against QuantForge's dedicated W4A16 decode kernel.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/w4a8_decode_layout.py M K N          # M <= 16, e.g. 16 4096 4096

The f16 ``matmul_transpose_b`` whose weight tile is built by a ``linalg.generic`` producer (fused into the loads) from the same nibbles the
W4A8 prefill kernel reads: one copy of the weights for prefill and decode (``e2e.W4A8DecodeGemm``, the "compact" mode of the vLLM method).
Device time, L2 flushed, interleaved rounds; ``DEC_OPTS`` overrides the schedule.
"""
import sys, os, time, ctypes, tempfile, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from torch.profiler import profile, ProfilerActivity
from quantforge.e2e import compile_ir, _descriptor, _descriptor_type, _current_raw_stream, W4A16Gemm
import quantforge.torch_ops as ops
from quantforge.quant import w4a8_quantize
from quantforge import w4a8_op

M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); G = 128
from quantforge.e2e import W4A8_DECODE_OPTS, W4A8DecodeGemm
opts = os.environ.get("DEC_OPTS")
dec = W4A8DecodeGemm.build(M, K, N, pipeline_opts=(opts + " accumulate=false") if opts else W4A8_DECODE_OPTS)
dev = "cuda"; g = torch.Generator(device=dev).manual_seed(0)
w = torch.randn(N, K, device=dev, generator=g) * 0.05
packed, gs, zp, sw = w4a8_quantize(w)
sc = (gs.float() * sw[:, None]).half().contiguous()
x = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
out = torch.zeros(M, N, device=dev)
def run_new():
    dec.launch(x, packed, zp, sc, out)
run_new(); torch.cuda.synchronize()
from quantforge import quant
ref = x.float() @ quant.w4a8_dequantize(packed, gs, zp, sw).t()
print("max err", (out - ref).abs().max().item(), "ref", ref.abs().max().item())
p16, s16, z16 = w4a8_op.from_w4a8(packed, gs, zp, sw)
k16 = W4A16Gemm.build(M, K, N, group_size=G, pad=True, pipeline_opts=ops.default_opts(ops.bucket_rows(M), K, N), acc="f32")
o16 = torch.zeros(M, N, device=dev)
def run_old():
    o16.zero_(); return k16.run(x, p16, s16, z16, out=o16)
flush = torch.zeros(256 * 2**20, dtype=torch.uint8, device=dev)
def warm():
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float16); t = time.time()
    while time.time() - t < 1: 
        for _ in range(20): a @ a
        torch.cuda.synchronize()
def best(f, names):
    for _ in range(3): f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20): flush.max(); f()
        torch.cuda.synchronize()
    ts = [e.device_time for e in p.events() if e.device_time > 0 and any(n in e.name for n in names)]
    return min(ts)
b = {"new": 1e9, "old": 1e9}
for _ in range(4):
    warm(); b["new"] = min(b["new"], best(run_new, ["dec"])); b["old"] = min(b["old"], best(run_old, ["a16_gemm"]))
print(f"M={M} K={K} N={N}: W4A8-layout f16 kernel {b['new']:.0f} us, W4A16 kernel {b['old']:.0f} us, ratio {b['new']/b['old']:.2f}")
