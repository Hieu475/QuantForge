"""One small W4A16 GEMM, compared with PyTorch. Exit code 0 = correct.

Meant to run under `compute-sanitizer` (see scripts/sanitize.sh): a single
launch keeps the sanitizer's runtime reasonable while still covering the
shared-memory staging, cp.async, Tensor Core and barrier logic.

    python python/quantforge/smoke.py ["<pipeline options>"] [M K N [group_size [bits]]]

group_size 0 / omitted = per-tensor scale; bits 4 (default), 8, or 9 = FP8 E4M3 weights (group-wise only).
"""
import re
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantforge.e2e import W4A16Gemm, pack_int4, reference_gemm  # noqa: E402


def main(argv):
    opts = argv[1] if len(argv) > 1 else ""
    M, K, N = (int(x) for x in argv[2:5]) if len(argv) >= 5 else (256, 128, 256)
    g = torch.Generator(device="cuda").manual_seed(0)
    act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    G = int(argv[5]) if len(argv) > 5 and int(argv[5]) else None
    bits = int(argv[6]) if len(argv) > 6 else 4
    lo, hi = (0, 16) if bits == 4 else (-128, 128)
    q = torch.randint(lo, hi, (K, N), device="cuda", dtype=torch.int8, generator=g)
    fp8 = bits == 9
    if fp8:   # random E4M3 values (NaN patterns excluded): the bytes are what the kernel reads
        w8 = ((torch.rand(K, N, device="cuda", generator=g) * 2 - 1) * 400).to(torch.float8_e4m3fn)
        q = w8.view(torch.int8)
        assert G, "FP8 weights need group-wise scales"
    if G:
        scale = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
        zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
    else:
        scale = torch.tensor(0.05, device="cuda", dtype=torch.float16)
        zp = torch.tensor(8, device="cuda", dtype=torch.int8)
    kern = W4A16Gemm.build(M, K, N, workdir=tempfile.mkdtemp(), pipeline_opts=opts,
                           group_size=G, bits=8 if fp8 else bits, fp8=fp8, pad=True)
    out = kern.run(act, pack_int4(q) if bits == 4 else q.contiguous(), scale, zp)
    if fp8:
        ref = act.float() @ (w8.float() * scale.float().repeat_interleave(G, 0)).half().float()  # f16 weights, as the kernel
    else:
        ref = reference_gemm(act, q, scale, zp)
    if "epilogue=swiglu" in opts:
        # the columns of q are gate_0, up_0, gate_1, up_1, ...: the kernel writes silu(gate_j) * up_j
        import torch.nn.functional as F
        sl = (slice(None), slice(0, None, 2)), (slice(None), slice(1, None, 2))
        gate = reference_gemm(act, q[sl[0]], scale[sl[0]] if scale.dim() else scale, zp[sl[0]] if zp.dim() else zp)
        up = reference_gemm(act, q[sl[1]], scale[sl[1]] if scale.dim() else scale, zp[sl[1]] if zp.dim() else zp)
        ref = F.silu(gate) * up
        out = out[:, : N // 2]
    m = re.search(r"epilogue=(relu|silu|gelu)", opts)
    if m:  # fused activation epilogue: out = act(A @ W)
        import torch.nn.functional as F
        ref = {"relu": F.relu, "silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh")}[m.group(1)](ref)
    err = (out - ref).abs().max().item()
    ok = err < 1e-4 * ref.abs().max().item() + 1e-3
    print(f"smoke M={M} K={K} N={N} G={G} bits={bits} opts={opts!r}: max|err|={err:.6f} {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
