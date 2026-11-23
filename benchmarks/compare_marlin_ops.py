"""The torch ops the vLLM plugin calls (W4A16 `w4a16_linear`, W4A8 `w4a8_hybrid`, compact layout) against vLLM's Marlin GEMM, per row count, in one CUDA-graph replay each.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/compare_marlin_ops.py --kn 4096x4096 --kn 4096x11008 --rows 16 32 64 96 128 192 256

Unlike `compare_marlin.py` (one kernel, one schedule) this times what a decode step runs: the op with the planned schedule (tile plans, atomic split-K, K-fold) including its
helper kernels (activation quantisation for W4A8, the zeroing of the output...). Each call is replayed from a CUDA graph (no launch gaps), the L2 is flushed before every replay (cold weights),
and the three variants are interleaved over several rounds after a warm-up (the laptop's clocks drift): the median of the replays, best round. `ratio` > 1: faster than Marlin.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import quantforge.torch_ops as qt  # noqa: E402
from quantforge import atomic_plan, int8_plan, kfold, quant, w4a8_op  # noqa: E402,F401
from quantforge.e2e import pack_int4  # noqa: E402

_FLUSH = torch.zeros(256 * 2 ** 20, dtype=torch.uint8, device="cuda")


def warm(seconds=1.0):
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()


def graph_of(fn):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def replay_time(g, act, n=40):
    ts = []
    for _ in range(n):
        _FLUSH.max()
        act.max()                                   # the activations are hot in L2 (the previous layer wrote them)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def build(M, K, N, G=128):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
    from vllm.scalar_type import scalar_types

    dev = "cuda"
    torch.manual_seed(0)
    w = torch.randn(K, N, device=dev, dtype=torch.float16) / 4
    act = torch.rand(M, K, device=dev, dtype=torch.float16) * 2 - 1
    _, marlin_q, marlin_s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, G, False)
    ws = marlin_make_workspace_new(w.device)
    _, q_w, s, _, _ = gptq_quantize_weights(w, scalar_types.uint4b8, G, False)
    q = q_w.reshape(K, N).to(torch.int8)
    s = s.reshape(K // G, N).to(torch.float16).contiguous()
    zp = torch.full_like(s, 8, dtype=torch.int8)
    packed = pack_int4(q)
    atomic_plan.preplan(K, N, G)                                      # the plugin plans the atomic split-K of every layer shape when the weights are installed
    folded = kfold.fold_k(packed, s, zp, kfold.plan_ksplit(K, N, G, 4)) if kfold.plan_ksplit(K, N, G, 4) > 1 else kfold.no_fold(torch.device(dev))
    p8, gs, zp8, sw = quant.w4a8_quantize(w.t().float() * 0.5)
    e = kfold.no_fold(torch.device(dev))
    scale_nk = (gs.float() * sw.float()[:, None]).half()
    gs_k, zp_k = quant.kmajor_scales(gs), quant.kmajor_scales(zp8)           # the compact layer keeps the integer scales K-major for the INT8 kernel (vllm_plugin)
    int8_plan.preplan(K, N)                                          # measured schedules of the INT8 kernel for this shape (once, cached)

    def marlin():
        return ops.marlin_gemm(act, None, marlin_q, None, marlin_s, None, None, None, g_idx, sort_idx, ws, scalar_types.uint4b8, M, N, K,
                               is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)

    def w4a16():
        return kfold.linear(act, packed, s, zp, folded, G, None, None)

    def w4a8():
        return torch.ops.quantforge.w4a8_hybrid(act, p8, gs, zp8, sw.float(), None, e[0], gs_k, zp_k, e[0], e[1], e[2], 128, scale_nk)

    return act, marlin, w4a16, w4a8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn", action="append", required=True, metavar="KxN")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 16, 32, 64, 96, 128, 192, 256])
    ap.add_argument("--rounds", type=int, default=3)
    ns = ap.parse_args()
    print(f"{'M x K x N':>18} {'Marlin':>8} {'W4A16':>8} {'W4A8':>8}   {'W4A16/Marlin':>12} {'W4A8/Marlin':>12}   (us per call, cold weights)")
    for kn in ns.kn:
        K, N = (int(x) for x in kn.lower().split("x"))
        for M in ns.rows:
            act, marlin, w4a16, w4a8 = build(M, K, N)
            gs = {n: graph_of(f) for n, f in (("marlin", marlin), ("w4a16", w4a16), ("w4a8", w4a8))}
            best = {n: float("inf") for n in gs}
            for _ in range(ns.rounds):
                warm()
                for n, g in gs.items():
                    best[n] = min(best[n], replay_time(g, act))
            print(f"{M:5d}x{K:5d}x{N:6d} {best['marlin']:8.1f} {best['w4a16']:8.1f} {best['w4a8']:8.1f}   {best['marlin'] / best['w4a16']:11.2f}x {best['marlin'] / best['w4a8']:11.2f}x", flush=True)


if __name__ == "__main__":
    main()
