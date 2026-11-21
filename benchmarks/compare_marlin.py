"""
QuantForge vs Marlin (vLLM's hand-written W4A16 CUDA kernel) vs cuBLAS fp16.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/compare_marlin.py \
        --shape 1x4096x4096 --shape 16x4096x11008 --shape 2048x4096x4096 [--group-size 128] [--json out.json]

Same weights for all three: a random fp16 matrix is quantized to INT4 with
vLLM's own helper (symmetric, zero point 8, group size G), then
  * Marlin   : vllm._custom_ops.marlin_gemm on the Marlin-repacked weights
  * QuantForge: the generated kernel (decode schedule for M <= 16) on the same
                INT4 values / scales (zero point 8)
  * cuBLAS   : fp16 GEMM with the pre-dequantised weight (4x the memory traffic)
The activations are re-touched after the flush so they are L2-resident, as the output of the
previous layer is (otherwise the result depends on whether a harness happens to copy them).
All numbers are *device* kernel time from the CUPTI trace (best of N), so host
launch overhead of the prototype runtime shim does not count against
QuantForge. The outputs are compared with each other and with the fp32 reference.

Needs: an NVIDIA GPU with sm_80+, torch, vllm. (pip install vllm)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantforge.e2e import W4A16Gemm, pack_int4  # noqa: E402
import quantforge.torch_ops as ops  # noqa: E402


_FLUSH = None


_WARM = []  # tensors that are L2-resident in a real model (activations: the previous layer wrote them)


def _flush_l2():
    """Evict the L2 (RTX 4050: 24 MB) so each call reads its weights from DRAM, as the
    layers of a real model do. Without this a 8 MB INT4 weight stays L2-resident across
    benchmark iterations and the result measures L2, not memory bandwidth."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _FLUSH.max()  # read, not write: a memset would leave dirty lines whose write-back slows the next kernel
    for t in _WARM:  # re-touch the activations so every kernel sees them hot in L2 (as after the previous layer)
        t.max()


def _warm_gpu(seconds=1.0):
    """Keep the GPU busy so it leaves its idle P-state before timing starts."""
    import time

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()


def device_time(fn, names, iters=30, flush=True):
    """Best per-call device time (s) of the kernels whose name contains one of `names`
    (summed per call when a call launches several)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                if flush:
                    _flush_l2()
                fn()
            torch.cuda.synchronize()
        kern = sorted(
            (e for e in prof.events() if e.device_time > 0 and any(n in e.name for n in names)),
            key=lambda e: e.time_range.start,
        )
        if not kern:
            raise RuntimeError(f"no kernel matching {names} in the trace")
        per_call = len(kern) // iters or 1
        times = [sum(k.device_time for k in kern[i:i + per_call]) for i in range(0, len(kern), per_call)]
        best = min(best, min(times) / 1e6)
    return best


def run(M, K, N, G, qf_opts, ns_warm=False, rounds=3):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    dev = "cuda"
    torch.manual_seed(0)
    w = torch.randn(K, N, device=dev, dtype=torch.float16) / 4
    act = (torch.rand(M, K, device=dev, dtype=torch.float16) * 2 - 1)
    w_ref, marlin_q, marlin_s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, G, False)
    workspace = marlin_make_workspace_new(w.device)

    # recover the INT4 values / scales vLLM quantised with, to feed QuantForge
    scale = (w.float().reshape(K // G, G, N).abs().amax(1) / 7)  # only used for shape; replaced below
    # gptq_quantize_weights: w_ref = (q - 8) * s  ->  q = round(w_ref / s) + 8, s per (group, n)
    from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
    w_ref2, q_w, s, _, _ = gptq_quantize_weights(w, scalar_types.uint4b8, G, False)
    assert torch.equal(w_ref2, w_ref)
    q = q_w.reshape(K, N).to(torch.int8)                       # 0..15 (zero point 8)
    s = s.reshape(K // G, N).to(torch.float16).contiguous()
    zp = torch.full_like(s, 8, dtype=torch.int8)
    packed = pack_int4(q)

    ref = act.float() @ w_ref.float()
    _WARM[:] = [act]

    def marlin():
        return ops.marlin_gemm(act, None, marlin_q, None, marlin_s, None, None, None, g_idx, sort_idx,
                               workspace, scalar_types.uint4b8, M, N, K, is_k_full=True,
                               use_atomic_add=False, use_fp32_reduce=True)

    if qf_opts == "autotune":
        from quantforge.autotune import autotune

        kern = autotune(M, K, N, group_size=G, verbose=True)
        qf_opts = kern.compiled.lowered_mlir.parent.name  # informational only
    else:
        kern = W4A16Gemm.build(M, K, N, group_size=G, pad=True, pipeline_opts=qf_opts, acc="f32")
    out_qf = torch.zeros(M, N, device=dev, dtype=torch.float32)

    def qf():
        out_qf.zero_()
        return kern.run(act, packed, s, zp, out=out_qf)

    wd = w_ref.contiguous()

    def cublas():
        return act @ wd

    res = {"M": M, "K": K, "N": N, "group_size": G, "quantforge_opts": qf_opts}
    err_m = (marlin().float() - ref).abs().max().item()
    err_q = (qf().float() - ref).abs().max().item()
    res.update(err_marlin=err_m, err_quantforge=err_q, ref_max=ref.abs().max().item())
    flush = not ns_warm
    # Interleave the kernels over several rounds (each preceded by a GPU warm-up that
    # brings the SM and memory clocks to their working P-state) and keep each one's best:
    # laptop clocks drift by +-30 % between sequential measurements.
    t_m = t_q = t_c = float("inf")
    for _ in range(rounds):
        _warm_gpu()
        t_m = min(t_m, device_time(marlin, ["marlin", "Marlin"], flush=flush))
        t_q = min(t_q, device_time(qf, ["a16_gemm"], flush=flush))
        t_c = min(t_c, device_time(cublas, ["gemm", "cutlass", "gemv", "sm80", "sm89", "ampere"], flush=flush))
    flops = 2 * M * N * K
    wbytes = K * N / 2
    for name, t in (("marlin", t_m), ("quantforge", t_q), ("cublas_fp16", t_c)):
        res[name] = {"us": t * 1e6, "tflops": flops / t / 1e12, "weight_gbs_int4": wbytes / t / 1e9}
    res["quantforge_vs_marlin"] = t_m / t_q  # > 1: QuantForge faster
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", required=True, metavar="MxKxN")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--warm", action="store_true", help="do not flush L2 between calls (weights stay cache-resident)")
    ap.add_argument("--autotune", action="store_true", help="tune QuantForge's tile options per shape (cached)")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved measurement rounds (best is kept)")
    ap.add_argument("--json")
    ns = ap.parse_args()
    out = []
    print(f"{'shape':>18} {'Marlin us':>10} {'QF us':>9} {'cuBLAS us':>10} {'QF/Marlin':>10}   max|err| (marlin / QF, ref max)")
    for sh in ns.shape:
        M, K, N = (int(x) for x in sh.lower().split("x"))
        opts = "autotune" if ns.autotune else ops.default_opts(ops.bucket_rows(M), K, N)   # the schedule the torch op really uses
        r = run(M, K, N, ns.group_size, opts, ns.warm, ns.rounds)
        out.append(r)
        print(f"{sh:>18} {r['marlin']['us']:10.1f} {r['quantforge']['us']:9.1f} {r['cublas_fp16']['us']:10.1f} "
              f"{r['quantforge_vs_marlin']:9.2f}x   {r['err_marlin']:.4f} / {r['err_quantforge']:.4f}  ({r['ref_max']:.1f})")
    if ns.json:
        Path(ns.json).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                             "results": out}, indent=2) + "\n")
        print("wrote", ns.json)


if __name__ == "__main__":
    main()
