"""QuantForge against Marlin, gemlite (Triton), bitsandbytes NF4 and cuBLAS fp16 on the same weights.

    QUANTFORGE_BUILD_DIR=<build> python benchmarks/compare_baselines.py \
        --shape 1x4096x4096 --shape 64x4096x4096 --shape 2048x4096x4096 [--json out.json]

Same method as compare_marlin.py (device kernel time from the CUPTI trace, L2 flushed by reading 256 MB before
every call, activations re-touched, kernels measured in interleaved rounds after a GPU warm-up, best of N).
All INT4 kernels get the same quantised weights (symmetric, zero point 8, group size G); bitsandbytes uses its
own NF4 code book on the same fp16 matrix (a different quantiser, listed for speed only).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_marlin as cm  # noqa: E402
from quantforge.e2e import W4A16Gemm, pack_int4  # noqa: E402


def run(M, K, N, G, rounds):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights
    from vllm.scalar_type import scalar_types
    from quantforge.autotune import autotune

    dev = "cuda"
    torch.manual_seed(0)
    w = torch.randn(K, N, device=dev, dtype=torch.float16) / 4
    act = torch.rand(M, K, device=dev, dtype=torch.float16) * 2 - 1
    w_ref, marlin_q, marlin_s, g_idx, sort_idx, _ = marlin_quantize(w, scalar_types.uint4b8, G, False)
    workspace = marlin_make_workspace_new(w.device)
    _, q_w, s, _, _ = gptq_quantize_weights(w, scalar_types.uint4b8, G, False)
    q = q_w.reshape(K, N).to(torch.int8)
    s = s.reshape(K // G, N).to(torch.float16).contiguous()
    zp = torch.full_like(s, 8, dtype=torch.int8)
    packed = pack_int4(q)
    ref = act.float() @ w_ref.float()
    cm._WARM[:] = [act]

    kern = autotune(M, K, N, group_size=G)
    out_qf = torch.zeros(M, N, device=dev, dtype=torch.float32)
    fns = {}
    fns["QuantForge"] = (lambda: (out_qf.zero_(), kern.run(act, packed, s, zp, out=out_qf))[1], ["a16_gemm"], out_qf)
    fns["Marlin"] = (lambda: ops.marlin_gemm(act, None, marlin_q, None, marlin_s, None, None, None, g_idx, sort_idx,
                                             workspace, scalar_types.uint4b8, M, N, K, is_k_full=True,
                                             use_atomic_add=False, use_fp32_reduce=True), ["marlin", "Marlin"], None)
    # gemlite: W_q [out, in] values 0..15, scales / zeros [out, in/G]
    try:
        from gemlite import DType, GemLiteLinear
        lin = GemLiteLinear(4, G, K, N, input_dtype=DType.FP16, output_dtype=DType.FP16)
        lin.pack(q.t().contiguous().to(torch.uint8), s.t().contiguous(), zp.t().contiguous().half(), None)
        fns["gemlite"] = (lambda: lin(act), ["gemv_INT", "gemm_splitK_INT", "gemm_INT", "gemm_splitK"], None)
        lin(act)
    except Exception as e:  # noqa: BLE001
        print("  (gemlite unavailable:", type(e).__name__, str(e)[:60], ")")
    try:
        import bitsandbytes as bnb
        wb = w.t().contiguous()                                   # [out, in] fp16
        qw, qs = bnb.functional.quantize_4bit(wb, blocksize=64, quant_type="nf4")
        fns["bitsandbytes NF4"] = (lambda: bnb.matmul_4bit(act, qw.t(), qs),
                                   ["4bit", "Dequantize", "gemv", "gemm", "cutlass", "sm89", "ampere"], None)
        fns["bitsandbytes NF4"][0]()
    except Exception as e:  # noqa: BLE001
        print("  (bitsandbytes unavailable:", type(e).__name__, str(e)[:60], ")")
    wd = w_ref.contiguous()
    fns["cuBLAS fp16"] = (lambda: act @ wd, ["gemm", "cutlass", "gemv", "sm80", "sm89", "ampere"], None)

    out_qf.zero_()
    errs = {"QuantForge": (kern.run(act, packed, s, zp, out=out_qf).float() - ref).abs().max().item(),
            "Marlin": (fns["Marlin"][0]().float() - ref).abs().max().item()}
    if "gemlite" in fns:
        errs["gemlite"] = (fns["gemlite"][0]().float() - ref).abs().max().item()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        cm._warm_gpu()
        for name, (fn, names, _) in fns.items():
            best[name] = min(best[name], cm.device_time(fn, names, flush=True))
    return {"M": M, "K": K, "N": N, "group_size": G, "us": {k: v * 1e6 for k, v in best.items()}, "max_err_vs_f32_ref": errs,
            "ref_max": ref.abs().max().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--json")
    ns = ap.parse_args()
    results = []
    for sh in ns.shape:
        M, K, N = (int(x) for x in sh.lower().split("x"))
        r = run(M, K, N, ns.group_size, ns.rounds)
        results.append(r)
        cells = "  ".join(f"{k}: {v:7.1f} us" for k, v in r["us"].items())
        print(f"{sh:>18}  {cells}", flush=True)
    if ns.json:
        Path(ns.json).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                             "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
