"""
Benchmark the QuantForge W4A16 GEMM against PyTorch's fp16 GEMM.

    QUANTFORGE_BUILD_DIR=<build> python python/quantforge/benchmark.py 2048 \
        "" "tensor-core=false" "fuse-dequant=false"

Reproducible results / regression checking:

    benchmark.py 1024 2048 4096 --json out.json            # write results + environment
    benchmark.py --shape 1x4096x4096 --group-size 128 --cold "decode=true"   # DRAM-resident weights
    benchmark.py 1024 2048 4096 --compare benchmarks/baseline_rtx4050_sm89.json
                                                           # exit 1 if >15 % slower

Sizes are the leading integer arguments; every following non-integer argument
is a --quantforge-gpu-pipeline option string ("" = defaults).

Each extra argument is a --quantforge-gpu-pipeline option string ("" = defaults).
QF_ACC=f16 switches the accumulator to f16 (full-rate Tensor Core on GeForce).
The PyTorch number uses a pre-dequantized fp16 weight (cuBLAS, Tensor Cores), so
it is an upper bound for what a compiled kernel can reach, not an apples-to-apples
INT4 comparison.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import warnings

import torch

warnings.filterwarnings("ignore", message="Warning: Profiler clears events")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantforge.e2e import W4A16Gemm, dequant_reference, pack_int4, reference_gemm  # noqa: E402


ACC = os.environ.get("QF_ACC", "f32")  # accumulator type: f32 (default) | f16


COLD = False        # --cold: flush the L2 before every timed call (weights come from DRAM)
_FLUSH = None


def _flush_l2(*hot):
    """Evict the L2 by *reading* 256 MB (a memset would leave dirty lines that slow the next kernel),
    then re-touch `hot` (the activations: in a model the previous layer has just written them)."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _FLUSH.max()
    for t in hot:
        t.max()


def _time_kernel(kern, act, packed, scale, zp, out, iters=20):
    """Best-of-`iters` kernel time in seconds (CUDA events around the launch)."""
    best = float("inf")
    for _ in range(iters):
        out.zero_()
        if COLD:
            _flush_l2(act)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        kern.run(act, packed, scale, zp, out=out)
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


def _kernel_time(kern, act, packed, scale, zp, out, iters=20):
    """Best-of-`iters` *device* time (s) of the generated kernel, from the CUPTI
    trace. Excludes the per-call host work of the prototype runtime shim
    (module load / launch through ctypes), which dominates tiny decode problems."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if COLD:
                _flush_l2(act)
            kern.run(act, packed, scale, zp, out=out)
        torch.cuda.synchronize()
    times = [e.device_time for e in prof.key_averages() if "a16_gemm" in e.key]
    if not times:  # profiler unavailable: fall back to event timing
        return _time_kernel(kern, act, packed, scale, zp, out, iters)
    # key_averages() averages over calls: use the per-call minimum from raw events
    raw = [e.device_time for e in prof.events() if "a16_gemm" in e.name and e.device_time > 0]
    return (min(raw) if raw else min(times)) / 1e6


def _run_size(shape, configs, group_size=None):
    """Benchmark one problem (M, K, N); returns (rows, torch_row)."""
    M, K, N = shape
    n = K
    g = torch.Generator(device="cuda").manual_seed(0)
    act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
    if group_size is None:
        scale = torch.tensor(0.05, device="cuda", dtype=torch.float16)
        zp = torch.tensor(8, device="cuda", dtype=torch.int8)
    else:
        scale = (torch.rand(K // group_size, N, device="cuda", generator=g) * 0.1 + 0.01).half()
        zp = torch.randint(0, 16, (K // group_size, N), device="cuda", dtype=torch.int8, generator=g)
    packed = pack_int4(q)
    ref = reference_gemm(act, q, scale, zp)
    flops = 2 * M * N * K

    kernels = {}
    for opts in configs:
        kern = W4A16Gemm.build(M, K, N, workdir=tempfile.mkdtemp(), pipeline_opts=opts, acc=ACC,
                               pad=True, group_size=group_size)
        out = torch.zeros(M, N, device="cuda", dtype=torch.float32 if ACC == "f32" else torch.float16)
        kern.run(act, packed, scale, zp, out=out)  # warm-up + JIT
        kernels[opts] = (kern, out, (out.float() - ref).abs().max().item())

    # Round-robin over the configurations so thermal / clock drift hits all
    # of them equally; report the best time of each.
    best = {opts: float("inf") for opts in configs}
    for _ in range(3):
        for opts in configs:
            kern, out, _ = kernels[opts]
            best[opts] = min(best[opts], _kernel_time(kern, act, packed, scale, zp, out))
    rows = [
        {
            "size": n,
            "m": M,
            "n": N,
            "group_size": group_size,
            "weight_gbs": (K * N / 2) / best[opts] / 1e9,
            "config": opts,
            "ms": best[opts] * 1e3,
            "gflops": flops / best[opts] / 1e9,
            "max_abs_err": kernels[opts][2],
        }
        for opts in configs
    ]

    w = dequant_reference(q, scale, zp)
    for _ in range(3):
        act @ w
    torch.cuda.synchronize()
    best_t = float("inf")
    for _ in range(20):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        act @ w
        e.record()
        e.synchronize()
        best_t = min(best_t, s.elapsed_time(e) / 1e3)
    torch_row = {"size": n, "m": M, "n": N, "group_size": group_size, "ms": best_t * 1e3, "gflops": flops / best_t / 1e9}
    return rows, torch_row


def environment():
    """Everything needed to judge whether two result files are comparable."""
    import platform
    import subprocess

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None

    props = torch.cuda.get_device_properties(0)
    repo = Path(__file__).resolve().parents[2]
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "accumulator": ACC,
        "git_commit": sh(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(sh(["git", "-C", str(repo), "status", "--porcelain"])),
    }


def _label(r):
    m, k, n = r.get("m", r["size"]), r["size"], r.get("n", r["size"])
    g = r.get("group_size")
    return (f"{n}^3" if m == k == n else f"{m}x{k}x{n}") + (f" g{g}" if g else "")


def compare(rows, baseline_path, tolerance):
    """Returns the list of regressions (config slower than baseline by > tolerance)."""
    base = json.loads(Path(baseline_path).read_text())
    base_gpu = base.get("environment", {}).get("gpu")
    if base_gpu != environment()["gpu"]:
        # Timings from another GPU say nothing about a regression.
        print(f"  baseline was recorded on '{base_gpu}', this is "
              f"'{environment()['gpu']}': comparison skipped")
        return []
    key = lambda r: (r["size"], r.get("m", r["size"]), r.get("n", r["size"]), r.get("group_size"), r["config"])  # noqa: E731
    ref = {key(r): r for r in base["results"]}
    bad = []
    for r in rows:
        b = ref.get(key(r))
        if b is None:
            continue
        ratio = r["gflops"] / b["gflops"]
        flag = "REGRESSION" if ratio < 1 - tolerance else "ok"
        print(f"  {_label(r)} {r['config'] or '<default>':30s} {r['gflops']:9.1f} vs baseline "
              f"{b['gflops']:9.1f} GFLOP/s ({ratio:5.2f}x) {flag}")
        if flag != "ok":
            bad.append((r, b))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("args", nargs="*", help="sizes (integers) followed by pipeline option strings")
    ap.add_argument("--json", metavar="PATH", help="write results + environment as JSON")
    ap.add_argument("--compare", metavar="BASELINE", help="compare with a JSON written by --json")
    ap.add_argument("--group-size", type=int, default=None, metavar="G",
                    help="group-wise scale/zero point along K (default: per-tensor)")
    ap.add_argument("--m", type=int, default=None, help="rows of the activation (default: = size)")
    ap.add_argument("--shape", action="append", default=[], metavar="MxKxN",
                    help="explicit problem shape (repeatable), e.g. 1x4096x11008")
    ap.add_argument("--tolerance", type=float, default=0.15, help="allowed slowdown (default 0.15)")
    ap.add_argument("--cold", action="store_true",
                    help="flush the L2 before every call, so the weights are read from DRAM as in a real "
                         "model (the default times a warm L2, where a small layer's weights stay cached)")
    ns = ap.parse_args(argv)

    sizes = []
    configs = []
    for a in ns.args:
        (sizes if a.isdigit() and not configs else configs).append(int(a) if a.isdigit() and not configs else a)
    shapes = [tuple(int(x) for x in sh.lower().split("x")) for sh in ns.shape]
    if sizes or not shapes:
        shapes += [(ns.m or n, n, n) for n in (sizes or [1024])]
    configs = configs or [""]
    global COLD
    COLD = ns.cold

    all_rows, torch_rows = [], []
    for shape in shapes:
        n = shape[1]
        rows, trow = _run_size(shape, configs, ns.group_size)
        all_rows += rows
        torch_rows.append(trow)
        for r in rows:
            print(f"{_label(r)} {r['config'] or '<default>':36s} {r['ms']:8.3f} ms {r['gflops']:9.1f} GFLOP/s {r['weight_gbs']:7.1f} GB/s(w)  max|err|={r['max_abs_err']:.5f}")
        print(f"{_label(trow)} {'torch fp16 gemm (cuBLAS)':36s} {trow['ms']:8.3f} ms {trow['gflops']:9.1f} GFLOP/s")

    if ns.json:
        Path(ns.json).parent.mkdir(parents=True, exist_ok=True)
        Path(ns.json).write_text(json.dumps(
            {"schema": 1, "environment": environment(), "results": all_rows, "torch": torch_rows},
            indent=2) + "\n")
        print("wrote", ns.json)
    if ns.compare:
        print(f"comparing with {ns.compare} (tolerance {ns.tolerance:.0%}):")
        if compare(all_rows, ns.compare, ns.tolerance):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
