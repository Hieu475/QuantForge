"""
Autotuner for the W4A16 / W8A16 kernels.

    from quantforge.autotune import autotune
    kern = autotune(M=1, K=4096, N=11008, group_size=128)      # compiles + times candidates
    out = kern.run(act, packed_w, scale, zero_point)

The best pipeline options per (GPU, bits, group size, M bucket, K, N) are cached
in a JSON file (default ~/.cache/quantforge/tune.json, override with
QUANTFORGE_TUNE_CACHE), so the search runs once per shape. Timing uses the
device time of the kernel from the CUPTI trace with the L2 flushed before every
call (weights are read from DRAM, as in a real model), best of a few runs.

Search space: block / warp tile, K tile and number of stages (see PIPELINE.md);
`decode=true` candidates for M <= 16, 128-row tiles otherwise.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

from .e2e import W4A16Gemm, pack_int4, padded_shape

_FLUSH = None


def _flush_l2():
    import torch

    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _FLUSH.max()  # read, not write: a memset would leave dirty lines whose write-back slows the next kernel


def device_time(fn, name_part="a16_gemm", iters=20, repeats=3, flush=True) -> float:
    """Best per-call device time (seconds) of the kernels whose name contains `name_part`."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    warnings.filterwarnings("ignore", message="Warning: Profiler clears events")
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                if flush:
                    _flush_l2()
                fn()
            torch.cuda.synchronize()
        ts = sorted((e.device_time for e in prof.events() if name_part in e.name and e.device_time > 0))
        if ts:
            best = min(best, ts[0] / 1e6)
    if best == float("inf"):
        raise RuntimeError(f"no kernel named *{name_part}* in the trace")
    return best


def candidates(M: int, bits: int = 4) -> list[str]:
    """Pipeline option strings to try for an activation with M rows."""
    if M <= 16:
        out = []
        for bn, wn in ((64, 16), (64, 32), (32, 8), (32, 16), (128, 16), (128, 32)):
            for bk in (64, 128):
                for stages in (1, 2):
                    out.append(f"decode=true block-tile-n={bn} warp-tile-n={wn} "
                               f"block-tile-k={bk} smem-stages={stages}")
        # one tile of look-ahead for the weight loads (rotating registers): -19 % on
        # K >> N shapes (few blocks, long K loop), neutral elsewhere; deeper never helped
        for bn, wn in ((64, 16), (128, 16)):
            for bk in (64, 128):  # a smaller K tile = less smem per block = more resident blocks
                out.append(f"decode=true block-tile-n={bn} warp-tile-n={wn} block-tile-k={bk} "
                           f"smem-stages=1 prefetch-distance=1")
            out.append(f"decode=true block-tile-n={bn} warp-tile-n={wn} block-tile-k=128 "
                       f"smem-stages=2 sink-dequant-stores=true prefetch-distance=1")
        return out
    out = [""]
    for opts in ("block-tile-k=64", "smem-stages=1", "block-tile-k=64 smem-stages=1",
                 "warp-tile-m=32 warp-tile-n=64", "warp-tile-m=64 warp-tile-n=32",
                 "warp-tile-m=128 warp-tile-n=32"):
        out.append(opts)
    # 3 stages of the 32-deep K tile: -11 % at 2048 x 4096 x 11008 (wave quantization / occupancy
    # effects: +4 % slower at 2048 x 4096 x 4096), hence a candidate, not the default
    if M >= 256:
        out.append("block-tile-k=32 smem-stages=3")
    # 64 x 128 block tiles: twice the blocks, so the last wave of a 512-row GEMM is fuller
    # (512 x 4096 x 4096: 128 -> 256 blocks for 60 resident slots; -5 % time)
    if M >= 256:
        for st in ("smem-stages=1", ""):
            out.append(f"block-tile-m=64 block-tile-n=128 warp-tile-m=32 warp-tile-n=64 block-tile-k=64 {st}".strip())
    if M <= 64:  # skinny tiles waste less work than 128 rows
        for bm in (32, 64):
            for wn in (32, 64):
                out.append(f"block-tile-m={bm} warp-tile-m=16 warp-tile-n={wn} block-tile-n=128 block-tile-k=64")
                out.append(f"block-tile-m={bm} warp-tile-m=16 warp-tile-n={wn} block-tile-n=128 block-tile-k=32")
    return out


def _cache_path() -> Path:
    return Path(os.environ.get("QUANTFORGE_TUNE_CACHE", Path.home() / ".cache" / "quantforge" / "tune.json"))


def _load() -> dict:
    try:
        return json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return {}


def _store(cache: dict) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
        tmp.replace(p)
    except OSError:
        pass  # a read-only home must not break tuning


def _m_bucket(M: int) -> str:
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        if M <= b:
            return str(b) if b > 16 else "<=16"
    return ">4096"


def cache_key(M, K, N, group_size, bits, gpu, acc):
    return f"{gpu}|w{bits}|g{group_size}|{acc}|M{_m_bucket(M)}|K{K}|N{N}"


def autotune(M: int, K: int, N: int, group_size: int | None = None, bits: int = 4, acc: str = "f32",
             chip: str = "sm_80", retune: bool = False, verbose: bool = False,
             max_candidates: int | None = None, epilogue: str = "none",
             accumulate: bool = True) -> W4A16Gemm:
    """Best-known kernel for this problem, tuning (and caching) on first use."""
    import torch

    gpu = torch.cuda.get_device_name(0)
    key = cache_key(M, K, N, group_size, bits, gpu, acc)
    cache = _load()
    if not retune and key in cache:
        return W4A16Gemm.build(M, K, N, group_size=group_size, bits=bits, acc=acc, chip=chip,
                               pad=True, pipeline_opts=cache[key]["opts"], epilogue=epilogue,
                               accumulate=accumulate)

    g = torch.Generator(device="cuda").manual_seed(0)
    act = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    if bits == 4:
        q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
        w = pack_int4(q)
    else:
        w = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8, generator=g)
    if group_size is None:
        scale = torch.tensor(0.05, device="cuda", dtype=torch.float16)
        zp = torch.tensor(8, device="cuda", dtype=torch.int8)
    else:
        scale = (torch.rand(K // group_size, N, device="cuda", generator=g) * 0.1 + 0.01).half()
        zp = torch.randint(0, 16, (K // group_size, N), device="cuda", dtype=torch.int8, generator=g)
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32 if acc == "f32" else torch.float16)

    results = []
    cands = candidates(M, bits)[: max_candidates]
    for opts in cands:
        try:
            kern = W4A16Gemm.build(M, K, N, group_size=group_size, bits=bits, acc=acc, chip=chip, pad=True,
                                   pipeline_opts=opts, workdir=tempfile.mkdtemp(prefix="qf_tune_"))

            def call(kern=kern):
                out.zero_()
                kern.run(act, w, scale, zp, out=out)

            t = device_time(call)
        except Exception as e:  # tile does not fit / budget exceeded / unsupported combination
            if verbose:
                print(f"  skip {opts!r}: {str(e).splitlines()[-1][:100]}")
            continue
        results.append((t, opts))
        if verbose:
            print(f"  {t * 1e6:9.1f} us  {opts!r}")
    if not results:
        raise RuntimeError("no candidate configuration compiled")
    results.sort()
    best_t, best = results[0]
    cache = _load()
    cache[key] = {"opts": best, "us": best_t * 1e6, "tried": len(results)}
    _store(cache)
    return W4A16Gemm.build(M, K, N, group_size=group_size, bits=bits, acc=acc, chip=chip, pad=True,
                           pipeline_opts=best, epilogue=epilogue, accumulate=accumulate)
