"""Per-shape tile schedule of the W4A8 INT8 GEMM (`e2e.W4A8GemmDq`, 32 .. 256 rows).

The schedule rule of `w4a8_op._int8_opts` was tuned on narrow layers (N <= 2048); on a square 4096 x 4096 layer at 64 rows it picked a 64 x 128 block with a 128-deep K tile that takes
115 us (cold weights) where a 64 x 64 block with a 256-deep K tile and 16 x 32 warp tiles takes 65 us (Marlin: 61-67 us): `ncu` showed the kernel waiting on global loads (long scoreboard 67 % of the
stalls), and a block that covers fewer columns with a deeper K tile keeps more weight bytes in flight per SM. The best schedule depends on the shape, so it is *measured* once per (GPU, rows, K, N), like the
W4A16 tile planner (`tile_plan.py`): `plan_int8` times a handful of schedules with CUDA events and cold weights and keeps the best when it beats the rule by 4 %; the decision lives in the autotune
cache (and in kernel packs); `w4a8_op._int8_opts` only looks it up. `QUANTFORGE_NO_TILE_PLAN=1` turns it off.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .e2e import recording_paused

_KEY_VERSION = 1
MIN_GAIN = 0.96            # a candidate must take <= 96 % of the rule's time (the repeat noise is 1-2 %)
ROWS = (32, 64, 96, 128, 160, 192, 224, 256)
_BASE = "thread-tile-k=32 smem-pad=16 accumulate=false"


def _o(bm: int, bn: int, bk: int, wm: int, wn: int, stages: int = 1, pd: int = 0) -> str:
    return (f"block-tile-m={bm} block-tile-n={bn} block-tile-k={bk} warp-tile-m={wm} warp-tile-n={wn} smem-stages={stages} "
            + (f"prefetch-distance={pd} " if pd else "") + _BASE)


# (block-tile-m, block-tile-n, block-tile-k, warp-tile-m, warp-tile-n, smem-stages, prefetch-distance): what won, over all schedules (block 32 / 64 / 128 x 32 / 64 / 128, K tile 64 / 128 / 256,
# 2 / 4 / 8 warps, 1-3 stages, look-ahead 0-2; cold weights) at 32, 64, 96, 128, 256 rows x 4096 x 4096, 64 rows x 4096 x 11008, 11008 x 4096 and 2048 x 2048
_SMALL = [(32, 128, 256, 32, 16, 2, 0), (32, 128, 256, 16, 32, 1, 0), (32, 64, 256, 16, 32, 2, 0), (32, 64, 256, 32, 16, 2, 0), (32, 32, 256, 16, 16, 1, 0),
          (64, 64, 256, 16, 32, 1, 0), (64, 64, 256, 32, 16, 1, 0), (64, 64, 256, 32, 32, 1, 0), (64, 64, 256, 64, 16, 1, 0),
          (64, 128, 256, 32, 32, 1, 0), (64, 128, 128, 32, 32, 1, 1)]
_LARGE = [(128, 128, 128, 64, 32, 1, 1), (128, 64, 128, 64, 32, 2, 0), (128, 128, 128, 32, 64, 3, 0),
          (64, 64, 128, 32, 32, 2, 0), (64, 64, 256, 32, 32, 1, 0), (32, 128, 256, 32, 16, 2, 0), (64, 128, 128, 32, 32, 1, 1)]


def candidates(rows: int, K: int, N: int) -> list:
    """The schedules timed for a `rows`-row GEMM (the winners of the sweeps: fewer columns per block and a deeper K tile than the large-batch default keep more weight bytes in
    flight at 32-96 rows, 128 x 128 / 128 x 64 blocks from 128 rows)."""
    out = []
    for bm, bn, bk, wm, wn, st, pd in (_SMALL if rows <= 96 else _LARGE + _SMALL[:3]):
        if bk > K or K % bk:                                   # a 256-deep tile needs K % 256 == 0: its 128-deep twin otherwise
            if bk == 256 and K % 128 == 0:
                bk = 128
            else:
                continue
        if rows % bm or N % bn or K // bk < 4:
            continue
        o = _o(bm, bn, bk, wm, wn, st, pd)
        if o not in out:
            out.append(o)
    return out


def _key(rows: int, K: int, N: int, gpu: str) -> str:
    return f"tilei8_{_KEY_VERSION}|{gpu}|r{rows}|K{K}|N{N}"


def _gpu(device: Optional[int] = None) -> str:
    device = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_name(device).replace(" ", "_")


def lookup(rows: int, K: int, N: int) -> Optional[str]:
    """The planned schedule of this shape, or None (use the rule). Never measures."""
    if os.environ.get("QUANTFORGE_NO_TILE_PLAN") or rows not in ROWS or not torch.cuda.is_available():
        return None
    from . import autotune as at

    return at._load().get(_key(rows, K, N, _gpu())) or None


@recording_paused
def plan_int8(rows: int, K: int, N: int, rule: str, device: Optional[int] = None, max_candidates: int = 40) -> Optional[str]:
    """Measure schedules of the INT8 kernel for `rows` rows and cache the winner (returns its options, or None when the rule stays)."""
    if os.environ.get("QUANTFORGE_NO_TILE_PLAN") or rows not in ROWS or K % 128 or N % 64:
        return None
    from . import autotune as at
    from .e2e import W4A8GemmDq, _toolchain_available

    key = _key(rows, K, N, _gpu(device))
    cache = at._load()
    if key in cache:
        return cache[key] or None
    if not _toolchain_available():
        return None
    device = torch.cuda.current_device() if device is None else device
    dev = torch.device("cuda", device)
    g = torch.Generator(device=dev).manual_seed(0)
    a = torch.randint(-100, 100, (rows, K), device=dev, dtype=torch.int8, generator=g)
    bp = torch.randint(-128, 127, (N, K // 2), device=dev, dtype=torch.int8, generator=g)
    from .quant import kmajor_scales

    gs = kmajor_scales(torch.randint(1, 6, (N, K // 128), device=dev, dtype=torch.int8, generator=g))      # K-major (with the padding row), as the compact layer keeps them
    zp = kmajor_scales(torch.randint(0, 16, (N, K // 128), device=dev, dtype=torch.int8, generator=g))
    sx = torch.rand(rows, device=dev, generator=g, dtype=torch.float32) * 0.01      # explicit: vLLM sets the default dtype to the model's (f16) while loading
    sw = torch.rand(N, device=dev, generator=g, dtype=torch.float32) * 0.01
    y = torch.empty(rows, N, device=dev, dtype=torch.float16)

    def time_of(opts: str) -> float:
        try:
            k = W4A8GemmDq.build(rows, K, N, bias=False, pipeline_opts=opts, kmajor=True)
        except Exception:  # noqa: BLE001 - not buildable for this shape: not a candidate
            return float("inf")
        return at.event_time(lambda: k.launch(a, bp, gs, zp, sx, sw, None, y))

    base = time_of(rule)
    best, best_t = None, float("inf")
    for opts in candidates(rows, K, N)[:max_candidates]:
        if opts == rule:
            continue
        t = time_of(opts)
        if t < best_t:
            best, best_t = opts, t
    choice = best if best is not None and best_t <= MIN_GAIN * base else None
    if os.environ.get("QUANTFORGE_PLAN_VERBOSE"):
        print(f"int8 plan rows={rows} K={K} N={N}: rule {base*1e6:.1f} us, best {best_t*1e6:.1f} us {best if choice else '(keep the rule)'}", flush=True)
    cache[key] = choice or ""
    at._store(cache)
    return choice


def preplan(K: int, N: int, rows=ROWS) -> None:
    """Decide the schedules of a layer's shape now (a layer is built once; the op only looks them up)."""
    if os.environ.get("QUANTFORGE_NO_TILE_PLAN") or not torch.cuda.is_available():
        return
    from . import w4a8_op

    for r in rows:
        plan_int8(r, K, N, w4a8_op._rule_opts(r, K, N))
