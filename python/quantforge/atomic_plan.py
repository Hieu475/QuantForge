"""Split-K with an in-kernel reduction for the long-K, narrow-N decode layers (``down_proj``, ``o_proj``): which split factor and tile.

``e2e.W4A16GemmAtomic`` cuts K into S slices; every block adds its [16 x bn] partial result with ``red.global.add.f32`` into a zeroed f32
buffer (``epilogue=atomic-add``), and one small kernel converts it to f16 (+ bias). Unlike K-folding (``kfold.py``) there is no second copy
of the weights and any 1..16 rows work. The extra work is a memset and the conversion (~1.5 us), so it only pays for layers where the plain
kernel cannot fill the GPU: 8960 x 1536 60 -> 45 us, 4864 x 896 26.7 -> 23 us on the RTX 4050 (Marlin 40.7 / 15.2); 1536 x 1536 and
896 x 896 do not gain and keep the plain kernel. ``plan_atomic`` measures that once per (GPU, shape) and caches the answer in the autotune
cache; ``QUANTFORGE_NO_ATOMIC=1`` turns the whole thing off.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import torch

MAX_N = 2048          # wider outputs fill the GPU with column blocks alone
MIN_RATIO = 2.5       # K / N: only long-K layers
CONVERT_US = 1.5      # memset + f32 -> f16 conversion kernel around the GEMM (measured)
MIN_GAIN = 0.88       # use it only if kernel + conversion take <= 88 % of the plain kernel
_KEY_VERSION = 1
_TILES = [(64, 16, 128), (128, 16, 128), (64, 16, 64), (128, 16, 64), (64, 8, 64)]   # (block-tile-n, warp-tile-n, block-tile-k)
# 32 / 64 rows (the skinny kernels; vLLM at 17..64 concurrent requests): a block holds one core's worth of K-loop latency, so the 16 - 20 column
# blocks of a narrow layer leave every SM with one block that alternates dequantisation and mma; splitting K gives two or three blocks per SM
# that overlap (8960 x 1536: 96 -> 75 us at 64 rows, 83 -> 52 us at 32). Wide outputs lose to the atomics (4096 x 4096 0.80-0.93x), the planner measures.
MAX_N_SKINNY = 3072
_TILES_SKINNY = [(32, 16, 64), (64, 16, 64), (64, 32, 64), (64, 16, 128), (128, 16, 64), (128, 16, 128), (128, 32, 64)]
_SPLITS_SKINNY = (2, 3, 4, 5, 6, 8)


from .e2e import recording_paused  # noqa: E402


def tile_opts(bn: int, wn: int, bk: int, rows: int = 16) -> str:
    if rows > 16:
        return f"block-tile-m={rows} warp-tile-m=32 warp-tile-n={wn} block-tile-n={bn} block-tile-k={bk} smem-stages=1 prefetch-distance=1"
    return f"decode=true block-tile-n={bn} warp-tile-n={wn} block-tile-k={bk} smem-stages=1 prefetch-distance=1"


def cp_async_ok(opts: str) -> bool:
    """Can the 16 x bk activation tile be copied with cp.async (needed for `valid-rows`): the items must split evenly over the threads."""
    g = lambda k: int(re.search(rf"{k}=(\d+)", opts).group(1))  # noqa: E731
    return (16 * g("block-tile-k") // 8) % (32 * g("block-tile-n") // g("warp-tile-n")) == 0


@recording_paused
def plan_atomic(K: int, N: int, group_size: int, device: Optional[int] = None, rows: int = 16, measure: bool = True) -> Optional[tuple]:
    """``(S, opts)`` or None (keep the plain kernel). Measured with cold weights, cached. ``rows``: 16 (the decode tile, 1..16 rows), 32 or 64.
    ``measure=False``: only look the decision up (autotune cache, kernel packs); an unknown shape gives None and nothing is cached. The op uses that: a
    planner starts torch profiler sessions, which must not happen inside an operator call (it corrupts the profiler of the caller); layers are planned
    when they are built (`preplan`)."""
    if os.environ.get("QUANTFORGE_NO_ATOMIC") or not group_size or K % group_size or rows not in (16, 32, 64):
        return None
    if rows == 16 and (N > MAX_N or K < MIN_RATIO * N or N % 64):
        return None
    if rows > 16 and (N > MAX_N_SKINNY or N % 32):
        return None
    from . import autotune as at
    from .e2e import W4A16GemmAtomic, pack_int4

    device = torch.cuda.current_device() if device is None else device
    gpu = torch.cuda.get_device_name(device).replace(" ", "_")
    key = f"atomic{_KEY_VERSION}|{gpu}|g{group_size}|" + (f"r{rows}|" if rows > 16 else "") + f"K{K}|N{N}"
    cache = at._load()
    if key in cache:
        v = cache[key]
        return None if not v else (int(v.split("|")[0]), v.split("|")[1])
    if not measure:
        return None
    from .e2e import _toolchain_available

    if not _toolchain_available():     # nothing to measure with: keep the plain kernel, and do not cache that decision
        return None
    groups = K // group_size
    dev = torch.device("cuda", device)
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.rand(rows, K, device=dev, generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int8, generator=g)
    packed = pack_int4(q)
    scale = (torch.rand(groups, N, device=dev, generator=g) * 0.1 + 0.01).half()
    zero = torch.randint(0, 16, (groups, N), device=dev, dtype=torch.int8, generator=g)
    out = torch.zeros(rows, N, device=dev, dtype=torch.float32)
    # the plain kernel (f16-store) launched directly: the op itself would ask for this plan again
    from . import torch_ops
    kf = torch_ops._kernel_f16(rows, K, N, group_size, False)
    if getattr(kf, "_fn", None) is None or kf.Mp != rows:
        return None
    y16 = torch.empty(rows, N, device=dev, dtype=torch.float16)
    plain = at.event_time(lambda: kf.launch(x, packed, scale, zero, None, y16))
    best = (float("inf"), None)
    tiles, splits, lo = (_TILES, (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16), 24) if rows == 16 else (_TILES_SKINNY, _SPLITS_SKINNY, 20)
    for bn, wn, bk in tiles:
        if N % bn:
            continue
        opts = tile_opts(bn, wn, bk, rows)
        blocks_n = N // bn
        # splits that give about 24-200 blocks in total (the GPU holds 80-100 resident blocks)
        for S in (s for s in splits if groups % s == 0 and (K // s) % bk == 0 and lo <= blocks_n * s <= 200):
            try:
                k = W4A16GemmAtomic.build(rows, K, N, group_size, S, pipeline_opts=opts)
            except Exception:  # noqa: BLE001 - not buildable: next candidate
                continue
            t = at.event_time(lambda: k.launch(x, packed, scale, zero, out)) + CONVERT_US * 1e-6
            if rows == 16 and not cp_async_ok(opts):
                t += 1.2e-6                       # the padded activation copy of the non-exact kernel
            if t < best[0]:
                best = (t, (S, opts))
    if os.environ.get("QUANTFORGE_PLAN_VERBOSE"):
        print(f"atomic plan rows={rows} K={K} N={N}: plain {plain*1e6:.1f} us, best {best[0]*1e6:.1f} us {best[1]}", flush=True)
    choice = best[1] if best[1] is not None and best[0] <= MIN_GAIN * plain else None
    cache[key] = f"{choice[0]}|{choice[1]}" if choice else ""
    at._store(cache)
    return choice


def preplan(K: int, N: int, group_size: int) -> None:
    """Decide the 16-, 32- and 64-row plans of a layer now (at weight loading), not at the first call of that size: that may be inside a CUDA graph capture."""
    if os.environ.get("QUANTFORGE_NO_ATOMIC") or not torch.cuda.is_available():
        return
    from . import tile_plan

    for rows in (16, 32, 64):      # the tile of the plain kernel first: the atomic plans are measured against it
        try:
            tile_plan.plan_tile(rows, K, N, group_size=group_size)
        except Exception:  # noqa: BLE001
            pass
    for rows in (16, 32, 64):
        try:
            plan_atomic(K, N, group_size, rows=rows)
        except Exception:  # noqa: BLE001 - a planner problem must never stop a model from loading
            pass
