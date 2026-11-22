"""Per-shape tile schedule of the 16-, 32- and 64-row GEMMs.

The default schedules of `torch_ops._SKINNY_OPTS` are the best single choice over a set of shapes, but the best tile of these compute-bound kernels
depends on how the column blocks fall on the SMs and on K: measured with cold weights on the RTX 4050 at 64 rows, a 32-deep K tile with a 64 x 32 warp
tile is 6-10 % faster for 1536 x 17920, 2048 x 16384, 3072 x 16384, 3584 x 18944 and 4096 x 28672 and 14 % faster at K x N = 4096 x 10240, a 64 x 16
warp tile with a 64-deep K tile 3-7 % for 2048 x 22016, 3584 x 37888 and 8192 x 3072; at 32 rows 4096 x 6144 is 21 % faster with a 16-row warp tile (and
the default wins for the long-K layers). ``plan_tile`` measures a handful of schedules once per (GPU, rows, K, N) and keeps the best when it beats the
default by 4 %; the decision lives in the autotune cache (and in kernel packs) and `torch_ops.default_opts` looks it up. ``QUANTFORGE_NO_TILE_PLAN=1`` turns it off.

16 rows (decode, N > 2048): the rule (`torch_ops._decode_opts`) takes a 64-deep K tile below N = 8192, which the cp.async copy of the 16 x bk activation tile
cannot split evenly over the threads, so the exact-row kernels of M in {1, 2, 4, 8} (`valid-rows`) cannot be built there and the op copies the activations to a
padded buffer first (one more kernel). The 16-row planner measures the rule on its real path (padded copy included) against the candidates that do support
`valid-rows` and against the other tiles: a 3584 x 4608 / 4096 x 4096 layer is ~4 % faster that way and 1536 x 17920, 3072 x 16384 4.6-6 %.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .e2e import recording_paused

_KEY_VERSION = 1
MIN_GAIN16 = 0.97          # decode: the measurement is a single kernel (+ copy), repeat noise < 1 %
MIN_GAIN = 0.96            # a candidate must take <= 96 % of the default's time (the repeat noise is 1-2 %)
_BASE = "block-tile-n=128 smem-stages=1 prefetch-distance=1"
# (warp-tile-m, warp-tile-n, block-tile-k); block-tile-m is the row count
_CANDIDATES_DECODE = [(bn, wn, bk) for bn in (64, 128) for wn in (16, 32) for bk in (64, 128) if wn <= bn]     # (block-tile-n, warp-tile-n, block-tile-k)
_CANDIDATES = {
    16: [],
    64: [(64, 32, 32), (32, 64, 32), (64, 16, 64), (32, 32, 64), (64, 64, 32)],
    32: [(32, 64, 32), (32, 16, 64), (32, 32, 64), (16, 32, 64), (16, 16, 64), (16, 32, 32)],
}
# 96 .. 256 rows (block-tile-m, warp-tile-m, warp-tile-n, block-tile-k); the empty options (the 128-row default tile) are a candidate too
_MID_ROWS = (96, 128, 160, 192, 224, 256)
_MID_BM64 = [(64, 32, 64, 64), (64, 64, 32, 32), (64, 32, 32, 32), (64, 32, 64, 32), (64, 32, 32, 64)]
_MID_BM32 = [(32, 16, 32, 64), (32, 32, 32, 32), (32, 16, 16, 64), (32, 32, 64, 32), (32, 16, 32, 128)]
for _r in _MID_ROWS:
    _CANDIDATES[_r] = []          # candidates come from `_mid_candidates`


def _mid_candidates(rows: int) -> list:
    return (_MID_BM64 if rows % 64 == 0 else []) + _MID_BM32


def _opts(rows: int, wm: int, wn: int, bk: int) -> str:
    return f"block-tile-m={rows} warp-tile-m={wm} warp-tile-n={wn} {_BASE} block-tile-k={bk}"


def _opts_mid(bm: int, wm: int, wn: int, bk: int) -> str:
    return f"block-tile-m={bm} warp-tile-m={wm} warp-tile-n={wn} {_BASE} block-tile-k={bk}"


def _opts16(bn: int, wn: int, bk: int) -> str:
    return f"decode=true block-tile-n={bn} warp-tile-n={wn} block-tile-k={bk} smem-stages=1 prefetch-distance=1"


def _key(rows: int, K: int, N: int, gpu: str) -> str:
    return f"tile{_KEY_VERSION}|{gpu}|r{rows}|K{K}|N{N}"


def _gpu(device: Optional[int] = None) -> str:
    device = torch.cuda.current_device() if device is None else device
    return torch.cuda.get_device_name(device).replace(" ", "_")


def lookup(rows: int, K: int, N: int) -> Optional[str]:
    """The planned schedule of this shape, or None (use the default). Never measures."""
    if os.environ.get("QUANTFORGE_NO_TILE_PLAN") or rows not in _CANDIDATES or not torch.cuda.is_available():
        return None
    from . import autotune as at

    v = at._load().get(_key(rows, K, N, _gpu()))
    if v == "default":
        from . import torch_ops

        return torch_ops._prefill_opts(rows, K, N)
    return v or None


@recording_paused
def plan_tile(rows: int, K: int, N: int, device: Optional[int] = None, group_size: int = 128) -> Optional[str]:
    """Measure the candidate schedules of a `rows`-row GEMM and cache the winner (returns its options, or None when the default stays)."""
    if os.environ.get("QUANTFORGE_NO_TILE_PLAN") or rows not in _CANDIDATES or N % 128 or K % group_size or K % 64:
        return None
    if rows == 16 and N <= 2048:       # the narrow outputs have their own autotuned schedules
        return None
    from . import autotune as at
    from . import torch_ops
    from .e2e import W4A16Gemm, _toolchain_available, pack_int4

    key = _key(rows, K, N, _gpu(device))
    cache = at._load()
    if key in cache:
        return cache[key] or None
    if not _toolchain_available():
        return None
    device = torch.cuda.current_device() if device is None else device
    dev = torch.device("cuda", device)
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.rand(rows, K, device=dev, generator=g) * 2 - 1).half()
    packed = pack_int4(torch.randint(0, 16, (K, N), device=dev, dtype=torch.int8, generator=g))
    scale = (torch.rand(K // group_size, N, device=dev, generator=g) * 0.1 + 0.01).half()
    zero = torch.randint(0, 16, (K // group_size, N), device=dev, dtype=torch.int8, generator=g)
    out = torch.empty(rows, N, device=dev, dtype=torch.float32)

    def time_of(opts: str) -> float:
        try:
            k = W4A16Gemm.build(rows, K, N, group_size=group_size, pad=False, pipeline_opts=opts, accumulate=False)
        except Exception:  # noqa: BLE001 - not buildable for this shape: not a candidate
            return float("inf")
        return at.event_time(lambda: k.launch(x, packed, scale, zero, out))

    if rows == 16:
        return _plan_decode(K, N, group_size, dev, key, cache, x, packed, scale, zero)

    base_opts = torch_ops._default_skinny(rows, K, N)
    base = time_of(base_opts)
    best, best_t = None, float("inf")
    if rows in _MID_ROWS:
        cands = [_opts_mid(*c) for c in _mid_candidates(rows) if K % c[3] == 0]
        cands.append(torch_ops._prefill_opts(rows, K, N))                 # the 128-row default tile
    else:
        cands = [_opts(rows, wm, wn, bk) for wm, wn, bk in _CANDIDATES[rows] if K % bk == 0 and wm <= rows]
    for opts in cands:
        if opts == base_opts:
            continue
        t = time_of(opts)
        if t < best_t:
            best, best_t = opts, t
    choice = best if best is not None and best_t <= MIN_GAIN * base else None
    if choice == "":
        choice = "default"                       # the 128-row default tile (empty options) is stored under a name
    if os.environ.get("QUANTFORGE_PLAN_VERBOSE"):
        print(f"tile plan rows={rows} K={K} N={N}: default {base*1e6:.1f} us, best {best_t*1e6:.1f} us {best if choice else '(keep default)'}", flush=True)
    cache[key] = choice or ""
    at._store(cache)
    return choice


def _plan_decode(K, N, G, dev, key, cache, x, packed, scale, zero):
    """16 rows: time each schedule as the op runs it for one row (M = 1): the exact-row kernel (`valid-rows`) when its activation tile can be copied with
    cp.async, else the 16-row kernel behind the copy of the activation row into a padded buffer."""
    from . import autotune as at
    from . import torch_ops
    from .e2e import W4A16GemmF16

    x1 = x[:1].contiguous()
    pad = torch.zeros(16, K, device=dev, dtype=torch.half)
    y = torch.empty(16, N, device=dev, dtype=torch.half)

    def time_of(opts: str) -> float:
        try:
            k = W4A16GemmF16.build(16, K, N, G, bias=False, pipeline_opts=opts, valid_rows=1)
            return at.event_time(lambda: k.launch(x1, packed, scale, zero, None, y))
        except Exception:  # noqa: BLE001 - no exact-row kernel for this tile (or not buildable): the padded path
            pass
        try:
            k = W4A16GemmF16.build(16, K, N, G, bias=False, pipeline_opts=opts)
        except Exception:  # noqa: BLE001
            return float("inf")

        def padded():
            pad[:1].copy_(x1)
            k.launch(pad, packed, scale, zero, None, y)

        return at.event_time(padded)

    base = time_of(torch_ops._decode_opts(N))
    best, best_t = None, float("inf")
    for bn, wn, bk in _CANDIDATES_DECODE:
        if N % bn or K % bk:
            continue
        opts = _opts16(bn, wn, bk)
        t = time_of(opts)
        if t < best_t:
            best, best_t = opts, t
    choice = best if best is not None and best != torch_ops._decode_opts(N) and best_t <= MIN_GAIN16 * base else None
    if os.environ.get("QUANTFORGE_PLAN_VERBOSE"):
        print(f"tile plan rows=16 K={K} N={N}: rule {base*1e6:.1f} us, best {best_t*1e6:.1f} us {best if choice else '(keep the rule)'}", flush=True)
    cache[key] = choice or ""
    at._store(cache)
    return choice
