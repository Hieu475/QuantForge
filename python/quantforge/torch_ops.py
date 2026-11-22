"""`torch.ops.quantforge.w4a16_gemm`: the generated kernels as a PyTorch custom op.

    import quantforge.torch_ops                      # registers the op
    y = torch.ops.quantforge.w4a16_gemm(x, packed, scale, zero_point, 128)   # x: [..., K] f16
    y = quantforge.torch_ops.w4a16_linear(x, packed, scale, zero_point, 128, bias)  # same, python

* ``packed`` int8 ``[K, N/2]`` (two INT4 per byte, low nibble = even column), ``scale`` f16 and
  ``zero_point`` int8, either ``[K/G, N]`` (group-wise) or scalars (``group_size=0``, per tensor).
  **W8A16**: a group-wise weight whose ``packed`` is a plain int8 ``[K, N]`` tensor (signed values,
  same scale / zero point layout; ``quant.quantize_groupwise_int8``) selects the 8-bit kernel -- the
  width is inferred from ``packed.shape[1] == scale.shape[1]``. A ``torch.float8_e4m3fn`` ``[K, N]``
  weight (``quant.quantize_fp8``) selects the FP8 kernel: ``fp8_to_f16(w) * scale``, the
  ``zero_point`` argument is ignored (pass zeros). No swiglu / per-tensor for 8 bit.
* Returns ``act(x @ dequant(W) (+ bias))`` (``activation`` = none | relu | silu | gelu | swiglu, fused) with the dtype of ``x`` (f16); accumulation is f32.
* Works under ``torch.compile`` (a fake implementation provides the output shape) and in CUDA
  graphs: the kernel is launched on the current stream with no synchronisation or allocation of
  its own. Compile / JIT-load happens on the first call of each shape, so **warm every shape up
  once before capturing a graph** (a module load is not allowed during capture).
* Kernels are built per (row bucket, K, N, group size) and cached (``bucket_rows``: <= 16, 32, 64, 96, 128, 160, 192, 224, 256,
  256, 384, 512, ... 4096, then multiples of 1024; the call's activations are zero-padded to the bucket);
  ``M <= 16`` uses the decode schedule, ``M <= 32`` / ``M <= 64`` 32- / 64-row tiles.
  ``set_autotune(True)`` measures the tile options once per shape (``quantforge.autotune``, cached
  on disk) instead of using the default schedule.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import torch

from .e2e import KernelUnavailable, W4A16Gemm
from .quant import unpack_int4

__all__ = ["w4a16_linear", "set_autotune", "clear_cache", "bucket_rows", "CPP_RUNTIME"]

def _decode_opts(N: int) -> str:
    """Default schedule of an M <= 16 call. The K tile is the knob that matters across LLM layer shapes: 128 wins for
    wide outputs (N = 14336: +18 %, the fused gate/up N = 28672: +28 %), 64 for narrow ones (N = 2048: +15 %,
    N = 6144: +10 %). One exception, from a sweep over N at K = 4096 (RTX 4050, 20 SMs, 80 resident 128-column
    blocks): with 81 to 96 column blocks (N = 10240 .. 12288, e.g. the 11008-wide MLP of Llama-7B) the grid is only
    slightly larger than one wave of the 128-deep K tile, and a 32-deep tile (smaller shared memory per block, so all
    blocks are resident) is 8-12 % faster (1 x 4096 x 11008: 150 -> 135 us); from ~104 blocks up it loses again.
    The same wave arithmetic gives a second band: 160 .. 200 column blocks (N = 20480 .. 25600, 32-deep tiles run in 2 waves of 96, the 128-deep
    ones in 3 waves of 80): 10-14 % faster (2048 x 20480: 136.7 -> 124.3 us, 4096 x 24576: 328 -> 302 us; N = 8192 .. 9728 gains only 1-6 %, less than the padded
    activation copy these kernels need, see `_kernel_f16`); 112 blocks (N = 14336) loses 14 % with the 32-deep tile and 224 (N = 28672) 4 %, so those stay at 128.
    Narrow outputs (N <= 2048) use 32 / 64 column blocks, as the autotuner picks for them. `set_autotune(True)` replaces the rule by
    a search."""
    if N <= 1152 and N % 32 == 0:       # narrow outputs need narrower blocks to fill 20 SMs (autotune winners; 4864 x 896: 49 -> 28 us)
        return "decode=true block-tile-n=32 warp-tile-n=8 block-tile-k=128 smem-stages=2"
    if N <= 2048 and N % 64 == 0:
        return "decode=true block-tile-n=64 warp-tile-n=16 block-tile-k=128 smem-stages=1 prefetch-distance=1"
    bk = 32 if (10240 <= N <= 12288 or 20480 <= N <= 25600) else 128 if N >= 8192 else 64
    return f"decode=true block-tile-n=128 warp-tile-n=16 block-tile-k={bk} smem-stages=1 prefetch-distance=1"


def _prefill_opts(M: int, K: int, N: int) -> str:
    """Default schedule of an M > 64 call. Large weights (K * N >= 2**25: 4096 x 11008, 11008 x 4096, ...)
    run 8-12 % faster with a 3-stage pipeline of 32-deep K tiles from 512 rows up (RTX 4050, cold
    weights; the 4096 x 4096 class and M < 512 do not gain, 256 x 11008 x 4096 loses 10 %)."""
    return "block-tile-k=32 smem-stages=3" if M >= 512 and K * N >= 1 << 25 else ""


_lock = threading.Lock()
_kernels: dict[tuple, W4A16Gemm] = {}
_autotune = False


def set_autotune(enabled: bool = True) -> None:
    """Tune tile options per shape on first use (slow the first time, then cached on disk)."""
    global _autotune
    _autotune = bool(enabled)


def clear_cache() -> None:
    with _lock:
        _kernels.clear()
        _kernels_f16.clear()
    if _cpp is not None:
        _cpp.clear_kernels()


# A kernel is compiled for a bucket of row counts, not for every M: a request with M rows runs the
# kernel of its bucket (activations zero-padded, extra rows dropped), so a server that sees prompt
# lengths 1..4096 compiles at most len(_BUCKETS) kernels per layer shape instead of one per length.
# M <= 16 is one decode kernel; 32 and 64 are the skinny kernels (32 / 64-row block tiles); the buckets
# above are multiples of the 64 / 128-row block tiles.
_BUCKETS = (16, 32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
# Skinny schedules (17..64 rows). The 4096 x 4096 class: the best of a sweep over tiles / K tile / stages / look-ahead
# (RTX 4050, cold weights): 32 rows 57.8 us (the old 64-row schedule: 68.6 us), 64 rows 87 us (97 us). Narrow outputs
# (N <= 2048, the o_proj / down_proj / qkv of small models) want smaller tiles: with 64 rows, 32-row blocks with a 16 / 32 wide
# warp tile are 1.2-1.6x faster than the 64-row block (4864 x 896: 77 -> 47.5 us; 896 x 896: 17.7 -> 10.8 us), and with 32 rows
# a 64-deep K tile helps the wide, short-K outputs (896 x 9728: 35.5 -> 32.2 us). Sweep: benchmarks/sweep_skinny.py.
_SKINNY_OPTS = {
    32: ("block-tile-m=32 warp-tile-m=16 warp-tile-n=32 block-tile-n=128 block-tile-k=128 "
         "smem-stages=1 prefetch-distance=1"),
    64: ("block-tile-m=64 warp-tile-m=32 warp-tile-n=64 block-tile-n=128 block-tile-k=64 "
         "smem-stages=1 prefetch-distance=1"),
}


def _skinny_opts(M: int, K: int, N: int) -> str:
    """Default schedule of a 32- or 64-row call (see the comment above `_SKINNY_OPTS`). Narrow outputs (N <= 2048) take 32 / 64
    column blocks: the sweep over 74 candidates (`benchmarks/sweep_skinny.py`, 8 shapes) found them 11-52 % faster than the
    128-column blocks, while for wide outputs no candidate beat the defaults by more than 4 %."""
    base = "smem-stages=1 prefetch-distance=1"
    if N <= 2048 and N % 64 == 0 and M in (32, 64):
        if K * N <= 1_200_000:                 # 896 x 896 / 896 x 1152: 32 rows 1.5x, 64 rows 1.24x
            bn = 32 if (M == 32 and N <= 1152 and N % 32 == 0) else 64
            return f"block-tile-m=32 warp-tile-m=16 warp-tile-n=16 block-tile-n={bn} block-tile-k=128 {base}"
        bk = 128 if M == 32 else 64
        return f"block-tile-m=32 warp-tile-m=32 warp-tile-n=16 block-tile-n=64 block-tile-k={bk} {base}"
    if M == 32 and N >= 8192 and K <= 2048:
        return ("block-tile-m=32 warp-tile-m=16 warp-tile-n=32 block-tile-n=128 block-tile-k=64 "
                "smem-stages=1 prefetch-distance=1")
    return _SKINNY_OPTS[M]


def _default_skinny(M: int, K: int, N: int) -> str:
    """The rule-based schedule of a 32- .. 256-row call (what `default_opts` returns when no plan exists)."""
    return _skinny_opts(M, K, N) if M <= 64 else _mid_opts(M, K, N)


def _mid_opts(M: int, K: int, N: int) -> str:
    """Schedule of a 96..256-row call (decode at 65-256 concurrent requests). The buckets are multiples of 32 so that a batch of 96 or 192 rows is not padded to
    128 / 256 (that wasted 25-33 % of the arithmetic of a compute-bound GEMM): 64-row blocks when the count is a multiple of 64, else 32-row blocks. Measured
    at 96 / 160 / 192 rows (cold weights): 8960 x 1536 263 -> 119 us, 4096 x 4096 150 -> 108, 1536 x 2048 54 -> 29, 1536 x 17920 244 -> 209; very wide
    outputs (N >= 20000) keep the 128-row default tile, which wins at 128 / 224 / 256 rows there (2048 x 22016: 379 / 739 against 397 / 815 us)."""
    if N >= 20000:
        return _prefill_opts(M, K, N)
    if M % 64 == 0:
        return "block-tile-m=64 warp-tile-m=32 warp-tile-n=64 block-tile-n=128 block-tile-k=64 smem-stages=1 prefetch-distance=1"
    return "block-tile-m=32 warp-tile-m=16 warp-tile-n=32 block-tile-n=128 block-tile-k=64 smem-stages=1 prefetch-distance=1"


def default_opts(M: int, K: int, N: int) -> str:
    """Pipeline options the op uses for a call with M rows when autotuning is off (M is the bucketed row count). 32 / 64 rows: the schedule
    `tile_plan.plan_tile` measured for this shape when there is one (autotune cache or kernel pack), else the rule."""
    if M <= 16:
        from . import tile_plan

        return tile_plan.lookup(16, K, N) or _decode_opts(N)
    if M <= 64:
        from . import tile_plan

        return tile_plan.lookup(M, K, N) or _skinny_opts(M, K, N)
    if M <= 256:
        from . import tile_plan

        return tile_plan.lookup(M, K, N) or _mid_opts(M, K, N)
    return _prefill_opts(M, K, N)


def bucket_rows(M: int) -> int:
    """Row count of the kernel that serves a call with M activation rows."""
    for b in _BUCKETS:
        if M <= b:
            return b
    return -(-M // 1024) * 1024


def weight_kind(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> int:
    """Weight format: 4 = packed INT4 [K, N/2], 8 = plain int8 [K, N], 9 = FP8 E4M3 ``float8_e4m3fn``
    [K, N] (the 8 / 9 formats are group-wise only)."""
    if packed.dtype == torch.float8_e4m3fn:
        return 9
    return 8 if group_size and packed.shape[1] == scale.shape[1] else 4


def _out_cols(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> int:
    return scale.shape[1] if group_size else packed.shape[1] * 2


def _kernel(M: int, K: int, N: int, group_size: int, device_index: int,
            activation: str = "none", overwrite: bool = False, kind: int = 4) -> W4A16Gemm:
    """The kernel serving M rows (see `bucket_rows`). `overwrite`: a kernel that writes
    C = A @ W without reading C (no bias; implied by swiglu)."""
    M = bucket_rows(M)
    key = (M, K, N, group_size, device_index, _autotune, activation, overwrite, kind)
    k = _kernels.get(key)
    if k is not None:
        return k
    with _lock:
        k = _kernels.get(key)
        if k is None:
            g = group_size or None
            if _autotune:
                from .autotune import autotune

                k = autotune(M, K, N, group_size=g, bits=4 if kind == 4 else 8, fp8=kind == 9,
                              epilogue=activation, accumulate=not overwrite)
            else:
                opts = default_opts(M, K, N)
                k = W4A16Gemm.build(M, K, N, group_size=g, pad=True, pipeline_opts=opts,
                                    bits=4 if kind == 4 else 8, fp8=kind == 9, epilogue=activation, accumulate=not overwrite)
            _kernels[key] = k
    return k


def _tile_threads(opts: str) -> tuple:
    """(threads per block, block-tile-k) of a decode option string."""
    import re
    g = lambda k, d: int(m.group(1)) if (m := re.search(rf"{k}=(\d+)", opts)) else d  # noqa: E731
    return 32 * (g("block-tile-n", 128) // g("warp-tile-n", 16)), g("block-tile-k", 64)


_kernels_fold: dict = {}


def _kernel_fold(R: int, K: int, N: int, group_size: int, kind: int):
    """The overwrite (f32 accumulator) kernel of the K-folded GEMM for exactly R <= 16 activation rows: a 16-row tile whose missing rows are
    zero-filled by cp.async (`valid-rows`), so the caller passes the activations as they are, no padded copy. The accumulator the caller allocates
    must have 16 rows (the kernel stores them all). Falls back to the bucket kernel (padded copy) when the shape cannot use cp.async for the
    activation tile (the tile must split evenly over the block's threads) or R = 16."""
    key = (R, K, N, group_size, kind)
    k = _kernels_fold.get(key)
    if k is not None:
        return k
    with _lock:
        k = _kernels_fold.get(key)
        if k is None:
            base = default_opts(16, K, N)
            opts = base
            # cp.async needs the 16 x bk activation tile to split evenly over the block's threads: take the default schedule if it does,
            # else 64-column blocks of 4 warps with the same K tile (9.5 us against 9.4 for the 1536 x 3 folded shape). A deeper K tile for the wide folds
            # (10752 columns) measured 57 us against 45 with the padded copy, so those keep the padded copy
            tail = base[base.index(" block-tile-k"):] if " block-tile-k" in base else ""
            cands = [base, "decode=true block-tile-n=64 warp-tile-n=16" + tail]
            for c in cands:
                th, kb = _tile_threads(c)
                if (16 * kb // 8) % th == 0 and K % kb == 0 and (c == base or N % 64 == 0):
                    opts = c
                    break
            threads, bk = _tile_threads(opts)
            if R < 16 and not _autotune and not os.environ.get("QUANTFORGE_NO_VALID_ROWS") and (16 * bk // 8) % threads == 0:
                try:
                    k = W4A16Gemm.build(16, K, N, group_size=group_size or None, pipeline_opts=f"{opts} valid-rows={R}",
                                        bits=4 if kind == 4 else 8, fp8=kind == 9, accumulate=False)
                    k.Mp = R
                except Exception:  # noqa: BLE001 - not buildable with valid-rows: the padded kernel below
                    k = None
            if k is not None:
                _kernels_fold[key] = k
    if k is None:                                  # (outside the lock: `_kernel` takes it too)
        k = _kernel(R, K, N, group_size, 0, "none", True, kind)
        _kernels_fold[key] = k
    return k


_kernels_atomic: dict = {}


def _kernel_atomic(M: int, K: int, N: int, group_size: int):
    """The split-K kernel with in-kernel atomic reduction (see `atomic_plan`), or a `_NoKernel` (not hot: the C++ op takes the plain path).
    M <= 16: M in {1, 2, 4, 8} exactly M rows (`valid-rows`) when the tile allows cp.async, otherwise a 16-row kernel on a padded copy.
    17 <= M <= 64: the 32- / 64-row kernel of the bucket on a padded copy."""
    from . import atomic_plan
    from .e2e import W4A16GemmAtomic

    rows = 16 if M <= 16 else bucket_rows(M)
    if rows > 64:
        return _NoKernel(16, -1, -1)
    plan = atomic_plan.plan_atomic(K, N, group_size, rows=rows, measure=False)   # planned when the layer was built (`atomic_plan.preplan`)
    if plan is None:
        return _NoKernel(16, -1, -1)
    S, opts = plan
    R = rows if rows > 16 else (M if M in (1, 2, 4, 8) and atomic_plan.cp_async_ok(opts) and not os.environ.get("QUANTFORGE_NO_VALID_ROWS") else 16)
    key = (R, K, N, group_size)
    k = _kernels_atomic.get(key)
    if k is None:
        with _lock:
            k = _kernels_atomic.get(key)
            if k is None:
                k = W4A16GemmAtomic.build(rows, K, N, group_size, S, pipeline_opts=opts, valid_rows=R if R < rows else 0)
                k.Mp, k.Kp, k.Np = R, K, N
                _kernels_atomic[key] = k
    return k


class _NoKernel:
    """Stands for a shape the f16-store kernel cannot serve (K or N not a multiple of its tile, autotune on): the C++ op then uses the f32 path."""

    _fn = _set_stream = None

    def __init__(self, Mp, Kp, Np):
        self.Mp, self.Kp, self.Np = Mp, Kp, Np


_kernels_f16: dict = {}


def _kernel_f16(M: int, K: int, N: int, group_size: int, has_bias: bool):
    """The W4A16 kernel with the f16-store epilogue for M rows (decode / skinny / prefill buckets), INT4 group-wise weights."""
    from .e2e import W4A16GemmF16, padded_shape

    # M in {1, 2, 4, 8} (the sizes of vLLM's CUDA graphs): a kernel for exactly M rows of a 16-row tile; its activation tile is zero-filled past
    # M and its stores skipped, so the caller passes x itself, no padded copy. Other M use the bucket's kernel on a padded copy.
    exact = M in (1, 2, 4, 8) and not os.environ.get("QUANTFORGE_NO_VALID_ROWS")
    Mb = M if exact else bucket_rows(M)
    key = (Mb, K, N, group_size, has_bias, exact)
    k = _kernels_f16.get(key)
    if k is not None:
        return k
    with _lock:
        k = _kernels_f16.get(key)
        if k is None:
            opts = default_opts(Mb, K, N)
            Mp, Kp, Np = padded_shape(Mb, K, N, group_size, opts)
            if _autotune or (Kp, Np) != (K, N) or os.environ.get("QUANTFORGE_NO_F16STORE"):
                k = _NoKernel(Mp, Kp, Np)
            else:
                try:
                    k = W4A16GemmF16.build(Mp, K, N, group_size, bias=has_bias, pipeline_opts=opts, valid_rows=Mb if exact and Mb < Mp else 0)
                    k.Mp, k.Kp, k.Np = (Mb if exact and Mb < Mp else Mp), K, N
                except RuntimeError:
                    if not (exact and Mb < Mp):
                        raise
                    # the schedule cannot zero-fill the activation tile with cp.async: the bucket's kernel on a padded copy
                    k = W4A16GemmF16.build(Mp, K, N, group_size, bias=has_bias, pipeline_opts=opts)
                    k.Mp, k.Kp, k.Np = Mp, K, N
            _kernels_f16[key] = k
    return k


_warned_dense = False
PACK_MISSES: list = []     # (M, K, N, activation) of every kernel that was neither in a pack nor compilable
DENSE_FALLBACKS = 0        # calls served by the dense fallback (a kernel pack covers a model when this stays 0)


def _dense_impl(x, packed, scale, zero_point, group_size, bias=None, activation="none"):
    """Last resort when a shape has no kernel (not in a kernel pack and no toolchain to compile it): dequantize the INT4 weight
    with torch and use cuBLAS. Correct but slow (the f16 weight is materialised on every call)."""
    global _warned_dense, DENSE_FALLBACKS
    DENSE_FALLBACKS += 1
    if not _warned_dense:
        import warnings
        warnings.warn("quantforge: no kernel for a layer shape (not in a kernel pack, no toolchain to compile it); "
                      "using a slow dense fallback for it. Build a pack for the model: `python -m quantforge.pack build`.")
        _warned_dense = True
    if weight_kind(packed, scale, group_size) != 4 or not group_size:
        raise KernelUnavailable("the dense fallback serves group-wise INT4 weights only")
    K = x.shape[-1]
    q = unpack_int4(packed).to(torch.float16)
    g = group_size
    w = (q.view(K // g, g, -1) - zero_point.to(torch.float16).unsqueeze(1)) * scale.unsqueeze(1)
    y = x.reshape(-1, K) @ w.reshape(K, -1)
    if bias is not None:
        y = y + bias.to(y.dtype)
    if activation == "swiglu":
        g_, u_ = y[:, 0::2].float(), y[:, 1::2].float()
        y = (torch.nn.functional.silu(g_) * u_).to(x.dtype)
    elif activation == "relu":
        y = torch.relu(y)
    elif activation == "silu":
        y = torch.nn.functional.silu(y)
    elif activation == "gelu":
        y = torch.nn.functional.gelu(y, approximate="tanh")
    return y.reshape(*x.shape[:-1], y.shape[-1])


def _python_impl(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
    activation: str = "none",
) -> torch.Tensor:
    """Reference / fallback implementation (also the hot path when the C++ runtime is unavailable)."""
    K = x.shape[-1]
    kind = weight_kind(packed, scale, group_size)
    if kind == 9:   # the kernels see FP8 as raw bytes
        packed = packed.view(torch.int8)
    N = _out_cols(packed, scale, group_size)
    swiglu = activation == "swiglu"
    cols = N // 2 if swiglu else N
    if swiglu and bias is not None:
        raise ValueError("activation='swiglu' does not take a bias")
    if swiglu and kind != 4:
        raise ValueError("activation='swiglu' needs an INT4 weight")
    overwrite = swiglu or bias is None
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    try:
        kern = _kernel(M, K, N, group_size, x.device.index or 0, activation, overwrite, kind)
    except KernelUnavailable:
        return _dense_impl(x, packed, scale, zero_point, group_size, bias, activation)
    if group_size and (kern.Kp, kern.Np) == (K, N) and x2.is_cuda and not (
            x2.data_ptr() % 16 or packed.data_ptr() % 16 or scale.data_ptr() % 16
            or zero_point.data_ptr() % 16) and packed.is_contiguous() and scale.is_contiguous() \
            and zero_point.is_contiguous():
        # hot path (only M may need padding): pad the activations, start the accumulator from
        # bias / nothing, launch, slice + convert -- three small kernels around the GEMM
        act = x2 if kern.Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, kern.Mp - M))
        if overwrite:   # the kernel writes every element it is going to be read back from
            acc = torch.empty(kern.Mp, N, dtype=torch.float32, device=x.device)
        else:
            acc = bias.float().expand(kern.Mp, N).contiguous()
        kern.launch(act, packed, scale, zero_point, acc)
        return acc[:M, :cols].to(x.dtype).reshape(*x.shape[:-1], cols)
    xk = x2 if kern.M == M else torch.nn.functional.pad(x2, (0, 0, 0, kern.M - M))   # rows of its bucket
    out = kern.run(xk, packed, scale, zero_point,
                   bias=bias.float() if bias is not None else None)[:M]
    return out.to(x.dtype).reshape(*x.shape[:-1], cols)


def _fake(x, packed, scale, zero_point, group_size, bias=None, activation="none"):
    n = _out_cols(packed, scale, group_size)
    cols = n // 2 if activation == "swiglu" else n
    return x.new_empty(*x.shape[:-1], cols)


def _prebuilt_ext():
    """The C++ runtime built ahead of time for this torch (a kernel pack's ``ext/<tag>/quantforge_rt.so``), or None: no compiler needed."""
    if os.environ.get("QUANTFORGE_NO_PREBUILT_EXT"):
        return None
    import importlib.util

    from .e2e import ext_tag, pack_dirs

    tag = ext_tag()
    for d in pack_dirs():
        so = d / "ext" / tag / "quantforge_rt.so"
        if so.exists():
            spec = importlib.util.spec_from_file_location("quantforge_rt", so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def _load_cpp_runtime():
    """Compile (once, cached by torch) and load the C++ dispatcher hot path; None if unavailable."""
    if os.environ.get("QUANTFORGE_NO_CPP"):
        return None
    try:
        import ctypes
        ext = _prebuilt_ext()
        if ext is None:
            from torch.utils import cpp_extension

            rt = Path(__file__).resolve().parent / "runtime"
            src = rt / "torch_runtime.cpp"
            cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            ext = cpp_extension.load(name="quantforge_rt", sources=[str(src), str(rt / "kfold_reduce.cu")], with_cuda=True,
                                     extra_include_paths=[f"{cuda}/include"], verbose=False)

        def builder(M, K, N, group_size, device_index, activation, overwrite, kind):
            try:
                return builder_(M, K, N, group_size, device_index, activation, overwrite, kind)
            except KernelUnavailable:      # not in a pack, no toolchain: the C++ op takes the Python fallback (the dense one)
                PACK_MISSES.append((M, K, N, activation))      # also for the optional kernels (atomic, f16-store, fold), which then fall back silently
                return 0, 0, 16, -1, -1

        def builder_(M, K, N, group_size, device_index, activation, overwrite, kind):
            addr = lambda f: (ctypes.cast(f, ctypes.c_void_p).value or 0) if f is not None else 0  # noqa: E731
            if activation in ("f16s", "f16sb"):               # the f16-store kernel (W4A16GemmF16)
                k = _kernel_f16(M, K, N, group_size, activation == "f16sb")
                return addr(k._fn), addr(k._set_stream), k.Mp, k.Kp, k.Np
            if activation == "atomic":                        # split-K with atomic reduction for long-K narrow layers (see atomic_plan)
                k = _kernel_atomic(M, K, N, group_size)
                return addr(k._fn), addr(k._set_stream), k.Mp, k.Kp, k.Np
            if activation == "foldvr":                        # K-folded GEMM: exact rows, no padded copy (see _kernel_fold)
                k = _kernel_fold(M, K, N, group_size, kind)
                return addr(k._fn), addr(k._set_stream), k.Mp, k.Kp, k.Np
            k = _kernel(M, K, N, group_size, device_index, activation, overwrite, kind)
            return addr(k._fn), addr(k._set_stream), k.Mp, k.Kp, k.Np

        ext.set_python_hooks(builder, _python_impl)
        return ext
    except Exception as e:  # noqa: BLE001 - any build problem: fall back to the Python op
        import warnings
        msg = str(e)
        # a failed build prints the compiler command first: show the lines that say what went wrong
        errs = [ln.strip() for ln in msg.splitlines() if "error" in ln.lower() and "ninja" not in ln.lower()]
        detail = " | ".join(errs[:2])[:400] if errs else (msg.splitlines()[0][:200] if msg else "")
        warnings.warn(f"quantforge: C++ torch runtime unavailable ({type(e).__name__}: {detail}); using the slower Python op "
                      f"(set QUANTFORGE_NO_CPP=1 to silence this)")
        return None


_cpp = _load_cpp_runtime()
if _cpp is not None:
    torch.library.register_fake("quantforge::w4a16_gemm")(_fake)
else:
    _w4a16_gemm = torch.library.custom_op("quantforge::w4a16_gemm", mutates_args=())(_python_impl)
    _w4a16_gemm.register_fake(_fake)

CPP_RUNTIME = _cpp is not None


def w4a16_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
    activation: str = "none",
    perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``act(x @ dequant(W) + bias)`` for an f16 ``x`` of shape ``[..., K]`` (see the module
    docstring); ``activation``: none | relu | silu | gelu (tanh form) | swiglu, fused into the kernel.
    ``swiglu``: ``packed`` / ``scale`` / ``zero_point`` hold the gate and up projections interleaved
    by column (``quantforge.quant.interleave_gate_up``); the result is ``silu(x @ Wg) * (x @ Wu)``,
    half as wide as the GEMM.
    ``perm``: row permutation of an act-order GPTQ weight (``quant.from_gptq_act_order``); the
    activations are gathered with it first (one extra small kernel)."""
    if x.dtype != torch.float16:
        raise TypeError(f"x must be float16, got {x.dtype}")
    if perm is not None:  # GPTQ act-order: the weight rows were permuted offline
        x = x.index_select(-1, perm.to(x.device))
    return torch.ops.quantforge.w4a16_gemm(x, packed, scale, zero_point, group_size, bias, activation)
