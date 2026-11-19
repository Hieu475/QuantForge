"""`torch.ops.quantforge.w4a16_gemm`: the generated kernels as a PyTorch custom op.

    import quantforge.torch_ops                      # registers the op
    y = torch.ops.quantforge.w4a16_gemm(x, packed, scale, zero_point, 128)   # x: [..., K] f16
    y = quantforge.torch_ops.w4a16_linear(x, packed, scale, zero_point, 128, bias)  # same, python

* ``packed`` int8 ``[K, N/2]`` (two INT4 per byte, low nibble = even column), ``scale`` f16 and
  ``zero_point`` int8, either ``[K/G, N]`` (group-wise) or scalars (``group_size=0``, per tensor).
  **W8A16**: a group-wise weight whose ``packed`` is a plain int8 ``[K, N]`` tensor (signed values,
  same scale / zero point layout; ``quant.quantize_groupwise_int8``) selects the 8-bit kernel -- the
  width is inferred from ``packed.shape[1] == scale.shape[1]``. No swiglu / per-tensor for 8 bit.
* Returns ``act(x @ dequant(W) (+ bias))`` (``activation`` = none | relu | silu | gelu | swiglu, fused) with the dtype of ``x`` (f16); accumulation is f32.
* Works under ``torch.compile`` (a fake implementation provides the output shape) and in CUDA
  graphs: the kernel is launched on the current stream with no synchronisation or allocation of
  its own. Compile / JIT-load happens on the first call of each shape, so **warm every shape up
  once before capturing a graph** (a module load is not allowed during capture).
* Kernels are built per (row bucket, K, N, group size) and cached (``bucket_rows``: <= 16, 64, 128,
  256, 384, 512, ... 4096, then multiples of 1024; the call's activations are zero-padded to the bucket);
  ``M <= 16`` uses the decode schedule, ``M <= 64`` a 64-row tile.
  ``set_autotune(True)`` measures the tile options once per shape (``quantforge.autotune``, cached
  on disk) instead of using the default schedule.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import torch

from .e2e import W4A16Gemm

__all__ = ["w4a16_linear", "set_autotune", "clear_cache", "bucket_rows", "CPP_RUNTIME"]

def _decode_opts(N: int) -> str:
    """Default schedule of an M <= 16 call. The K tile is the only knob that matters across LLM layer
    shapes: 128 wins clearly for wide outputs (N = 14336: +18 %, the fused gate/up N = 28672: +28 %,
    N = 8192..11008: +2..7 %), 64 for narrow ones (N = 2048: +15 %, N = 6144: +10 %); measured on the
    RTX 4050 (benchmarks in the commit message). `set_autotune(True)` replaces the rule by a search."""
    bk = 128 if N >= 8192 else 64
    return f"decode=true block-tile-n=128 warp-tile-n=16 block-tile-k={bk} smem-stages=1 prefetch-distance=1"
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
    if _cpp is not None:
        _cpp.clear_kernels()


# A kernel is compiled for a bucket of row counts, not for every M: a request with M rows runs the
# kernel of its bucket (activations zero-padded, extra rows dropped), so a server that sees prompt
# lengths 1..4096 compiles at most len(_BUCKETS) kernels per layer shape instead of one per length.
# M <= 16 is one decode kernel; the buckets above are multiples of the 64 / 128-row block tiles.
_BUCKETS = (16, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
_SKINNY_OPTS = ("block-tile-m=64 block-tile-n=128 warp-tile-m=32 warp-tile-n=64 "
                "block-tile-k=64 smem-stages=1")


def bucket_rows(M: int) -> int:
    """Row count of the kernel that serves a call with M activation rows."""
    for b in _BUCKETS:
        if M <= b:
            return b
    return -(-M // 1024) * 1024


def weight_bits(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> int:
    """4 for packed INT4 [K, N/2], 8 for plain int8 [K, N] (group-wise only)."""
    return 8 if group_size and packed.shape[1] == scale.shape[1] else 4


def _out_cols(packed: torch.Tensor, scale: torch.Tensor, group_size: int) -> int:
    return scale.shape[1] if group_size else packed.shape[1] * 2


def _kernel(M: int, K: int, N: int, group_size: int, device_index: int,
            activation: str = "none", overwrite: bool = False, bits: int = 4) -> W4A16Gemm:
    """The kernel serving M rows (see `bucket_rows`). `overwrite`: a kernel that writes
    C = A @ W without reading C (no bias; implied by swiglu)."""
    M = bucket_rows(M)
    key = (M, K, N, group_size, device_index, _autotune, activation, overwrite, bits)
    k = _kernels.get(key)
    if k is not None:
        return k
    with _lock:
        k = _kernels.get(key)
        if k is None:
            g = group_size or None
            if _autotune:
                from .autotune import autotune

                k = autotune(M, K, N, group_size=g, bits=bits, epilogue=activation, accumulate=not overwrite)
            else:
                opts = _decode_opts(N) if M <= 16 else _SKINNY_OPTS if M <= 64 else ""
                k = W4A16Gemm.build(M, K, N, group_size=g, pad=True, pipeline_opts=opts, bits=bits,
                                    epilogue=activation, accumulate=not overwrite)
            _kernels[key] = k
    return k


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
    bits = weight_bits(packed, scale, group_size)
    N = _out_cols(packed, scale, group_size)
    swiglu = activation == "swiglu"
    cols = N // 2 if swiglu else N
    if swiglu and bias is not None:
        raise ValueError("activation='swiglu' does not take a bias")
    if swiglu and bits != 4:
        raise ValueError("activation='swiglu' needs an INT4 weight")
    overwrite = swiglu or bias is None
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    kern = _kernel(M, K, N, group_size, x.device.index or 0, activation, overwrite, bits)
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


def _load_cpp_runtime():
    """Compile (once, cached by torch) and load the C++ dispatcher hot path; None if unavailable."""
    if os.environ.get("QUANTFORGE_NO_CPP"):
        return None
    try:
        import ctypes
        from torch.utils import cpp_extension

        src = Path(__file__).resolve().parent / "runtime" / "torch_runtime.cpp"
        cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        ext = cpp_extension.load(name="quantforge_rt", sources=[str(src)], with_cuda=True,
                                 extra_include_paths=[f"{cuda}/include"], verbose=False)

        def builder(M, K, N, group_size, device_index, activation, overwrite, bits):
            k = _kernel(M, K, N, group_size, device_index, activation, overwrite, bits)
            addr = lambda f: ctypes.cast(f, ctypes.c_void_p).value  # noqa: E731
            return addr(k._fn), addr(k._set_stream), k.Mp, k.Kp, k.Np

        ext.set_python_hooks(builder, _python_impl)
        return ext
    except Exception as e:  # noqa: BLE001 - any build problem: fall back to the Python op
        import warnings
        warnings.warn(f"quantforge: C++ torch runtime unavailable ({type(e).__name__}: "
                      f"{str(e).splitlines()[0][:120] if str(e) else ''}); using the Python op")
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
