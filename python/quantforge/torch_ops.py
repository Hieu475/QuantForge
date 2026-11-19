"""`torch.ops.quantforge.w4a16_gemm`: the generated kernels as a PyTorch custom op.

    import quantforge.torch_ops                      # registers the op
    y = torch.ops.quantforge.w4a16_gemm(x, packed, scale, zero_point, 128)   # x: [..., K] f16
    y = quantforge.torch_ops.w4a16_linear(x, packed, scale, zero_point, 128, bias)  # same, python

* ``packed`` int8 ``[K, N/2]`` (two INT4 per byte, low nibble = even column), ``scale`` f16 and
  ``zero_point`` int8, either ``[K/G, N]`` (group-wise) or scalars (``group_size=0``, per tensor).
* Returns ``act(x @ dequant(W) (+ bias))`` (``activation`` = none | relu | silu | gelu | swiglu, fused) with the dtype of ``x`` (f16); accumulation is f32.
* Works under ``torch.compile`` (a fake implementation provides the output shape) and in CUDA
  graphs: the kernel is launched on the current stream with no synchronisation or allocation of
  its own. Compile / JIT-load happens on the first call of each shape, so **warm every shape up
  once before capturing a graph** (a module load is not allowed during capture).
* Kernels are built per (M, K, N, group size) and cached; ``M <= 16`` uses the decode schedule.
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

__all__ = ["w4a16_linear", "set_autotune", "clear_cache", "CPP_RUNTIME"]

_DECODE_OPTS = "decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1"
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


def _kernel(M: int, K: int, N: int, group_size: int, device_index: int,
            activation: str = "none", overwrite: bool = False) -> W4A16Gemm:
    """`overwrite`: a kernel that writes C = A @ W without reading C (no bias; implied by swiglu)."""
    key = (M, K, N, group_size, device_index, _autotune, activation, overwrite)
    k = _kernels.get(key)
    if k is not None:
        return k
    with _lock:
        k = _kernels.get(key)
        if k is None:
            g = group_size or None
            if _autotune:
                from .autotune import autotune

                k = autotune(M, K, N, group_size=g, epilogue=activation, accumulate=not overwrite)
            else:
                opts = _DECODE_OPTS if M <= 16 else ""
                k = W4A16Gemm.build(M, K, N, group_size=g, pad=True, pipeline_opts=opts,
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
    N = packed.shape[1] * 2
    swiglu = activation == "swiglu"
    cols = N // 2 if swiglu else N
    if swiglu and bias is not None:
        raise ValueError("activation='swiglu' does not take a bias")
    overwrite = swiglu or bias is None
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    kern = _kernel(M, K, N, group_size, x.device.index or 0, activation, overwrite)
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
    out = kern.run(x2, packed, scale, zero_point, bias=bias.float() if bias is not None else None)
    return out.to(x.dtype).reshape(*x.shape[:-1], cols)


def _fake(x, packed, scale, zero_point, group_size, bias=None, activation="none"):
    cols = packed.shape[1] if activation == "swiglu" else packed.shape[1] * 2
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

        def builder(M, K, N, group_size, device_index, activation, overwrite):
            k = _kernel(M, K, N, group_size, device_index, activation, overwrite)
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
) -> torch.Tensor:
    """``act(x @ dequant(W) + bias)`` for an f16 ``x`` of shape ``[..., K]`` (see the module
    docstring); ``activation``: none | relu | silu | gelu (tanh form) | swiglu, fused into the kernel.
    ``swiglu``: ``packed`` / ``scale`` / ``zero_point`` hold the gate and up projections interleaved
    by column (``quantforge.quant.interleave_gate_up``); the result is ``silu(x @ Wg) * (x @ Wu)``,
    half as wide as the GEMM."""
    if x.dtype != torch.float16:
        raise TypeError(f"x must be float16, got {x.dtype}")
    return torch.ops.quantforge.w4a16_gemm(x, packed, scale, zero_point, group_size, bias, activation)
