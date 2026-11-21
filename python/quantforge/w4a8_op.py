"""``torch.ops.quantforge.w4a8_hybrid``: one linear layer with W4A8 weights for the calls with more than 16 rows (prefill, large batches)
and the W4A16 decode kernels for the others, both made from the SAME quantised weight. With empty ``packed16`` (the "compact" mode) the decode reads the W4A8 layout itself through
``e2e.W4A8DecodeGemm`` (1.1-1.7x slower, no second copy of the nibbles); ``scale_nk`` is then ``gs * sw`` in f16, ``[N, K/G]``.

    y = torch.ops.quantforge.w4a8_hybrid(x, p8, gs, zp, sw, bias, packed16, scale16, zero16, packed_f, scale_f, zero_f, group, scale_nk)

``p8 / gs / zp / sw`` are the W4A8 tensors of ``quant.w4a8_quantize`` (nibbles packed along K); ``packed16 / scale16 / zero16`` the same values
in the W4A16 layout (nibbles along N, scale = gs * sw, zero = zp: ``from_w4a8``), ``*_f`` their K-folded copies (``kfold``; empty = not folded).
The row count is looked at INSIDE the op, so a graph compiled by torch.compile / captured by CUDA graphs has no Python branch on a
dynamic dimension. The kernels compile on first use per (rows bucket, K, N) and are cached on disk; the per-token activation quantisation is
eager here (a few small kernels), the compiled fused variants of ``nn.py`` are for the Hugging Face path.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import kfold, quant

_KERNELS: dict = {}
_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


def from_w4a8(packed: torch.Tensor, gs: torch.Tensor, zp: torch.Tensor, sw: torch.Tensor):
    """The W4A16 layout of a W4A8 weight: ``(packed int8 [K, N/2], scale f16 [K/G, N], zero int8 [K/G, N])``. The dequantised weights are
    equal: ``(q - zp) * gs * sw``; only the group scale ``gs * sw`` is rounded to f16."""
    nib = quant.unpack_k8(packed)                                                    # [N, K]
    q = nib.t().contiguous().to(torch.int8)                                          # [K, N]
    scale = (gs.float() * sw[:, None]).t().contiguous().half()                       # [K/G, N]
    return quant.pack_int4(q).contiguous(), scale, zp.t().contiguous()


def _bucket(m: int) -> int:
    for b in _BUCKETS:
        if m <= b:
            return b
    return -(-m // 4096) * 4096


def _kernel(Mp: int, K: int, N: int, has_bias: bool):
    key = (Mp, K, N, has_bias)
    k = _KERNELS.get(key)
    if k is None:
        from .e2e import W4A8GemmDq

        k = _KERNELS[key] = W4A8GemmDq.build(Mp, K, N, bias=has_bias)
    return k


def _decode_kernel(K: int, N: int, has_bias: bool, rows: int = 16):
    key = ("dec", K, N, has_bias, rows)
    k = _KERNELS.get(key)
    if k is None:
        from .e2e import W4A8DecodeGemm

        k = _KERNELS[key] = W4A8DecodeGemm.build(16, K, N, f16_out=True, bias=has_bias, valid_rows=rows if rows < 16 else 0)
    return k


_PAD16: dict = {}


def _padded16(x2: torch.Tensor) -> torch.Tensor:
    """x2 (M <= 16 rows) in the first rows of a reusable zero-initialised [16, K] buffer: one copy kernel instead of pad's allocate + fill + copy.
    The rows past M keep an earlier call's values: a GEMM row depends only on its own activation row and the extra output rows are dropped.
    One buffer per (device, stream, K), as in the C++ runtime."""
    M, K = x2.shape
    if M == 16:
        return x2.contiguous()
    key = (x2.device.index, torch.cuda.current_stream(x2.device).cuda_stream, K)
    buf = _PAD16.get(key)
    if buf is None:
        buf = _PAD16[key] = torch.zeros(16, K, device=x2.device, dtype=torch.float16)
    buf[:M].copy_(x2)
    return buf


def _quantize_tokens(x2: torch.Tensor):
    from .nn import _int8_quantize_tokens        # torch.compile'd (one fused kernel pair, dynamic shapes): compiled by the first big call, i.e.
    xq, s = _int8_quantize_tokens(x2)           # vLLM's profiling run, before any CUDA graph is captured
    return xq, s.reshape(-1)


def _impl(x: torch.Tensor, p8: torch.Tensor, gs: torch.Tensor, zp: torch.Tensor, sw: torch.Tensor, bias: Optional[torch.Tensor],
          packed16: torch.Tensor, scale16: torch.Tensor, zero16: torch.Tensor, packed_f: torch.Tensor, scale_f: torch.Tensor,
          zero_f: torch.Tensor, group: int, scale_nk: torch.Tensor) -> torch.Tensor:
    lead, K, N = x.shape[:-1], x.shape[-1], p8.shape[0]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    if M <= 16 and packed16.numel() == 0:                             # compact: decode from the W4A8 layout itself (no second copy)
        b = None if bias is None else bias.contiguous()
        if M in (1, 2, 4, 8) and x2.is_contiguous() and x2.data_ptr() % 16 == 0:         # exactly M rows: no padded copy (valid-rows kernel)
            out = torch.empty(M, N, device=x.device, dtype=torch.float16)
            _decode_kernel(K, N, bias is not None, M).launch(x2, p8, zp, scale_nk, out, b)
            return out.reshape(*lead, N)
        out = torch.empty(16, N, device=x.device, dtype=torch.float16)
        _decode_kernel(K, N, bias is not None).launch(_padded16(x2), p8, zp, scale_nk, out, b)
        return out[:M].reshape(*lead, N)
    if M <= 16:                                                       # decode: the W4A16 kernels (memory bound, same bytes)
        y = kfold.linear(x2, packed16, scale16, zero16, (packed_f, scale_f, zero_f), group, bias, None)
        return y.reshape(*lead, N)
    Mp = _bucket(M)
    k = _kernel(Mp, K, N, bias is not None)
    xq, s_x = _quantize_tokens(x2)
    if Mp != M:                                                       # zero rows (scale 0) up to the bucket
        xq = torch.nn.functional.pad(xq, (0, 0, 0, Mp - M))
        s_x = torch.nn.functional.pad(s_x, (0, Mp - M))
    y = torch.empty(Mp, N, device=x.device, dtype=torch.float16)
    k.launch(xq, p8, gs, zp, s_x.contiguous(), sw, None if bias is None else bias.float().contiguous(), y)
    return (y[:M] if Mp != M else y).reshape(*lead, N)


def _fake(x, p8, gs, zp, sw, bias, packed16, scale16, zero16, packed_f, scale_f, zero_f, group, scale_nk):
    return x.new_empty((*x.shape[:-1], p8.shape[0]), dtype=torch.float16)


_op = torch.library.custom_op("quantforge::w4a8_hybrid", mutates_args=())(_impl)
_op.register_fake(_fake)
