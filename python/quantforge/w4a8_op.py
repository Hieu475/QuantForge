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


def _decode_kernel(K: int, N: int):
    key = ("dec", K, N)
    k = _KERNELS.get(key)
    if k is None:
        from .e2e import W4A8DecodeGemm

        k = _KERNELS[key] = W4A8DecodeGemm.build(16, K, N)
    return k


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
        xp = torch.nn.functional.pad(x2, (0, 0, 0, 16 - M)) if M < 16 else x2
        out = torch.empty(16, N, device=x.device, dtype=torch.float32)
        _decode_kernel(K, N).launch(xp.contiguous(), p8, zp, scale_nk, out)
        y = out[:M].to(torch.float16)
        return (y if bias is None else y + bias).reshape(*lead, N)
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
