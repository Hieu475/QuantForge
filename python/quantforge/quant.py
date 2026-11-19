"""
Quantization helpers for the W4A16 kernels: group-wise quantization, and
repacking of the two common checkpoint layouts (GPTQ, AWQ) into the layout the
QuantForge kernels read.

QuantForge layout (what `W4A16Gemm.run` takes)
    packed  int8  [K, N/2]    low nibble = even column, high nibble = odd column
    scale   f16   [K/G, N]    one scale per (group of G rows along K, column)
    zero    int8  [K/G, N]    one zero point per (group, column), values 0..15
    dequant(w[k, n]) = (q[k, n] - zero[k // G, n]) * scale[k // G, n]

Checkpoint layouts handled here (all tensors on CPU or GPU, int32 = 8 nibbles)
    GPTQ   qweight int32 [K/8, N]   8 consecutive *k* of one column per word
           qzeros  int32 [K/G, N/8] 8 consecutive *n* per word, stored as zero-1
                                    (classic AutoGPTQ "v1" format; pass
                                    zeros_offset=0 for formats that store them as is)
           scales  f16   [K/G, N]
           g_idx   int32 [K]        optional; must be the plain k // G ordering
                                    (actorder checkpoints need a row permutation
                                    that the kernel does not implement; rejected)
    AWQ    qweight int32 [K, N/8]   8 consecutive *n* per word, in the order
                                    [0, 2, 4, 6, 1, 3, 5, 7]
           qzeros  int32 [K/G, N/8] same packing, stored as is
           scales  f16   [K/G, N]
"""

from __future__ import annotations

import torch

# AWQ stores logical column j of an 8-wide word at nibble position AWQ_ORDER.index(j)
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: integer tensor [K, N] with values 0..15  ->  int8 [K, N/2] (low nibble = even column)."""
    return ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: int8 [K, N/2] -> int8 [K, N] with values 0..15."""
    p = packed.to(torch.int16) & 0xFF
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = (p & 0xF).to(torch.int8)
    out[:, 1::2] = ((p >> 4) & 0xF).to(torch.int8)
    return out


def quantize_groupwise(w: torch.Tensor, group_size: int = 128, symmetric: bool = False):
    """Round-to-nearest INT4 quantization of w[K, N], one (scale, zero) per group of
    `group_size` rows of every column.

    Returns (q uint-valued int8 [K, N] in 0..15, scale f16 [K/G, N], zero int8 [K/G, N]).
    Asymmetric: min/max mapped to 0..15; symmetric: zero fixed at 8.
    """
    K, N = w.shape
    if K % group_size:
        raise ValueError(f"group_size {group_size} must divide K={K}")
    g = w.float().reshape(K // group_size, group_size, N)
    if symmetric:
        amax = g.abs().amax(dim=1).clamp(min=1e-8)
        scale = amax / 7.0
        zero = torch.full_like(scale, 8.0)
    else:
        lo, hi = g.amin(dim=1), g.amax(dim=1)
        scale = ((hi - lo) / 15.0).clamp(min=1e-8)
        zero = (-lo / scale).round().clamp(0, 15)
    scale16 = scale.half()
    q = (g / scale16.float().unsqueeze(1) + zero.unsqueeze(1)).round().clamp(0, 15)
    return q.reshape(K, N).to(torch.int8), scale16, zero.to(torch.int8)


def quantize_groupwise_int8(w: torch.Tensor, group_size: int = 128):
    """Symmetric round-to-nearest INT8 quantization of w[K, N], one scale per group of
    `group_size` rows of every column (zero point 0).

    Returns (q int8 [K, N] in -127..127, scale f16 [K/G, N], zero int8 [K/G, N] == 0)."""
    K, N = w.shape
    if K % group_size:
        raise ValueError(f"group_size {group_size} must divide K={K}")
    g = w.float().reshape(K // group_size, group_size, N)
    scale = (g.abs().amax(dim=1) / 127.0).clamp(min=1e-8).half()
    q = (g / scale.float().unsqueeze(1)).round().clamp(-127, 127)
    return q.reshape(K, N).to(torch.int8), scale, torch.zeros_like(scale, dtype=torch.int8)


def _unpack_words_along_last(words: torch.Tensor, order=None) -> torch.Tensor:
    """int32 [A, B] -> int32 [A, 8*B]: nibble i of word b is logical column 8*b + order[i]^-1."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    nib = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], dim=-1)  # [A, B, 8] by position
    if order is not None:
        nib = nib[..., [order.index(j) for j in range(8)]]  # position -> logical column
    return nib.reshape(words.shape[0], -1).to(torch.int32)


def _pack_words_along_last(vals: torch.Tensor, order=None) -> torch.Tensor:
    """Inverse of _unpack_words_along_last (used by tests / tooling)."""
    A, N = vals.shape
    v = vals.reshape(A, N // 8, 8).to(torch.int64)
    if order is not None:
        v = v[..., order]  # position i holds logical column order[i]
    word = torch.zeros(A, N // 8, dtype=torch.int64, device=vals.device)
    for i in range(8):
        word |= (v[..., i] & 0xF) << (4 * i)
    word = torch.where(word >= 2**31, word - 2**32, word)
    return word.to(torch.int32)


def from_gptq(qweight, qzeros, scales, g_idx=None, zeros_offset: int = 1):
    """GPTQ checkpoint tensors -> (packed int8 [K, N/2], scale f16 [K/G, N], zero int8 [K/G, N])."""
    K = qweight.shape[0] * 8
    N = qweight.shape[1]
    G = K // scales.shape[0]
    if g_idx is not None:
        expect = torch.arange(K, device=g_idx.device) // G
        if not torch.equal(g_idx.to(expect.dtype), expect):
            raise NotImplementedError(
                "GPTQ act-order (permuted g_idx) checkpoints need a row permutation; "
                "re-quantize without desc_act or permute the rows offline"
            )
    w = qweight.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], dim=1).reshape(K, N)  # k = 8*row + i
    zero = (_unpack_words_along_last(qzeros) + zeros_offset) & 0xF
    return pack_int4(q.to(torch.int8)), scales.half().contiguous(), zero.to(torch.int8)


def from_awq(qweight, qzeros, scales):
    """AWQ checkpoint tensors -> (packed int8 [K, N/2], scale f16 [K/G, N], zero int8 [K/G, N])."""
    q = _unpack_words_along_last(qweight, AWQ_ORDER)  # [K, N]
    zero = _unpack_words_along_last(qzeros, AWQ_ORDER)
    return pack_int4(q.to(torch.int8)), scales.half().contiguous(), zero.to(torch.int8)


# Writers for the same layouts (tests, converting our own quantization output)
def to_gptq(q, scale, zero, zeros_offset: int = 1):
    K, N = q.shape
    qv = q.to(torch.int64).reshape(K // 8, 8, N)
    word = torch.zeros(K // 8, N, dtype=torch.int64, device=q.device)
    for i in range(8):
        word |= (qv[:, i] & 0xF) << (4 * i)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    qz = _pack_words_along_last((zero.to(torch.int32) - zeros_offset) & 0xF)
    return word, qz, scale


def to_awq(q, scale, zero):
    return _pack_words_along_last(q.to(torch.int32), AWQ_ORDER), _pack_words_along_last(
        zero.to(torch.int32), AWQ_ORDER), scale


def interleave_gate_up(gate, up):
    """Fuse the gate and up projections of a gated MLP into one weight for
    ``W4A16Gemm.build(..., epilogue="swiglu")``.

    gate, up: (packed [K, N/2] int8, scale f16 [K/G, N], zero int8 [K/G, N]) in QuantForge layout
    with the same K, N and group size. Returns the same triple for N' = 2N with column 2j = gate j
    and column 2j + 1 = up j (so every thread of the MMA accumulator holds a (gate_j, up_j) pair);
    the kernel's output is silu(x @ W_gate) * (x @ W_up) of shape [M, N]."""
    (pg, sg, zg), (pu, su, zu) = gate, up
    if pg.shape != pu.shape or sg.shape != su.shape or zg.shape != zu.shape:
        raise ValueError("gate and up must have identical shapes")
    qg, qu = unpack_int4(pg), unpack_int4(pu)                       # [K, N]
    inter = lambda a, b: torch.stack((a, b), dim=-1).reshape(a.shape[0], -1)  # noqa: E731
    return pack_int4(inter(qg, qu)), inter(sg, su).contiguous(), inter(zg, zu).contiguous()
