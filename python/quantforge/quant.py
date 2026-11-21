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


def quantize_fp8(w: torch.Tensor, group_size: int = 128):
    """FP8 (E4M3) quantization of w[K, N] with one f16 scale per group of `group_size` rows of every
    column: ``w ~ w8.float() * scale`` with the group's absolute maximum mapped to 448.

    Returns (w8 ``torch.float8_e4m3fn`` [K, N], scale f16 [K/G, N], zero int8 [K/G, N] == 0). The zero
    point is only there to keep the operand list of the GEMM; FP8 dequantisation ignores it."""
    K, N = w.shape
    if K % group_size:
        raise ValueError(f"group_size {group_size} must divide K={K}")
    g = w.float().reshape(K // group_size, group_size, N)
    scale = (g.abs().amax(dim=1) / 448.0).clamp(min=1e-8).half()
    w8 = (g / scale.float().unsqueeze(1)).clamp(-448, 448).reshape(K, N).to(torch.float8_e4m3fn)
    return w8, scale, torch.zeros_like(scale, dtype=torch.int8)


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
                "GPTQ act-order (permuted g_idx) checkpoints need a row permutation: "
                "use from_gptq_act_order() and pass perm= to w4a16_linear"
            )
    w = qweight.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], dim=1).reshape(K, N)  # k = 8*row + i
    zero = (_unpack_words_along_last(qzeros) + zeros_offset) & 0xF
    return pack_int4(q.to(torch.int8)), scales.half().contiguous(), zero.to(torch.int8)


def from_gptq8(qweight, qzeros, scales, g_idx=None, zeros_offset: int = 1):
    """8-bit GPTQ checkpoint tensors -> (q int8 [K, N], scale f16 [K/G, N], zero int8 [K/G, N]).

    GPTQ stores unsigned bytes (0..255) four to an int32 along K; the kernels take int8, so both the weights and the zero
    points are shifted by 128 (``(q - 128) - (zero - 128) == q - zero``). Act-order checkpoints: ``from_gptq8_act_order``."""
    K, N = qweight.shape[0] * 4, qweight.shape[1]
    G = K // scales.shape[0]
    if g_idx is not None:
        expect = torch.arange(K, device=g_idx.device) // G
        if not torch.equal(g_idx.to(expect.dtype), expect):
            raise NotImplementedError("act-order 8-bit GPTQ: use from_gptq8_act_order()")
    w = qweight.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(w >> (8 * i)) & 0xFF for i in range(4)], dim=1).reshape(K, N)            # k = 4*row + i
    zw = qzeros.to(torch.int64) & 0xFFFFFFFF
    zero = torch.stack([(zw >> (8 * i)) & 0xFF for i in range(4)], dim=-1).reshape(qzeros.shape[0], -1)
    zero = (zero + zeros_offset) & 0xFF
    return (q - 128).to(torch.int8).contiguous(), scales.half().contiguous(), (zero - 128).to(torch.int8)


def from_gptq8_act_order(qweight, qzeros, scales, g_idx, zeros_offset: int = 1):
    """8-bit GPTQ with ``desc_act=True`` -> (q, scale, zero, perm), the counterpart of ``from_gptq_act_order``."""
    K = qweight.shape[0] * 4
    perm = act_order_permutation(g_idx, K // scales.shape[0])
    q, scale, zero = from_gptq8(qweight, qzeros, scales, None, zeros_offset)
    return q[perm.to(q.device)].contiguous(), scale, zero, perm


def act_order_permutation(g_idx, group_size: int):
    """Row permutation that makes an act-order GPTQ checkpoint group-contiguous.

    ``g_idx[k]`` is the quantisation group of input row k (``desc_act=True`` checkpoints
    store the rows in the original order but the groups are scattered). Returns ``perm``
    (int64 [K]) such that rows ``perm`` form groups of ``group_size`` consecutive rows:
    ``W[perm]`` has the plain ``i // G`` group layout and ``x @ W == x[..., perm] @ W[perm]``.
    Raises ValueError if the groups are not all exactly ``group_size`` rows."""
    g = g_idx.to(torch.int64)
    counts = torch.bincount(g)
    if counts.numel() * group_size != g.numel() or not bool((counts == group_size).all()):
        raise ValueError("g_idx must assign exactly group_size rows to every group")
    return torch.argsort(g, stable=True)


def from_gptq_act_order(qweight, qzeros, scales, g_idx, zeros_offset: int = 1):
    """GPTQ checkpoint with ``desc_act=True`` -> (packed, scale, zero, perm).

    packed / scale / zero are in QuantForge layout with the rows already permuted to be
    group-contiguous; ``perm`` (int64 [K]) must be applied to the activations:
    ``y = x[..., perm] @ W`` (``w4a16_linear(..., perm=perm)`` does it)."""
    K = qweight.shape[0] * 8
    G = K // scales.shape[0]
    perm = act_order_permutation(g_idx, G)
    packed, scale, zero = from_gptq(qweight, qzeros, scales, None, zeros_offset)
    q = unpack_int4(packed)[perm.to(packed.device)]
    return pack_int4(q.contiguous()), scale, zero, perm


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


def concat_columns(*parts):
    """Fuse several projections of the SAME input into one GEMM (e.g. attention q / k / v): the
    weights are concatenated along N (INT4, INT8 and FP8 alike). Each part is (packed [K, N_i/2] int8, scale f16 [K/G, N_i],
    zero int8 [K/G, N_i]) in QuantForge layout with the same K and group size, N_i even. Returns
    the same triple for N = sum(N_i); the output of the fused GEMM is the concatenation of the
    outputs (split it with ``out.split([N_1, N_2, ...], -1)``; concatenate the biases the same way).
    Quantisation groups run along K, so nothing about the values changes."""
    if not parts:
        raise ValueError("nothing to concatenate")
    k, g = parts[0][0].shape[0], parts[0][1].shape[0]
    for packed, scale, zero in parts:
        # INT4 weights are packed two per byte ([K, N/2]); INT8 / FP8 weights are [K, N]
        if packed.shape[0] != k or scale.shape[0] != g or zero.shape != scale.shape \
                or packed.shape[1] * 2 != scale.shape[1] and packed.shape[1] != scale.shape[1]:
            raise ValueError("parts must share K and the group size and have consistent shapes")
    return (torch.cat([p[0] for p in parts], dim=1).contiguous(),
            torch.cat([p[1] for p in parts], dim=1).contiguous(),
            torch.cat([p[2] for p in parts], dim=1).contiguous())
