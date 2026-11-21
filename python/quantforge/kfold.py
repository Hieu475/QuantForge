"""K-folding: fill the GPU with a long-K, narrow-N decode GEMM without split-K in the compiler.

A decode GEMM ``[M, K] x [K, N]`` with a small N (the ``down_proj`` / ``o_proj`` of every transformer: N = 896 .. 2048, K up
to 9 x N) has only 12-24 column blocks for 20 SMs and cannot stream the weights fast enough (0.55-0.75x of Marlin). Folding
cuts K into S slices and lays them side by side as columns of one wider GEMM:

    W' = [W_0 | W_1 | ... | W_{S-1}]  with  W_s = W[s*K/S:(s+1)*K/S, :]      shape [K/S, S*N]
    A' = rows (m, s) of x:  A'[m*S + s] = x[m, s*K/S:(s+1)*K/S]               shape [M*S, K/S]   (a free reshape of x)
    C' = A' @ W'          shape [M*S, S*N];   y[m] = sum_s C'[m*S + s, s*N:(s+1)*N]

The kernel computes every (slice, slice) pair, S times more than needed, but the tensor-core tile already has 16 rows, so
while ``M * S <= 16`` the extra work is free; the weights are still read exactly once, by S times as many blocks with a
K loop S times shorter. The result is a sum of S diagonal blocks (one small torch kernel). Needs ``K % (S * group) == 0``;
costs a second copy of the weights (the plain layout is kept for the calls with more than ``16 // S`` rows).

    S = plan_ksplit(K, N, group)                    # measured once per shape (cached), 1 = do not fold
    packed_f, scale_f, zero_f = fold_k(packed, scale, zero, S)
    y = torch.ops.quantforge.w4a16_gemm_kfold(x, packed, scale, zero, packed_f, scale_f, zero_f, group, bias)
"""
from __future__ import annotations

from typing import Optional

import torch

from . import torch_ops  # noqa: F401  (registers quantforge::w4a16_gemm)

MAX_N = 2048          # wider outputs already have enough blocks (measured: no gain from N = 4096)
MIN_GAIN = 0.85       # fold only if the folded GEMM takes <= 85 % of the plain one
_KEY_VERSION = 2


def fold_k(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, S: int):
    """Lay S slices of K side by side as columns (works for int4 / int8 / fp8, all of them are ``[K, *]``)."""
    K, G = packed.shape[0], scale.shape[0]
    if S < 2 or K % S or G % S:
        raise ValueError(f"cannot fold K={K} ({G} groups) into {S} slices")
    cat = lambda t: torch.cat([t[i * (t.shape[0] // S):(i + 1) * (t.shape[0] // S)] for i in range(S)], dim=1).contiguous()  # noqa: E731
    return cat(packed), cat(scale), cat(zero)


def _folded_gemm(x2, packed_f, scale_f, zero_f, S, group, bias):
    M = x2.shape[0]
    N = scale_f.shape[1] // S
    yf = torch.ops.quantforge.w4a16_gemm(x2.reshape(M * S, -1), packed_f, scale_f, zero_f, group, None, "none")
    y = yf.view(M, S, S, N).diagonal(dim1=1, dim2=2).sum(-1, dtype=torch.float32).to(torch.float16)
    return y if bias is None else y + bias


def _factor(scale: torch.Tensor, scale_f: torch.Tensor) -> int:
    """The fold factor is read from the tensor shapes, never passed as a constant: a graph compiled once (vLLM caches its
    compiled graphs across runs) stays correct for weights folded differently; an empty ``scale_f`` means 'not folded'."""
    return scale_f.shape[1] // scale.shape[1] if scale_f.numel() else 1


def _kfold_impl(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, packed_f: torch.Tensor,
                scale_f: torch.Tensor, zero_f: torch.Tensor, group_size: int,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    S = _factor(scale, scale_f)
    if S > 1 and x2.shape[0] * S <= 16:
        y = _folded_gemm(x2, packed_f, scale_f, zero_f, S, group_size, bias)
    else:
        y = torch.ops.quantforge.w4a16_gemm(x2, packed, scale, zero, group_size, bias, "none")
    return y.reshape(*lead, y.shape[-1])


def _kfold_fake(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, packed_f: torch.Tensor,
                scale_f: torch.Tensor, zero_f: torch.Tensor, group_size: int,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], scale.shape[1]), dtype=torch.float16)


_kfold_op = torch.library.custom_op("quantforge::w4a16_gemm_kfold", mutates_args=())(_kfold_impl)
_kfold_op.register_fake(_kfold_fake)


def no_fold(device) -> tuple:
    """Placeholder folded weights of a layer that is not folded (the op then always runs the plain kernel)."""
    e = torch.empty(0, 0, device=device)
    return e.to(torch.int8), e.to(torch.float16), e.to(torch.int8)


def linear(x: torch.Tensor, packed, scale, zero, folded, group_size: int,
           bias: Optional[torch.Tensor] = None, perm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``x @ dequant(W) + bias`` with the folded weights ``folded = (packed_f, scale_f, zero_f)`` (``no_fold`` for none);
    the calls with more than ``16 // S`` rows take the plain kernel. ``x``: f16 ``[..., K]``."""
    if perm is not None:
        x = x.index_select(-1, perm.to(x.device))
    return torch.ops.quantforge.w4a16_gemm_kfold(x, packed, scale, zero, *folded, group_size, bias)


def plan_ksplit(K: int, N: int, group_size: int, kind: int = 4, device: int | None = None) -> int:
    """Best fold factor for a ``[K, N]`` weight at one row (1 = leave it alone). Measured on the real op, cached in the
    autotune cache; among the factors within 5 % of the best the smallest wins (it serves more rows: ``M * S <= 16``)."""
    from . import autotune as at

    if kind not in (4, 8) or N > MAX_N or not group_size or K % group_size:
        return 1
    device = torch.cuda.current_device() if device is None else device
    gpu = torch.cuda.get_device_name(device).replace(" ", "_")
    key = f"kfold{_KEY_VERSION}|{gpu}|k{kind}|g{group_size}|K{K}|N{N}"
    cache = at._load()
    if key in cache:
        return int(cache[key])
    groups = K // group_size
    cands = [s for s in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16) if groups % s == 0 and (S := s) and s * N <= 1 << 15]
    if not cands:
        return 1
    dev = torch.device("cuda", device)
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.rand(1, K, device=dev, generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int8, generator=g)
    packed = ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.int8) if kind == 4 else (q - 8).to(torch.int8)
    scale = (torch.rand(groups, N, device=dev, generator=g) * 0.1 + 0.01).half()
    zero = torch.randint(0, 16, (groups, N), device=dev, dtype=torch.int8, generator=g)
    plain = at.device_time(lambda: torch.ops.quantforge.w4a16_gemm(x, packed, scale, zero, group_size, None, "none"))
    times = {}
    for s in cands:
        pf, sf, zf = fold_k(packed, scale, zero, s)
        times[s] = at.device_time(lambda: torch.ops.quantforge.w4a16_gemm(
            x.reshape(s, -1), pf, sf, zf, group_size, None, "none"))
    best = min(times.values())
    s_best = min(s for s, t in times.items() if t <= 1.05 * best)
    choice = s_best if times[s_best] <= MIN_GAIN * plain else 1
    cache[key] = choice
    at._store(cache)
    return choice
