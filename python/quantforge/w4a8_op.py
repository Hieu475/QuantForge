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

import os
from typing import Optional

import torch

from . import kfold, quant
from .e2e import KernelUnavailable

_KERNELS: dict = {}
_BUCKETS = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


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


def _int8_opts(Mp: int, K: int, N: int, kmajor: bool = False) -> str:
    """The schedule the INT8 kernel runs for Mp rows: the one `int8_plan` measured for this shape when there is one (autotune cache or kernel pack; the plans are
    measured with the K-major scale layout, which is what the compact layer keeps), else the rule."""
    from . import int8_plan

    return (int8_plan.lookup(Mp, K, N) if kmajor else None) or _rule_opts(Mp, K, N)


def _rule_opts(Mp: int, K: int, N: int) -> str:
    """Rule-based schedule of the INT8 kernel for Mp (96 .. 256: multiples of 32) rows. The 128 x 128 block with 64 x 64 warp tiles of the large-batch default leaves most of the GPU idle
    or half-used below ~512 rows; measured with cold weights (us, default -> chosen): 8960 x 1536 at 128 rows 183 -> 108 (32 x 128 blocks), at 256 rows 186 -> 139
    (64 x 64 blocks); 1536 x 17920 at 128 rows 165 -> 138, at 256 rows 266 -> 243; the rows between (96 .. 224) have no 128-row padding any more (1536 x 17920 at 192: 190)."""
    from .e2e import W4A8_GEMM_OPTS

    if Mp > 256:
        return W4A8_GEMM_OPTS

    def o(bm, bn, wm, wn):
        return (f"block-tile-m={bm} block-tile-n={bn} block-tile-k=128 warp-tile-m={wm} warp-tile-n={wn} smem-stages=1 "
                "thread-tile-k=32 smem-pad=16 accumulate=false")

    if Mp <= 64 and K % 128 == 0 and K // (256 if K % 256 == 0 else 128) >= 4 and N % 64 == 0:     # a K loop shorter than the pipeline depth is not worth it (and not safe)
        # 32 / 64 rows (the compact layout runs 17-64 requests on the INT8 kernel): 64-column blocks with 16 x 32 warp tiles and a 256-deep K tile; measured (us, old -> new)
        # 8960 x 1536 93 -> 54 (32 rows), 94 -> 60 (64 rows); 1536 x 1536 16.6 -> 11.5; 1536 x 2048 18.6 -> 12.8; wide outputs 2-7 %
        bk = 256 if K % 256 == 0 else 128
        narrow = N <= 2048 or K >= 4 * N
        if Mp == 32 or narrow or N % 128:
            return (f"block-tile-m=32 block-tile-n=64 block-tile-k={bk} warp-tile-m=16 warp-tile-n=32 smem-stages=2 thread-tile-k=32 smem-pad=16 accumulate=false")
        return (f"block-tile-m=64 block-tile-n=128 block-tile-k=128 warp-tile-m=32 warp-tile-n=32 smem-stages=1 prefetch-distance=1 thread-tile-k=32 "
                "smem-pad=16 accumulate=false")
    if Mp <= 96 or (K >= 4 * N and Mp <= 128):
        return o(32, 128, 32, 32)
    if Mp == 128 and N < 8192:
        return W4A8_GEMM_OPTS
    if N % 64 == 0 and Mp % 64 == 0:
        return o(64, 64, 32, 32)
    return o(32, 128, 32, 32)


def _kernel(Mp: int, K: int, N: int, has_bias: bool, kmajor: bool = False, group: int = 128):
    key = (Mp, K, N, has_bias, kmajor) if group == 128 else (Mp, K, N, has_bias, kmajor, group)
    k = _KERNELS.get(key)
    if k is None:
        from .e2e import W4A8GemmDq

        k = _KERNELS[key] = W4A8GemmDq.build(Mp, K, N, bias=has_bias, pipeline_opts=_int8_opts(Mp, K, N, kmajor), kmajor=kmajor, group=group)
    return k


def _decode_opts(K: int, N: int) -> str:
    """Schedule of the single-layout decode kernel (`W4A8DecodeGemm`, 16 rows). The fixed `bn64 wn16 bk128 stages=3` of the first version loses 4-38 % to a
    256-deep K tile with 8-wide warp tiles (cold weights, us, old -> new): 4096 x 4096 89.8 -> 57.5, 8960 x 1536 83.2 -> 51.8, 11008 x 2048 109.5 -> 76.7,
    1536 x 1536 13.0 -> 10.3, 3584 x 4608 58.4 -> 51.5, 1536 x 17920 82.8 -> 80.0; 32-column blocks for the narrow outputs (N <= 1536). K must be a
    multiple of 256 (else the old schedule)."""
    from .e2e import W4A8_DECODE_OPTS

    if K % 256 or N % 64 or K // 256 < 4:                 # (a K loop of fewer than 4 tiles is not pipelined safely: the first schedule)
        return W4A8_DECODE_OPTS
    bn = 32 if N <= 1536 else 64
    return f"decode=true block-tile-n={bn} warp-tile-n=8 block-tile-k=256 smem-stages=3 accumulate=false"


def _decode_kernel(K: int, N: int, has_bias: bool, rows: int = 16, group: int = 128):
    key = ("dec", K, N, has_bias, rows) if group == 128 else ("dec", K, N, has_bias, rows, group)
    k = _KERNELS.get(key)
    if k is None:
        from .e2e import W4A8DecodeGemm

        k = _KERNELS[key] = W4A8DecodeGemm.build(16, K, N, f16_out=True, bias=has_bias, valid_rows=rows if rows < 16 else 0,
                                                 pipeline_opts=_decode_opts(K, N), group=group)
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


def _quantize_tokens(x2: torch.Tensor, Mp: int = 0, rotated: bool = False):
    """Per-token INT8 quantisation of the activations, padded to `Mp` rows (zero rows, scale 0). One CUDA kernel of the C++ runtime (`runtime/act_quant.cu`) when it is loaded,
    else the torch.compile'd function of nn.py plus padding copies; both give the same bits."""
    M = x2.shape[0]
    Mp = Mp or M
    if rotated:                                  # x2 is NOT rotated yet: quantise x2 H (callers check `can_fuse_rotation` first)
        return torch.ops.quantforge.quantize_rows_rot(x2, Mp)
    if _native_quant() and x2.is_contiguous() and x2.shape[1] % 16 == 0:
        return torch.ops.quantforge.quantize_rows(x2, Mp)
    from .nn import _int8_quantize_tokens        # torch.compile'd (one fused kernel pair, dynamic shapes): compiled by the first big call, i.e.
    xq, s = _int8_quantize_tokens(x2)           # vLLM's profiling run, before any CUDA graph is captured
    s = s.reshape(-1)
    if Mp != M:
        xq = torch.nn.functional.pad(xq, (0, 0, 0, Mp - M))
        s = torch.nn.functional.pad(s, (0, Mp - M))
    return xq, s


_NATIVE = None


def _native_quant() -> bool:
    """The `quantize_rows` op exists (the runtime is loaded and built from a source that has it) and `QUANTFORGE_NO_NATIVE_QUANT` is not set."""
    global _NATIVE
    if _NATIVE is None:
        from . import torch_ops  # noqa: F401 - loads the runtime

        _NATIVE = not os.environ.get("QUANTFORGE_NO_NATIVE_QUANT") and hasattr(torch.ops.quantforge, "quantize_rows")
    return _NATIVE


def can_fuse_rotation(x2: torch.Tensor) -> bool:
    """The rotation of the activations can ride in the quantisation kernel (`quantize_rows_rot`: f16 CUDA rows, K a multiple of 128 up to 24576)."""
    return (_native_quant() and hasattr(torch.ops.quantforge, "quantize_rows_rot") and x2.is_cuda and x2.dtype == torch.float16 and x2.is_contiguous()
            and x2.shape[1] <= 24576 and x2.shape[1] % 128 == 0 and x2.data_ptr() % 16 == 0)


def _min_int8_rows() -> int:
    """Calls with at most this many rows use the W4A16 kernels (when the W4A16 layout is kept), more rows the INT8 tensor-core kernel.
    The crossover is a measurement (see the CHANGELOG); `QUANTFORGE_W4A8_MIN_ROWS` overrides it."""
    v = os.environ.get("QUANTFORGE_W4A8_MIN_ROWS")
    return int(v) if v else _DEFAULT_MIN_INT8_ROWS


_DEFAULT_MIN_INT8_ROWS = 64     # measured: the INT8 kernel has a floor (~170 us for 1536 x 17920 at any M <= 128), the W4A16 skinny kernels win up to 64 rows, INT8 from ~96


FALLBACKS = 0          # calls served by the fallbacks below (no kernel in a pack, no toolchain); a pack covers a model when this stays 0
_warned = False


def _fallback(x2, p8, gs, zp, sw, bias, packed16, scale16, zero16, packed_f, scale_f, zero_f, group):
    """No W4A8 kernel for this shape (not in a kernel pack, no toolchain to compile it): the W4A16 kernels when the W4A16 layout is kept, else a dense
    torch matmul with the dequantised weight (correct, slow). Warns once."""
    global FALLBACKS, _warned
    FALLBACKS += 1
    if not _warned:
        import warnings
        warnings.warn("quantforge: no W4A8 kernel for a layer shape (not in a kernel pack, no toolchain to compile it); using a slower fallback for it. "
                      "Build a pack with `python -m quantforge.pack build --w4a8`.")
        _warned = True
    if packed16.numel() > 0:
        return kfold.linear(x2, packed16, scale16, zero16, (packed_f, scale_f, zero_f), group, bias, None)
    w = quant.w4a8_dequantize(p8, gs, zp, sw.float(), group).half()
    y = x2 @ w.t()
    return y if bias is None else y + bias


def _impl(x: torch.Tensor, p8: torch.Tensor, gs: torch.Tensor, zp: torch.Tensor, sw: torch.Tensor, bias: Optional[torch.Tensor],
          packed16: torch.Tensor, scale16: torch.Tensor, zero16: torch.Tensor, packed_f: torch.Tensor, scale_f: torch.Tensor,
          zero_f: torch.Tensor, group: int, scale_nk: torch.Tensor, rotate: bool = False) -> torch.Tensor:
    lead, K, N = x.shape[:-1], x.shape[-1], p8.shape[0]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    fuse = False
    if rotate:     # activations x H (the weights were rotated when they were quantised): up to 16 rows / the W4A16 branch as a kernel of its own, the INT8 branch inside its quantisation kernel
        int8_branch = M > 16 and not (M <= _min_int8_rows() and packed16.numel() > 0)
        fuse = int8_branch and can_fuse_rotation(x2)
        if not fuse:
            x2 = _rotate_impl(x2.contiguous())
    if M <= 16 and packed16.numel() == 0:                             # compact: decode from the W4A8 layout itself (no second copy)
        b = None if bias is None else bias.contiguous()
        try:
            if M in (1, 2, 4, 8) and x2.is_contiguous() and x2.data_ptr() % 16 == 0:     # exactly M rows: no padded copy (valid-rows kernel)
                out = torch.empty(M, N, device=x.device, dtype=torch.float16)
                _decode_kernel(K, N, bias is not None, M, group).launch(x2, p8, zp, scale_nk, out, b)
                return out.reshape(*lead, N)
            out = torch.empty(16, N, device=x.device, dtype=torch.float16)
            _decode_kernel(K, N, bias is not None, 16, group).launch(_padded16(x2), p8, zp, scale_nk, out, b)
            return out[:M].reshape(*lead, N)
        except KernelUnavailable:
            return _fallback(x2, p8, gs, zp, sw, bias, packed16, scale16, zero16, packed_f, scale_f, zero_f, group).reshape(*lead, N)
    if M <= _min_int8_rows() and packed16.numel() > 0:                # decode and small batches: the W4A16 kernels (memory bound / skinny, same bytes)
        y = kfold.linear(x2, packed16, scale16, zero16, (packed_f, scale_f, zero_f), group, bias, None)
        return y.reshape(*lead, N)
    Mp = _bucket(M)
    kmajor = packed16.numel() == 0 and scale16.numel() > 0          # compact layer: gs / zp also kept as int8 [K/128, N] (scale16 / zero16 carry them)
    gk, zk = (scale16, zero16) if kmajor else (gs, zp)
    try:
        k = _kernel(Mp, K, N, bias is not None, kmajor, group)
    except KernelUnavailable:
        if fuse:
            x2 = _rotate_impl(x2)
        return _fallback(x2, p8, gs, zp, sw, bias, packed16, scale16, zero16, packed_f, scale_f, zero_f, group).reshape(*lead, N)
    xq, s_x = _quantize_tokens(x2, Mp, rotated=fuse)                                # padded to the bucket: zero rows, scale 0
    y = torch.empty(Mp, N, device=x.device, dtype=torch.float16)
    k.launch(xq, p8, gk, zk, s_x.contiguous(), sw, None if bias is None else bias.float().contiguous(), y)
    return (y[:M] if Mp != M else y).reshape(*lead, N)


def _fake(x, p8, gs, zp, sw, bias, packed16, scale16, zero16, packed_f, scale_f, zero_f, group, scale_nk, rotate=False):
    return x.new_empty((*x.shape[:-1], p8.shape[0]), dtype=torch.float16)


_op = torch.library.custom_op("quantforge::w4a8_hybrid", mutates_args=())(_impl)
_op.register_fake(_fake)


# ---- block-Hadamard rotation (research/rotation_ppl.py): x -> x H on blocks of 128 along K, the weights are rotated the same way before quantisation ----
_H128 = {}


def hadamard_matrix(n: int = 128, device="cuda") -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard matrix [n, n] f32 (symmetric)."""
    H = torch.ones(1, 1, dtype=torch.float32)  # explicit: vLLM sets the default dtype to f16 while loading
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / n ** 0.5).to(device)


def rotate_weight(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> w H (block-diagonal, blocks of 128 along K), float32."""
    N, K = w.shape
    return (w.float().reshape(N, K // 128, 128) @ hadamard_matrix(128, w.device)).reshape(N, K)


def _rotate_impl(x: torch.Tensor) -> torch.Tensor:
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    if x2.is_cuda and x2.dtype == torch.float16 and K % 128 == 0 and _native_rotate():
        return torch.ops.quantforge.hadamard128(x2.contiguous()).reshape(x.shape)
    H = _H128.get(x.device)
    if H is None:
        H = _H128[x.device] = hadamard_matrix(128, x.device)
    return (x2.float().reshape(-1, K // 128, 128) @ H).reshape(x.shape).to(x.dtype)


_ROT = None


def _native_rotate() -> bool:
    global _ROT
    if _ROT is None:
        from . import torch_ops  # noqa: F401

        _ROT = not os.environ.get("QUANTFORGE_NO_NATIVE_QUANT") and hasattr(torch.ops.quantforge, "hadamard128")
    return _ROT


_rotate_op = torch.library.custom_op("quantforge::rotate128", mutates_args=())(_rotate_impl)
_rotate_op.register_fake(lambda x: x.new_empty(x.shape))           # contiguous, like the real result (empty_like would copy x's strides)


def rotate(x: torch.Tensor) -> torch.Tensor:
    """x H on blocks of 128 along the last dimension (f16 CUDA: one kernel of the C++ runtime)."""
    return torch.ops.quantforge.rotate128(x)
