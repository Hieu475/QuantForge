"""Mixture-of-experts building blocks: grouped W4A16 GEMMs (one kernel launch for all experts of a layer).

The kernels are the ordinary dense kernels with two additions made by the compiler (``ExpertBatchPass.cpp``,
pipeline options ``expert-problems`` / ``expert-roles``): ``grid.z`` is a list of *problems* (one row block each), and the
expert of every problem is read from a device array, ``meta[z]``; a negative entry is padding and costs nothing. The
rows are sorted by expert and padded to the row tile, so each row block belongs to one expert. All of it is
static-shape and free of host synchronisation, hence usable inside CUDA graphs.

    meta   int32 [P]                 expert of problem z (-1: unused)
    rows   f16   [P*Mcap, K]         activations, sorted and padded by expert
    packed int8  [E, K, N/2]         INT4 weights of all experts (``quantforge.e2e.pack_int4`` layout)
    scale  f16   [E, K/G, N]         zero   int8 [E, K/G, N]
    out    f16 / f32 [P*Mcap, N]     (swiglu: only the first N/2 columns are meaningful)
"""
from __future__ import annotations

import ctypes
import tempfile

from .e2e import (W4A16Gemm, _current_raw_stream, _descriptor, _descriptor_type, compile_ir, tile_sizes, w4a16_f16_ir,
                  w4a16_gemm_ir, _check_shape)

ROLES = "reeer"          # host arguments (rows, packed, scale, zero, out): rows / expert / expert / expert / rows


class GroupedGemm:
    """``out[z*Mcap:(z+1)*Mcap] = rows[z*Mcap:(z+1)*Mcap] @ dequant(W[meta[z]])`` for every problem z with ``meta[z] >= 0``.

    ``kind="f16"``: f16 output (``epilogue=f16-store``). ``kind="swiglu"``: gate and up projections interleaved by column
    (``quant.interleave_gate_up``), f32 output of ``silu(gate) * up`` in the first N/2 columns."""

    def __init__(self, compiled, name: str, kind: str, Mcap: int, K: int, N: int, E: int, P: int, group_size: int):
        self.kind, self.Mcap, self.K, self.N, self.E, self.P, self.group_size = kind, Mcap, K, N, E, P, group_size
        self.compiled = compiled
        self._lib = ctypes.CDLL(str(compiled.lib_path))
        self._fn = getattr(self._lib, f"_mlir_ciface_{name}")
        self._fn.argtypes = [ctypes.c_void_p] * 7
        self._fn.restype = None
        self._set_stream = self._lib.mgpuSetStream
        self._set_stream.argtypes = [ctypes.c_void_p]
        self._result = _descriptor_type(2)()

    @classmethod
    def build(cls, Mcap: int, K: int, N: int, E: int, P: int, group_size: int = 128, kind: str = "f16",
              pipeline_opts: str = "", workdir=None, chip: str = "sm_80") -> "GroupedGemm":
        if kind not in ("f16", "swiglu"):
            raise ValueError("kind must be 'f16' or 'swiglu'")
        if P < 1:
            raise ValueError("need at least one problem")
        _check_shape(Mcap, K, N, tile_sizes(pipeline_opts))
        if kind == "swiglu":
            name = "w4a16_gemm"
            ir = w4a16_gemm_ir(Mcap, K, N, name, "f32", group_size, pipeline_opts)
            opts = f"{pipeline_opts} epilogue=swiglu"
        else:
            name = "w4a16_gemm_f16"
            ir = w4a16_f16_ir(Mcap, K, N, group_size, name, False)
            opts = f"{pipeline_opts} accumulate=false epilogue=f16-store"
        opts = f"{opts} expert-problems={P} expert-roles={ROLES}".strip()
        with tempfile.TemporaryDirectory() as tmp:
            return cls(compile_ir(ir, workdir or tmp, chip=chip, pipeline_opts=opts), name, kind, Mcap, K, N, E, P, group_size)

    def launch(self, rows, packed, scale, zero, out, meta) -> None:
        """Asynchronous, on PyTorch's current stream. Shapes as in the module docstring; every tensor contiguous and 16-byte aligned."""
        import torch

        M = self.Mcap
        descs = [_descriptor(t) for t in (rows[:M], packed[0], scale[0], zero[0], out[:M], meta)]
        self._set_stream(_current_raw_stream(torch, rows.device.index or 0))
        self._fn(ctypes.byref(self._result), *(ctypes.byref(d) for d in descs))


# --------------------------------------------------------------------------- #
# the MoE feed-forward op: route -> grouped GEMM (gate/up + SwiGLU) -> grouped GEMM (down) -> weighted sum
# --------------------------------------------------------------------------- #
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

_T_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
MAX_TOKENS = _T_BUCKETS[-1]            # larger batches run in chunks of this many tokens
_kernels: dict = {}
_lock = threading.RLock()


def _row_tile(tokens: int, top_k: int, E: int) -> int:
    """Row tile of the grouped GEMMs: the average number of rows per expert decides how much padding a tile may waste."""
    avg = tokens * top_k / E
    return 16 if avg <= 8 else 32 if avg <= 24 else 64 if avg <= 56 else 128


def _problems(tokens: int, top_k: int, E: int, mcap: int) -> int:
    """Static upper bound of the row blocks: every expert wastes at most mcap - 1 padding rows."""
    return -(-(tokens * top_k + E * (mcap - 1)) // mcap)


def _opts(mcap: int, K: int, N: int) -> str:
    from . import torch_ops as t

    return t.default_opts(mcap, K, N)


def _kernel(kind: str, mcap: int, K: int, N: int, E: int, P: int, group: int) -> GroupedGemm:
    key = (kind, mcap, K, N, E, P, group)
    k = _kernels.get(key)
    if k is None:
        with _lock:
            k = _kernels.get(key)
            if k is None:
                k = _kernels[key] = GroupedGemm.build(mcap, K, N, E, P, group, kind, _opts(mcap, K, N))
    return k


def _fast_ops() -> bool:
    """The CUDA routing / gather / combine ops of the C++ runtime (``runtime/moe_kernels.cu``) are loaded (not with ``QUANTFORGE_NO_CPP``, nor from a
    kernel pack built before they existed): else the same arithmetic runs as torch ops."""
    global _FAST
    if _FAST is None:
        from . import torch_ops  # noqa: F401 - importing it builds / loads the runtime

        _FAST = not os.environ.get("QUANTFORGE_NO_MOE_OPS") and hasattr(torch.ops.quantforge, "moe_route")
    return _FAST


_FAST = None


def bucket_tokens(T: int) -> int:
    for b in _T_BUCKETS:
        if T <= b:
            return b
    return MAX_TOKENS


def route(topk_ids: torch.Tensor, E: int, mcap: int, P: int):
    """Sort the (token, slot) pairs by expert and pad every expert to a multiple of ``mcap`` rows. No host synchronisation, static shapes.

    Returns ``(dest, token, dest_sorted, meta, valid)``: ``dest`` row of the padded buffer of every pair in token-major order, ``token`` the token of
    each pair in sorted order, ``dest_sorted`` its row, ``meta`` int32 [P] (expert of each row block, -1 padding), ``valid`` the pairs whose id is
    in ``[0, E)`` (vLLM marks dummy / dropped experts with -1: they go to the scratch row ``P * mcap`` and are weighted 0)."""
    T, k = topk_ids.shape
    n = T * k
    flat = topk_ids.reshape(-1).to(torch.int64)
    dev = flat.device
    valid = (flat >= 0) & (flat < E)
    fe = torch.where(valid, flat, torch.full_like(flat, E))             # invalid pairs sort last, into a sentinel bucket
    order = torch.argsort(fe, stable=True)
    sorted_e = fe[order]
    counts = torch.zeros(E + 1, dtype=torch.int64, device=dev).scatter_add_(0, fe, torch.ones_like(fe))
    padded = (counts[:E] + mcap - 1) // mcap * mcap
    pend = torch.cumsum(padded, 0)
    pstart = torch.cat((pend - padded, torch.full((1,), P * mcap, dtype=torch.int64, device=dev)))
    cstart = torch.cumsum(counts, 0) - counts
    rank = torch.arange(n, device=dev) - cstart[sorted_e]
    dest_sorted = torch.where(sorted_e < E, pstart[sorted_e] + rank, pstart[sorted_e])      # the sentinel bucket: the scratch row
    blocks = torch.arange(P, device=dev) * mcap
    e_of_block = torch.searchsorted(pend, blocks, right=True)
    meta = torch.where(e_of_block < E, e_of_block, torch.full_like(e_of_block, -1)).to(torch.int32)
    dest = torch.empty_like(dest_sorted)
    dest[order] = dest_sorted                                          # row of pair i (token-major)
    return dest, order // k, dest_sorted, meta, valid


def moe_ffn_impl(x, topk_ids, topk_w, packed13, scale13, zero13, packed2, scale2, zero2, group_size: int) -> torch.Tensor:
    """``sum_s w[t, s] * W2[e] (silu(W1g[e] x[t]) * W1u[e] x[t])`` over the top-k experts ``e = topk_ids[t, s]`` of every token.

    ``packed13 [E, H, 2I/2]`` holds gate and up interleaved by column (``quant.interleave_gate_up``), ``packed2 [E, I, H/2]``;
    scales / zero points as in the module docstring. f16 activations."""
    T, H = x.shape
    E = packed13.shape[0]
    I = packed2.shape[1]
    if T > MAX_TOKENS:
        return torch.cat([moe_ffn_impl(x[i:i + MAX_TOKENS], topk_ids[i:i + MAX_TOKENS], topk_w[i:i + MAX_TOKENS], packed13, scale13,
                                       zero13, packed2, scale2, zero2, group_size) for i in range(0, T, MAX_TOKENS)])
    k = topk_ids.shape[1]
    Tb = bucket_tokens(T)
    mcap = _row_tile(Tb, k, E)
    P = _problems(Tb, k, E, mcap)
    k1 = _kernel("swiglu", mcap, H, 2 * I, E, P, group_size)
    k2 = _kernel("f16", mcap, I, H, E, P, group_size)
    R = P * mcap
    if _fast_ops():
        ops = torch.ops.quantforge
        dest, row_src, meta = ops.moe_route(topk_ids.contiguous(), E, mcap, P)
        rows = ops.moe_gather(x.contiguous(), row_src, k)
        g1 = torch.empty(R, 2 * I, dtype=torch.float32, device=x.device)
        k1.launch(rows, packed13, scale13, zero13, g1, meta)
        h = ops.moe_cast_rows(g1, meta, mcap, I)
        y = torch.empty(R + 1, H, dtype=torch.float16, device=x.device)
        k2.launch(h, packed2, scale2, zero2, y, meta)
        return ops.moe_combine(y, dest, topk_w.contiguous())
    dest, token, dest_sorted, meta, valid = route(topk_ids, E, mcap, P)
    rows = torch.zeros(R + 1, H, dtype=torch.float16, device=x.device)       # + the scratch row of the invalid pairs
    rows[dest_sorted] = x[token]
    g1 = torch.empty(R, 2 * I, dtype=torch.float32, device=x.device)
    k1.launch(rows, packed13, scale13, zero13, g1, meta)
    h = g1[:, :I].to(torch.float16).contiguous()
    y = torch.empty(R + 1, H, dtype=torch.float16, device=x.device)
    k2.launch(h, packed2, scale2, zero2, y, meta)
    y[R].zero_()
    w = torch.where(valid.view(T, k), topk_w.float(), torch.zeros_like(topk_w, dtype=torch.float32))
    picked = y[dest].view(T, k, H).float()
    return (picked * w.unsqueeze(-1)).sum(1).to(torch.float16)


def moe_ffn_reference(x, topk_ids, topk_w, packed13, scale13, zero13, packed2, scale2, zero2, group_size: int) -> torch.Tensor:
    """Dense torch version (dequantises the selected experts); the oracle of the tests and the fallback without a toolchain."""
    from .quant import unpack_int4

    def dq(p, s, z):
        q = unpack_int4(p).float()
        g = q.shape[0] // s.shape[0]
        return (q - z.float().repeat_interleave(g, 0)) * s.float().repeat_interleave(g, 0)

    T, H = x.shape
    out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for t in range(T):
        for s in range(topk_ids.shape[1]):
            e = int(topk_ids[t, s])
            gu = x[t].float() @ dq(packed13[e], scale13[e], zero13[e])
            h = torch.nn.functional.silu(gu[0::2]) * gu[1::2]
            out[t] += topk_w[t, s].float() * (h.half().float() @ dq(packed2[e], scale2[e], zero2[e]))
    return out.half()


# --------------------------------------------------------------------------- #
# conversion from vLLM's WNA16 MoE weight layout (what its GPTQ / AWQ MoE loader fills) to the grouped-GEMM layout
# --------------------------------------------------------------------------- #
def _nibbles(qw: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n/2] -> int8 [..., n]: low nibble first."""
    return torch.stack((qw & 0xF, qw >> 4), dim=-1).reshape(*qw.shape[:-1], qw.shape[-1] * 2).to(torch.int8)


def _pack_last(q: torch.Tensor) -> torch.Tensor:
    """int8 [..., n] 0..15 -> int8 [..., n/2]: low nibble = even index (``e2e.pack_int4`` along the last dim)."""
    return ((q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)).to(torch.int8)


def from_wna16(w13_qweight, w13_scales, w13_qzeros, w2_qweight, w2_scales, w2_qzeros, sym_zero: int = 8):
    """vLLM MoE WNA16 tensors -> ``(packed13, scale13, zero13, packed2, scale2, zero2)`` for ``moe_ffn_impl``.

    vLLM: ``w13_qweight`` uint8 [E, 2I, H/2] (gate rows, then up rows; low nibble = even k), ``w13_scales`` [E, 2I, H/G], ``w13_qzeros`` uint8
    [E, 2I/2, H/G] (two zero points per byte along the output dim, already +1 for GPTQ) or None (symmetric: zero = ``sym_zero``),
    ``w2_qweight`` uint8 [E, H, I/2] ... Ours: gate / up interleaved by column, K-major ([E, K, N/2], [E, K/G, N])."""
    E, N13, _ = w13_qweight.shape
    I = N13 // 2

    def zeros(qz, rows, groups):
        if qz is None:
            return torch.full((qz_shape(rows, groups)), sym_zero, dtype=torch.int8, device=w13_qweight.device)
        return _nibbles(qz.permute(0, 2, 1).contiguous()).permute(0, 2, 1)      # [E, N, groups]

    def qz_shape(rows, groups):
        return (E, rows, groups)

    def inter(t):                                                              # [E, 2I, ...] gate;up -> gate,up interleaved rows
        return torch.stack((t[:, :I], t[:, I:]), dim=2).reshape(t.shape)

    q13 = inter(_nibbles(w13_qweight))                                         # [E, 2I, H]
    s13 = inter(w13_scales)                                                    # [E, 2I, H/G]
    z13 = inter(zeros(w13_qzeros, N13, w13_scales.shape[2]))
    packed13 = _pack_last(q13.permute(0, 2, 1).contiguous())                   # [E, H, I]  (= 2I / 2 bytes per row)
    scale13 = s13.permute(0, 2, 1).contiguous().half()
    zero13 = z13.permute(0, 2, 1).contiguous()
    H = w2_qweight.shape[1]
    packed2 = _pack_last(_nibbles(w2_qweight).permute(0, 2, 1).contiguous())   # [E, I, H/2]
    scale2 = w2_scales.permute(0, 2, 1).contiguous().half()
    zero2 = zeros(w2_qzeros, H, w2_scales.shape[2]).permute(0, 2, 1).contiguous()
    return packed13, scale13, zero13, packed2, scale2, zero2


def prepare(E: int, H: int, I: int, top_k: int, group_size: int, max_tokens: int = MAX_TOKENS) -> None:
    """Compile (or load from a pack) the kernels of every token bucket now, so that nothing is built inside a CUDA graph capture."""
    for Tb in _T_BUCKETS:
        if Tb > max(max_tokens, 1) and Tb != bucket_tokens(max_tokens):
            continue
        mcap = _row_tile(Tb, top_k, E)
        P = _problems(Tb, top_k, E, mcap)
        _kernel("swiglu", mcap, H, 2 * I, E, P, group_size)
        _kernel("f16", mcap, I, H, E, P, group_size)
