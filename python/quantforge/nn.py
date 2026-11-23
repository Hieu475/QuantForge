"""``torch.nn`` integration: quantise the Linear layers of any PyTorch / Hugging Face model.

    import quantforge.nn as qnn
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    qnn.quantize_model(model, fmt="int4", group_size=128)     # decoder Linears -> QuantLinear on the GPU
    qnn.quantize_model(model, fmt="int8", only=("lm_head",))   # e.g. a lossless INT8 head afterwards

``fmt``: ``int4`` (packed, asymmetric RTN), ``int8`` (symmetric RTN) or ``fp8`` (E4M3), all with one
f16 scale per ``group_size`` input rows of every output column. The layers run on the QuantForge
kernels through ``torch.ops.quantforge.w4a16_gemm`` (see ``torch_ops``), so they work eagerly, under
``torch.compile`` and in CUDA graphs. The weights are quantised round-to-nearest from the dense
weights: for a better quality at 4 bit quantise with GPTQ / AWQ elsewhere and wrap the result with
``QuantLinear.from_quantized``.

Llama / Qwen / Mistral style decoders can additionally be *fused* (``fuse=True``, the default): the
``q_proj`` / ``k_proj`` / ``v_proj`` of an attention block become one GEMM, and a SiLU-gated MLP
(``gate_proj`` / ``up_proj`` / ``down_proj``) runs gate + up as one GEMM with the ``swiglu`` epilogue
(INT4; INT8 / FP8 use a fused-silu gate GEMM and an up GEMM).
"""
from __future__ import annotations

import gc
import math
import warnings
from typing import Iterable, Optional

import torch
import torch.nn as nn

from . import quant
from . import torch_ops as _ops

__all__ = ["QuantLinear", "FP8W8A8Linear", "INT8W8A8Linear", "quantize_model", "load_quantized_checkpoint", "enable_kfold", "FORMATS"]

FORMATS = ("int4", "int8", "fp8")
W4A8_FORMAT = "w4a8"          # INT4 weights, INT8 activations on the INT8 tensor cores, see W4A8Linear
INT8_W8A8_FORMAT = "int8_w8a8"   # INT8 weights *and* activations on the INT8 tensor cores (cuBLASLt), see INT8W8A8Linear
W8A8_FORMAT = "fp8_w8a8"      # FP8 weights *and* activations on the Ada / Hopper FP8 tensor cores (cuBLASLt), see FP8W8A8Linear


def _pick_group(in_features: int, group_size: int) -> Optional[int]:
    """`group_size`, or the largest of 64 / 32 / 16 that divides `in_features` (None: no valid group)."""
    for g in (group_size, 64, 32, 16):
        if g <= group_size and g > 0 and in_features % g == 0:
            return g
    return None


class QuantLinear(nn.Module):
    """``y = x @ dequant(W) + bias`` on the QuantForge kernels.

    Buffers: ``packed`` (int8 ``[K, N/2]`` for int4, int8 ``[K, N]`` for int8, ``float8_e4m3fn`` ``[K, N]``
    for fp8), ``scale`` (f16 ``[K/G, N]``), ``zero`` (int8 ``[K/G, N]``) and an optional f16 ``bias``."""

    def __init__(self, packed, scale, zero, bias, group_size: int, fmt: str, perm=None):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}")
        self.fmt, self.group_size = fmt, group_size
        self.in_features, self.out_features = packed.shape[0], scale.shape[1]
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.register_buffer("bias", bias)
        self.register_buffer("perm", perm)      # GPTQ act-order: input columns are gathered with it
        self.ksplit = 1                          # > 1: second copy of the weights folded for narrow-N decode (see kfold.py)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, fmt: str = "int4", group_size: int = 128,
                    device="cuda") -> "QuantLinear":
        """Quantise a dense ``nn.Linear`` (round-to-nearest) and place the result on `device`."""
        g = _pick_group(lin.in_features, group_size)
        if g is None:
            raise ValueError(f"in_features={lin.in_features} has no usable group size (<= {group_size})")
        if fmt == "int4" and lin.out_features % 2:
            raise ValueError("int4 needs an even out_features")
        w = lin.weight.data.t()                                              # [K, N] view
        K, N = w.shape
        # quantise in chunks of rows (groups run along K): a 150k-row lm_head must not need several
        # fp32 copies of itself on a small GPU
        step = max(g, (1 << 22) // max(N, 1) // g * g)
        parts = []
        for r in range(0, K, step):
            wc = w[r:r + step].to(device, torch.float32).contiguous()
            if fmt == "fp8":
                packed, scale, zero = quant.quantize_fp8(wc, g)
            elif fmt == "int8":
                packed, scale, zero = quant.quantize_groupwise_int8(wc, g)
            elif fmt == "int4":
                q, scale, zero = quant.quantize_groupwise(wc, g)
                packed = quant.pack_int4(q)
            else:
                raise ValueError(f"fmt must be one of {FORMATS}")
            parts.append((packed, scale, zero))
            del wc
        packed, scale, zero = (torch.cat([p[i] for p in parts]) for i in range(3))
        bias = None if lin.bias is None else lin.bias.data.to(device, torch.float16)
        return cls(packed.contiguous(), scale.contiguous(), zero.contiguous(), bias, g, fmt)

    @classmethod
    def from_quantized(cls, packed, scale, zero, bias=None, group_size: int | None = None,
                       fmt: str = "int4", perm=None) -> "QuantLinear":
        """Wrap tensors already in QuantForge layout (e.g. from ``quant.from_gptq`` / ``from_awq``;
        ``perm`` from ``quant.from_gptq_act_order``)."""
        g = group_size or packed.shape[0] // scale.shape[0]
        return cls(packed, scale, zero, bias, g, fmt, perm)

    def enable_kfold(self, ksplit: Optional[int] = None) -> int:
        """Fold K for the calls with few rows (``kfold.py``): a long-K, narrow-N layer (``down_proj``, ``o_proj`` of a
        small model) decodes up to 1.6x faster. Keeps a second copy of the weights. ``ksplit=None``: measure the best
        factor (cached); returns the factor in use (1 = not worth it)."""
        from . import kfold

        kind = 8 if self.fmt == "int8" else 4 if self.fmt == "int4" else 0
        S = kfold.plan_ksplit(self.in_features, self.out_features, self.group_size, kind) if ksplit is None else ksplit
        if kind == 4 and self.group_size > 0:
            from . import atomic_plan

            atomic_plan.preplan(self.in_features, self.out_features, self.group_size)
        if S > 1 and kind in (4, 8):
            pf, sf, zf = kfold.fold_k(self.packed, self.scale, self.zero, S)
            self.register_buffer("packed_f", pf, persistent=False)
            self.register_buffer("scale_f", sf, persistent=False)
            self.register_buffer("zero_f", zf, persistent=False)
            self.ksplit = S
        return self.ksplit

    _atomic_planned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xh = x if dt == torch.float16 else x.half()
        if not self._atomic_planned and self.fmt == "int4" and self.group_size > 0 and x.is_cuda and not torch.compiler.is_compiling():
            self._atomic_planned = True      # the op only looks plans up (it must not start profiler sessions inside an operator call): decide them once, here
            from . import atomic_plan

            atomic_plan.preplan(self.in_features, self.out_features, self.group_size)
        if self.ksplit > 1:
            from . import kfold

            y = kfold.linear(xh, self.packed, self.scale, self.zero, (self.packed_f, self.scale_f, self.zero_f),
                             self.group_size, self.bias, self.perm)
        else:
            y = _ops.w4a16_linear(xh, self.packed, self.scale, self.zero, self.group_size, self.bias, perm=self.perm)
        return y if dt == torch.float16 else y.to(dt)

    def dequantized_weight(self) -> torch.Tensor:
        """The dense f16 ``[out, in]`` weight the kernels multiply with (reference / debugging)."""
        g = self.group_size
        if self.fmt == "fp8":
            q, zero = self.packed.float(), torch.zeros_like(self.zero)
        elif self.fmt == "int8":
            q, zero = self.packed.float(), self.zero
        else:
            q, zero = quant.unpack_int4(self.packed).float(), self.zero
        w = (q - zero.float().repeat_interleave(g, 0)) * self.scale.float().repeat_interleave(g, 0)
        w = w.half().t().contiguous()                                     # [out, in] in permuted order
        if self.perm is not None:                                         # back to the original input order
            w = w[:, torch.argsort(self.perm)]
        return w

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, fmt={self.fmt}, "
                f"group_size={self.group_size}, bias={self.bias is not None}, act_order={self.perm is not None}")


@torch.no_grad()
def enable_kfold(model: nn.Module) -> dict:
    """``QuantLinear.enable_kfold()`` on every layer (also inside the fused attention / MLP blocks). Returns
    ``{"folded": n, "layers": total, "factors": {S: count}}``."""
    factors: dict[int, int] = {}
    total = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            total += 1
            S = m.enable_kfold()
            if S > 1:
                factors[S] = factors.get(S, 0) + 1
    return {"folded": sum(factors.values()), "layers": total, "factors": factors}


def _maybe_compile(fn):
    try:                                   # fuse the element-wise chains into a few kernels (the FP8 GEMM itself is cuBLASLt)
        return torch.compile(fn, dynamic=True)
    except Exception:                      # noqa: BLE001
        return fn


@_maybe_compile
def _fp8_quantize_tokens(x2):
    s = (x2.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp(min=1e-8)                # per-token scale
    return (x2.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s


@_maybe_compile
def _fp8_quantize_tensor(x2):
    s = (x2.abs().amax().float() / 448.0).clamp(min=1e-8)                                     # one scale for the whole tensor
    return (x2.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn), s


@_maybe_compile
def _fp8_epilogue(y, s_x, w_scale, bias, dtype):
    out = (y * s_x * w_scale.float()).to(dtype)
    return out if bias is None else out + bias.to(dtype)


class FP8W8A8Linear(nn.Module):
    """FP8 (E4M3) weights **and** activations: ``y = (quant(x) @ W8^T) * s_x * s_w + bias`` on the FP8 tensor cores.

    Default (``per_token=True``): weights with a per-output-channel scale ``s_w``, activations with a dynamic per-token scale
    ``s_x`` (amax / 448); ``per_token=False``: one scale per tensor on both sides, applied by cuBLASLt inside the GEMM (no epilogue
    pass, but its speed is erratic across shapes: 0.35x to 1.5x of fp16 in the layer benchmark). The product runs
    through ``torch._scaled_mm`` (cuBLASLt) with unit tensor scales and the two scale vectors applied to the f32 result, because
    the row-wise scaling mode of ``_scaled_mm`` is not available on sm_89. This is a *compute* format: on an RTX 4050 the FP8 GEMM
    runs at ~1.8-2.6x the speed of the fp16 one for M >= 512 (prefill, large batches), but the weights take twice the bytes of
    INT4, so for batch-1 decode ``QuantLinear`` (INT4) is faster. Needs an sm_89+ GPU. Not generated by the QuantForge compiler:
    LLVM 18's NVPTX backend cannot emit the FP8 ``mma`` instruction (see the blog)."""

    def __init__(self, w8_nk, w_scale, bias, per_token: bool = True):
        super().__init__()
        self.per_token = per_token
        self.in_features, self.out_features = w8_nk.shape[1], w8_nk.shape[0]
        self.register_buffer("weight", w8_nk)                      # float8_e4m3fn [out, in]
        self.register_buffer("w_scale", w_scale)                   # f16 [out]
        self.register_buffer("bias", bias)
        self.register_buffer("_one", torch.ones((), dtype=torch.float32, device=w8_nk.device))

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, device="cuda", per_token: bool = True) -> "FP8W8A8Linear":
        w = lin.weight.data.to(device, torch.float32)
        if per_token:        # per-output-channel weight scale + per-token activation scale (more accurate, ~2 extra passes)
            scale = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-8)
            w8 = (w / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            scale = scale.half()
        else:                # one scale per tensor on both sides: cuBLASLt applies them inside the GEMM, f16 out, no epilogue pass
            scale = (w.abs().amax() / 448.0).clamp(min=1e-8)
            w8 = (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        bias = None if lin.bias is None else lin.bias.data.to(device, torch.float16)
        return cls(w8, scale, bias, per_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.per_token:
            x8, s_x = _fp8_quantize_tokens(x2)
            # unit tensor scales and an f32 result: the raw FP8 products (up to 448 * 448 * K) overflow f16 before the scales are applied
            y = torch._scaled_mm(x8, self.weight.t(), scale_a=self._one, scale_b=self._one, out_dtype=torch.float32)
            y = _fp8_epilogue(y, s_x, self.w_scale, self.bias, torch.float16 if x.dtype == torch.float32 else x.dtype)
        else:
            x8, s_x = _fp8_quantize_tensor(x2)
            y = torch._scaled_mm(x8, self.weight.t(), scale_a=s_x, scale_b=self.w_scale, bias=self.bias, out_dtype=torch.float16)
            y = y if x.dtype in (torch.float16, torch.float32) else y.to(x.dtype)
        return y.reshape(*shape[:-1], self.out_features)

    def dequantized_weight(self) -> torch.Tensor:
        scale = self.w_scale.float()
        return (self.weight.float() * (scale[:, None] if self.per_token else scale)).half()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fmt=fp8_w8a8"


@_maybe_compile
def _int8_quantize_tokens(x2):
    s = (x2.abs().amax(dim=1, keepdim=True).float() / 127.0).clamp(min=1e-8)                 # per-token scale
    return (x2.float() / s).round().clamp(-127, 127).to(torch.int8), s


@_maybe_compile
def _silu_mul_quantize(gu, inter: int):
    """``silu(gate) * up`` of a ``[M, 2 * inter]`` gate / up output and its per-token INT8 quantisation in one compiled kernel pair
    (the input of ``down_proj``): saves the two element-wise passes over the ``[M, inter]`` activations."""
    g, u = gu[:, :inter].float(), gu[:, inter:].float()
    a = g * torch.sigmoid(g) * u
    s = (a.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    return (a / s).round().clamp(-127, 127).to(torch.int8), s


@_maybe_compile
def _rmsnorm_quantize(x2, weight, eps: float):
    """RMSNorm of ``[M, K]`` (the Llama / Qwen / Mistral form: ``weight * (x * rsqrt(mean(x^2) + eps)).to(x.dtype)``) and the per-token INT8
    quantisation of its output in one compiled function: returns the fp16 output (other consumers need it), the int8 values and the scales."""
    h = x2.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    y = weight * h.to(x2.dtype)
    yf = y.float()
    s = (yf.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    return y, (yf / s).round().clamp(-127, 127).to(torch.int8), s


class INT8W8A8Linear(nn.Module):
    """INT8 weights (per-output-channel scale) **and** activations (dynamic per-token scale) on the INT8 tensor cores:
    ``y = (quant(x) @ W8^T) * s_x * s_w + bias``. Two backends: ``"quantforge"`` (default; the GEMM and the dequantisation epilogue
    ``i32 -> f32 * s_x[m] * s_w[n] + bias[n] -> f16`` are ONE kernel generated by the QuantForge compiler, ``Int8GemmDq``; needs
    ``out_features % 128 == 0`` and ``in_features % 64 == 0``, rows are padded to a bucket) and ``"cublaslt"`` (``torch._int_mm``
    with the epilogue as separate torch ops, the layer shapes the compiled kernel does not take use it).

    On an RTX 4050 the INT8 ``mma`` runs at 2.0x the fp16 one (88 against 44 TOPS, ``benchmarks/mma_peak.cu``) and cuBLASLt reaches 80-83
    TOPS against 37-38 TFLOP/s for fp16 *when B is column-major* (``weight.t()`` of the row-major ``[out, in]`` weight is exactly that;
    a row-major B falls onto a path at 24 TOPS). ``_int_mm`` needs more than 16 rows, so calls with fewer rows use a dequantised fp16
    matmul (slow, correct): this is a prefill / large-batch format. Quality: plain per-token / per-channel INT8 without smoothing, see the blog."""

    _TILE_M, _TILE_N, _TILE_K = 128, 128, 64     # the compiled kernel's block tile (M is padded to a multiple, N and K must be)

    def __init__(self, w8_nk, w_scale, bias, backend: str = "quantforge"):
        super().__init__()
        self.in_features, self.out_features = w8_nk.shape[1], w8_nk.shape[0]
        self.register_buffer("weight", w8_nk)                      # int8 [out, in]
        self.register_buffer("w_scale", w_scale)                   # f16 [out]
        self.register_buffer("bias", bias)
        self.backend = backend
        self.use_compiled = (backend == "quantforge" and self.out_features % self._TILE_N == 0
                             and self.in_features % self._TILE_K == 0)
        if self.use_compiled:                                      # f32 copies for the fused epilogue
            self.register_buffer("_w_scale32", w_scale.float().contiguous(), persistent=False)
            self.register_buffer("_bias32", (bias if bias is not None else torch.zeros(self.out_features, device=w8_nk.device)).float().contiguous(),
                                 persistent=False)
        self._kernels: dict = {}

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, device="cuda", backend: str = "quantforge") -> "INT8W8A8Linear":
        w = lin.weight.data.to(device, torch.float32)
        scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        w8 = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
        bias = None if lin.bias is None else lin.bias.data.to(device, torch.float16)
        if lin.in_features % 8 or lin.out_features % 8:
            raise ValueError("int8_w8a8 needs in_features and out_features that are multiples of 8")
        return cls(w8, scale.half(), bias, backend)

    @staticmethod
    def _bucket(m: int) -> int:
        """Rows the kernel is compiled for: a multiple of 128 from a coarse ladder (a kernel per distinct M would compile for seconds each)."""
        for b in (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192):
            if m <= b:
                return b
        return -(-m // 4096) * 4096

    def _build_kernel(self, Mp: int, K: int):
        from .e2e import INT8_GEMM_OPTS, Int8GemmDq, int8_gemm_swizzle

        return Int8GemmDq.build(Mp, K, self.out_features, bias=True,
                                pipeline_opts=f"{INT8_GEMM_OPTS} block-swizzle={int8_gemm_swizzle(K, self.out_features)}")

    def _launch(self, k, xq, s_x, y) -> None:
        k.launch(xq, self.weight, s_x, self._w_scale32, self._bias32, y)

    @classmethod
    def concat(cls, layers) -> "INT8W8A8Linear":
        """One layer whose output channels are those of `layers` (they read the same input): q / k / v, gate / up."""
        w = torch.cat([m.weight for m in layers], 0).contiguous()
        sc = torch.cat([m.w_scale for m in layers])
        sizes = [m.out_features for m in layers]
        if any(m.bias is not None for m in layers):
            bias = torch.cat([m.bias if m.bias is not None else torch.zeros(n, dtype=torch.float16, device=w.device)
                              for m, n in zip(layers, sizes)])
        else:
            bias = None
        return cls(w, sc, bias, layers[0].backend)

    def _compiled_forward(self, x2: torch.Tensor, dt) -> torch.Tensor:
        xq, s_x = _int8_quantize_tokens(x2)
        return self._gemm_quantized(xq, s_x.reshape(-1), dt)

    def _gemm_quantized(self, xq: torch.Tensor, s_x: torch.Tensor, dt) -> torch.Tensor:
        """The compiled GEMM + dequantisation on activations that are already int8 ``[M, K]`` with their per-token scales ``[M]``."""
        (M, K), Mp = xq.shape, self._bucket(xq.shape[0])
        k = self._kernels.get(Mp)
        if k is None:
            k = self._kernels[Mp] = self._build_kernel(Mp, K)
        if Mp != M:                                                       # zero rows (scale 0) up to the bucket
            xq = torch.nn.functional.pad(xq, (0, 0, 0, Mp - M))
            s_x = torch.nn.functional.pad(s_x, (0, Mp - M))
        y = torch.empty(Mp, self.out_features, device=xq.device, dtype=torch.float16)
        self._launch(k, xq, s_x.contiguous(), y)
        y = y[:M] if Mp != M else y
        return y if dt == torch.float16 else y.to(dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        dt = torch.float16 if x.dtype == torch.float32 else x.dtype
        if self.use_compiled and x2.shape[0] > 16 and x2.is_cuda:
            return self._compiled_forward(x2, dt).reshape(*shape[:-1], self.out_features)
        if x2.shape[0] <= 16:                                      # _int_mm needs > 16 rows
            y = x2.to(torch.float16) @ self.dequantized_weight().t()
            y = y if self.bias is None else y + self.bias
            return y.to(dt).reshape(*shape[:-1], self.out_features)
        xq, s_x = _int8_quantize_tokens(x2)
        y = torch._int_mm(xq, self.weight.t())                     # column-major B: the fast cuBLASLt path
        y = _fp8_epilogue(y, s_x, self.w_scale, self.bias, dt)
        return y.reshape(*shape[:-1], self.out_features)

    def dequantized_weight(self) -> torch.Tensor:
        return (self.weight.float() * self.w_scale.float()[:, None]).half()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fmt=int8_w8a8"


class W4A8Linear(INT8W8A8Linear):
    """INT4 weights (progressive quantisation: INT8 per channel, then INT4 per group of 128 with an integer scale and zero point) and INT8
    activations (dynamic per token), on the INT8 tensor cores, in ONE compiled kernel (``W4A8GemmDq``): the weights are the only
    4-bit traffic and are expanded to int8 while the tile is staged, the accumulation is int32 over all of K and the only float
    scale left (per output channel) is applied in the store. Weight memory is half of INT8's (a quarter of fp16's). Prefill /
    large-batch format like ``INT8W8A8Linear`` (fewer than 17 rows use a dequantised fp16 matmul); needs ``out_features % 128 == 0``
    and ``in_features % 128 == 0``, other layers are left dense by ``quantize_model``."""

    _TILE_N, _TILE_K = 128, 128

    def __init__(self, packed, group_scale, group_zero, w_scale, bias):
        nn.Module.__init__(self)
        self.out_features, self.in_features = packed.shape[0], packed.shape[1] * 2
        self.register_buffer("weight", packed)                     # int8 [out, in/2]: two nibbles per byte (zero point 8)
        self.register_buffer("group_scale", group_scale)           # int8 [out, in/128]
        self.register_buffer("group_zero", group_zero)             # int8 [out, in/128]
        self.register_buffer("w_scale", w_scale)                   # f32 [out]
        self.register_buffer("bias", bias)
        self.backend = "quantforge"
        self.rotate = False        # True: the weight was quantised after a block-Hadamard rotation (w4a8_op.rotate_weight); the forward rotates its input the same way
        self.group = self.in_features // group_scale.shape[1]       # 128 (default) or 64 / 32 along K
        self.use_compiled = self.out_features % self._TILE_N == 0 and self.in_features % self._TILE_K == 0
        self.register_buffer("_w_scale32", w_scale.float().contiguous(), persistent=False)
        self.register_buffer("_bias32", (bias if bias is not None else torch.zeros(self.out_features, device=packed.device)).float().contiguous(),
                             persistent=False)
        self._kernels: dict = {}

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, device="cuda", backend: str = "quantforge", group: int = 128) -> "W4A8Linear":
        from .quant import w4a8_quantize

        if lin.in_features % 128 or lin.out_features % 128:
            raise ValueError("w4a8 needs in_features and out_features that are multiples of 128")
        if lin.in_features % group:
            raise ValueError(f"in_features {lin.in_features} is not a multiple of the group size {group}")
        packed, gs, zp, sw = w4a8_quantize(lin.weight.data.to(device, torch.float32), group)
        bias = None if lin.bias is None else lin.bias.data.to(device, torch.float16)
        return cls(packed, gs, zp, sw, bias)

    @classmethod
    def concat(cls, layers) -> "W4A8Linear":
        cat = lambda name: torch.cat([getattr(m, name) for m in layers], 0).contiguous()      # noqa: E731
        sizes = [m.out_features for m in layers]
        w = layers[0].weight
        if any(m.bias is not None for m in layers):
            bias = torch.cat([m.bias if m.bias is not None else torch.zeros(n, dtype=torch.float16, device=w.device)
                              for m, n in zip(layers, sizes)])
        else:
            bias = None
        out = cls(cat("weight"), cat("group_scale"), cat("group_zero"), cat("w_scale"), bias)
        out.rotate = layers[0].rotate
        return out

    def _build_kernel(self, Mp: int, K: int):
        from .e2e import W4A8GemmDq

        return W4A8GemmDq.build(Mp, K, self.out_features, bias=True, group=self.group)

    def _launch(self, k, xq, s_x, y) -> None:
        k.launch(xq, self.weight, self.group_scale, self.group_zero, s_x, self._w_scale32, self._bias32, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        dt = torch.float16 if x.dtype == torch.float32 else x.dtype
        if self.rotate:
            from .w4a8_op import _rotate_impl

            x2 = _rotate_impl(x2.to(torch.float16).contiguous())
        if self.use_compiled and x2.shape[0] > 16 and x2.is_cuda:
            return self._compiled_forward(x2, dt).reshape(*shape[:-1], self.out_features)
        y = x2.to(torch.float16) @ self.dequantized_weight().t()
        y = y if self.bias is None else y + self.bias
        return y.to(dt).reshape(*shape[:-1], self.out_features)

    def dequantized_weight(self) -> torch.Tensor:
        from .quant import w4a8_dequantize

        return w4a8_dequantize(self.weight, self.group_scale, self.group_zero, self.w_scale, self.group).half()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fmt=w4a8"


# --------------------------------------------------------------------------- fusion
class _QuantQKV(nn.Module):
    """q / k / v projections of one attention block as ONE GEMM. The attention calls ``q_proj(x)``,
    ``k_proj(x)``, ``v_proj(x)`` on the same tensor: the first call runs the fused GEMM and the other two
    take their slice (keyed on the identity *and* the in-place version of the input)."""

    def __init__(self, q: QuantLinear, k: QuantLinear, v: QuantLinear):
        super().__init__()
        self.group_size, self.fmt = q.group_size, q.fmt
        packed, scale, zero = quant.concat_columns(
            *((m.packed, m.scale, m.zero) for m in (q, k, v)))
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.sizes = [m.out_features for m in (q, k, v)]
        if any(m.bias is not None for m in (q, k, v)):
            bias = torch.cat([m.bias if m.bias is not None else
                              torch.zeros(n, dtype=torch.float16, device=q.packed.device)
                              for m, n in zip((q, k, v), self.sizes)])
        else:
            bias = None
        self.register_buffer("bias", bias)
        self._key, self._outs = None, None

    def parts(self, x: torch.Tensor):
        try:
            version = x._version
        except RuntimeError:       # inference-mode tensors have no version counter
            version = None
        key = (id(x), version)
        if self._key != key or self._outs is None:
            xh = x if x.dtype == torch.float16 else x.half()
            outs = _ops.w4a16_linear(xh, self.packed, self.scale, self.zero, self.group_size,
                                     self.bias).split(self.sizes, -1)
            self._outs = tuple(o if x.dtype == torch.float16 else o.to(x.dtype) for o in outs)
            self._key, self._x = key, x      # keep x alive so that its id cannot be reused
        return self._outs


class _QKVPart(nn.Module):
    def __init__(self, fused: _QuantQKV, index: int):
        super().__init__()
        self.fused, self.index = fused, index

    def forward(self, x):
        take = getattr(self.fused, "take", None)
        return take(x, self.index) if take is not None else self.fused.parts(x)[self.index]


class _QuantGatedMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``. INT4: gate / up interleaved into one GEMM with the ``swiglu``
    epilogue; INT8 / FP8: a gate GEMM with a fused silu epilogue and an up GEMM."""

    def __init__(self, gate: QuantLinear, up: QuantLinear, down: nn.Module):
        super().__init__()
        self.down_proj, self.group_size = down, gate.group_size
        self.fused = gate.fmt == "int4"
        if self.fused:
            packed, scale, zero = quant.interleave_gate_up(
                (gate.packed, gate.scale, gate.zero), (up.packed, up.scale, up.zero))
            self.register_buffer("packed", packed)
            self.register_buffer("scale", scale)
            self.register_buffer("zero", zero)
        else:
            self.gate_proj, self.up_proj = gate, up
        self.has_bias = gate.bias is not None or up.bias is not None
        if self.has_bias and self.fused:
            raise ValueError("a gated MLP with biases cannot use the fused swiglu kernel")

    def forward(self, x):
        xh = x if x.dtype == torch.float16 else x.half()
        if self.fused:
            h = _ops.w4a16_linear(xh, self.packed, self.scale, self.zero, self.group_size, None, "swiglu")
        else:
            g = _ops.w4a16_linear(xh, self.gate_proj.packed, self.gate_proj.scale, self.gate_proj.zero,
                                  self.group_size, self.gate_proj.bias, "silu")
            h = g * self.up_proj(xh)
        return self.down_proj(h if x.dtype == torch.float16 else h.to(x.dtype))


class _Int8Shared(nn.Module):
    """Several ``INT8W8A8Linear`` layers that read the SAME input (q / k / v; gate / up) as ONE layer: the activations are quantised
    once and one GEMM (one launch, a wider N) computes all the outputs; each original layer becomes a ``_QKVPart`` that takes its
    slice (the first call computes, the others reuse it: keyed on the identity and the in-place version of the input, as ``_QuantQKV``)."""

    def __init__(self, layers):
        super().__init__()
        sizes = [m.out_features for m in layers]
        self.inner = type(layers[0]).concat(layers)
        self.sizes = sizes
        self._key, self._outs, self._taken = None, None, 0
        self.pre_hits = 0           # how many calls used the quantisation done by the norm
        self._pre = None            # (x, int8 x, scales) left by a norm that already quantised its output (see _install_norm_quantisation)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        pre, self._pre = self._pre, None
        if pre is not None and pre[0] is x and x.dtype == torch.float16 and self.inner.use_compiled:
            self.pre_hits += 1
            return self.inner._gemm_quantized(pre[1], pre[2], torch.float16).reshape(*x.shape[:-1], -1)
        return self.inner(x)

    def parts(self, x: torch.Tensor):
        try:
            version = x._version
        except RuntimeError:       # inference-mode tensors have no version counter
            version = None
        key = (id(x), version)
        if self._key != key or self._outs is None:
            self._outs = tuple(self._run(x).split(self.sizes, -1))
            self._key, self._x, self._taken = key, x, 0      # keep x alive so that its id cannot be reused
        return self._outs

    def full(self, x: torch.Tensor) -> torch.Tensor:
        """The un-split output ``[..., sum(sizes)]`` (for a consumer that wants all the parts at once)."""
        return self._run(x)

    def take(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """Part ``index`` of the shared GEMM. The cache (the outputs of the last call, ~80 MB for a gate / up at 4096 rows) is dropped
        as soon as every part was taken: otherwise every layer would hold its last activations until the next forward."""
        out = self.parts(x)[index]
        self._taken += 1
        if self._taken >= len(self.sizes):
            self._key = self._outs = self._x = None
            self._taken = 0
        return out


def _fuse_int8(model: nn.Module, fuse_activation: bool = True, fuse_norm: bool = True) -> tuple[int, int]:
    """Share the activation quantisation and the GEMM between q/k/v and between gate/up of INT8 W8A8 blocks; returns (#qkv, #gate-up)."""
    n_qkv = n_mlp = 0
    for module in list(model.modules()):
        for names, counter in ((("q_proj", "k_proj", "v_proj"), "qkv"), (("gate_proj", "up_proj"), "mlp")):
            layers = [getattr(module, n, None) for n in names]
            if not all(isinstance(m, INT8W8A8Linear) for m in layers):
                continue
            if len({(m.in_features, m.backend, type(m)) for m in layers}) != 1:
                continue
            shared = _Int8Shared(layers)
            for i, n in enumerate(names):
                setattr(module, n, _QKVPart(shared, i))
            if counter == "qkv":
                n_qkv += 1
            else:
                n_mlp += 1
                down = getattr(module, "down_proj", None)
                if isinstance(down, type(layers[0])) and _is_silu(module) and all(m.bias is None for m in layers) \
                        and down.in_features == layers[0].out_features and fuse_activation:
                    module.forward = _int8_gated_mlp_forward(shared, down, layers[0].out_features, module.forward)
    if fuse_norm:
        for module in list(model.modules()):
            for norm_name, owner, proj in (("input_layernorm", "self_attn", "q_proj"), ("post_attention_layernorm", "mlp", "gate_proj")):
                norm, target = getattr(module, norm_name, None), getattr(getattr(module, owner, None), proj, None)
                if isinstance(target, _QKVPart) and isinstance(target.fused, _Int8Shared):
                    _install_norm_quantisation(norm, target.fused)
    return n_qkv, n_mlp


def _install_norm_quantisation(norm: nn.Module, shared: "_Int8Shared") -> bool:
    """Make the RMSNorm that feeds a shared INT8 / W4A8 layer quantise its own output: one compiled function computes the norm and the int8
    values + scales, the shared layer finds them (matched on the identity of the tensor the norm returned) and skips its own quantisation.
    Only the Llama / Qwen / Mistral form of RMSNorm; fp16, more than 16 rows (as the layer itself)."""
    name = type(norm).__name__.lower()
    if not name.endswith("rmsnorm") or "gemma" in name or not hasattr(norm, "variance_epsilon") or not hasattr(norm, "weight"):
        return False
    original = norm.forward

    def forward(x):
        rows = x.numel() // x.shape[-1]
        if x.dtype != torch.float16 or norm.weight.dtype != torch.float16 or rows <= 16 or not (x.is_cuda and shared.inner.use_compiled):
            return original(x)
        y, xq, s = _rmsnorm_quantize(x.reshape(rows, -1), norm.weight, float(norm.variance_epsilon))
        y = y.reshape(x.shape)
        shared._pre = (y, xq, s.reshape(-1))
        return y
    norm.forward = forward
    return True


def _int8_gated_mlp_forward(shared: "_Int8Shared", down, inter: int, fallback):
    """``down(silu(gate(x)) * up(x))`` of an INT8 / W4A8 block with the activation and the quantisation of ``down``'s input fused
    (``_silu_mul_quantize``); calls with 16 rows or fewer, or layers the compiled kernel does not take, use the generic path."""
    def forward(x):
        rows = x.numel() // x.shape[-1]
        if rows <= 16 or not (shared.inner.use_compiled and down.use_compiled and x.is_cuda):
            return fallback(x)
        dt = torch.float16 if x.dtype == torch.float32 else x.dtype
        gu = shared.full(x).reshape(rows, 2 * inter)
        xq, s_x = _silu_mul_quantize(gu, inter)
        y = down._gemm_quantized(xq, s_x.reshape(-1), dt)
        return y.reshape(*x.shape[:-1], down.out_features)
    return forward


def _is_silu(module: nn.Module) -> bool:
    act = getattr(module, "act_fn", None) or getattr(module, "activation_fn", None)
    return act is not None and "silu" in type(act).__name__.lower()


def _fuse(model: nn.Module) -> tuple[int, int]:
    """Fuse q/k/v and gated MLPs of already quantised blocks; returns (#qkv, #mlp)."""
    n_qkv = n_mlp = 0
    for module in list(model.modules()):
        names = ("q_proj", "k_proj", "v_proj")
        if all(isinstance(getattr(module, n, None), QuantLinear) for n in names):
            ql = [getattr(module, n) for n in names]
            if len({(m.fmt, m.group_size, m.in_features) for m in ql}) == 1 \
                    and all(m.perm is None for m in ql):
                fused = _QuantQKV(*ql)
                for i, n in enumerate(names):
                    setattr(module, n, _QKVPart(fused, i))
                n_qkv += 1
        # Phi-3 style: gate and up stacked in ONE Linear (outputs [gate | up]) -> split, then fuse as below
        gu = getattr(module, "gate_up_proj", None)
        if isinstance(gu, QuantLinear) and hasattr(module, "down_proj") and _is_silu(module) \
                and gu.out_features % 2 == 0 and gu.perm is None and gu.bias is None:
            half = gu.out_features // 2
            cols = (half // 2, half // 2) if gu.fmt == "int4" else (half, half)     # packed columns per half
            parts = [QuantLinear(gu.packed[:, a:a + c].contiguous(), gu.scale[:, h * half:(h + 1) * half].contiguous(),
                                 gu.zero[:, h * half:(h + 1) * half].contiguous(), None, gu.group_size, gu.fmt)
                     for h, (a, c) in enumerate(((0, cols[0]), (cols[0], cols[1])))]
            module.forward = _QuantGatedMLP(parts[0], parts[1], module.down_proj).forward
            module._qf_mlp = module.forward.__self__
            del module.gate_up_proj
            n_mlp += 1
            continue
        names = ("gate_proj", "up_proj", "down_proj")
        if all(hasattr(module, n) for n in names) and _is_silu(module) \
                and isinstance(module.gate_proj, QuantLinear) and isinstance(module.up_proj, QuantLinear):
            g, u = module.gate_proj, module.up_proj
            if (g.fmt, g.group_size, g.in_features, g.out_features) == \
                    (u.fmt, u.group_size, u.in_features, u.out_features) \
                    and not (g.fmt == "int4" and (g.bias is not None or u.bias is not None)) \
                    and g.perm is None and u.perm is None:
                module.forward = _QuantGatedMLP(g, u, module.down_proj).forward
                module._qf_mlp = module.forward.__self__          # keep the fused module's buffers alive
                del module.gate_proj, module.up_proj
                n_mlp += 1
    return n_qkv, n_mlp


@torch.no_grad()
def quantize_model(model: nn.Module, fmt: str = "int4", group_size: int = 128,
                   skip: Iterable[str] = ("lm_head",), only: Optional[Iterable[str]] = None,
                   fuse: bool = True, device="cuda", verbose: bool = False, w8a8_per_token: bool = True, kfold: bool = False,
                   int8_backend: str = "quantforge") -> dict:
    """Replace the ``nn.Linear`` layers of `model` (in place) with :class:`QuantLinear`.

    ``skip``: name substrings that stay dense (default: the ``lm_head``); ``only``: if given, quantise
    only layers whose name contains one of these substrings (e.g. ``only=("lm_head",)`` with
    ``fmt="int8"`` to quantise just the head). Layers whose ``in_features`` is not a multiple of any
    group size <= ``group_size`` (down to 16), or int4 layers with an odd ``out_features``, are left
    dense with a warning. The layers are quantised one at a time on `device` and the dense weights are
    freed immediately, so a model larger than the GPU in fp16 can be converted when it is loaded on the
    CPU (the non-Linear parameters are moved to `device` at the end). Tied weights (``lm_head`` shared
    with the embeddings) stay with the embedding when the head is quantised.

    Returns ``{"quantized": n, "skipped": [names], "bytes_before": b, "bytes_after": b, "fused_qkv": n,
    "fused_mlp": n}``."""
    if fmt not in FORMATS + (W8A8_FORMAT, INT8_W8A8_FORMAT, W4A8_FORMAT):
        raise ValueError(f"fmt must be one of {FORMATS + (W8A8_FORMAT, INT8_W8A8_FORMAT, W4A8_FORMAT)}")
    skip, only = tuple(skip), None if only is None else tuple(only)
    stats = {"quantized": 0, "skipped": [], "bytes_before": 0, "bytes_after": 0}
    targets = []
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{cname}" if name else cname
            if any(s in full for s in skip) and not (only and any(o in full for o in only)):
                continue
            if only is not None and not any(o in full for o in only):
                continue
            targets.append((mod, cname, child, full))
    for mod, cname, lin, full in targets:
        try:
            if fmt == W4A8_FORMAT:
                ql = W4A8Linear.from_linear(lin, device, group=group_size)
            elif fmt == INT8_W8A8_FORMAT:
                ql = INT8W8A8Linear.from_linear(lin, device, backend=int8_backend)
            elif fmt == W8A8_FORMAT:
                ql = FP8W8A8Linear.from_linear(lin, device, per_token=w8a8_per_token)
            else:
                ql = QuantLinear.from_linear(lin, fmt, group_size, device)
        except ValueError as e:
            warnings.warn(f"quantforge: {full} left dense ({e})")
            stats["skipped"].append(full)
            continue
        stats["bytes_before"] += lin.weight.numel() * lin.weight.element_size()
        stats["bytes_after"] += sum(b.numel() * b.element_size() for b in (
            (ql.weight, ql.w_scale) + ((ql.group_scale, ql.group_zero) if fmt == W4A8_FORMAT else ()) if fmt in (W8A8_FORMAT, INT8_W8A8_FORMAT, W4A8_FORMAT) else (ql.packed, ql.scale, ql.zero)))
        setattr(mod, cname, ql)
        stats["quantized"] += 1
        if verbose:
            print(f"  {full}: {tuple(lin.weight.shape)} -> {getattr(ql, 'fmt', fmt)}")
        del lin
        if stats["quantized"] % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    stats["fused_qkv"], stats["fused_mlp"] = (_fuse_int8(model) if fmt in (INT8_W8A8_FORMAT, W4A8_FORMAT) else _fuse(model)) if fuse else (0, 0)
    model.to(device)       # embeddings / norms / any dense layer left; the quantised buffers are there
    if kfold:
        stats["kfold"] = enable_kfold(model)
    gc.collect()
    torch.cuda.empty_cache()
    return stats


@torch.no_grad()
def load_quantized_checkpoint(model: nn.Module, tensors, method: str = "gptq", fuse: bool = True,
                              device="cuda", zeros_offset: int = 1, verbose: bool = False, kfold: bool = False) -> dict:
    """Turn the Linear layers of `model` into :class:`QuantLinear` from an AWQ / GPTQ checkpoint.

    `model` is any (dense) instance of the architecture, e.g. ``AutoModelForCausalLM.from_pretrained(base)``;
    `tensors` is the checkpoint (a ``.safetensors`` path or a ``{name: tensor}`` dict) with
    ``<layer>.qweight / qzeros / scales`` (and ``g_idx`` for GPTQ, ``bias`` if the layer has one).
    ``method``: ``"gptq"`` (4-bit, ``desc_act`` act-order supported through a row permutation), ``"awq"``
    (GEMM version, 4-bit) or ``"fp8"`` (an FP8 E4M3 checkpoint with per-output-channel ``<layer>.weight_scale``, e.g.
    the ``compressed-tensors`` "FP8-dynamic" models: only the *weights* are used, the activations stay fp16). ``zeros_offset``: 1 for classic GPTQ ``checkpoint_format: gptq`` files, 0 for
    ``gptq_v2``. Non-Linear tensors that exist in both the checkpoint and the model (embeddings, norms)
    are copied; unquantised Linear layers keep their weights. Returns ``{"quantized": n, "act_order": n,
    "fused_qkv": n, "fused_mlp": n}``."""
    if method not in ("gptq", "awq", "fp8"):
        raise ValueError("method must be 'gptq', 'awq' or 'fp8'")
    if isinstance(tensors, (str, bytes)) or hasattr(tensors, "__fspath__"):
        from safetensors import safe_open
        f = safe_open(str(tensors), framework="pt", device="cpu")
        get, keys = f.get_tensor, set(f.keys())
    else:
        get, keys = tensors.__getitem__, set(tensors.keys())
    stats = {"quantized": 0, "act_order": 0}
    quantised_names = set()
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{cname}" if name else cname
            if method == "fp8":
                if f"{full}.weight_scale" not in keys or f"{full}.weight" not in keys:
                    continue
                w8 = get(f"{full}.weight")
                if w8.dtype != torch.float8_e4m3fn:
                    continue
                w8 = w8.to(device).t().contiguous()                              # [K, N]
                K_, N_ = w8.shape
                G_ = 128 if K_ % 128 == 0 else math.gcd(K_, 128)
                scale = get(f"{full}.weight_scale").to(device, torch.float16).reshape(1, N_)   # per output channel
                scale = scale.expand(K_ // G_, N_).contiguous()
                zero = torch.zeros_like(scale, dtype=torch.int8)
                bias = get(f"{full}.bias").to(device, torch.float16) if f"{full}.bias" in keys else (
                    None if child.bias is None else child.bias.data.to(device, torch.float16))
                setattr(mod, cname, QuantLinear.from_quantized(w8, scale, zero, bias, G_, "fp8"))
                quantised_names.add(full)
                stats["quantized"] += 1
                continue
            if f"{full}.qweight" not in keys:
                continue
            qw, qz, sc = (get(f"{full}.{k}").to(device) for k in ("qweight", "qzeros", "scales"))
            perm = None
            if method == "awq":
                packed, scale, zero = quant.from_awq(qw, qz, sc)
            else:
                gi = get(f"{full}.g_idx").to(device) if f"{full}.g_idx" in keys else None
                K = qw.shape[0] * 8
                G = K // sc.shape[0]
                if gi is not None and not torch.equal(gi.long(), torch.arange(K, device=device) // G):
                    packed, scale, zero, perm = quant.from_gptq_act_order(qw, qz, sc, gi, zeros_offset)
                    stats["act_order"] += 1
                else:
                    packed, scale, zero = quant.from_gptq(qw, qz, sc, None, zeros_offset)
            bias = get(f"{full}.bias").to(device, torch.float16) if f"{full}.bias" in keys else (
                None if child.bias is None else child.bias.data.to(device, torch.float16))
            setattr(mod, cname, QuantLinear.from_quantized(
                packed.contiguous(), scale.contiguous(), zero.contiguous(), bias, None, "int4", perm))
            quantised_names.add(full)
            stats["quantized"] += 1
            if verbose:
                print(f"  {full}: {method} int4 g{packed.shape[0] // scale.shape[0]}")
            del qw, qz, sc
    if not stats["quantized"]:
        raise ValueError("no quantised tensors (<layer>.qweight, or <layer>.weight + weight_scale for fp8) in the "
                         "checkpoint match a Linear layer of the model")
    own = model.state_dict()
    for k in keys:
        if k in own and not any(k.startswith(q + ".") for q in quantised_names) \
                and own[k].shape == get(k).shape:
            own[k].copy_(get(k))
    stats["fused_qkv"], stats["fused_mlp"] = _fuse(model) if fuse else (0, 0)
    model.to(device)
    if kfold:
        stats["kfold"] = enable_kfold(model)
    gc.collect()
    torch.cuda.empty_cache()
    return stats


