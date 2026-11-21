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

__all__ = ["QuantLinear", "FP8W8A8Linear", "quantize_model", "load_quantized_checkpoint", "enable_kfold", "FORMATS"]

FORMATS = ("int4", "int8", "fp8")
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
        if S > 1 and kind in (4, 8):
            pf, sf, zf = kfold.fold_k(self.packed, self.scale, self.zero, S)
            self.register_buffer("packed_f", pf, persistent=False)
            self.register_buffer("scale_f", sf, persistent=False)
            self.register_buffer("zero_f", zf, persistent=False)
            self.ksplit = S
        return self.ksplit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xh = x if dt == torch.float16 else x.half()
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
        return self.fused.parts(x)[self.index]


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
                   fuse: bool = True, device="cuda", verbose: bool = False, w8a8_per_token: bool = True, kfold: bool = False) -> dict:
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
    if fmt not in FORMATS + (W8A8_FORMAT,):
        raise ValueError(f"fmt must be one of {FORMATS + (W8A8_FORMAT,)}")
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
            if fmt == W8A8_FORMAT:
                ql = FP8W8A8Linear.from_linear(lin, device, per_token=w8a8_per_token)
            else:
                ql = QuantLinear.from_linear(lin, fmt, group_size, device)
        except ValueError as e:
            warnings.warn(f"quantforge: {full} left dense ({e})")
            stats["skipped"].append(full)
            continue
        stats["bytes_before"] += lin.weight.numel() * lin.weight.element_size()
        stats["bytes_after"] += sum(b.numel() * b.element_size() for b in (
            (ql.weight, ql.w_scale) if fmt == W8A8_FORMAT else (ql.packed, ql.scale, ql.zero)))
        setattr(mod, cname, ql)
        stats["quantized"] += 1
        if verbose:
            print(f"  {full}: {tuple(lin.weight.shape)} -> {ql.fmt} g{ql.group_size}")
        del lin
        if stats["quantized"] % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    stats["fused_qkv"], stats["fused_mlp"] = _fuse(model) if fuse else (0, 0)
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


