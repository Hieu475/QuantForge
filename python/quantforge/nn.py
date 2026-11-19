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
import warnings
from typing import Iterable, Optional

import torch
import torch.nn as nn

from . import quant
from . import torch_ops as _ops

__all__ = ["QuantLinear", "quantize_model", "FORMATS"]

FORMATS = ("int4", "int8", "fp8")


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

    def __init__(self, packed, scale, zero, bias, group_size: int, fmt: str):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}")
        self.fmt, self.group_size = fmt, group_size
        self.in_features, self.out_features = packed.shape[0], scale.shape[1]
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        self.register_buffer("bias", bias)

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
        w = lin.weight.data.to(device, torch.float32).t().contiguous()       # [K, N]
        if fmt == "fp8":
            packed, scale, zero = quant.quantize_fp8(w, g)
        elif fmt == "int8":
            packed, scale, zero = quant.quantize_groupwise_int8(w, g)
        elif fmt == "int4":
            q, scale, zero = quant.quantize_groupwise(w, g)
            packed = quant.pack_int4(q)
        else:
            raise ValueError(f"fmt must be one of {FORMATS}")
        bias = None if lin.bias is None else lin.bias.data.to(device, torch.float16)
        return cls(packed.contiguous(), scale.contiguous(), zero.contiguous(), bias, g, fmt)

    @classmethod
    def from_quantized(cls, packed, scale, zero, bias=None, group_size: int | None = None,
                       fmt: str = "int4") -> "QuantLinear":
        """Wrap tensors already in QuantForge layout (e.g. from ``quant.from_gptq`` / ``from_awq``)."""
        g = group_size or packed.shape[0] // scale.shape[0]
        return cls(packed, scale, zero, bias, g, fmt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        y = _ops.w4a16_linear(x if dt == torch.float16 else x.half(), self.packed, self.scale,
                              self.zero, self.group_size, self.bias)
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
        return w.half().t().contiguous()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, fmt={self.fmt}, "
                f"group_size={self.group_size}, bias={self.bias is not None}")


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
    act = getattr(module, "act_fn", None)
    return act is not None and "silu" in type(act).__name__.lower()


def _fuse(model: nn.Module) -> tuple[int, int]:
    """Fuse q/k/v and gated MLPs of already quantised blocks; returns (#qkv, #mlp)."""
    n_qkv = n_mlp = 0
    for module in list(model.modules()):
        names = ("q_proj", "k_proj", "v_proj")
        if all(isinstance(getattr(module, n, None), QuantLinear) for n in names):
            ql = [getattr(module, n) for n in names]
            if len({(m.fmt, m.group_size, m.in_features) for m in ql}) == 1:
                fused = _QuantQKV(*ql)
                for i, n in enumerate(names):
                    setattr(module, n, _QKVPart(fused, i))
                n_qkv += 1
        names = ("gate_proj", "up_proj", "down_proj")
        if all(hasattr(module, n) for n in names) and _is_silu(module) \
                and isinstance(module.gate_proj, QuantLinear) and isinstance(module.up_proj, QuantLinear):
            g, u = module.gate_proj, module.up_proj
            if (g.fmt, g.group_size, g.in_features, g.out_features) == \
                    (u.fmt, u.group_size, u.in_features, u.out_features) \
                    and not (g.fmt == "int4" and (g.bias is not None or u.bias is not None)):
                module.forward = _QuantGatedMLP(g, u, module.down_proj).forward
                module._qf_mlp = module.forward.__self__          # keep the fused module's buffers alive
                del module.gate_proj, module.up_proj
                n_mlp += 1
    return n_qkv, n_mlp


@torch.no_grad()
def quantize_model(model: nn.Module, fmt: str = "int4", group_size: int = 128,
                   skip: Iterable[str] = ("lm_head",), only: Optional[Iterable[str]] = None,
                   fuse: bool = True, device="cuda", verbose: bool = False) -> dict:
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
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
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
            ql = QuantLinear.from_linear(lin, fmt, group_size, device)
        except ValueError as e:
            warnings.warn(f"quantforge: {full} left dense ({e})")
            stats["skipped"].append(full)
            continue
        stats["bytes_before"] += lin.weight.numel() * lin.weight.element_size()
        stats["bytes_after"] += sum(b.numel() * b.element_size() for b in (ql.packed, ql.scale, ql.zero))
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
    gc.collect()
    torch.cuda.empty_cache()
    return stats
