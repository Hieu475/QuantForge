"""vLLM plugin: run the QuantForge W4A16 kernels inside vLLM (``quantization="quantforge"``).

    # install the package (it registers itself through the `vllm.general_plugins` entry point), then:
    from vllm import LLM
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", quantization="quantforge", dtype="float16")

It accepts the checkpoints of vLLM's own ``gptq`` / ``gptq_marlin`` (4 and 8 bit, symmetric or asymmetric, group size 32 / 64 / 128,
``desc_act`` act-order included) and ``awq`` methods. The checkpoint tensors are loaded exactly as vLLM loads them (so
tensor parallelism and fused q/k/v and gate/up layers work unchanged) and converted once, after loading, into the
QuantForge layout; every linear layer then calls ``torch.ops.quantforge.w4a16_gemm``, which works under vLLM's
``torch.compile`` and CUDA-graph capture. The kernels are JIT-compiled per (layer shape, row bucket) on first use and cached
on disk, so the first start of a model takes a while (about 0.3 s per kernel) and later starts do not.

K-folding (``kfold.py``) keeps a second copy of the narrow-output layers: ``QUANTFORGE_KFOLD=0`` disables it,
``QUANTFORGE_KFOLD_BUDGET_MB`` (default 256) caps the extra memory.

Formats: GPTQ 4-bit and 8-bit (symmetric or not, act-order included) and AWQ 4-bit with zero points. Limitations: no MoE layers, no
AWQ with a symmetric / absent zero point, f16 / bf16 activations (bf16 is converted to f16 around the op).
"""
from __future__ import annotations

import os

NAME = "quantforge"


import torch

try:
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.auto_awq import BaseAWQLinearMethod
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig, AutoGPTQLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.layers.quantization.utils.gptq_utils import get_linear_quant_method
    from vllm.scalar_type import scalar_types
    HAVE_VLLM = True
except ImportError:                      # vLLM is not installed: the module is importable but registers nothing
    HAVE_VLLM = False


def register() -> None:
    """Entry point (`vllm.general_plugins`). The classes below are defined at import time, at module level, because vLLM
    pickles the quantization config to its engine process; importing this module is what registers the method."""
    if not HAVE_VLLM:
        raise RuntimeError("vLLM is not installed")


_FOLD_BYTES = [0]            # bytes of folded weight copies made so far in this process (see QUANTFORGE_KFOLD_BUDGET_MB)

if HAVE_VLLM:

    def _install(layer: torch.nn.Module, packed, scale, zero, perm, group: int, drop: tuple) -> None:
        """Replace the checkpoint tensors of `layer` by the QuantForge layout (shared by the GPTQ and AWQ methods)."""
        from . import kfold

        for name in drop:
            if hasattr(layer, name):
                delattr(layer, name)
        param = lambda t: torch.nn.Parameter(t.contiguous(), requires_grad=False)      # noqa: E731
        layer.qf_packed, layer.qf_scale, layer.qf_zero = param(packed), param(scale.half()), param(zero)
        layer.qf_perm = param(perm) if perm is not None else None
        layer.qf_group = group
        # narrow-N decode layers get a second, folded copy of the weights (kfold.py); the others a placeholder, so that
        # every layer calls the same op and the graph vLLM compiles (and caches) does not depend on this choice
        S = 1
        if os.environ.get("QUANTFORGE_KFOLD", "1") != "0":
            S = kfold.plan_ksplit(packed.shape[0], scale.shape[1], group, 4 if packed.shape[1] * 2 == scale.shape[1] else 8)
            extra = packed.numel() * packed.element_size() + scale.numel() * 2 + zero.numel()
            budget = float(os.environ.get("QUANTFORGE_KFOLD_BUDGET_MB", "256")) * 2 ** 20
            if S > 1 and _FOLD_BYTES[0] + extra > budget:        # the folded copy is extra memory: stop when the budget is spent
                S = 1
            elif S > 1:
                _FOLD_BYTES[0] += extra
        folded = kfold.fold_k(packed, scale.half(), zero, S) if S > 1 else kfold.no_fold(packed.device)
        layer.qf_packed_f, layer.qf_scale_f, layer.qf_zero_f = (param(t) for t in folded)

    def _apply(layer: torch.nn.Module, x: torch.Tensor, bias):
        from . import kfold

        xh = x if x.dtype == torch.float16 else x.half()
        y = kfold.linear(xh, layer.qf_packed, layer.qf_scale, layer.qf_zero,
                         (layer.qf_packed_f, layer.qf_scale_f, layer.qf_zero_f), layer.qf_group,
                         None if bias is None else bias.half(), layer.qf_perm)
        return y if x.dtype == torch.float16 else y.to(x.dtype)

    class QuantForgeAWQLinearMethod(BaseAWQLinearMethod):
        """AWQ-format weights (4-bit, with zero points), QuantForge kernels."""

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from . import quant

            packed, scale, zero = quant.from_awq(layer.qweight.data, layer.qzeros.data, layer.scales.data)
            _install(layer, packed, scale, zero, None, packed.shape[0] // layer.scales.shape[0], ("qweight", "qzeros", "scales"))

        def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            return _apply(layer, x, bias)

    class QuantForgeLinearMethod(AutoGPTQLinearMethod):
        """GPTQ-format weights, QuantForge kernels."""

        def __init__(self, quant_config: "QuantForgeConfig") -> None:     # no Marlin check: our kernels run on sm_80+
            self.quant_config = quant_config
            self.input_dtype = None
            self.quant_type = quant_config.quant_type

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from . import quant

            cfg = self.quant_config
            qweight, qzeros, scales = layer.qweight.data, layer.qzeros.data, layer.scales.data
            g_idx = layer.g_idx.data if cfg.desc_act else None
            zeros_offset = 0 if str(cfg.full_config.get("checkpoint_format", "")).lower() == "gptq_v2" else 1
            perm = None
            if g_idx is not None and g_idx.numel() and not torch.equal(
                    g_idx.long(), torch.arange(g_idx.numel(), device=g_idx.device) // (g_idx.numel() // scales.shape[0])):
                from_ao = quant.from_gptq8_act_order if cfg.weight_bits == 8 else quant.from_gptq_act_order
                packed, scale, zero, perm = from_ao(qweight, qzeros, scales, g_idx, zeros_offset)
            elif cfg.weight_bits == 8:
                packed, scale, zero = quant.from_gptq8(qweight, qzeros, scales, None, zeros_offset)
            else:
                packed, scale, zero = quant.from_gptq(qweight, qzeros, scales, None, zeros_offset)
            _install(layer, packed, scale, zero, perm, qweight.shape[0] * (32 // cfg.weight_bits) // scales.shape[0],
                     ("qweight", "qzeros", "scales", "g_idx"))

        def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            return _apply(layer, x, bias)

    @register_quantization_config(NAME)
    class QuantForgeConfig(AutoGPTQConfig):
        """vLLM's GPTQ config with the QuantForge linear methods; also reads AWQ (4-bit, zero-point) checkpoints."""

        # (bits, symmetric): asymmetric checkpoints are fine for QuantForge too (it keeps the zero points)
        TYPE_MAP = {(4, True): scalar_types.uint4b8, (4, False): scalar_types.uint4,
                    (8, True): scalar_types.uint8b128, (8, False): scalar_types.uint8}
        awq = False                      # True for an AWQ checkpoint
        pack_factor = 8
        modules_to_not_convert: list = []

        @classmethod
        def get_name(cls):
            return NAME

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @classmethod
        def from_config(cls, config):
            if str(config.get("quant_method", "")).lower() != "awq":
                return super().from_config(config)
            bits = cls.get_from_keys(config, ["w_bit", "bits"])
            group = cls.get_from_keys(config, ["q_group_size", "group_size"])
            if bits != 4 or not cls.get_from_keys(config, ["zero_point"]) or group <= 0:
                raise NotImplementedError("quantforge reads 4-bit AWQ checkpoints with zero points and a group size")
            self = cls(bits, group, False, False, cls.get_from_keys_or(config, ["lm_head"], default=False), {}, config, None)
            self.awq, self.pack_factor = True, 32 // bits
            self.modules_to_not_convert = cls.get_from_keys_or(config, ["modules_to_not_convert"], None) or []
            return self

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            if user_quant == NAME and str(hf_quant_cfg.get("quant_method", "")).lower() in ("gptq", "awq"):
                return NAME
            return None

        def get_quant_method(self, layer, prefix):
            if self.awq:
                if isinstance(layer, LinearBase) or (isinstance(layer, ParallelLMHead) and self.lm_head_quantized):
                    if is_layer_skipped(prefix, self.modules_to_not_convert, self.packed_modules_mapping, skip_with_substr=True):
                        return UnquantizedLinearMethod()
                    return QuantForgeAWQLinearMethod(self)
                return None
            # linear layers only (get_linear_quant_method returns None for anything else, e.g. MoE experts)
            return get_linear_quant_method(self, layer, prefix, QuantForgeLinearMethod)
