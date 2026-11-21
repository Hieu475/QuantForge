"""vLLM plugin: run the QuantForge W4A16 kernels inside vLLM (``quantization="quantforge"``).

    # install the package (it registers itself through the `vllm.general_plugins` entry point), then:
    from vllm import LLM
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", quantization="quantforge", dtype="float16")

It accepts the same 4-bit GPTQ checkpoints as vLLM's own ``gptq`` / ``gptq_marlin`` methods (symmetric or asymmetric, group
size 32 / 64 / 128, ``desc_act`` act-order included). The checkpoint tensors are loaded exactly as vLLM loads them (so
tensor parallelism and fused q/k/v and gate/up layers work unchanged) and converted once, after loading, into the
QuantForge layout; every linear layer then calls ``torch.ops.quantforge.w4a16_gemm``, which works under vLLM's
``torch.compile`` and CUDA-graph capture. The kernels are JIT-compiled per (layer shape, row bucket) on first use and cached
on disk, so the first start of a model takes a while (about 0.3 s per kernel) and later starts do not.

Limitations: 4-bit only (no 8-bit GPTQ, no AWQ yet), no MoE layers, f16 / bf16 activations (bf16 is converted to f16 around the op).
"""
from __future__ import annotations

import os

NAME = "quantforge"


import torch

try:
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig, AutoGPTQLinearMethod
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


if HAVE_VLLM:

    class QuantForgeLinearMethod(AutoGPTQLinearMethod):
        """GPTQ-format weights, QuantForge kernels."""

        def __init__(self, quant_config: "QuantForgeConfig") -> None:     # no Marlin check: our kernels run on sm_80+
            self.quant_config = quant_config
            self.input_dtype = None
            self.quant_type = quant_config.quant_type

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from . import quant
            from . import torch_ops  # noqa: F401  (registers the op)

            cfg = self.quant_config
            qweight, qzeros, scales = layer.qweight.data, layer.qzeros.data, layer.scales.data
            g_idx = layer.g_idx.data if cfg.desc_act else None
            zeros_offset = 0 if str(cfg.full_config.get("checkpoint_format", "")).lower() == "gptq_v2" else 1
            perm = None
            if g_idx is not None and g_idx.numel() and not torch.equal(
                    g_idx.long(), torch.arange(g_idx.numel(), device=g_idx.device) // (g_idx.numel() // scales.shape[0])):
                packed, scale, zero, perm = quant.from_gptq_act_order(qweight, qzeros, scales, g_idx, zeros_offset)
            else:
                packed, scale, zero = quant.from_gptq(qweight, qzeros, scales, None, zeros_offset)
            group = qweight.shape[0] * 8 // scales.shape[0]
            for name in ("qweight", "qzeros", "scales", "g_idx"):
                if hasattr(layer, name):
                    delattr(layer, name)
            param = lambda t: torch.nn.Parameter(t.contiguous(), requires_grad=False)      # noqa: E731
            layer.qf_packed, layer.qf_scale, layer.qf_zero = param(packed), param(scale.half()), param(zero)
            layer.qf_perm = param(perm) if perm is not None else None
            layer.qf_group = group
            # narrow-N decode layers get a second, folded copy of the weights (kfold.py); the others a placeholder, so that
            # every layer calls the same op and the graph vLLM compiles (and caches) does not depend on this choice
            from . import kfold

            S = 1
            if os.environ.get("QUANTFORGE_KFOLD", "1") != "0":
                S = kfold.plan_ksplit(packed.shape[0], scale.shape[1], group)
            folded = kfold.fold_k(packed, scale.half(), zero, S) if S > 1 else kfold.no_fold(packed.device)
            layer.qf_packed_f, layer.qf_scale_f, layer.qf_zero_f = (param(t) for t in folded)

        def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            import quantforge.torch_ops as ops

            xh = x if x.dtype == torch.float16 else x.half()
            from . import kfold

            y = kfold.linear(xh, layer.qf_packed, layer.qf_scale, layer.qf_zero,
                             (layer.qf_packed_f, layer.qf_scale_f, layer.qf_zero_f), layer.qf_group,
                             None if bias is None else bias.half(), layer.qf_perm)
            return y if x.dtype == torch.float16 else y.to(x.dtype)

    @register_quantization_config(NAME)
    class QuantForgeConfig(AutoGPTQConfig):
        """vLLM's GPTQ config with the QuantForge linear method."""

        # (bits, symmetric): the 4-bit asymmetric checkpoints are fine for QuantForge too (it keeps the zero points)
        TYPE_MAP = {(4, True): scalar_types.uint4b8, (4, False): scalar_types.uint4, (8, True): scalar_types.uint8b128}

        @classmethod
        def get_name(cls):
            return NAME

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            if user_quant == NAME and str(hf_quant_cfg.get("quant_method", "")).lower() == "gptq":
                return NAME
            return None

        def get_quant_method(self, layer, prefix):
            # linear layers only (get_linear_quant_method returns None for anything else, e.g. MoE experts)
            return get_linear_quant_method(self, layer, prefix, QuantForgeLinearMethod)
