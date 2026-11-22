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

``quantization="quantforge_w4a8"`` quantises an ordinary fp16 / bf16 checkpoint to W4A8 while it loads, or loads a W4A8 checkpoint written by
``w4a8_checkpoint.save_w4a8`` (GPTQ quality, no fp16 round trip) (``QuantForgeW4A8Config``, ``w4a8_op.py``; tested on vLLM 0.25.1 and 0.34.0).

K-folding (``kfold.py``) keeps a second copy of the narrow-output layers: ``QUANTFORGE_KFOLD=0`` disables it,
``QUANTFORGE_KFOLD_BUDGET_MB`` (default 256) caps the extra memory.

Needs vLLM >= 0.24 (the generation tests pass on 0.24.0, 0.25.1 and 0.34.0; the releases in between were not run). Formats: GPTQ 4-bit and 8-bit (symmetric or not, act-order included) and AWQ 4-bit with zero points. Limitations: no MoE layers, no
AWQ with a symmetric / absent zero point, f16 / bf16 activations (bf16 is converted to f16 around the op).
"""
from __future__ import annotations

import os

NAME = "quantforge"


import torch

try:
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.auto_awq import BaseAWQLinearMethod
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig, AutoGPTQLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.layers.quantization.utils.gptq_utils import get_linear_quant_method
    from vllm.scalar_type import scalar_types
    HAVE_VLLM = True
except ImportError as _e:                # vLLM missing, or older than 0.24 (before auto_gptq / auto_awq): registers nothing
    HAVE_VLLM = False
    _IMPORT_ERROR = _e

MIN_VLLM = "0.24"                        # first release with AutoGPTQConfig / BaseAWQLinearMethod; run on 0.24.0, 0.25.1, 0.34.0


def register() -> None:
    """Entry point (`vllm.general_plugins`), called by vLLM on every start, whatever model it loads: it must never raise. The
    classes below are defined at import time, at module level, because vLLM pickles the quantization config to its engine
    process; importing this module is what registers the method."""
    if not HAVE_VLLM:
        import warnings

        try:
            from importlib.metadata import version
            found = f"vLLM {version('vllm')}"
        except Exception:                # noqa: BLE001
            found = "no vLLM"
        warnings.warn(f"quantforge: the 'quantforge' quantization method is not available ({found}; needs vLLM >= {MIN_VLLM}: "
                      f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR})", stacklevel=2)


_FOLD_BYTES = [0]            # bytes of folded weight copies made so far in this process (see QUANTFORGE_KFOLD_BUDGET_MB)

if HAVE_VLLM:

    def _layer_skipped(prefix: str, ignored: list, mapping) -> bool:
        """``is_layer_skipped`` with substring matching: the keyword changed between vLLM releases (``skip_with_substr=True``
        up to 0.2x, ``match_mode="substring"`` in 0.34)."""
        import inspect

        kw = {"match_mode": "substring"} if "match_mode" in inspect.signature(is_layer_skipped).parameters \
            else {"skip_with_substr": True}
        return is_layer_skipped(prefix, ignored, mapping, **kw)

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
        if packed.shape[1] * 2 == scale.shape[1] and group > 0:      # INT4: decide the atomic split-K plans of this shape now
            from . import atomic_plan

            atomic_plan.preplan(packed.shape[0], scale.shape[1], group)
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

        # attributes that vLLM's own __init__ sets and its create_weights reads (it differs between releases: 0.34 reads
        # use_batch_invariant, 0.24 reads input_dtype); defaults as class attributes so that a new one cannot break us
        use_batch_invariant = False
        input_dtype = None
        kernel = None

        def __init__(self, quant_config: "QuantForgeConfig") -> None:     # no Marlin check: our kernels run on sm_80+
            self.quant_config = quant_config
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
                    if _layer_skipped(prefix, self.modules_to_not_convert, self.packed_modules_mapping):
                        return UnquantizedLinearMethod()
                    return QuantForgeAWQLinearMethod(self)
                return None
            # linear layers only (get_linear_quant_method returns None for anything else, e.g. MoE experts)
            return get_linear_quant_method(self, layer, prefix, QuantForgeLinearMethod)


    class QuantForgeW4A8Method(UnquantizedLinearMethod):
        """Online W4A8 quantisation of an ordinary fp16 / bf16 checkpoint (``quantization="quantforge_w4a8"``): after loading, every linear layer
        whose sizes are multiples of 128 is quantised to 4-bit weights with integer group scales (``quant.w4a8_quantize``, round-to-nearest) and
        calls ``torch.ops.quantforge.w4a8_hybrid``: the compiled W4A8 kernel (INT8 tensor cores) above 16 rows, the W4A16 decode kernels below,
        both from the same weight. Memory: about one byte per weight (the two layouts), a quarter of fp16 would need dropping the decode copy."""

        def __init__(self, checkpoint: bool = False) -> None:
            super().__init__()
            self.checkpoint = checkpoint          # True: the checkpoint holds W4A8 tensors (w4a8_checkpoint.save_w4a8); False: online quantisation

        def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype,
                           **extra_weight_attrs):
            if self.checkpoint:
                from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter
                from vllm.model_executor.utils import set_weight_attrs

                N, K = sum(output_partition_sizes), input_size_per_partition
                if N % 128 or K % 128:
                    raise ValueError(f"a W4A8 checkpoint layer needs sizes that are multiples of 128 (got {N} x {K})")
                loader = extra_weight_attrs.pop("weight_loader")
                mk = lambda cols, dt: ModelWeightParameter(data=torch.empty(N, cols, dtype=dt), input_dim=1, output_dim=0,   # noqa: E731
                                                           weight_loader=loader)
                # the names are those of ``nn.W4A8Linear`` buffers, which is what ``save_w4a8`` writes
                layer.register_parameter("weight", mk(K // 2, torch.int8))
                layer.register_parameter("group_scale", mk(K // 128, torch.int8))
                layer.register_parameter("group_zero", mk(K // 128, torch.int8))
                layer.register_parameter("w_scale", ChannelQuantScaleParameter(data=torch.empty(N, dtype=torch.float32), output_dim=0,
                                                                              weight_loader=loader))
                for name in ("weight", "group_scale", "group_zero", "w_scale"):
                    set_weight_attrs(getattr(layer, name), extra_weight_attrs)
                layer.qf_w4a8 = True
                return
            super().create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype,
                                   **extra_weight_attrs)
            # the fp16 weight of a layer that will be quantised is held on the CPU while the checkpoint is read: vLLM moves each module to the
            # GPU only while it processes it (``device_loading_context``), so the GPU never holds the whole fp16 model (QUANTFORGE_W4A8_LOAD_ON_CPU=0
            # keeps the old behaviour). Layers that stay dense (sizes not multiples of 128) must remain on the GPU.
            if (os.environ.get("QUANTFORGE_W4A8_LOAD_ON_CPU", "1") != "0" and input_size_per_partition % 128 == 0
                    and sum(output_partition_sizes) % 128 == 0 and params_dtype in (torch.float16, torch.bfloat16)):
                layer.weight.data = layer.weight.data.cpu()

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from . import kfold, quant, w4a8_op

            if self.checkpoint:
                packed, gs, zp, sw = (layer.weight.data, layer.group_scale.data, layer.group_zero.data, layer.w_scale.data)
                for name in ("weight", "group_scale", "group_zero", "w_scale"):
                    delattr(layer, name)
                w = packed
                N, K = packed.shape[0], packed.shape[1] * 2
            else:
                w = layer.weight.data
                N, K = w.shape
                layer.qf_w4a8 = bool(N % 128 == 0 and K % 128 == 0 and w.is_cuda and w.dtype in (torch.float16, torch.bfloat16))
                if not layer.qf_w4a8:
                    return super().process_weights_after_loading(layer)
                parts = [quant.w4a8_quantize(w[r:r + 2048].float()) for r in range(0, N, 2048)]       # bounded float temporaries
                packed, gs, zp, sw = (torch.cat([p[i] for p in parts]) for i in range(4))
            param = lambda t: torch.nn.Parameter(t.contiguous(), requires_grad=False)      # noqa: E731
            if os.environ.get("QUANTFORGE_W4A8_DECODE", "compact") == "compact":
                # one copy of the nibbles: the decode reads the W4A8 layout itself (slower, a quarter of fp16's memory)
                e = kfold.no_fold(w.device)
                if not self.checkpoint:
                    del layer.weight
                layer.qf_packed, layer.qf_scale, layer.qf_zero = param(e[0]), param(e[1]), param(e[2])
                layer.qf_packed_f, layer.qf_scale_f, layer.qf_zero_f = param(e[0]), param(e[1]), param(e[2])
                layer.qf_group = 128
                layer.qf8_scale_nk = param((gs.float() * sw.float()[:, None]).half())
            else:
                p16, s16, z16 = w4a8_op.from_w4a8(packed, gs, zp, sw.float())
                _install(layer, p16, s16, z16, None, 128, () if self.checkpoint else ("weight",))
                layer.qf8_scale_nk = param(torch.empty(0, device=w.device, dtype=torch.float16))
            layer.qf8_packed, layer.qf8_gs, layer.qf8_zp, layer.qf8_sw = param(packed), param(gs), param(zp), param(sw.float())

        def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            if not layer.qf_w4a8:
                return super().apply(layer, x, bias)
            xh = x if x.dtype == torch.float16 else x.half()
            y = torch.ops.quantforge.w4a8_hybrid(
                xh, layer.qf8_packed, layer.qf8_gs, layer.qf8_zp, layer.qf8_sw, None if bias is None else bias.half(),
                layer.qf_packed, layer.qf_scale, layer.qf_zero, layer.qf_packed_f, layer.qf_scale_f, layer.qf_zero_f, layer.qf_group,
                layer.qf8_scale_nk)
            return y if x.dtype == torch.float16 else y.to(x.dtype)

    @register_quantization_config("quantforge_w4a8")
    class QuantForgeW4A8Config(QuantizationConfig):
        """W4A8: either an unquantised checkpoint quantised while loading (no quantization_config needed), or a checkpoint written by
        ``w4a8_checkpoint.save_w4a8`` (GPTQ or round-to-nearest; its config.json says ``"quant_method": "quantforge_w4a8"``)."""

        def __init__(self, checkpoint: bool = False) -> None:
            super().__init__()
            self.checkpoint = checkpoint

        @classmethod
        def get_name(cls):
            return "quantforge_w4a8"

        @classmethod
        def get_supported_act_dtypes(cls):
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @staticmethod
        def get_config_filenames() -> list:
            return []                                  # vLLM then builds the config with no arguments

        @classmethod
        def from_config(cls, config):
            return cls(checkpoint=str(config.get("quant_method", "")).lower() == "quantforge_w4a8")

        def get_quant_method(self, layer, prefix):
            return QuantForgeW4A8Method(self.checkpoint) if isinstance(layer, LinearBase) else None
