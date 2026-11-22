"""Write a model quantised with ``quantize_model_gptq(..., fuse=False)`` as an ordinary GPTQ checkpoint (the AutoGPTQ / GPTQModel tensor layout:
``qweight`` int32 ``[K/8, N]``, ``qzeros`` int32 ``[K/G, N/8]``, ``scales`` f16 ``[K/G, N]``, ``g_idx`` int32 ``[K]`` per linear layer, everything
else dense fp16), so that vLLM loads it with ``quantization="gptq_marlin"`` or ``"quantforge"`` and any other GPTQ reader can use it.

    from quantforge.gptq_checkpoint import save_gptq
    quantize_model_gptq(model, calib_ids, 128, fuse=False)      # fuse=False: the q/k/v and gate/up layers stay separate, as in a checkpoint
    save_gptq(model, "out_dir", tokenizer)

``benchmarks/make_gptq_checkpoint.py`` does all of it from a Hugging Face model id. The embedding / head are left as they are (fp16).
Layers quantised with act-order (a ``perm``) are not supported yet (an error).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from . import quant
from .nn import QuantLinear


def gptq_tensors(m: QuantLinear) -> dict:
    """The four GPTQ tensors of one quantised layer (``qweight``, ``qzeros``, ``scales``, ``g_idx``) plus its bias."""
    q = quant.unpack_int4(m.packed)                                   # [K, N] 0..15
    K = q.shape[0]
    G = m.group_size
    qw, qz, scales = quant.to_gptq(q, m.scale, m.zero)
    if m.perm is not None:
        raise NotImplementedError("act-order layers cannot be written yet")
    g_idx = (torch.arange(K, device=q.device) // G).to(torch.int32)
    out = {"qweight": qw.contiguous(), "qzeros": qz.contiguous(), "scales": scales.half().contiguous(), "g_idx": g_idx.contiguous()}
    if m.bias is not None:
        out["bias"] = m.bias.half().contiguous()
    return out


def save_gptq(model: nn.Module, out_dir, tokenizer=None) -> Path:
    """Save `model` (a Hugging Face causal LM whose decoder linear layers are all ``QuantLinear``, unfused) to `out_dir` in GPTQ format."""
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    groups = set()
    quant_names = set()
    for name, m in model.named_modules():
        if type(m).__name__ in ("_QKVPart", "_Int8Shared"):
            raise ValueError("save the model quantised with fuse=False (the shared q/k/v and gate/up layers are not a checkpoint format)")
        if isinstance(m, QuantLinear):
            if m.fmt != "int4":
                raise ValueError(f"{name}: only INT4 layers can be written as a GPTQ checkpoint (format {m.fmt})")
            if m.perm is not None:
                raise NotImplementedError(f"{name}: act-order layers cannot be written yet (quantise without act_order)")
            quant_names.add(name)
            groups.add(m.group_size)
            for k, v in gptq_tensors(m).items():
                tensors[f"{name}.{k}"] = v.detach().cpu()
        elif isinstance(m, nn.Linear) and "lm_head" not in name and name:
            raise ValueError(f"{name} is still a dense nn.Linear: quantise every decoder linear layer first")
    if not quant_names:
        raise ValueError("no QuantLinear layer in the model")
    if len(groups) != 1:
        raise ValueError(f"layers with several group sizes {sorted(groups)} cannot be written")
    for name, p in model.state_dict().items():
        if any(name.startswith(q + ".") for q in quant_names):
            continue
        tensors[name] = p.detach().cpu().contiguous()
    if "lm_head.weight" not in tensors and getattr(model.config, "tie_word_embeddings", False) is False and hasattr(model, "lm_head"):
        tensors["lm_head.weight"] = model.lm_head.weight.detach().cpu().contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(out)
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = {"quant_method": "gptq", "bits": 4, "group_size": groups.pop(), "desc_act": False, "sym": False,
                                  "true_sequential": True, "checkpoint_format": "gptq", "lm_head": False}
    cfg["torch_dtype"] = cfg["dtype"] = "float16"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    return out
