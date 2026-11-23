"""Write a quantised Hugging Face model as a W4A8 checkpoint that vLLM loads with ``quantization="quantforge_w4a8"``.

    import quantforge.gptq as qgptq, quantforge.w4a8_checkpoint as ck
    qgptq.quantize_model_gptq(model, ids, fmt="w4a8", fuse=False)        # or quantize_model(model, "w4a8", fuse=False)
    ck.save_w4a8(model, "Qwen2.5-0.5B-W4A8", tokenizer)
    # vLLM:  LLM("Qwen2.5-0.5B-W4A8", quantization="quantforge_w4a8", dtype="float16")

The checkpoint is the ``state_dict`` of the model: every decoder linear layer is four tensors (``weight`` int8 ``[N, K/2]`` in the
``quant.pack_k8`` layout, ``group_scale`` / ``group_zero`` int8 ``[N, K/128]``, ``w_scale`` f32 ``[N]``, plus its ``bias``), everything else
(embeddings, norms, the head) is the dense fp16 tensor, and ``config.json`` gets ``quantization_config = {"quant_method": "quantforge_w4a8"}``.
Because the weights come already quantised, vLLM never holds the fp16 model and no round-to-nearest step runs: GPTQ quality is kept.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn

from .nn import W4A8Linear

FORMAT_VERSION = 1


def _vllm_name(name: str) -> str:
    """The name vLLM gives the (fused) layer a Hugging Face linear belongs to."""
    for part in ("q_proj", "k_proj", "v_proj"):
        name = name.replace(part, "qkv_proj")
    for part in ("gate_proj", "up_proj"):
        name = name.replace(part, "gate_up_proj")
    return name


def save_w4a8(model: nn.Module, out_dir, tokenizer=None) -> Path:
    """Save `model` (a Hugging Face causal LM whose decoder linear layers are all ``W4A8Linear``, unfused) to `out_dir`."""
    out = Path(out_dir)
    for name, m in model.named_modules():
        if type(m).__name__ in ("_QKVPart", "_Int8Shared"):
            raise ValueError("save the model quantised with fuse=False (the shared q/k/v and gate/up layers are not a checkpoint format)")
        if isinstance(m, nn.Linear) and "lm_head" not in name:
            raise ValueError(f"{name} is still a dense nn.Linear (sizes not multiples of 128?): vLLM's W4A8 method needs every decoder "
                             "linear layer quantised")
    if not any(isinstance(m, W4A8Linear) for m in model.modules()):
        raise ValueError("no W4A8Linear layer in the model")
    model.save_pretrained(out, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    rot_of = {name: m.rotate for name, m in model.named_modules() if isinstance(m, W4A8Linear)}
    unrotated = sorted({_vllm_name(n) for n, r in rot_of.items() if not r})
    for n, r in rot_of.items():
        if r and _vllm_name(n) in unrotated:
            raise ValueError(f"{n} is rotated but another layer that shares its input (and fuses with it in vLLM) is not")
    groups = {m.group for m in model.modules() if isinstance(m, W4A8Linear)}
    if len(groups) != 1:
        raise ValueError("the W4A8 layers must all have the same group size")
    cfg["quantization_config"] = {"quant_method": "quantforge_w4a8", "group_size": groups.pop(), "format_version": FORMAT_VERSION,
                                  "layout": "pack_k8", "rotated": any(rot_of.values()), "unrotated_layers": unrotated}
    cfg["torch_dtype"] = cfg["dtype"] = "float16"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return out
