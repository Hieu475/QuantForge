"""GPTQ post-training quantisation (Frantar et al., 2022) producing QuantForge layers.

    import quantforge.gptq as qgptq
    ids = qgptq.calibration_ids(tokenizer, texts, nsamples=128, seqlen=512)       # [nsamples, seqlen]
    qgptq.quantize_model_gptq(model, ids, group_size=128)                         # Llama / Qwen / Mistral ...

Layer by layer, the decoder blocks are run on the calibration tokens (with the previously quantised blocks
in place), the Hessian ``H = 2 X^T X / n`` of every Linear's input is accumulated, and the weights are
quantised column by column with the error of each column compensated in the not yet quantised ones (via
the Cholesky factor of ``H^-1``). Round-to-nearest ignores the activations; GPTQ typically recovers most of
the perplexity it loses (see the README for measurements).

``act_order=True`` quantises the columns in order of decreasing Hessian diagonal (better quality for small
groups); the layer then carries an input permutation (``QuantLinear.perm``), which disables the q/k/v
fusion of those layers. Only 4-bit asymmetric group-wise weights are produced (the QuantForge INT4
layout), with scales rounded to f16 *before* the error is propagated, so the optimisation sees the same
dequantised weights the kernels use.
"""
from __future__ import annotations

import gc
import math
import time
from typing import Optional

import torch
import torch.nn as nn

from . import quant
from .nn import QuantLinear, W4A8Linear, _fuse, _fuse_int8

__all__ = ["gptq_quantize_weight", "gptq_quantize_weight_w4a8", "calibration_ids", "quantize_model_gptq"]


@torch.no_grad()
def gptq_quantize_weight(W: torch.Tensor, H: torch.Tensor, group_size: int = 128, blocksize: int = 128,
                         damp: float = 0.01, act_order: bool = False, bits: int = 4, clip: bool = False):
    """Quantise ``W [out, in]`` given the input Hessian ``H [in, in]``.

    Returns ``(q, scale, zero, perm)`` in QuantForge orientation: ``q`` int8 ``[in, out]`` (values
    ``0 .. 2^bits-1``, rows already in the permuted order if ``perm`` is not None), ``scale`` f16
    ``[in/G, out]``, ``zero`` int8 ``[in/G, out]``, ``perm`` int64 ``[in]`` or None."""
    W = W.clone().float()
    H = H.clone().float()
    rows, cols = W.shape
    if cols % group_size:
        raise ValueError(f"in_features {cols} is not a multiple of the group size {group_size}")
    blocksize = max(group_size, blocksize // group_size * group_size)
    qmax = 2 ** bits - 1
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

    codes = torch.empty(rows, cols, dtype=torch.int8, device=W.device)
    scales, zeros = [], []
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % group_size == 0:              # parameters of the next group from the *updated* weights
                g = W1[:, i:i + group_size]
                lo, hi = g.amin(dim=1), g.amax(dim=1)
                scale = ((hi - lo) / qmax).clamp(min=1e-8).half()
                zero = (-lo / scale.float()).round().clamp(0, qmax)
                if clip:      # shrink the range of each row when that lowers its rounding error in the group
                    best = ((g - (((g / scale.float().unsqueeze(1) + zero.unsqueeze(1)).round().clamp(0, qmax)
                                   - zero.unsqueeze(1)) * scale.float().unsqueeze(1))) ** 2).sum(1)
                    for p in (0.97, 0.94, 0.9, 0.85, 0.8):
                        sc = (((hi - lo) * p) / qmax).clamp(min=1e-8).half()
                        zr = (-(lo * p) / sc.float()).round().clamp(0, qmax)
                        e = ((g - (((g / sc.float().unsqueeze(1) + zr.unsqueeze(1)).round().clamp(0, qmax)
                                   - zr.unsqueeze(1)) * sc.float().unsqueeze(1))) ** 2).sum(1)
                        better = e < best
                        best = torch.where(better, e, best)
                        scale = torch.where(better, sc, scale)
                        zero = torch.where(better, zr, zero)
                scales.append(scale)
                zeros.append(zero.to(torch.int8))
                sf, zf = scale.float(), zero
            w = W1[:, i]
            q = (w / sf + zf).round().clamp(0, qmax)
            codes[:, i1 + i] = q.to(torch.int8)
            err = (w - (q - zf) * sf) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes.t().contiguous(), torch.stack(scales).contiguous(), torch.stack(zeros).contiguous(), perm


@torch.no_grad()
def rotate_hessian(H: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Hessian of the inputs after the block-Hadamard rotation x -> x B (B block diagonal, blocks of `block`, symmetric orthonormal): B^T H B."""
    from .w4a8_op import hadamard_matrix

    n = H.shape[0] // block
    Hm = hadamard_matrix(block, H.device)
    return torch.einsum("pi,apbq,qj->aibj", Hm, H.float().reshape(n, block, n, block), Hm).reshape(H.shape)


def gptq_quantize_weight_w4a8(W: torch.Tensor, H: torch.Tensor, group_size: int = 128, blocksize: int = 128, damp: float = 0.01):
    """GPTQ for the W4A8 format (``quant.w4a8_quantize``): the weight is divided by its per-channel scale (taken from the original
    weight, as round-to-nearest does) so that it lives in int8 steps, and every group of `group_size` columns gets an integer scale
    and zero point (``quant.w4a8_group_params``) chosen from the weights *as updated by the compensation of the earlier columns*.

    Returns ``(packed int8 [out, in/2], gs int8 [out, in/G], zp int8 [out, in/G], sw f32 [out])``."""
    W = W.clone().float()
    H = H.clone().float()
    rows, cols = W.shape
    if cols % group_size or cols % 2:
        raise ValueError(f"in_features {cols} is not a multiple of the group size {group_size}")
    blocksize = max(group_size, blocksize // group_size * group_size)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    sw = (W.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    W = W / sw[:, None]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

    codes = torch.empty(rows, cols, dtype=torch.uint8, device=W.device)
    gss, zps = [], []
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if i % group_size == 0:
                gs, zp = quant.w4a8_group_params(W1[:, i:i + group_size].clamp(-127, 127))
                gss.append(gs)
                zps.append(zp)
            w = W1[:, i]
            q = (w / gs + zp).round().clamp(0, 15)
            codes[:, i1 + i] = q.to(torch.uint8)
            err = (w - (q - zp) * gs) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    packed = quant.pack_k8(codes)
    return (packed, torch.stack(gss, 1).to(torch.int8).contiguous(), torch.stack(zps, 1).to(torch.int8).contiguous(), sw)


def calibration_ids(tokenizer, texts, nsamples: int = 128, seqlen: int = 512, seed: int = 0) -> torch.Tensor:
    """``nsamples`` random windows of ``seqlen`` tokens of the concatenated `texts` (a str or a list)."""
    if not isinstance(texts, str):
        texts = "\n\n".join(texts)
    ids = tokenizer(texts, return_tensors="pt").input_ids[0]
    if ids.numel() <= seqlen:
        raise ValueError("calibration text is shorter than one window")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seqlen, (nsamples,), generator=g)
    return torch.stack([ids[s:s + seqlen] for s in starts.tolist()])


class _Stop(Exception):
    pass


class _Catcher(nn.Module):
    """Stands in for the first decoder layer: records its inputs and aborts the forward pass."""

    def __init__(self, layer):
        super().__init__()
        self.layer, self.inputs, self.kwargs = layer, [], None

    def forward(self, hidden_states, *args, **kwargs):
        self.inputs.append(hidden_states.detach())
        if self.kwargs is None:
            self.kwargs = kwargs
        raise _Stop


def _find_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find the decoder layers (expected model.layers)")


@torch.no_grad()
def quantize_model_gptq(model: nn.Module, calib_ids: torch.Tensor, group_size: int = 128,
                        damp: float = 0.01, act_order: bool = False, blocksize: int = 128, clip: bool = False,
                        fuse: bool = True, device="cuda", verbose: bool = True, rtn: tuple = (),
                        token_clip: float = 10.0, fmt: str = "int4", rotate: bool = False) -> dict:
    """GPTQ-quantise every Linear inside the decoder layers of `model` (fp16, e.g. loaded on the CPU) to
    INT4 and replace them with :class:`QuantLinear`. ``calib_ids`` is ``[nsamples, seqlen]`` token ids
    (see :func:`calibration_ids`); the ``lm_head`` / embeddings are left alone (use
    ``quantize_model(model, 'int8', only=('lm_head',), fuse=False)`` for the head afterwards).
    ``token_clip`` (default 10, 0 = off): before a calibration token enters the Hessian its norm is capped at
    ``token_clip`` times the median token norm. A few "massive activation" tokens (the attention sink; the inputs of the
    ``down_proj`` of one early layer have a token 300 to 800 times the median norm in Llama-3.2 and Qwen2.5) otherwise
    dominate the Hessian, and the compensation then ruins other tokens: without the cap GPTQ was *worse than
    round-to-nearest* on the whole Wikitext-2 test set for Llama-3.2-1B (24.08 against 22.28; with the cap 20.19). A cap of 30
    fixed the average but left rare windows broken (window 70: loss 5.9 against 3.0 for fp16); 10 leaves none (0 of 564 windows
    worse than round-to-nearest) and costs 0.07 perplexity points against 30; 5 costs 0.2.

    ``rtn``: exact names ``"<layer index>.<module path>"`` (e.g. ``"1.mlp.down_proj"``) of Linears that are quantised
    with plain round-to-nearest instead of GPTQ (see the note on massive-activation layers in the blog / README).
    ``fmt="w4a8"`` produces :class:`W4A8Linear` layers (INT4 weights with integer group scale and zero point, see ``gptq_quantize_weight_w4a8``;
    layers whose sizes are not multiples of 128 stay dense; ``act_order`` / ``clip`` are not available there).
    ``rotate`` (``fmt="w4a8"`` only): quantise in the block-Hadamard basis (`w4a8_op.rotate_weight`; the layer's Hessian is rotated with it), the layers then rotate their
    input (``W4A8Linear.rotate``) and the checkpoint is marked rotated.
    Returns ``{"quantized": n, "seconds": s, "fused_qkv": n, "fused_mlp": n, "act_order": n}``."""
    t0 = time.time()
    if rotate and fmt != "w4a8":
        raise ValueError("rotate is available for fmt='w4a8' only")
    model.config.use_cache = False
    layers = _find_layers(model)
    # everything outside the decoder layers (embeddings, rotary, final norm, head) runs on the GPU; the layers
    # themselves are moved one at a time below
    def to_device_except_layers(module):
        for child in module.children():
            if child is layers:
                continue
            if any(m is layers for m in child.modules()):
                to_device_except_layers(child)
            else:
                child.to(device)
        for p in module.parameters(recurse=False):
            p.data = p.data.to(device)
        for name, b in module.named_buffers(recurse=False):
            module._buffers[name] = b.to(device)

    to_device_except_layers(model)
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    ids = calib_ids.to(device)
    try:
        for i in range(ids.shape[0]):
            try:
                model(ids[i:i + 1])
            except _Stop:
                pass
    finally:
        layers[0] = catcher.layer
    inps, kwargs = catcher.inputs, catcher.kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in ("past_key_values", "past_key_value")}
    kwargs["use_cache"] = False
    n_tokens = len(inps) * inps[0].shape[1]
    if fmt not in ("int4", "w4a8"):
        raise ValueError('fmt must be "int4" or "w4a8"')
    if fmt == "w4a8" and (act_order or clip):
        raise ValueError("act_order / clip are not available for w4a8")
    stats = {"quantized": 0, "act_order": 0, "outlier_ratio": {}}

    for li, layer in enumerate(layers):
        layer.to(device)
        targets = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}
        hess = {n: torch.zeros(m.in_features, m.in_features, device=device) for n, m in targets.items()}

        ratio = {}                              # largest token norm / median token norm seen at each Linear input

        def make_hook(n):
            def hook(_, args):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                nrm = x.norm(dim=1)
                ratio[n] = max(ratio.get(n, 0.0), (nrm.max() / nrm.median().clamp(min=1e-6)).item())
                if token_clip > 0:
                    cap = token_clip * nrm.median().clamp(min=1e-6)
                    xh = x * (cap / nrm.clamp(min=1e-6)).clamp(max=1.0).unsqueeze(1)
                else:
                    xh = x
                hess[n].addmm_(xh.t(), xh, alpha=2.0 / n_tokens)
            return hook

        handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in targets.items()]
        for x in inps:
            layer(x, **kwargs)
        for h in handles:
            h.remove()
        rtn_set = set(rtn)
        for n, m in targets.items():
            if fmt == "w4a8":
                if m.in_features % 128 or m.out_features % 128:
                    continue                                    # stays dense (the kernel needs multiples of 128)
                from .w4a8_op import rotate_weight

                if f"{li}.{n}" in rtn_set:
                    if rotate:
                        packed, gs, zp, sw = quant.w4a8_quantize(rotate_weight(m.weight.data.to(device)))
                        ql = W4A8Linear(packed, gs, zp, sw, None if m.bias is None else m.bias.data.to(device, torch.float16))
                    else:
                        ql = W4A8Linear.from_linear(m, device)
                else:
                    W = m.weight.data.to(device)
                    packed, gs, zp, sw = (gptq_quantize_weight_w4a8(rotate_weight(W), rotate_hessian(hess[n]), group_size, blocksize, damp) if rotate
                                          else gptq_quantize_weight_w4a8(W, hess[n], group_size, blocksize, damp))
                    bias = None if m.bias is None else m.bias.data.to(device, torch.float16)
                    ql = W4A8Linear(packed, gs, zp, sw, bias)
                ql.rotate = rotate
                parent = layer.get_submodule(n.rsplit(".", 1)[0]) if "." in n else layer
                setattr(parent, n.rsplit(".", 1)[-1], ql)
                stats["quantized"] += 1
                continue
            if f"{li}.{n}" in rtn_set:
                q, scale, zero = quant.quantize_groupwise(m.weight.data.to(device).float().t().contiguous(), group_size)
                perm = None
            else:
                q, scale, zero, perm = gptq_quantize_weight(m.weight.data.to(device), hess[n], group_size,
                                                            blocksize, damp, act_order, clip=clip)
            bias = None if m.bias is None else m.bias.data.to(device, torch.float16)
            ql = QuantLinear.from_quantized(quant.pack_int4(q), scale, zero, bias, group_size, "int4",
                                            None if perm is None else perm.to(device))
            parent = layer.get_submodule(n.rsplit(".", 1)[0]) if "." in n else layer
            setattr(parent, n.rsplit(".", 1)[-1], ql)
            stats["quantized"] += 1
            stats["act_order"] += perm is not None
        for n, r in ratio.items():
            stats["outlier_ratio"][f"{li}.{n}"] = r
        del hess
        # the next layer sees the outputs of the quantised one
        new_inps = []
        for x in inps:
            out = layer(x, **kwargs)
            new_inps.append((out[0] if isinstance(out, tuple) else out).detach())
        inps = new_inps
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(f"  layer {li + 1}/{len(layers)} quantised ({time.time() - t0:.0f} s)", flush=True)
    stats["fused_qkv"], stats["fused_mlp"] = (_fuse_int8(model) if fmt == "w4a8" else _fuse(model)) if fuse else (0, 0)
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    stats["seconds"] = time.time() - t0
    return stats
