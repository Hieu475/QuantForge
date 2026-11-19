"""
Real AWQ / GPTQ checkpoints (Qwen2.5-0.5B-Instruct, 4-bit, group size 128) through the QuantForge
kernels, against vLLM's own AWQ-dequantize / GPTQ-GEMM kernels as an independent oracle.

Only the few tensors a test needs are downloaded (HTTP Range requests on the safetensors file,
a few MB in total) and cached in ~/.cache/quantforge/ckpt. Skipped when offline
(QUANTFORGE_OFFLINE=1 or no network) or when vLLM is not installed.
"""
import json
import os
import struct
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402
from quantforge import quant  # noqa: E402
from quantforge.e2e import W4A16Gemm  # noqa: E402

AWQ = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
GPTQ = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
_DTYPES = {"I32": torch.int32, "F16": torch.float16}
_CACHE = Path(os.environ.get("QUANTFORGE_CKPT_CACHE", Path.home() / ".cache" / "quantforge" / "ckpt"))


def _range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _tensors(repo, names):
    """{name: cpu tensor}; downloads only these tensors of model.safetensors (cached)."""
    out, missing = {}, []
    for n in names:
        f = _CACHE / repo.replace("/", "--") / f"{n}.pt"
        if f.exists():
            out[n] = torch.load(f)
        else:
            missing.append(n)
    if missing:
        if os.environ.get("QUANTFORGE_OFFLINE"):
            raise unittest.SkipTest("offline: checkpoint tensors not cached")
        try:
            url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
            n = struct.unpack("<Q", _range(url, 0, 7))[0]
            header = json.loads(_range(url, 8, 8 + n - 1))
            for name in missing:
                dt, shape, (a, b) = (header[name][k] for k in ("dtype", "shape", "data_offsets"))
                raw = bytearray(_range(url, 8 + n + a, 8 + n + b - 1))
                t = torch.frombuffer(raw, dtype=_DTYPES[dt]).reshape(shape).clone()
                f = _CACHE / repo.replace("/", "--") / f"{name}.pt"
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(t, f)
                out[name] = t
        except (OSError, KeyError) as e:   # URLError / timeouts are OSErrors
            raise unittest.SkipTest(f"checkpoint not reachable: {e}")
    return out


def _vllm_ops():
    try:
        from vllm import _custom_ops as ops
        return ops
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"vLLM not available: {e}")


LAYERS = [  # (module, K, N): gate/up, down and attention projections of two decoder layers
    ("model.layers.0.mlp.gate_proj", 896, 4864),
    ("model.layers.0.mlp.down_proj", 4864, 896),
    ("model.layers.0.self_attn.k_proj", 896, 128),
    ("model.layers.11.self_attn.q_proj", 896, 896),
]


def _act(M, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()


def test_real_awq_checkpoint():
    _require_gpu()
    ops = _vllm_ops()
    for mod, K, N in LAYERS:
        t = _tensors(AWQ, [f"{mod}.qweight", f"{mod}.qzeros", f"{mod}.scales"])
        qw, qz, sc = (t[f"{mod}.{k}"].cuda() for k in ("qweight", "qzeros", "scales"))
        assert qw.shape == (K, N // 8)
        W = ops.awq_dequantize(qw, sc, qz, 0, 0, 0)            # vLLM's dequantisation [K, N] f16
        packed, scale, zero = quant.from_awq(qw, qz, sc)
        for M, opts in ((1, "decode=true"), (16, "decode=true"), (64, "")):
            act = _act(M, K)
            kern = W4A16Gemm.build(M, K, N, group_size=128, pad=True, pipeline_opts=opts)
            out = kern.run(act, packed, scale, zero)
            ref = act.float() @ W.float()
            err = (out - ref).abs().max().item()
            assert err < 1e-4 * ref.abs().max().item() + 1e-3, (mod, M, err)


def _gptq_reference_weights(qw, qz, sc, K, N):
    """f32 weights straight from the GPTQ file format (own unpacking, independent of
    quantforge.quant): qweight packs 8 consecutive K rows per int32, qzeros 8 columns per int32
    holding zero - 1."""
    shifts = torch.arange(0, 32, 4, device=qw.device, dtype=torch.int32)
    q = ((qw.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N).float()
    z = ((qz.unsqueeze(-1) >> shifts) & 0xF).reshape(qz.shape[0], N).float() + 1
    return (q - z.repeat_interleave(128, 0)) * sc.float().repeat_interleave(128, 0)


def test_real_gptq_checkpoint():
    """Primary oracle: an f32 reference built from the checkpoint with independent code.
    vLLM's gptq_gemm is only a loose cross-check: it accumulates in f16, so it is off by
    ~0.6 % of the output range at K = 4864 while QuantForge matches the f32 reference to 5e-4."""
    _require_gpu()
    ops = _vllm_ops()
    for mod, K, N in LAYERS:
        t = _tensors(GPTQ, [f"{mod}.{k}" for k in ("qweight", "qzeros", "scales", "g_idx")])
        qw, qz, sc, gi = (t[f"{mod}.{k}"].cuda() for k in ("qweight", "qzeros", "scales", "g_idx"))
        assert qw.shape == (K // 8, N)
        assert torch.equal(gi, torch.arange(K, device="cuda", dtype=torch.int32) // 128), \
            "this test covers desc_act=False checkpoints"
        W = _gptq_reference_weights(qw, qz, sc, K, N)
        packed, scale, zero = quant.from_gptq(qw, qz, sc, gi)
        for M, opts in ((1, "decode=true"), (16, "decode=true"), (64, "")):
            act = _act(M, K, seed=3)
            ref = act.float() @ W
            kern = W4A16Gemm.build(M, K, N, group_size=128, pad=True, pipeline_opts=opts)
            out = kern.run(act, packed, scale, zero)
            err = (out - ref).abs().max().item()
            assert err < 1e-4 * ref.abs().max().item() + 3e-3, (mod, M, err)
            vllm = ops.gptq_gemm(act, qw, qz, sc, gi, False, False, 4).float()
            assert (vllm - ref).abs().max().item() < 1.5e-2 * ref.abs().max().item() + 1e-2, (mod, M)


if __name__ == "__main__":
    unittest.main()
