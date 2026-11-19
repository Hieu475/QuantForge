"""CPU-only tests of python/quantforge/quant.py (checkpoint repacking, quantization)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import pytest  # noqa: E402
import torch  # noqa: E402

from quantforge import quant  # noqa: E402


def _rand_q(K, N, G, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (K, N), dtype=torch.int8, generator=g)
    scale = (torch.rand(K // G, N, generator=g) * 0.1 + 0.01).half()
    zero = torch.randint(0, 16, (K // G, N), dtype=torch.int8, generator=g)
    return q, scale, zero


def test_pack_unpack_roundtrip():
    q, _, _ = _rand_q(32, 64, 32)
    assert torch.equal(quant.unpack_int4(quant.pack_int4(q)), q)


@pytest.mark.parametrize("offset", [0, 1])
def test_gptq_roundtrip(offset):
    q, scale, zero = _rand_q(64, 32, 32)
    packed, s, z = quant.from_gptq(*quant.to_gptq(q, scale, zero, offset), zeros_offset=offset)
    assert torch.equal(quant.unpack_int4(packed), q)
    assert torch.equal(z, zero) and torch.equal(s, scale)


def test_awq_roundtrip():
    q, scale, zero = _rand_q(64, 32, 32)
    packed, s, z = quant.from_awq(*quant.to_awq(q, scale, zero))
    assert torch.equal(quant.unpack_int4(packed), q)
    assert torch.equal(z, zero) and torch.equal(s, scale)


def test_awq_word_layout_matches_reference_packer():
    """Independent re-implementation of AutoAWQ's packing loop."""
    q, scale, zero = _rand_q(16, 16, 16)
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    ref = torch.zeros(16, 2, dtype=torch.int64)
    for col in range(2):
        for i in range(8):
            ref[:, col] |= q[:, col * 8 + order[i]].to(torch.int64) << (4 * i)
    ref = torch.where(ref >= 2**31, ref - 2**32, ref).to(torch.int32)
    assert torch.equal(quant.to_awq(q, scale, zero)[0], ref)


def test_gptq_rejects_act_order():
    q, scale, zero = _rand_q(64, 32, 32)
    qw, qz, s = quant.to_gptq(q, scale, zero)
    g_idx = torch.arange(64, dtype=torch.int32) // 32
    quant.from_gptq(qw, qz, s, g_idx)                      # plain order: fine
    with pytest.raises(NotImplementedError):
        quant.from_gptq(qw, qz, s, g_idx.flip(0))


def test_gptq_act_order_matches_reference():
    K, N, G = 64, 32, 16
    q, scale, zero = _rand_q(K, N, G)
    qw, qz, s = quant.to_gptq(q, scale, zero)
    g_idx = (torch.randperm(K) % (K // G)).to(torch.int32)    # rows scattered over the groups
    # reference: row k is dequantised with the scale / zero of group g_idx[k]
    deq = (q.float() - zero.float()[g_idx.long()]) * scale.float()[g_idx.long()]
    packed, sc, zp, perm = quant.from_gptq_act_order(qw, qz, s, g_idx)
    deq_perm = (quant.unpack_int4(packed).float() - zp.float().repeat_interleave(G, 0)) \
        * sc.float().repeat_interleave(G, 0)
    x = torch.randn(3, K)
    assert torch.allclose(x[:, perm] @ deq_perm, x @ deq, atol=1e-4)
    with pytest.raises(ValueError):
        quant.act_order_permutation(torch.zeros(K, dtype=torch.int32), G)


@pytest.mark.parametrize("symmetric", [False, True])
def test_quantize_groupwise_error_is_small(symmetric):
    w = torch.randn(256, 64)
    q, scale, zero = quant.quantize_groupwise(w, 64, symmetric)
    assert q.min() >= 0 and q.max() <= 15
    deq = (q.float() - zero.float().repeat_interleave(64, 0)) * scale.float().repeat_interleave(64, 0)
    assert (deq - w).abs().max() <= scale.float().repeat_interleave(64, 0).max() * 0.51 + 1e-3
