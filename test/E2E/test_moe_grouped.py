"""Grouped W4A16 GEMM (one launch, grid.z problems, expert chosen per problem from a device array)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402
from quantforge import quant  # noqa: E402
from quantforge.e2e import dequant_reference, pack_int4  # noqa: E402
from quantforge.moe import GroupedGemm  # noqa: E402

OPTS = {16: "block-tile-m=16 block-tile-n=64 warp-tile-m=16 warp-tile-n=16 block-tile-k=64",
        64: "block-tile-m=64 block-tile-n=128 warp-tile-m=32 warp-tile-n=64 block-tile-k=64"}


def _experts(E, K, N, G, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, 16, (E, K, N), device="cuda", dtype=torch.int8, generator=g)
    sc = (torch.rand(E, K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
    zp = torch.randint(0, 16, (E, K // G, N), device="cuda", dtype=torch.int8, generator=g)
    return q, sc, zp


@pytest.mark.parametrize("Mcap", [16, 64])
def test_grouped_f16_matches_per_problem_reference(Mcap):
    _require_gpu()
    E, K, N, G = 5, 256, 256, 64
    meta_l = [2, 0, 0, 4, -1, 1, 3, -1]
    P = len(meta_l)
    q, sc, zp = _experts(E, K, N, G, 0)
    packed = torch.stack([pack_int4(q[e]) for e in range(E)]).contiguous()
    g = torch.Generator(device="cuda").manual_seed(1)
    rows = (torch.rand(P * Mcap, K, device="cuda", generator=g) * 2 - 1).half()
    meta = torch.tensor(meta_l, dtype=torch.int32, device="cuda")
    out = torch.full((P * Mcap, N), 123.0, device="cuda", dtype=torch.half)         # padding problems must stay untouched
    kern = GroupedGemm.build(Mcap, K, N, E, P, G, "f16", OPTS[Mcap])
    kern.launch(rows, packed, sc, zp, out, meta)
    for p, e in enumerate(meta_l):
        blk = out[p * Mcap:(p + 1) * Mcap].float()
        if e < 0:
            assert (blk == 123.0).all(), "a padding problem wrote its output"
        else:
            ref = rows[p * Mcap:(p + 1) * Mcap].float() @ dequant_reference(q[e], sc[e], zp[e]).float()
            assert (blk - ref).abs().max().item() < 1e-2 * ref.abs().max().item(), (p, e)


def test_grouped_swiglu_matches_per_problem_reference():
    _require_gpu()
    Mcap, E, K, I, G = 16, 4, 256, 128, 64
    meta_l = [3, 1, 1, -1, 0]
    P = len(meta_l)
    qg, sg, zg = _experts(E, K, I, G, 2)
    qu, su, zu = _experts(E, K, I, G, 3)
    inter = [quant.interleave_gate_up((pack_int4(qg[e]), sg[e], zg[e]), (pack_int4(qu[e]), su[e], zu[e])) for e in range(E)]
    packed = torch.stack([t[0] for t in inter]).contiguous()
    sc = torch.stack([t[1] for t in inter]).contiguous()
    zp = torch.stack([t[2] for t in inter]).contiguous()
    g = torch.Generator(device="cuda").manual_seed(4)
    rows = (torch.rand(P * Mcap, K, device="cuda", generator=g) * 2 - 1).half()
    meta = torch.tensor(meta_l, dtype=torch.int32, device="cuda")
    out = torch.zeros(P * Mcap, 2 * I, device="cuda", dtype=torch.float32)
    kern = GroupedGemm.build(Mcap, K, 2 * I, E, P, G, "swiglu", OPTS[16])
    kern.launch(rows, packed, sc, zp, out, meta)
    for p, e in enumerate(meta_l):
        if e < 0:
            continue
        a = rows[p * Mcap:(p + 1) * Mcap].float()
        gate = a @ dequant_reference(qg[e], sg[e], zg[e]).float()
        up = a @ dequant_reference(qu[e], su[e], zu[e]).float()
        ref = torch.nn.functional.silu(gate) * up
        got = out[p * Mcap:(p + 1) * Mcap, :I]
        assert (got - ref).abs().max().item() < 1e-2 * ref.abs().max().item() + 1e-3, (p, e)


def _moe_weights(E, H, I, G, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    p13, s13, z13, p2, s2, z2 = [], [], [], [], [], []
    for _ in range(E):
        w = [torch.randn(H, I, device="cuda", generator=gen) * 0.05 for _ in range(3)]        # gate, up, down^T
        parts = []
        for m in (w[0], w[1]):
            q, s, z = quant.quantize_groupwise(m, G)
            parts.append((pack_int4(q), s, z))
        pa, sa, za = quant.interleave_gate_up(*parts)
        p13.append(pa), s13.append(sa), z13.append(za)
        q, s, z = quant.quantize_groupwise(w[2].t().contiguous(), G)                          # [I, H]
        p2.append(pack_int4(q)), s2.append(s), z2.append(z)
    st = lambda l: torch.stack(l).contiguous()  # noqa: E731
    return st(p13), st(s13), st(z13), st(p2), st(s2), st(z2)


@pytest.mark.parametrize("T,top_k", [(1, 2), (3, 2), (17, 2), (100, 3), (300, 2)])
def test_moe_ffn_matches_dense_reference(T, top_k):
    _require_gpu()
    from quantforge import moe

    E, H, I, G = 8, 256, 256, 64
    w = _moe_weights(E, H, I, G, 7)
    gen = torch.Generator(device="cuda").manual_seed(8)
    x = (torch.randn(T, H, device="cuda", generator=gen) * 0.5).half()
    logits = torch.randn(T, E, device="cuda", generator=gen)
    topk_w, topk_ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).float()
    got = moe.moe_ffn_impl(x, topk_ids, topk_w, *w, G)
    ref = moe.moe_ffn_reference(x, topk_ids, topk_w, *w, G)
    assert got.shape == ref.shape and got.dtype == torch.float16
    assert (got.float() - ref.float()).abs().max().item() < 2e-2 * ref.float().abs().max().item() + 2e-3


@pytest.mark.parametrize("sym", [False, True])
def test_from_wna16_builds_the_grouped_layout(sym):
    """vLLM's MoE WNA16 tensors (N-major, low nibble = even k, qzeros packed along N) convert to our K-major gate/up-interleaved layout."""
    _require_gpu()
    from quantforge import moe

    E, H, I, G = 3, 256, 128, 64
    g = torch.Generator(device="cuda").manual_seed(1)
    ri = lambda *s: torch.randint(0, 16, s, device="cuda", dtype=torch.int8, generator=g)  # noqa: E731
    rs = lambda *s: torch.rand(*s, device="cuda", generator=g).half()  # noqa: E731
    qg, qu, q2 = ri(E, H, I), ri(E, H, I), ri(E, I, H)
    sg, su, s2 = rs(E, H // G, I), rs(E, H // G, I), rs(E, I // G, H)
    zg, zu, z2 = (ri(E, H // G, I), ri(E, H // G, I), ri(E, I // G, H)) if not sym else (None,) * 3
    pack_k = lambda q_kn: moe._pack_last(q_kn.permute(0, 2, 1).contiguous()).view(torch.uint8)  # [E,K,N] -> [E,N,K/2]  # noqa: E731
    w13_q = torch.cat([pack_k(qg).view(torch.int8), pack_k(qu).view(torch.int8)], 1).view(torch.uint8)
    w13_s = torch.cat([sg, su], 2).permute(0, 2, 1).contiguous()
    pack_n = lambda z_gn: moe._pack_last(z_gn).permute(0, 2, 1).contiguous().view(torch.uint8)  # [E,G',N] -> [E,N/2,G']  # noqa: E731
    w13_z = None if sym else pack_n(torch.cat([zg, zu], 2))
    w2_q = pack_k(q2)
    w2_s = s2.permute(0, 2, 1).contiguous()
    w2_z = None if sym else pack_n(z2)
    p13, sc13, zp13, p2, sc2, zp2 = moe.from_wna16(w13_q, w13_s, w13_z, w2_q, w2_s, w2_z)

    inter = lambda a, b: torch.stack((a, b), dim=-1).reshape(*a.shape[:-1], -1)  # noqa: E731
    assert torch.equal(p13, moe._pack_last(inter(qg, qu)))
    assert torch.equal(sc13, inter(sg, su))
    assert torch.equal(zp13, torch.full_like(zp13, 8) if sym else inter(zg, zu))
    assert torch.equal(p2, moe._pack_last(q2))
    assert torch.equal(sc2, s2)
    assert torch.equal(zp2, torch.full_like(zp2, 8) if sym else z2)


def test_moe_ffn_ignores_invalid_expert_ids():
    """vLLM fills dummy / dropped routing entries with -1: they must contribute nothing (and not index out of bounds)."""
    _require_gpu()
    from quantforge import moe

    E, H, I, G, T, k = 8, 256, 256, 64, 12, 3
    w = _moe_weights(E, H, I, G, 9)
    gen = torch.Generator(device="cuda").manual_seed(10)
    x = (torch.randn(T, H, device="cuda", generator=gen) * 0.5).half()
    ids = torch.stack([torch.randperm(E, device="cuda", generator=gen)[:k] for _ in range(T)])
    tw = torch.softmax(torch.randn(T, k, device="cuda", generator=gen), -1)
    ids_bad = ids.clone().int()
    ids_bad[2] = -1                                                   # a whole token without experts
    ids_bad[5, 1] = -1                                                # one dropped slot
    ids_bad[7, 0] = E + 3                                             # out of range
    ref_w = tw.clone()
    ref_w[2] = 0
    ref_w[5, 1] = 0
    ref_w[7, 0] = 0
    ref = moe.moe_ffn_reference(x, ids, ref_w, *w, G)                 # same experts, the invalid slots weighted 0
    got = moe.moe_ffn_impl(x, ids_bad, tw, *w, G)
    assert (got.float() - ref.float()).abs().max().item() < 2e-2 * ref.float().abs().max().item() + 2e-3
    assert got[2].abs().max().item() == 0.0
