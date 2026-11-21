"""Split-K in the compiler (experimental, `split-k=S` pipeline option): the result equals the unsplit kernel's."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402

DECODE = "decode=true block-tile-n=64 warp-tile-n=16 block-tile-k=128 smem-stages=1 prefetch-distance=1"


@pytest.mark.parametrize("M,K,N,S", [(16, 1024, 256, 4), (16, 1536, 384, 3), (32, 1024, 256, 2)])
def test_split_k_matches_the_unsplit_kernel_and_the_reference(M, K, N, S):
    _require_gpu()
    from quantforge.e2e import W4A16Gemm, pack_int4

    G = 128
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.rand(M, K, device="cuda", generator=g) * 2 - 1).half()
    q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
    scale = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
    zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
    ref = x.float() @ ((q.float() - zp.float().repeat_interleave(G, 0)) * scale.float().repeat_interleave(G, 0))
    outs = {}
    for sk in (1, S):
        k = W4A16Gemm.build(M, K, N, group_size=G, pad=False, pipeline_opts=f"{DECODE} split-k={sk}", accumulate=False)
        out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        k.launch(x, pack_int4(q), scale, zp, out)
        torch.cuda.synchronize()
        assert not torch.isnan(out).any()
        assert (out - ref).abs().max() < 1e-3 * ref.abs().max(), sk
        outs[sk] = out
    assert (outs[S] - outs[1]).abs().max() < 1e-3 * ref.abs().max()
