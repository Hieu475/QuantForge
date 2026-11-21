"""Compiler-generated INT8 GEMM on the Tensor Cores (experimental): bit-exact against cuBLASLt's int32 result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from test_w4a16_e2e import _require_gpu  # noqa: E402


@pytest.mark.parametrize("M,K,N,extra", [(256, 512, 256, ""), (512, 1024, 384, ""), (256, 512, 512, " block-swizzle=4")])
def test_int8_gemm_is_bit_exact(M, K, N, extra):
    _require_gpu()
    from quantforge.e2e import INT8_GEMM_OPTS, Int8Gemm

    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8, generator=g)
    bt = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8, generator=g)
    c = torch.full((M, N), 12345, device="cuda", dtype=torch.int32)            # overwritten, not accumulated into
    Int8Gemm.build(M, K, N, pipeline_opts=INT8_GEMM_OPTS + extra).launch(a, bt, c)
    torch.cuda.synchronize()
    assert torch.equal(c, torch._int_mm(a, bt.t()))
    ext = a.double() @ bt.double().t()                                         # and against exact arithmetic (sums < 2**53)
    assert torch.equal(c.double(), ext)


@pytest.mark.parametrize("bm,bn,bk,wm,wn", [(64, 128, 64, 32, 64), (64, 128, 32, 64, 64), (128, 256, 32, 64, 64), (256, 128, 64, 64, 64),
                                             (128, 64, 128, 32, 32), (64, 64, 64, 32, 32), (128, 128, 128, 64, 64)])
def test_int8_gemm_is_exact_for_block_n_different_from_block_m(bm, bn, bk, wm, wn):
    """Regression: the promotion pass recognised the transposed-B tile [bn x bk] only when bn happened to equal bm / wm; otherwise it
    promoted the warp tile into a buffer shared by all warps and half of the output (or more) was wrong."""
    _require_gpu()
    from quantforge.e2e import Int8Gemm

    M, K, N = 512, 512, 512
    g = torch.Generator(device="cuda").manual_seed(1)
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8, generator=g)
    bt = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8, generator=g)
    opts = (f"block-tile-m={bm} block-tile-n={bn} block-tile-k={bk} warp-tile-m={wm} warp-tile-n={wn} smem-stages=2 "
            "thread-tile-k=32 smem-pad=16 accumulate=false")
    c = torch.full((M, N), -1, device="cuda", dtype=torch.int32)
    Int8Gemm.build(M, K, N, pipeline_opts=opts).launch(a, bt, c)
    torch.cuda.synchronize()
    assert torch.equal(c, torch._int_mm(a, bt.t()))


@pytest.mark.parametrize("M,K,N,bias", [(256, 512, 256, True), (512, 1024, 384, False), (256, 512, 512, True)])
def test_int8_gemm_with_the_dequantisation_fused_into_the_store(M, K, N, bias):
    """y[f16] = (A @ Bt^T) * sx[m] * sw[n] + bias[n] in ONE kernel: equal to the reference computed in f32 and rounded once."""
    _require_gpu()
    from quantforge.e2e import Int8GemmDq

    g = torch.Generator(device="cuda").manual_seed(3)
    a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8, generator=g)
    bt = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8, generator=g)
    sx = torch.rand(M, device="cuda", generator=g) * 0.02 + 0.001
    sw = torch.rand(N, device="cuda", generator=g) * 0.02 + 0.001
    b = torch.randn(N, device="cuda", generator=g)
    ref = (torch._int_mm(a, bt.t()).float() * sx[:, None] * sw[None, :] + (b[None, :] if bias else 0)).half()
    y = torch.full((M, N), 9.0, device="cuda", dtype=torch.float16)            # overwritten everywhere
    Int8GemmDq.build(M, K, N, bias=bias).launch(a, bt, sx, sw, b, y)
    torch.cuda.synchronize()
    assert torch.equal(y, ref)
