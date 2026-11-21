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
