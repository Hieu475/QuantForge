// Per-token INT8 quantisation of the activations of the W4A8 op (w4a8_op.py), padded to the row bucket in the same kernel.
//
//   quantize_rows(x[M, K] f16, Mp) -> (xq[Mp, K] int8, sx[Mp] f32)
//
// Row r < M: s = max(amax(|x[r]|) * (1/127), 1e-8), xq = clamp(rint(x / s), -127, 127), sx = s.   Rows M .. Mp - 1: zeros, scale 0 (their output rows are dropped).
// Bit-identical to the torch expression it replaces (`(x.float() / s).round().clamp(-127, 127)`: IEEE division, round half to even), but one kernel instead of a
// compiled reduction + elementwise pair plus two padding copies, and no Triton / inductor at run time. Static shapes, no host synchronisation: graph capturable.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__global__ void quantize_rows_kernel(const __half *__restrict__ x, int M, int K, int8_t *__restrict__ xq, float *__restrict__ sx) {
  const int r = blockIdx.x;
  int8_t *out = xq + static_cast<long long>(r) * K;
  if (r >= M) {
    for (int c = threadIdx.x * 16; c < K; c += blockDim.x * 16)
      *reinterpret_cast<int4 *>(out + c) = make_int4(0, 0, 0, 0);
    if (threadIdx.x == 0)
      sx[r] = 0.f;
    return;
  }
  const __half *row = x + static_cast<long long>(r) * K;
  float amax = 0.f;
  for (int c = threadIdx.x * 8; c < K; c += blockDim.x * 8) {
    const int4 raw = *reinterpret_cast<const int4 *>(row + c);
    const __half2 *h = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h[i]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0)
    red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0)
      red[0] = v;
  }
  __syncthreads();
  const float s = fmaxf(red[0] * (1.0f / 127.0f), 1e-8f);      // torch divides a tensor by a scalar as a multiplication by the (float) reciprocal
  for (int c = threadIdx.x * 8; c < K; c += blockDim.x * 8) {
    const int4 raw = *reinterpret_cast<const int4 *>(row + c);
    const __half2 *h = reinterpret_cast<const __half2 *>(&raw);
    int8_t q[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(h[i]);
      q[2 * i] = static_cast<int8_t>(fminf(fmaxf(rintf(f.x / s), -127.f), 127.f));
      q[2 * i + 1] = static_cast<int8_t>(fminf(fmaxf(rintf(f.y / s), -127.f), 127.f));
    }
    *reinterpret_cast<int2 *>(out + c) = *reinterpret_cast<int2 *>(q);
  }
  if (threadIdx.x == 0)
    sx[r] = s;
}

// x[M, K] f16 -> [Mp, K] f16: the rows of x, then zeros (one kernel instead of at::constant_pad_nd's allocate + fill + copy).
__global__ void pad_rows_kernel(const int4 *__restrict__ x, int M, int64_t vecPerRow, int4 *__restrict__ out) {
  const int r = blockIdx.x;
  const int4 zero = make_int4(0, 0, 0, 0);
  for (int64_t c = threadIdx.x; c < vecPerRow; c += blockDim.x)
    out[static_cast<int64_t>(r) * vecPerRow + c] = r < M ? x[static_cast<int64_t>(r) * vecPerRow + c] : zero;
}

} // namespace

at::Tensor pad_rows_cuda(const at::Tensor &x, int64_t Mp) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) * x.element_size() % 16 == 0 && x.data_ptr() != nullptr &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "pad_rows: x must be a contiguous, 16-byte aligned CUDA [M, K] tensor with 16-byte rows");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(Mp >= M, "pad_rows: Mp must be at least M");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor out = at::empty({Mp, K}, x.options());
  if (Mp > 0)
    pad_rows_kernel<<<static_cast<unsigned>(Mp), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const int4 *>(x.data_ptr()), static_cast<int>(M), K * x.element_size() / 16, reinterpret_cast<int4 *>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::tuple<at::Tensor, at::Tensor> quantize_rows_cuda(const at::Tensor &x, int64_t Mp) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.size(1) % 16 == 0,
              "quantize_rows: x must be a contiguous f16 CUDA [M, K] tensor, K a multiple of 16");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(Mp >= M, "quantize_rows: Mp must be at least M");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor xq = at::empty({Mp, K}, x.options().dtype(at::kChar));
  at::Tensor sx = at::empty({Mp}, x.options().dtype(at::kFloat));
  if (Mp > 0)
    quantize_rows_kernel<<<static_cast<unsigned>(Mp), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), static_cast<int>(M), static_cast<int>(K), xq.data_ptr<int8_t>(), sx.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {xq, sx};
}

TORCH_LIBRARY_FRAGMENT(quantforge, m) {
  m.def("quantize_rows(Tensor x, int Mp) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(quantforge, CUDA, m) {
  m.impl("quantize_rows", &quantize_rows_cuda);
}
