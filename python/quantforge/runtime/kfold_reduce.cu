// Epilogue of the K-folded GEMM (kfold.py): y[m, n] = sum_s acc[m*S + s, s*N + n] (+ bias[n]), written as f16.
// The folded GEMM computes every (slice, slice) pair; only the S diagonal blocks are wanted. One kernel replaces the
// f32 -> f16 conversion, the strided reduction and the final cast / bias add that torch would launch separately.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__global__ void kfold_reduce_kernel(const float *__restrict__ acc, const __half *__restrict__ bias,
                                    __half *__restrict__ out, int64_t S, int64_t N) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  const int64_t m = blockIdx.y, ld = S * N;
  float v = 0.f;
  for (int64_t s = 0; s < S; ++s)
    v += acc[(m * S + s) * ld + s * N + n];
  if (bias != nullptr)
    v += __half2float(bias[n]);
  out[m * N + n] = __float2half(v);
}

} // namespace

at::Tensor kfold_reduce_cuda(const at::Tensor &acc, int64_t M, int64_t S, int64_t N,
                             const c10::optional<at::Tensor> &bias) {
  TORCH_CHECK(acc.is_cuda() && acc.scalar_type() == at::kFloat && acc.is_contiguous() && acc.size(1) == S * N,
              "kfold_reduce: acc must be a contiguous f32 [rows, S*N] CUDA tensor");
  TORCH_CHECK(acc.size(0) >= M * S, "kfold_reduce: acc has fewer than M*S rows");
  const at::Tensor b = bias.has_value() ? bias->contiguous() : at::Tensor();
  TORCH_CHECK(!bias.has_value() || (b.scalar_type() == at::kHalf && b.numel() == N), "kfold_reduce: bias must be f16 [N]");
  c10::cuda::CUDAGuard guard(acc.device());
  at::Tensor out = at::empty({M, N}, acc.options().dtype(at::kHalf));
  const int threads = 128;
  dim3 grid(static_cast<unsigned>((N + threads - 1) / threads), static_cast<unsigned>(M));
  kfold_reduce_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      acc.data_ptr<float>(), bias.has_value() ? reinterpret_cast<const __half *>(b.data_ptr<at::Half>()) : nullptr,
      reinterpret_cast<__half *>(out.data_ptr<at::Half>()), S, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
