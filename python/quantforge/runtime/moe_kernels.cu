// Glue of the grouped-GEMM mixture-of-experts feed-forward (moe.py): routing, gather, cast and combine as four small kernels, in place
// of the ~30 elementwise / sort / scan kernels torch would launch for the same index arithmetic. All shapes are static and nothing
// synchronises with the host, so the ops run inside CUDA graphs.
//
//   moe_route(topk_ids[T, k] int32/int64, E, mcap, P) -> (dest[T*k] int32, row_src[P*mcap] int32, meta[P] int32)
//       The (token, slot) pairs are grouped by expert and every expert is padded to a multiple of `mcap` rows (the row tile of the GEMMs).
//       dest[i]    row of the padded buffer that pair i (token-major) occupies; P*mcap (the scratch row) for an id outside [0, E) (vLLM uses -1)
//       row_src[r] index of the (token, slot) pair whose token's activations row r holds (token = pair / k), -1 for a padding row
//       meta[j]    expert of row block j (rows j*mcap .. (j+1)*mcap), -1 when the block lies past the last expert (the grouped kernel skips it)
//       One thread block: the order inside an expert is whatever the shared-memory atomics give (every row is independent, so the result
//       does not depend on it).
//   moe_gather(x[T, H] f16, row_src[R]) -> rows[R + 1, H] f16: copies the rows that have a source; the others are left uninitialised (never read back).
//   moe_cast_rows(g[R, W] f32, meta, mcap, I) -> h[R, I] f16: converts the first I columns of the rows of active blocks.
//   moe_combine(y[R + 1, H] f16, dest[T*k], w[T, k] f32/f16, T, k) -> out[T, H] f16 = sum_s w[t, s] * y[dest[t, s]] (pairs with dest >= R skipped).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kRouteThreads = 1024;

__device__ __forceinline__ float as_float(float v) { return v; }
__device__ __forceinline__ float as_float(__half v) { return __half2float(v); }

template <typename Id>
__global__ void moe_route_kernel(const Id *__restrict__ ids, int n, int E, int mcap, int P, int *__restrict__ dest,
                                 int *__restrict__ row_src, int *__restrict__ meta, int *__restrict__ scratch_rank) {
  extern __shared__ int sm[];
  int *count = sm;           // E + 1
  int *pstart = sm + E + 1;  // E + 1
  int *pend = sm + 2 * (E + 1);
  const int R = P * mcap;
  for (int e = threadIdx.x; e <= E; e += blockDim.x)
    count[e] = 0;
  for (int r = threadIdx.x; r < R; r += blockDim.x)
    row_src[r] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    long long v = static_cast<long long>(ids[i]);
    int e = (v >= 0 && v < E) ? static_cast<int>(v) : E;
    scratch_rank[i] = atomicAdd(&count[e], 1);   // rank inside the expert (also stored for the second pass)
    dest[i] = e;                                   // the expert for now
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      int padded = (count[e] + mcap - 1) / mcap * mcap;
      pstart[e] = acc;
      acc += padded;
      pend[e] = acc;
    }
    pstart[E] = R;   // the scratch row of the invalid ids
    pend[E] = R;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int e = dest[i];
    int row = (e < E) ? pstart[e] + scratch_rank[i] : R;
    dest[i] = row;
    if (e < E)
      row_src[row] = i;   // pair index for now; the token is derived by the caller side (gather uses i / k)
  }
  for (int j = threadIdx.x; j < P; j += blockDim.x) {
    const int start = j * mcap;
    int lo = 0, hi = E;   // first expert whose range ends after `start`
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (pend[mid] > start)
        hi = mid;
      else
        lo = mid + 1;
    }
    meta[j] = (lo < E) ? lo : -1;
  }
}

__global__ void moe_gather_kernel(const __half *__restrict__ x, const int *__restrict__ row_src, int k, int H, __half *__restrict__ rows) {
  const int r = blockIdx.x;
  const int pair = row_src[r];
  if (pair < 0)
    return;
  const int4 *src = reinterpret_cast<const int4 *>(x + static_cast<long long>(pair / k) * H);
  int4 *dst = reinterpret_cast<int4 *>(rows + static_cast<long long>(r) * H);
  for (int c = threadIdx.x; c < H / 8; c += blockDim.x)
    dst[c] = src[c];
}

__global__ void moe_cast_rows_kernel(const float *__restrict__ g, const int *__restrict__ meta, int mcap, int W, int I, __half *__restrict__ h) {
  const int r = blockIdx.x;
  if (meta[r / mcap] < 0)
    return;
  const float *src = g + static_cast<long long>(r) * W;
  __half *dst = h + static_cast<long long>(r) * I;
  for (int c = threadIdx.x * 2; c < I; c += blockDim.x * 2) {
    float2 v = *reinterpret_cast<const float2 *>(src + c);
    *reinterpret_cast<__half2 *>(dst + c) = __floats2half2_rn(v.x, v.y);
  }
}

template <typename W>
__global__ void moe_combine_kernel(const __half *__restrict__ y, const int *__restrict__ dest, const W *__restrict__ w, int k, int H, int R,
                                   __half *__restrict__ out) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x * 2; c < H; c += blockDim.x * 2) {
    float a0 = 0.f, a1 = 0.f;
    for (int s = 0; s < k; ++s) {
      const int row = dest[t * k + s];
      if (row >= R)
        continue;
      const float wt = as_float(w[t * k + s]);
      const __half2 v = *reinterpret_cast<const __half2 *>(y + static_cast<long long>(row) * H + c);
      a0 += wt * __low2float(v);
      a1 += wt * __high2float(v);
    }
    *reinterpret_cast<__half2 *>(out + static_cast<long long>(t) * H + c) = __floats2half2_rn(a0, a1);
  }
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_route_cuda(const at::Tensor &topk_ids, int64_t E, int64_t mcap, int64_t P) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() && topk_ids.dim() == 2, "moe_route: topk_ids must be a contiguous CUDA [T, k] tensor");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong, "moe_route: topk_ids must be int32 or int64");
  const int64_t n = topk_ids.numel();
  const size_t smem = 3 * (E + 1) * sizeof(int);
  TORCH_CHECK(smem <= 48 * 1024, "moe_route: too many experts for the single-block router");
  c10::cuda::CUDAGuard guard(topk_ids.device());
  auto opts = topk_ids.options().dtype(at::kInt);
  at::Tensor dest = at::empty({n}, opts), row_src = at::empty({P * mcap}, opts), meta = at::empty({P}, opts), rank = at::empty({n}, opts);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (topk_ids.scalar_type() == at::kInt)
    moe_route_kernel<int><<<1, kRouteThreads, smem, stream>>>(topk_ids.data_ptr<int>(), n, E, mcap, P, dest.data_ptr<int>(),
                                                               row_src.data_ptr<int>(), meta.data_ptr<int>(), rank.data_ptr<int>());
  else
    moe_route_kernel<int64_t><<<1, kRouteThreads, smem, stream>>>(topk_ids.data_ptr<int64_t>(), n, E, mcap, P, dest.data_ptr<int>(),
                                                                   row_src.data_ptr<int>(), meta.data_ptr<int>(), rank.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dest, row_src, meta};
}

at::Tensor moe_gather_cuda(const at::Tensor &x, const at::Tensor &row_src, int64_t k) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.size(1) % 8 == 0,
              "moe_gather: x must be a contiguous f16 CUDA [T, H] tensor, H a multiple of 8");
  const int64_t R = row_src.numel(), H = x.size(1);
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor rows = at::empty({R + 1, H}, x.options());
  moe_gather_kernel<<<static_cast<unsigned>(R), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half *>(x.data_ptr<at::Half>()), row_src.data_ptr<int>(), static_cast<int>(k), static_cast<int>(H),
      reinterpret_cast<__half *>(rows.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rows;
}

at::Tensor moe_cast_rows_cuda(const at::Tensor &g, const at::Tensor &meta, int64_t mcap, int64_t I) {
  TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == at::kFloat && g.dim() == 2 && I % 2 == 0 && g.size(1) >= I,
              "moe_cast_rows: g must be a contiguous f32 CUDA [R, W>=I] tensor, I even");
  const int64_t R = g.size(0);
  c10::cuda::CUDAGuard guard(g.device());
  at::Tensor h = at::empty({R, I}, g.options().dtype(at::kHalf));
  moe_cast_rows_kernel<<<static_cast<unsigned>(R), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      g.data_ptr<float>(), meta.data_ptr<int>(), static_cast<int>(mcap), static_cast<int>(g.size(1)), static_cast<int>(I),
      reinterpret_cast<__half *>(h.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}

at::Tensor moe_combine_cuda(const at::Tensor &y, const at::Tensor &dest, const at::Tensor &w) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kHalf && y.dim() == 2 && y.size(1) % 2 == 0,
              "moe_combine: y must be a contiguous f16 CUDA [R+1, H] tensor");
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2 && (w.scalar_type() == at::kFloat || w.scalar_type() == at::kHalf),
              "moe_combine: w must be a contiguous f32 / f16 CUDA [T, k] tensor");
  const int64_t T = w.size(0), k = w.size(1), H = y.size(1), R = y.size(0) - 1;
  c10::cuda::CUDAGuard guard(y.device());
  at::Tensor out = at::empty({T, H}, y.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (w.scalar_type() == at::kFloat)
    moe_combine_kernel<float><<<static_cast<unsigned>(T), 256, 0, stream>>>(reinterpret_cast<const __half *>(y.data_ptr<at::Half>()),
                                                                             dest.data_ptr<int>(), w.data_ptr<float>(), k, H, R,
                                                                             reinterpret_cast<__half *>(out.data_ptr<at::Half>()));
  else
    moe_combine_kernel<__half><<<static_cast<unsigned>(T), 256, 0, stream>>>(
        reinterpret_cast<const __half *>(y.data_ptr<at::Half>()), dest.data_ptr<int>(), reinterpret_cast<const __half *>(w.data_ptr<at::Half>()), k, H,
        R, reinterpret_cast<__half *>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_FRAGMENT(quantforge, m) {
  m.def("moe_route(Tensor topk_ids, int E, int mcap, int P) -> (Tensor, Tensor, Tensor)");
  m.def("moe_gather(Tensor x, Tensor row_src, int k) -> Tensor");
  m.def("moe_cast_rows(Tensor g, Tensor meta, int mcap, int I) -> Tensor");
  m.def("moe_combine(Tensor y, Tensor dest, Tensor w) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantforge, CUDA, m) {
  m.impl("moe_route", &moe_route_cuda);
  m.impl("moe_gather", &moe_gather_cuda);
  m.impl("moe_cast_rows", &moe_cast_rows_cuda);
  m.impl("moe_combine", &moe_combine_cuda);
}
