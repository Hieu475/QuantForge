// C++ hot path of `torch.ops.quantforge.w4a16_gemm`.
//
// The Python version of the op costs ~60 us of interpreter time per call (custom_op machinery,
// shape checks, memref descriptors), which is more than the kernel itself for the small layers of
// a 0.5B model. This extension registers the same op in the dispatcher and keeps the whole hot
// path in C++: look the compiled kernel up by shape, pad the activations, start the accumulator,
// build the five memref descriptors on the stack, call the generated `_mlir_ciface_*` function on
// the current stream, slice + convert the result.
//
// Python is called back only to (a) build / load a kernel for a shape seen for the first time
// (`builder`) and (b) for the cases the hot path does not cover (`fallback`: K or N needing
// padding, per-tensor parameters, unaligned tensors).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace py = pybind11;

at::Tensor kfold_reduce_cuda(const at::Tensor &acc, int64_t M, int64_t S, int64_t N,
                             const c10::optional<at::Tensor> &bias); // kfold_reduce.cu

namespace {

struct Kernel {
  // result + up to six memref descriptors (a kernel with fewer arguments ignores the extra ones: x86-64 C calling convention)
  void (*fn)(void *, void *, void *, void *, void *, void *, void *) = nullptr;
  void (*set_stream)(void *) = nullptr;
  int64_t Mp = 0, Kp = 0, Np = 0;
  bool hot = false; // K and N need no padding
};

using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int, std::string, bool, int>;
struct KeyHash {
  size_t operator()(const Key &k) const {
    size_t h = std::hash<std::string>()(std::get<5>(k));
    auto mix = [&](int64_t v) { h = h * 1000003u ^ std::hash<int64_t>()(v); };
    mix(std::get<0>(k)); mix(std::get<1>(k)); mix(std::get<2>(k));
    mix(std::get<3>(k)); mix(std::get<4>(k)); mix(std::get<6>(k) ? 1 : 0);
    mix(std::get<7>(k));
    return h;
  }
};

std::mutex g_mutex;

// Zero-padded activation buffers for small row counts (decode / skinny buckets). An M-row call
// copies its rows into the front of a reusable [Mp, K] buffer: one copy kernel instead of the
// allocate + fill + copy that at::constant_pad_nd launches. The rows past M keep whatever an earlier
// call left there: a GEMM row only depends on its own activation row, so they can only change the
// output rows that are dropped (the buffer starts zeroed, so it never holds NaN / garbage memory).
// One buffer per (device, stream, Mp, K): calls on different streams never share one.
using ScratchKey = std::tuple<int, void *, int64_t, int64_t>;
std::map<ScratchKey, at::Tensor> g_scratch;
constexpr int64_t kScratchMaxRows = 64;

at::Tensor padded_activations(const at::Tensor &x2, int64_t Mp) {
  const int64_t M = x2.size(0), K = x2.size(1);
  if (Mp > kScratchMaxRows)
    return at::constant_pad_nd(x2, {0, 0, 0, Mp - M});
  ScratchKey key{x2.device().index(),
                 c10::cuda::getCurrentCUDAStream(x2.device().index()).stream(), Mp, K};
  at::Tensor buf;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_scratch.find(key);
    if (it == g_scratch.end())
      it = g_scratch.emplace(key, at::zeros({Mp, K}, x2.options())).first;
    buf = it->second;
  }
  buf.narrow(0, 0, M).copy_(x2);
  return buf;
}
std::unordered_map<Key, Kernel, KeyHash> g_kernels;
py::object *g_builder = nullptr;  // (M, K, N, group, device, activation) -> (fn, set_stream, Mp, Kp, Np)
py::object *g_fallback = nullptr; // the Python implementation

struct Desc2 { // MLIR memref descriptor, rank 2
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

struct Desc1 { // MLIR memref descriptor, rank 1
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

inline Desc1 describe1(const at::Tensor &t) { return {t.data_ptr(), t.data_ptr(), 0, t.size(0), t.stride(0)}; }

inline Desc2 describe(const at::Tensor &t) {
  return {t.data_ptr(), t.data_ptr(), 0, {t.size(0), t.size(1)}, {t.stride(0), t.stride(1)}};
}

inline bool aligned16(const at::Tensor &t) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0; }

inline bool cpp_no_atomic() {
  static const bool off = std::getenv("QUANTFORGE_NO_ATOMIC") != nullptr;
  return off;
}

inline bool cpp_no_f16store() {
  static const bool off = std::getenv("QUANTFORGE_NO_F16STORE") != nullptr;
  return off;
}

Kernel lookup(int64_t M, int64_t K, int64_t N, int64_t group, int device, const std::string &act,
              bool overwrite, int bits) {
  Key key{M, K, N, group, device, act, overwrite, bits};
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(key);
    if (it != g_kernels.end())
      return it->second;
  }
  py::gil_scoped_acquire gil; // first use of this shape: compile / load it in Python
  py::tuple r = (*g_builder)(M, K, N, group, device, act, overwrite, bits);
  Kernel k;
  k.fn = reinterpret_cast<decltype(k.fn)>(r[0].cast<uintptr_t>());
  k.set_stream = reinterpret_cast<decltype(k.set_stream)>(r[1].cast<uintptr_t>());
  k.Mp = r[2].cast<int64_t>();
  k.Kp = r[3].cast<int64_t>();
  k.Np = r[4].cast<int64_t>();
  k.hot = group > 0 && k.Kp == K && k.Np == N;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernels[key] = k;
  return k;
}

at::Tensor call_fallback(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &scale,
                         const at::Tensor &zero, int64_t group, const c10::optional<at::Tensor> &bias,
                         const std::string &act) {
  py::gil_scoped_acquire gil;
  py::object out = (*g_fallback)(x, packed, scale, zero, group, bias, act);
  return out.cast<at::Tensor>();
}

// `dst`: where the f16-store kernel may write its result directly when the row count needs no padding (the chunk loop of `w4a16_gemm` passes the slice of
// its output; without it every chunk of a long prefill was computed into a fresh buffer and then copied: 73 MB per chunk for N = 17920, ~12 % of a prefill).
at::Tensor w4a16_gemm_rows(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &scale,
                           const at::Tensor &zero, int64_t group_size, const c10::optional<at::Tensor> &bias,
                           std::string activation, const c10::optional<at::Tensor> &dst = c10::nullopt) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "x must be a float16 CUDA tensor");
  TORCH_CHECK(x.dim() >= 1 && packed.dim() == 2, "bad shapes");
  // INT4: packed [K, N/2]. INT8 (group-wise only): plain int8 [K, N], i.e. as wide as `scale`.
  TORCH_CHECK(group_size == 0 || scale.dim() == 2, "group-wise scale must be [K/G, N]");
  const int64_t N = group_size > 0 ? scale.size(1) : packed.size(1) * 2;
  // weight kind: 4 = INT4, 8 = int8, 9 = FP8 E4M3 (the kernels read FP8 as raw int8 bytes)
  const bool fp8 = packed.scalar_type() == at::kFloat8_e4m3fn;
  const int bits = fp8 ? 9 : ((group_size > 0 && packed.size(1) == N) ? 8 : 4);
  TORCH_CHECK(!fp8 || group_size > 0, "FP8 weights need group-wise scales");
  const int64_t K = x.size(-1);
  const int64_t M = x.numel() / K;
  c10::cuda::CUDAGuard guard(x.device());
  const bool swiglu = activation == "swiglu";
  TORCH_CHECK(!(swiglu && bias.has_value()), "activation='swiglu' does not take a bias");
  TORCH_CHECK(!(swiglu && bits != 4), "activation='swiglu' needs an INT4 weight");
  // Plain GEMM (no activation): the kernel overwrites an f32 accumulator and ONE fused kernel (kfold_reduce, S = 1) adds the bias and
  // writes f16 -- instead of expanding the bias into the accumulator (2 torch kernels) and converting afterwards (1 more).
  const bool post = activation == "none" && !swiglu;
  const bool overwrite = swiglu || post || !bias.has_value(); // the kernel need not read C
  const int64_t cols = swiglu ? N / 2 : N;
  Kernel k = lookup(M, K, N, group_size, x.device().index(), activation, overwrite, bits);

  at::Tensor x2 = x.reshape({M, K}).contiguous();
  const at::Tensor pk = fp8 ? packed.view(at::kChar) : packed;
  if (!k.hot || !aligned16(x2) || !aligned16(pk) || !aligned16(scale) || !aligned16(zero) ||
      !pk.is_contiguous() || !scale.is_contiguous() || !zero.is_contiguous())
    return call_fallback(x, packed, scale, zero, group_size, bias, activation);

  // Long-K narrow-N layers (down_proj, o_proj), 1..16 rows: K split over blocks that add their partial results atomically to a zeroed f32 buffer
  // (atomic_plan.py decides per shape; not hot = keep the plain kernel), then one kernel converts to f16 (+ bias). No weight copy, any row count.
  if (post && bits == 4 && group_size > 0 && M <= 64 && !cpp_no_atomic()) {
    Kernel ka = lookup(M, K, N, group_size, x.device().index(), "atomic", true, 4);
    if (ka.hot && ka.fn) {
      at::Tensor act = (ka.Mp == M) ? x2 : padded_activations(x2, ka.Mp);
      at::Tensor acc = at::zeros({std::max<int64_t>(ka.Mp, 16), N}, x.options().dtype(at::kFloat));   // 16 rows for the decode tile, else the bucket
      Desc2 a0 = describe(act), a1 = describe(pk), a2 = describe(scale), a3 = describe(zero), a4 = describe(acc), resa{};
      ka.set_stream(c10::cuda::getCurrentCUDAStream(x.device().index()).stream());
      ka.fn(&resa, &a0, &a1, &a2, &a3, &a4, nullptr);
      c10::optional<at::Tensor> ba;
      if (bias.has_value())
        ba = bias->scalar_type() == at::kHalf ? *bias : bias->to(at::kHalf);
      std::vector<int64_t> shapea(x.sizes().begin(), x.sizes().end());
      shapea.back() = N;
      return kfold_reduce_cuda(acc, M, 1, N, ba).reshape(shapea);
    }
  }

  // INT4, group-wise, no activation: the kernel with the f16-store epilogue writes the f16 result (+ bias) itself, one kernel, no f32 buffer.
  // `lookup` returns hot = false when that kernel cannot be built for the shape (then the f32 path below runs).
  if (post && bits == 4 && group_size > 0 && !cpp_no_f16store()) {
    Kernel k16 = lookup(M, K, N, group_size, x.device().index(), bias.has_value() ? "f16sb" : "f16s", true, 4);
    at::Tensor bh;
    if (bias.has_value())
      bh = (bias->scalar_type() == at::kHalf ? *bias : bias->to(at::kHalf)).contiguous();
    if (k16.hot && k16.fn && (!bias.has_value() || aligned16(bh))) {
      at::Tensor act16 = (k16.Mp == M) ? x2 : padded_activations(x2, k16.Mp);
      at::Tensor out16 = (dst.has_value() && k16.Mp == M && dst->is_contiguous() && dst->size(0) == M && dst->size(1) == N) ? *dst
                                                                                                                         : at::empty({k16.Mp, N}, x.options());
      Desc2 e0 = describe(act16), e1 = describe(pk), e2 = describe(scale), e3 = describe(zero), eo = describe(out16), res16{};
      Desc1 eb{};
      if (bias.has_value())
        eb = describe1(bh);
      k16.set_stream(c10::cuda::getCurrentCUDAStream(x.device().index()).stream());
      if (bias.has_value())
        k16.fn(&res16, &e0, &e1, &e2, &e3, &eb, &eo);
      else
        k16.fn(&res16, &e0, &e1, &e2, &e3, &eo, nullptr);
      std::vector<int64_t> shape16(x.sizes().begin(), x.sizes().end());
      shape16.back() = N;
      return out16.narrow(0, 0, M).reshape(shape16);
    }
  }

  at::Tensor act = (k.Mp == M) ? x2 : padded_activations(x2, k.Mp);
  at::Tensor acc = overwrite
                       ? at::empty({k.Mp, N}, x.options().dtype(at::kFloat))
                       : bias->to(at::kFloat).expand({k.Mp, N}).contiguous();
  Desc2 d0 = describe(act), d1 = describe(pk), d2 = describe(scale), d3 = describe(zero),
        d4 = describe(acc), result{};
  k.set_stream(c10::cuda::getCurrentCUDAStream(x.device().index()).stream());
  k.fn(&result, &d0, &d1, &d2, &d3, &d4, nullptr);

  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
  shape.back() = cols;
  if (post) {
    c10::optional<at::Tensor> b16;
    if (bias.has_value())
      b16 = bias->scalar_type() == at::kHalf ? *bias : bias->to(at::kHalf);
    return kfold_reduce_cuda(acc, M, 1, N, b16).reshape(shape);
  }
  return acc.narrow(0, 0, M).narrow(1, 0, cols).to(x.scalar_type()).reshape(shape);
}

// The kernels accumulate in f32 and the result is converted to f16 afterwards: for a big prefill batch (vLLM profiles with
// 8192+ tokens) that f32 buffer alone is M x N x 4 bytes (720 MB for N = 22016). Long inputs are therefore processed in
// chunks of kMaxRows rows written straight into the f16 result, so the temporary stays bounded.
constexpr int64_t kMaxRows = 2048;

at::Tensor w4a16_gemm(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &scale,
                      const at::Tensor &zero, int64_t group_size, const c10::optional<at::Tensor> &bias,
                      std::string activation) {
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "bad shapes");
  const int64_t K = x.size(-1);
  const int64_t M = x.numel() / K;
  if (M <= kMaxRows)
    return w4a16_gemm_rows(x, packed, scale, zero, group_size, bias, activation);
  const int64_t N = group_size > 0 ? scale.size(1) : packed.size(1) * 2;
  const int64_t cols = activation == "swiglu" ? N / 2 : N;
  at::Tensor x2 = x.reshape({M, K});
  at::Tensor out = at::empty({M, cols}, x.options());
  for (int64_t r = 0; r < M; r += kMaxRows) {
    const int64_t n = std::min<int64_t>(kMaxRows, M - r);
    at::Tensor dst = out.narrow(0, r, n);
    at::Tensor res = w4a16_gemm_rows(x2.narrow(0, r, n), packed, scale, zero, group_size, bias, activation, dst);
    if (res.data_ptr() != dst.data_ptr())    // not written in place (a padded or non-f16-store chunk): copy it
      dst.copy_(res);
  }
  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
  shape.back() = cols;
  return out.reshape(shape);
}

// K-folded GEMM (kfold.py): S slices of K laid side by side as columns, used while M * S <= 16 rows. The generated kernel
// writes an f32 accumulator [rows, S*N]; one small kernel adds the S diagonal blocks (+ bias) and writes f16. For more
// rows, or without the folded weights, it is the plain op.
at::Tensor w4a16_gemm_kfold(const at::Tensor &x, const at::Tensor &packed, const at::Tensor &scale,
                            const at::Tensor &zero, const at::Tensor &packed_f, const at::Tensor &scale_f,
                            const at::Tensor &zero_f, int64_t group_size, const c10::optional<at::Tensor> &bias) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() >= 1, "x must be a float16 CUDA tensor");
  const int64_t K = x.size(-1);
  const int64_t M = x.numel() / K;
  const int64_t S = scale_f.numel() ? scale_f.size(1) / scale.size(1) : 1;
  if (S <= 1 || M * S > 16)
    return w4a16_gemm(x, packed, scale, zero, group_size, bias, "none");
  const int64_t N = scale.size(1), Nf = S * N, Kf = K / S;
  const int bits = (packed_f.size(1) == Nf) ? 8 : 4;
  c10::cuda::CUDAGuard guard(x.device());
  // "foldvr": exactly M * S rows (cp.async zero-fills the rest of the 16-row tile); the accumulator then has 16 rows (the kernel stores them all)
  Kernel k = lookup(M * S, Kf, Nf, group_size, x.device().index(), "foldvr", /*overwrite=*/true, bits);
  at::Tensor x2 = x.reshape({M * S, Kf}).contiguous();
  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
  shape.back() = N;
  if (!k.hot || !aligned16(x2) || !aligned16(packed_f) || !aligned16(scale_f) || !aligned16(zero_f) ||
      !packed_f.is_contiguous() || !scale_f.is_contiguous() || !zero_f.is_contiguous()) {
    // not on the hot path: the plain ops (same result, a few more kernels)
    at::Tensor yf = w4a16_gemm(x2, packed_f, scale_f, zero_f, group_size, c10::nullopt, "none");
    at::Tensor y = yf.view({M, S, S, N}).diagonal(0, 1, 2).sum(at::IntArrayRef({-1}), false, at::kFloat).to(at::kHalf);
    if (bias.has_value())
      y = y + *bias;
    return y.reshape(shape);
  }
  at::Tensor act = (k.Mp == M * S) ? x2 : padded_activations(x2, k.Mp);
  at::Tensor acc = at::empty({std::max<int64_t>(k.Mp, 16), Nf}, x.options().dtype(at::kFloat));
  Desc2 d0 = describe(act), d1 = describe(packed_f), d2 = describe(scale_f), d3 = describe(zero_f),
        d4 = describe(acc), result{};
  k.set_stream(c10::cuda::getCurrentCUDAStream(x.device().index()).stream());
  k.fn(&result, &d0, &d1, &d2, &d3, &d4, nullptr);
  return kfold_reduce_cuda(acc, M, S, N, bias).reshape(shape);
}

void set_python_hooks(py::object builder, py::object fallback) {
  delete g_builder;
  delete g_fallback;
  g_builder = new py::object(std::move(builder));   // intentionally leaked at exit
  g_fallback = new py::object(std::move(fallback));
}

void clear_kernels() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernels.clear();
  g_scratch.clear();
}

} // namespace

TORCH_LIBRARY(quantforge, m) {
  m.def("w4a16_gemm(Tensor x, Tensor packed, Tensor scale, Tensor zero_point, int group_size, "
        "Tensor? bias=None, str activation=\"none\") -> Tensor");
}

TORCH_LIBRARY_FRAGMENT(quantforge, m) {
  m.def("w4a16_gemm_kfold(Tensor x, Tensor packed, Tensor scale, Tensor zero_point, Tensor packed_f, Tensor scale_f, "
        "Tensor zero_f, int group_size, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantforge, CUDA, m) {
  m.impl("w4a16_gemm", &w4a16_gemm);
  m.impl("w4a16_gemm_kfold", &w4a16_gemm_kfold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_python_hooks", &set_python_hooks);
  m.def("clear_kernels", &clear_kernels);
  m.attr("HAS_KFOLD") = true;
}
