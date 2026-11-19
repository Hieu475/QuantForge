// PROTOTYPE (not part of the library): see the "Small batch" section of python/quantforge/PIPELINE.md for the measurements.
// Prototype: W4A16 GEMM with B kept in registers (Marlin-style), to bound what the MLIR pipeline could reach.
//   C[M,N] (f32, zero-initialised, atomically accumulated over K slices) = A[M,K] f16 @ dequant(W)[K,N]
//   dequant(W)[k,n] = (q[k,n] - z[k/G,n]) * s[k/G,n],  G = 128
// Weights are repacked offline so that every thread loads 16 B (32 nibbles) per k16 step that are exactly
// its B fragments of 8 n8-tiles (a 64-column strip); dequantisation (lop3 + hsub2 + hmul2) goes straight
// into the mma.m16n8k16 operands: no shared memory, no barrier for B.  A is staged through shared memory
// with cp.async (STAGES deep) and read with ldmatrix.
// build: nvcc -arch=sm_89 -O3 -std=c++17 bir.cu -o bir
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

constexpr int G = 128;      // quantisation group = k chunk
#ifndef WARPS_
#define WARPS_ 4
#endif
constexpr int WARPS = WARPS_;
#ifndef MINB
#define MINB 1
#endif

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hsub2u(uint32_t a, uint32_t b) {
  uint32_t r; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
// (q - z) * s for the two nibbles selected by 0x000F000F of (w >> sh)
__device__ __forceinline__ uint32_t dq(uint32_t w, int sh, uint32_t zt, uint32_t s2) {
  uint32_t v = ((w >> sh) & 0x000F000Fu) | 0x64006400u;   // half2 (1024 + q0, 1024 + q1)
  return hmul2u(hsub2u(v, zt), s2);
}

// MT = M / 16 row tiles, STAGES = cp.async depth for A, PF = B prefetch distance in k16 steps
template <int MT, int STAGES, int PF>
__global__ void __launch_bounds__(WARPS * 32, MINB)
gemm(const __half *__restrict__ A, const uint4 *__restrict__ Bp, const uint4 *__restrict__ Sp,
     const uint2 *__restrict__ Zp, float *__restrict__ C, int K, int N, int klen) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int M = MT * 16;
  constexpr int STAGE_BYTES = M * G * 2;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int nblocks64 = N / 64;
  const int nb = blockIdx.x * WARPS + warp;          // this warp's 64-column strip
  const int k_begin = blockIdx.y * klen, nchunks = klen / G;

  float acc[MT][8][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  // ---- A staging: the block loads A[:, k0 : k0 + G] (M rows x 256 B), 16 B per cp.async
  auto stage_A = [&](int chunk, int buf) {
    uint8_t *dst = smem + buf * STAGE_BYTES;
    const __half *src = A + k_begin + chunk * G;
    for (int i = threadIdx.x; i < M * (G / 8); i += WARPS * 32) {
      int row = i / (G / 8), seg = i % (G / 8);
      cp_async16(dst + row * (G * 2) + ((seg ^ (row & 7)) * 16), src + (size_t)row * K + seg * 8);
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nchunks) stage_A(s, s);
    cp_commit();
  }

  // ---- B: ring of PF uint4 (one per k16 step), scales / zeros per chunk
  const int nsteps = nchunks * (G / 16);
  const size_t bstride = (size_t)nblocks64 * 32;     // uint4 per k16 row of blocks
  const uint4 *bptr = Bp + ((size_t)(k_begin / 16)) * bstride + (size_t)nb * 32 + lane;
  uint4 bring[PF];
#pragma unroll
  for (int p = 0; p < PF; ++p) bring[p] = (p < nsteps) ? __ldg(bptr + (size_t)p * bstride) : make_uint4(0, 0, 0, 0);

  const size_t sidx = ((size_t)(k_begin / G)) * nblocks64 * 8 + (size_t)nb * 8 + lane / 4;
  uint4 sc_next = __ldg(Sp + sidx);
  uint2 zr_next = __ldg(Zp + sidx);

  for (int c = 0; c < nchunks; ++c) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {   // issue the A copy for chunk c + STAGES - 1 (its buffer was consumed one iteration ago)
      int nxt = c + STAGES - 1;
      if (nxt < nchunks) stage_A(nxt, nxt % STAGES);
      cp_commit();
    }
    // scales of this chunk -> half2 (s, s) and (1024 + z) per column of the thread
    uint32_t s2[8], zt[8];
    {
      uint32_t sw[4] = {sc_next.x, sc_next.y, sc_next.z, sc_next.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t half_bits = (sw[j / 2] >> ((j & 1) * 16)) & 0xFFFFu;
        s2[j] = half_bits | (half_bits << 16);
      }
      uint32_t zw[2] = {zr_next.x, zr_next.y};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t z = (zw[j / 4] >> ((j % 4) * 8)) & 0xFFu;
        uint32_t h = 0x6400u | z;
        zt[j] = h | (h << 16);
      }
    }
    if (c + 1 < nchunks) {
      sc_next = __ldg(Sp + sidx + (size_t)(c + 1) * nblocks64 * 8);
      zr_next = __ldg(Zp + sidx + (size_t)(c + 1) * nblocks64 * 8);
    }
    const uint8_t *abuf = smem + (c % STAGES) * STAGE_BYTES;

#pragma unroll
    for (int ks = 0; ks < G / 16; ++ks) {
      const int step = c * (G / 16) + ks;
      uint4 b = bring[ks % PF];                       // PF divides G/16 = 8 -> static index after unroll
      if (step + PF < nsteps) bring[ks % PF] = __ldg(bptr + (size_t)(step + PF) * bstride);
      uint32_t w[4] = {b.x, b.y, b.z, b.w};
      uint32_t bf[8][2];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        bf[2 * t][0] = dq(w[t], 0, zt[2 * t], s2[2 * t]);
        bf[2 * t][1] = dq(w[t], 4, zt[2 * t], s2[2 * t]);
        bf[2 * t + 1][0] = dq(w[t], 8, zt[2 * t + 1], s2[2 * t + 1]);
        bf[2 * t + 1][1] = dq(w[t], 12, zt[2 * t + 1], s2[2 * t + 1]);
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        uint32_t a[4];
        int row = mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int seg = ks * 2 + (lane >> 4);
        ldsm4(a[0], a[1], a[2], a[3], abuf + row * (G * 2) + ((seg ^ (row & 7)) * 16));
#pragma unroll
        for (int j = 0; j < 8; ++j) mma16816(acc[mt][j], a, bf[j][0], bf[j][1]);
      }
    }
  }
  cp_wait<0>();

  // ---- C += acc  (K slices accumulate atomically)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int row = mt * 16 + lane / 4, col = nb * 64 + j * 8 + (lane % 4) * 2;
      atomicAdd(&C[(size_t)row * N + col], acc[mt][j][0]);
      atomicAdd(&C[(size_t)row * N + col + 1], acc[mt][j][1]);
      atomicAdd(&C[(size_t)(row + 8) * N + col], acc[mt][j][2]);
      atomicAdd(&C[(size_t)(row + 8) * N + col + 1], acc[mt][j][3]);
    }
}

// reference: C = A @ dequant(W) with the original (unpacked) layout, f32
__global__ void ref_gemm(const __half *A, const uint8_t *q, const __half *S, const uint8_t *Z, float *C,
                         int M, int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    float w = ((float)q[(size_t)k * N + n] - (float)Z[(size_t)(k / G) * N + n]) * __half2float(S[(size_t)(k / G) * N + n]);
    acc += __half2float(A[(size_t)m * K + k]) * __half2float(__float2half(w));   // weights as f16, like the kernel
  }
  C[(size_t)m * N + n] = acc;
}

__global__ void flush_kernel(const float4 *p, size_t n, float *sink) {
  float s = 0; for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += p[i].x;
  if (s == 123.456f) *sink = s;
}

template <int MT, int STAGES, int PF>
float run(const __half *dA, const uint4 *dB, const uint4 *dS, const uint2 *dZ, float *dC, int K, int N, int KS,
          float *flushbuf, size_t flushn, float *sink, bool verify, const float *dref, int reps = 30) {
  constexpr int M = MT * 16;
  size_t smem = (size_t)STAGES * M * G * 2;
  auto kern = gemm<MT, STAGES, PF>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  dim3 grid(N / (64 * WARPS), KS), block(WARPS * 32);
  int klen = K / KS;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float best = 1e9;
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(dC, 0, (size_t)M * N * 4));
    flush_kernel<<<80, 256>>>((const float4 *)flushbuf, flushn / 4, sink);
    cudaEventRecord(e0);
    kern<<<grid, block, smem>>>(dA, dB, dS, dZ, dC, K, N, klen);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    if (r >= 3) best = std::min(best, ms);
  }
  CK(cudaGetLastError());
  if (verify) {
    std::vector<float> h((size_t)M * N), hr((size_t)M * N);
    CK(cudaMemcpy(h.data(), dC, h.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(), dref, hr.size() * 4, cudaMemcpyDeviceToHost));
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < h.size(); ++i) { maxerr = std::max(maxerr, (double)fabsf(h[i] - hr[i])); maxref = std::max(maxref, (double)fabsf(hr[i])); }
    printf("    verify: max|err| = %.5f (max|ref| = %.2f)%s\n", maxerr, maxref, maxerr < 1e-3 * maxref + 1e-2 ? "  OK" : "  MISMATCH");
  }
  return best * 1000.f;   // us
}

template <int MT>
void bench(int K, int N, std::mt19937 &rng) {
  constexpr int M = MT * 16;
  printf("M=%d K=%d N=%d group %d\n", M, K, N, G);
  std::vector<uint8_t> q((size_t)K * N), z((size_t)(K / G) * N);
  std::vector<__half> s((size_t)(K / G) * N), a((size_t)M * K);
  for (auto &v : q) v = rng() & 15;
  for (auto &v : z) v = rng() & 15;
  for (auto &v : s) v = __float2half(0.01f + (rng() % 1000) * 1e-4f);
  for (auto &v : a) v = __float2half(((int)(rng() % 2001) - 1000) * 1e-3f);
  // repack: Bp[kk][nb][lane] (uint4), Sp[g][nb][lane/4] (8 halves), Zp[g][nb][lane/4] (8 bytes)
  int nb64 = N / 64;
  std::vector<uint4> Bp((size_t)(K / 16) * nb64 * 32);
  for (int kk = 0; kk < K / 16; ++kk)
    for (int nb = 0; nb < nb64; ++nb)
      for (int lane = 0; lane < 32; ++lane) {
        int g = lane / 4, tq = lane % 4, k0 = kk * 16 + 2 * tq;
        uint32_t w[4];
        for (int t = 0; t < 4; ++t) {
          uint32_t word = 0;
          for (int u = 0; u < 2; ++u) {                       // tile 2t + u
            int col = nb * 64 + (2 * t + u) * 8 + g;
            uint32_t qa = q[(size_t)(k0) * N + col], qb = q[(size_t)(k0 + 1) * N + col];
            uint32_t qc = q[(size_t)(k0 + 8) * N + col], qd = q[(size_t)(k0 + 9) * N + col];
            word |= qa << (4 * (2 * u + 0)) | qc << (4 * (2 * u + 1)) | qb << (4 * (2 * u + 4)) | qd << (4 * (2 * u + 5));
          }
          w[t] = word;
        }
        Bp[((size_t)kk * nb64 + nb) * 32 + lane] = make_uint4(w[0], w[1], w[2], w[3]);
      }
  std::vector<uint4> Sp((size_t)(K / G) * nb64 * 8);
  std::vector<uint2> Zp((size_t)(K / G) * nb64 * 8);
  for (int g = 0; g < K / G; ++g)
    for (int nb = 0; nb < nb64; ++nb)
      for (int g8 = 0; g8 < 8; ++g8) {
        uint16_t sh[8]; uint8_t zz[8];
        for (int j = 0; j < 8; ++j) { int col = nb * 64 + j * 8 + g8; sh[j] = *reinterpret_cast<uint16_t *>(&s[(size_t)g * N + col]); zz[j] = z[(size_t)g * N + col]; }
        uint32_t sw[4], zw[2] = {0, 0};
        for (int i = 0; i < 4; ++i) sw[i] = sh[2 * i] | (uint32_t)sh[2 * i + 1] << 16;
        for (int j = 0; j < 8; ++j) zw[j / 4] |= (uint32_t)zz[j] << (8 * (j % 4));
        size_t idx = ((size_t)g * nb64 + nb) * 8 + g8;
        Sp[idx] = make_uint4(sw[0], sw[1], sw[2], sw[3]); Zp[idx] = make_uint2(zw[0], zw[1]);
      }
  __half *dA, *dS; uint8_t *dq, *dz; uint4 *dB, *dSp; uint2 *dZp; float *dC, *dref, *flushbuf, *sink;
  CK(cudaMalloc(&dA, a.size() * 2)); CK(cudaMemcpy(dA, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dq, q.size())); CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dz, z.size())); CK(cudaMemcpy(dz, z.data(), z.size(), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dS, s.size() * 2)); CK(cudaMemcpy(dS, s.data(), s.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dB, Bp.size() * 16)); CK(cudaMemcpy(dB, Bp.data(), Bp.size() * 16, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dSp, Sp.size() * 16)); CK(cudaMemcpy(dSp, Sp.data(), Sp.size() * 16, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dZp, Zp.size() * 8)); CK(cudaMemcpy(dZp, Zp.data(), Zp.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dC, (size_t)M * N * 4)); CK(cudaMalloc(&dref, (size_t)M * N * 4));
  size_t flushn = 256ull << 20 >> 2;
  CK(cudaMalloc(&flushbuf, flushn * 4)); CK(cudaMemset(flushbuf, 0, flushn * 4)); CK(cudaMalloc(&sink, 4));
  ref_gemm<<<dim3((N + 127) / 128, M), 128>>>(dA, dq, dS, dz, dref, M, K, N);
  CK(cudaDeviceSynchronize());
  for (int w = 0; w < 30; ++w) flush_kernel<<<80, 256>>>((const float4 *)flushbuf, flushn / 4, sink);  // GPU warm-up
  struct Cfg { int ks; const char *name; };
  for (int ks : {1, 2, 4, 8}) {
    if ((K / ks) % G || N / (64 * WARPS) * ks < 1) continue;
    float t = run<MT, 3, 4>(dA, dB, dSp, dZp, dC, K, N, ks, flushbuf, flushn, sink, ks == 4, dref);
    printf("  KS=%d stages=3 PF=4: %7.1f us   (%.1f GB/s of INT4 weights)\n", ks, t, (double)K * N / 2 / (t * 1e-6) / 1e9);
  }
  for (int ks : {4}) {
    float t2 = run<MT, 2, 4>(dA, dB, dSp, dZp, dC, K, N, ks, flushbuf, flushn, sink, false, dref);
    float t4 = run<MT, 4, 4>(dA, dB, dSp, dZp, dC, K, N, ks, flushbuf, flushn, sink, false, dref);
    float p2 = run<MT, 3, 2>(dA, dB, dSp, dZp, dC, K, N, ks, flushbuf, flushn, sink, false, dref);
    float p8 = run<MT, 3, 8>(dA, dB, dSp, dZp, dC, K, N, ks, flushbuf, flushn, sink, false, dref);
    printf("  KS=4 variants: stages2 %.1f  stages4 %.1f  PF2 %.1f  PF8 %.1f us\n", t2, t4, p2, p8);
  }
}

int main(int argc, char **argv) {
  std::mt19937 rng(1);
  int K = argc > 1 ? atoi(argv[1]) : 4096, N = argc > 2 ? atoi(argv[2]) : 4096;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s, %d SMs\n", p.name, p.multiProcessorCount);
  bench<2>(K, N, rng);
  bench<4>(K, N, rng);
  return 0;
}
