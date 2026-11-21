// Build and run:  nvcc -arch=sm_89 -O3 -o mma_peak benchmarks/mma_peak.cu && ./mma_peak   (use the sm_ of your GPU; the e4m3 line needs sm_89+)
// Peak throughput of the Tensor Core `mma.sync` instructions of this GPU, registers only (no memory traffic).
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define ITERS 4096
#define NACC 8

__global__ void k_f16_f32(float *out, unsigned seed) {
  float d[NACC][4] = {}; unsigned a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3, b0 = seed + 4, b1 = seed + 5;
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < NACC; ++j)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float s = 0; for (int j = 0; j < NACC; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
  if (s == 123.456f) out[0] = s;
}
__global__ void k_f16_f16(float *out, unsigned seed) {
  unsigned d[NACC][2] = {}; unsigned a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3, b0 = seed + 4, b1 = seed + 5;
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < NACC; ++j)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                   : "+r"(d[j][0]), "+r"(d[j][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  unsigned s = 0; for (int j = 0; j < NACC; ++j) s ^= d[j][0] ^ d[j][1];
  if (s == 123456u) out[0] = s;
}
__global__ void k_s8(float *out, unsigned seed) {
  int d[NACC][4] = {}; unsigned a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3, b0 = seed + 4, b1 = seed + 5;
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < NACC; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+r"(d[j][0]), "+r"(d[j][1]), "+r"(d[j][2]), "+r"(d[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  int s = 0; for (int j = 0; j < NACC; ++j) s ^= d[j][0] ^ d[j][1] ^ d[j][2] ^ d[j][3];
  if (s == 123456) out[0] = s;
}
__global__ void k_e4m3(float *out, unsigned seed) {
  float d[NACC][4] = {}; unsigned a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3, b0 = seed + 4, b1 = seed + 5;
  for (int i = 0; i < ITERS; ++i)
#pragma unroll
    for (int j = 0; j < NACC; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(d[j][0]), "+f"(d[j][1]), "+f"(d[j][2]), "+f"(d[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float s = 0; for (int j = 0; j < NACC; ++j) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
  if (s == 123.456f) out[0] = s;
}

template <typename F> double run(F kernel, double ops_per_mma, int sms, const char *name) {
  float *out; cudaMalloc(&out, 16);
  int blocks = sms * 4, threads = 128;               // 16 warps per SM
  for (int w = 0; w < 5; ++w) kernel<<<blocks, threads>>>(out, 1u);   // warm up / reach sustained clocks
  cudaDeviceSynchronize();
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  double best = 0;
  for (int r = 0; r < 5; ++r) {
    cudaEventRecord(a); kernel<<<blocks, threads>>>(out, 1u); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    double warps = (double)blocks * threads / 32, total = warps * ITERS * NACC * ops_per_mma;
    best = std::max(best, total / (ms * 1e-3) / 1e12);
  }
  printf("%-26s %7.1f T(FL)OPS\n", name, best);
  cudaFree(out); return best;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
  int sms = p.multiProcessorCount;
  double f = run(k_f16_f32, 2.0 * 16 * 8 * 16, sms, "fp16 in, fp32 accumulate");
  double h = run(k_f16_f16, 2.0 * 16 * 8 * 16, sms, "fp16 in, fp16 accumulate");
  double i = run(k_s8, 2.0 * 16 * 8 * 32, sms, "int8 in, int32 accumulate");
  double e = run(k_e4m3, 2.0 * 16 * 8 * 32, sms, "fp8 e4m3 in, fp32 accumulate");
  printf("ratios vs fp16/fp32acc: fp16acc %.2fx  int8 %.2fx  fp8 %.2fx\n", h / f, i / f, e / f);
}
