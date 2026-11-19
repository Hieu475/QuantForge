// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true prefetch-distance=1 smem-stages=1' %s | FileCheck --check-prefix=ACC %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true prefetch-distance=1 smem-stages=1 accumulate=false' %s | FileCheck --check-prefix=OVW %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true prefetch-distance=1 smem-stages=1 epilogue=swiglu' %s | FileCheck --check-prefix=SWIGLU %s

// accumulate=false (C = A @ W): the accumulators start from zero, so the initial f32 loads of C
// are gone -- but nothing else: the weight / scale / zero-point loads (also the ones carried through
// the K loop by prefetch-distance) must stay. epilogue=swiglu implies it and stores one value per
// (gate_j, up_j) pair: half the global stores, no ld.global.f32 either.
// (Numerics: test/E2E/test_w4a16_e2e.py::test_overwrite_output_never_reads_c / test_swiglu_*.)

// ACC:        ld.global.f32
// ACC:        ld.global.v4.b32

// OVW-NOT:    ld.global.f32
// OVW:        ld.global.v4.b32
// OVW:        ld.global.v2.u32

// SWIGLU-NOT: ld.global.f32
// SWIGLU:     ld.global.v4.b32
// SWIGLU:     ex2.approx.ftz.f32
func.func @w4a16_gemm(%act: tensor<16x512xf16>, %wp: tensor<512x64xi8>,
                      %scale: tensor<4x128xf16>, %zp: tensor<4x128xi8>,
                      %c: tensor<16x128xf32>) -> tensor<16x128xf32> {
  %w = qf.unpack %wp : tensor<512x64xi8> -> tensor<512x128xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<512x128xi8>, tensor<4x128xf16>, tensor<4x128xi8> -> tensor<512x128xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x128xf16>) outs(%c : tensor<16x128xf32>) -> tensor<16x128xf32>
  return %r : tensor<16x128xf32>
}
