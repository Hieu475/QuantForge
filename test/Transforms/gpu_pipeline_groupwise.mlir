// RUN: quantforge-opt --quantforge-gpu-pipeline %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='fuse-dequant=false' %s | FileCheck --check-prefix=UNFUSED %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='vectorize-loads=false' %s | FileCheck --check-prefix=SCALAR %s

// RUN: quantforge-opt --quantforge-gpu-pipeline %s | FileCheck --check-prefix=NOCVT %s

// Group-wise (G=64 along K) scale / zero point through the whole pipeline:
// still ONE kernel; the per-group scale (f16) and zero point (i8) are read
// from global memory while the B tile is staged: the 8 per-element loads of a
// 16-byte work item are merged into one 16-byte scale load and one 8-byte
// zero-point load (vectorize-loads, default); without it they stay scalar.
// (Numerics: test/E2E/test_w4a16_e2e.py::test_groupwise_*.)

// The dequantisation must stay packed (f16x2 magic-number arithmetic): a scalar
// int -> f16 convert per weight means vectorize-dequant stopped matching.
// NOCVT-NOT: cvt.rn.f16

// CHECK-NOT:  qf.dequant
// CHECK-NOT:  linalg.
// CHECK-DAG:  gpu.binary @w4a16_gemm_kernel
// CHECK-NOT:  gpu.binary @w4a16_gemm_kernel_0
// CHECK-DAG:  mma.sync.aligned.m16n8k16
// CHECK-DAG:  ld.global.v4.b32
// CHECK-NOT:  ld.global.u16
// SCALAR-DAG: ld.global.u16
// SCALAR-DAG: ld.global.u8
// UNFUSED-DAG: gpu.binary @w4a16_gemm_kernel_0
func.func @w4a16_gemm(%act: tensor<128x128xf16>, %wp: tensor<128x64xi8>,
                      %scale: tensor<2x128xf16>, %zp: tensor<2x128xi8>,
                      %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %w = qf.unpack %wp : tensor<128x64xi8> -> tensor<128x128xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<128x128xi8>, tensor<2x128xf16>, tensor<2x128xi8> -> tensor<128x128xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<128x128xf16>, tensor<128x128xf16>) outs(%c : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %r : tensor<128x128xf32>
}
