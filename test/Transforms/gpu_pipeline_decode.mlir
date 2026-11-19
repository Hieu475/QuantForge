// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true block-tile-n=128 warp-tile-n=32 smem-stages=2' %s | FileCheck --check-prefix=OVERRIDE %s

// Skinny-M (decode) schedule: one 16-row tile covers M = 16; the K tile is 128
// wide, the dequantisation stays fused (ONE kernel) and the packed weights are
// read with wide loads. Explicit options override the decode defaults.
// (Numerics + speed: test/E2E, benchmarks/compare_marlin.py.)

// CHECK-NOT:  linalg.
// CHECK-DAG:  gpu.binary @w4a16_gemm_kernel
// CHECK-NOT:  gpu.binary @w4a16_gemm_kernel_0
// CHECK-DAG:  .maxntid 32, 4, 1
// CHECK-DAG:  mma.sync.aligned.m16n8k16
// CHECK-DAG:  ld.global.u32
// OVERRIDE-DAG: .maxntid 32, 4, 1
// OVERRIDE-DAG: cp.async.wait_group
func.func @w4a16_gemm(%act: tensor<16x256xf16>, %wp: tensor<256x128xi8>,
                      %scale: tensor<f16>, %zp: tensor<i8>,
                      %c: tensor<16x256xf32>) -> tensor<16x256xf32> {
  %w = qf.unpack %wp : tensor<256x128xi8> -> tensor<256x256xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<256x256xi8>, tensor<f16>, tensor<i8> -> tensor<256x256xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x256xf16>, tensor<256x256xf16>) outs(%c : tensor<16x256xf32>) -> tensor<16x256xf32>
  return %r : tensor<16x256xf32>
}
