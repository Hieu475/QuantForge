// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true' %s | FileCheck %s

// FP8 E4M3 weights through the whole pipeline: one kernel, and the dequantisation (bit trick of
// lower-dequant-to-arith) is recognised by vectorize-dequant and stays packed: f16x2 multiplies,
// no 16-bit shifts / converts per weight (the scalar form has 128 `shl.b16` in this kernel).
// (Numerics: test/E2E/test_torch_op.py::test_fp8_weights_*, scripts/sanitize.sh.)

// CHECK-NOT:  qf.dequant
// CHECK-DAG:  gpu.binary @w8a16_gemm_kernel
// CHECK-NOT:  gpu.binary @w8a16_gemm_kernel_0
// CHECK-DAG:  mma.sync.aligned.m16n8k16
// CHECK-DAG:  mul.rn.f16x2
// CHECK-NOT:  shl.b16
// CHECK-NOT:  cvt.rn.f16
func.func @w8a16_gemm(%act: tensor<16x512xf16>, %wp: tensor<512x256xi8>,
                      %scale: tensor<4x256xf16>, %zp: tensor<4x256xi8>,
                      %c: tensor<16x256xf32>) -> tensor<16x256xf32> {
  
  %wf = qf.dequant %wp, %scale, %zp {fp8} : tensor<512x256xi8>, tensor<4x256xf16>, tensor<4x256xi8> -> tensor<512x256xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x256xf16>) outs(%c : tensor<16x256xf32>) -> tensor<16x256xf32>
  return %r : tensor<16x256xf32>
}
