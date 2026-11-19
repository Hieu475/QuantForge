// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true smem-stages=1 prefetch-distance=2' --mlir-print-ir-before=gpu-kernel-outlining %s 2>&1 >/dev/null | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true smem-stages=2 sink-dequant-stores=true prefetch-distance=1' --mlir-print-ir-before=gpu-kernel-outlining %s 2>&1 >/dev/null | FileCheck --check-prefix=DOUBLE %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true smem-stages=1' --mlir-print-ir-before=gpu-kernel-outlining %s 2>&1 >/dev/null | FileCheck --check-prefix=OFF %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true prefetch-distance=2' %s | FileCheck --check-prefix=PTX %s

// quantforge-prefetch-weight-loads: the weight loads (8 packed-byte vectors,
// 1 scale, 1 zero point per thread and tile) of the K loop are issued
// `distance` tiles ahead and carried through the loop as iter_args
// (2 accumulators + distance * 10 carried values). Results are bit-identical
// (test/E2E/test_w4a16_e2e.py::test_prefetch_*).

// CHECK:       scf.for {{.*}} iter_args({{.*}}) -> (vector<2x2xf32>, vector<2x2xf32>, vector<4xi8>, vector<8xf16>, vector<8xi8>, {{.*}}
// CHECK:       gpu.barrier
// DOUBLE:      scf.for {{.*}} iter_args({{.*}}) -> (vector<2x2xf32>, vector<2x2xf32>, vector<4xi8>, vector<8xf16>, vector<8xi8>, {{.*}}
// OFF:         scf.for {{.*}} iter_args({{.*}}) -> (vector<2x2xf32>, vector<2x2xf32>)
// PTX:         gpu.binary @w4a16_gemm_kernel
// PTX-NOT:     cvt.rn.f16
func.func @w4a16_gemm(%act: tensor<16x512xf16>, %wp: tensor<512x64xi8>,
                      %scale: tensor<4x128xf16>, %zp: tensor<4x128xi8>,
                      %c: tensor<16x128xf32>) -> tensor<16x128xf32> {
  %w = qf.unpack %wp : tensor<512x64xi8> -> tensor<512x128xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<512x128xi8>, tensor<4x128xf16>, tensor<4x128xi8> -> tensor<512x128xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x128xf16>) outs(%c : tensor<16x128xf32>) -> tensor<16x128xf32>
  return %r : tensor<16x128xf32>
}
