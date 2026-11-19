// RUN: quantforge-opt --quantforge-gpu-pipeline %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_89 features=+ptx80' %s | FileCheck --check-prefix=SM89 %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='fuse-dequant=false' %s | FileCheck --check-prefix=UNFUSED %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='tensor-core=false' %s | FileCheck --check-prefix=FMA %s

// =====================================================================
// End-to-end: qf IR -> PTX. Checks the structural contract of the pipeline:
//   * default: INT4 dequant is fused into the GEMM's shared-memory staging, so
//     there is exactly ONE gpu.binary and no device buffer for f16 weights
//   * the kernel stages tiles through shared memory and uses Tensor Core
//     mma.sync + ldmatrix (tensor-core=false falls back to FMA code)
//   * fuse-dequant=false keeps the dequantization as a separate kernel
//   * host function keeps its name and gets a C-interface wrapper
//   * no qf./tensor./linalg./vector. op survives
// (Numerical correctness is covered by test/E2E/test_w4a16_e2e.py.)
// =====================================================================

// CHECK-NOT:  qf.unpack
// CHECK-NOT:  qf.dequant
// CHECK-NOT:  linalg.
// CHECK-NOT:  tensor.
// CHECK-NOT:  vector.
// CHECK-NOT:  mgpuMemAlloc
// CHECK-DAG:  llvm.func @w4a16_gemm(
// CHECK-DAG:  llvm.func @_mlir_ciface_w4a16_gemm(
// CHECK-DAG:  gpu.launch_func
// CHECK-DAG:  gpu.binary @w4a16_gemm_kernel
// CHECK-NOT:  gpu.binary @w4a16_gemm_kernel_0
// CHECK-DAG:  .visible .entry w4a16_gemm_kernel
// CHECK-DAG:  .target sm_80
// CHECK-DAG:  .shared
// CHECK-DAG:  ldmatrix
// CHECK-DAG:  mma.sync.aligned.m16n8k16
// CHECK-DAG:  cp.async.ca.shared.global
// CHECK-DAG:  st.shared.v4.b32
// CHECK-NOT:  div.u
// SM89-DAG:   .target sm_89
// UNFUSED-DAG: llvm.call @mgpuMemAlloc
// UNFUSED-DAG: llvm.call @mgpuMemFree
// UNFUSED-DAG: gpu.binary @w4a16_gemm_kernel
// UNFUSED-DAG: gpu.binary @w4a16_gemm_kernel_0
// FMA-DAG:    gpu.binary @w4a16_gemm_kernel
// FMA-DAG:    .shared
// FMA-NOT:    mma.sync
// FMA-NOT:    ldmatrix
func.func @w4a16_gemm(%act: tensor<128x128xf16>, %wp: tensor<128x64xi8>,
                      %scale: tensor<f16>, %zp: tensor<i8>,
                      %c: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %w = qf.unpack %wp : tensor<128x64xi8> -> tensor<128x128xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<128x128xi8>, tensor<f16>, tensor<i8> -> tensor<128x128xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<128x128xf16>, tensor<128x128xf16>) outs(%c : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %r : tensor<128x128xf16>
}
