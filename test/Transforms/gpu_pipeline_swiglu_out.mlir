// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1 epilogue=swiglu' %s | FileCheck %s
// RUN: not quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1 epilogue=swiglu' %S/Inputs/swiglu_out_f32_arg.mlir 2>&1 | FileCheck %s --check-prefix=BAD

// W4A16 GEMM over gate/up columns interleaved by pairs with the SwiGLU epilogue, whose result is stored as f16 into its own [M, N/2] argument
// (`quantforge.swiglu_out`): one kernel, the f32 accumulator buffer is never written, silu(gate) * up is truncated in the kernel and stored with a
// 2-byte store (the f32 [M, N] argument stays untouched). An argument of another element type is a compile error, not a silent miscompile.

// CHECK:      gpu.binary @w4a16_gemm_kernel
// CHECK-NOT:  gpu.binary @w4a16_gemm_kernel_0
// CHECK-DAG:  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// CHECK-DAG:  cvt.rn.f16.f32
// CHECK-DAG:  ex2.approx
// CHECK-DAG:  st.global.b16
// BAD: quantforge.swiglu_out must be a rank-2 f16 memref
func.func @w4a16_gemm(%act: tensor<16x512xf16>, %wp: tensor<512x128xi8>, %scale: tensor<4x256xf16>, %zp: tensor<4x256xi8>,
                      %c: tensor<16x256xf32>, %y: tensor<16x128xf16> {quantforge.swiglu_out}) -> tensor<16x256xf32> {
  %w = qf.unpack %wp : tensor<512x128xi8> -> tensor<512x256xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<512x256xi8>, tensor<4x256xf16>, tensor<4x256xi8> -> tensor<512x256xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x256xf16>) outs(%c : tensor<16x256xf32>) -> tensor<16x256xf32>
  return %r : tensor<16x256xf32>
}
