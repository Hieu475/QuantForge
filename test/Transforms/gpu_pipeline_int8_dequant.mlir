// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 block-tile-m=128 block-tile-n=128 block-tile-k=64 warp-tile-m=64 warp-tile-n=64 smem-stages=1 thread-tile-k=32 smem-pad=16 accumulate=false epilogue=int8-dequant' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 block-tile-m=128 block-tile-n=128 block-tile-k=64 warp-tile-m=64 warp-tile-n=64 smem-stages=1 thread-tile-k=32 smem-pad=16 accumulate=false' %s | FileCheck --check-prefix=PLAIN %s

// INT8 GEMM with the dequantisation fused into the store (epilogue=int8-dequant): the program is `matmul (i32)` followed by an
// elementwise `y = f16(c * sx[m] * sw[n] + bias[n])`. The pass folds the elementwise kernel into the GEMM's final stores: ONE kernel,
// the int32 intermediate buffer is gone (no device allocation left), the result is stored as f16 pairs with one 4-byte store.
// Without the epilogue option the same program is two kernels (GEMM + elementwise) and keeps its int32 buffer.

// CHECK-DAG:  gpu.binary @i8_gemm_dq_kernel
// CHECK-NOT:  gpu.binary @i8_gemm_dq_kernel_0
// CHECK-DAG:  mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32
// CHECK-DAG:  cvt.rn.f32.s32
// CHECK-DAG:  cvt.rn.f16.f32
// CHECK-DAG:  st.global.b32
// CHECK-NOT:  mgpuMemAlloc
// PLAIN-DAG:  gpu.binary @i8_gemm_dq_kernel
// PLAIN-DAG:  gpu.binary @i8_gemm_dq_kernel_0
// PLAIN-DAG:  mgpuMemAlloc
#mmn = affine_map<(m, n) -> (m, n)>
#mm = affine_map<(m, n) -> (m)>
#mn = affine_map<(m, n) -> (n)>
func.func @i8_gemm_dq(%a: tensor<256x512xi8>, %bt: tensor<256x512xi8>,
                      %sx: tensor<256xf32> {quantforge.int8_row_scale}, %sw: tensor<256xf32> {quantforge.int8_col_scale},
                      %bias: tensor<256xf32> {quantforge.int8_bias}, %y: tensor<256x256xf16> {quantforge.int8_out}) -> tensor<256x256xf16> {
  %e = tensor.empty() : tensor<256x256xi32>
  %c = linalg.matmul_transpose_b ins(%a, %bt : tensor<256x512xi8>, tensor<256x512xi8>) outs(%e : tensor<256x256xi32>) -> tensor<256x256xi32>
  %r = linalg.generic {indexing_maps = [#mmn, #mm, #mn, #mn, #mmn], iterator_types = ["parallel", "parallel"], quantforge.no_tile, quantforge.no_vectorize}
      ins(%c, %sx, %sw, %bias : tensor<256x256xi32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%y : tensor<256x256xf16>) {
    ^bb0(%ci: i32, %s: f32, %w: f32, %b: f32, %o: f16):
      %f = arith.sitofp %ci : i32 to f32
      %t = arith.mulf %f, %s : f32
      %u = arith.mulf %t, %w : f32
      %v = arith.addf %u, %b : f32
      %h = arith.truncf %v : f32 to f16
      linalg.yield %h : f16
  } -> tensor<256x256xf16>
  return %r : tensor<256x256xf16>
}
