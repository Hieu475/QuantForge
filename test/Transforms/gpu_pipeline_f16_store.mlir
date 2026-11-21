// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1 epilogue=f16-store' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1 accumulate=false' %s | FileCheck --check-prefix=PLAIN %s

// INT4 W4A16 GEMM whose f32 accumulator is converted to f16 (+ an f16 bias) by the kernel itself (epilogue=f16-store): the elementwise kernel
// and the f32 buffer are gone, the pairs of results are stored as one 4-byte f16 store. Without the option the same program is two kernels
// (GEMM + elementwise) and keeps its f32 buffer.

// CHECK-DAG:  gpu.binary @w4a16_f16_kernel
// CHECK-NOT:  gpu.binary @w4a16_f16_kernel_0
// CHECK-DAG:  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// CHECK-DAG:  cvt.f32.f16
// CHECK-DAG:  cvt.rn.f16.f32
// CHECK-DAG:  st.global.b32
// CHECK-NOT:  mgpuMemAlloc
// PLAIN-DAG:  gpu.binary @w4a16_f16_kernel
// PLAIN-DAG:  gpu.binary @w4a16_f16_kernel_0
// PLAIN-DAG:  mgpuMemAlloc
#mmn = affine_map<(m, n) -> (m, n)>
#mn = affine_map<(m, n) -> (n)>
func.func @w4a16_f16(%act: tensor<16x512xf16>, %wp: tensor<512x128xi8>, %scale: tensor<4x256xf16>, %zp: tensor<4x256xi8>,
                     %bias: tensor<256xf16> {quantforge.int8_bias}, %y: tensor<16x256xf16> {quantforge.int8_out}) -> tensor<16x256xf16> {
  %w = qf.unpack %wp : tensor<512x128xi8> -> tensor<512x256xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<512x256xi8>, tensor<4x256xf16>, tensor<4x256xi8> -> tensor<512x256xf16>
  %e = tensor.empty() : tensor<16x256xf32>
  %c = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x256xf16>) outs(%e : tensor<16x256xf32>) -> tensor<16x256xf32>
  %r = linalg.generic {indexing_maps = [#mmn, #mn, #mmn], iterator_types = ["parallel", "parallel"], quantforge.no_tile, quantforge.no_vectorize}
      ins(%c, %bias : tensor<16x256xf32>, tensor<256xf16>) outs(%y : tensor<16x256xf16>) {
    ^bb0(%ci: f32, %b: f16, %o: f16):
      %bf = arith.extf %b : f16 to f32
      %v = arith.addf %ci, %bf : f32
      %h = arith.truncf %v : f32 to f16
      linalg.yield %h : f16
  } -> tensor<16x256xf16>
  return %r : tensor<16x256xf16>
}
