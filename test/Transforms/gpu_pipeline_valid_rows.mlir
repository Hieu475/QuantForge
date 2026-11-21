// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=32 warp-tile-n=8 block-tile-k=128 smem-stages=2 epilogue=f16-store valid-rows=3' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=32 warp-tile-n=8 block-tile-k=128 smem-stages=2 epilogue=f16-store' %s | FileCheck --check-prefix=FULL %s
// RUN: not quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1 epilogue=f16-store valid-rows=3' %s 2>&1 | FileCheck --check-prefix=ERR %s

// valid-rows=R: the kernel keeps its 16-row tile but the caller passes an [R, K] activation tensor and gets [R, N] back. The cp.async copies of the
// activation tile carry a src-size operand (rows >= R read nothing and are zero-filled) and the stores of those rows are guarded; without the
// option the copies are plain 16-byte ones. A copy that is not cp.async cannot honour it, so the pass refuses instead of reading past the tensor.

// CHECK: cp.async.ca.shared.global [{{.*}}], [{{.*}}], 16, %r{{[0-9]+}};
// CHECK: st.global.b32
// FULL: cp.async.ca.shared.global [{{.*}}], [{{.*}}], 16;
// ERR: error: quantforge-smem-promotion: valid-rows needs the cp.async copy
#mmn = affine_map<(m, n) -> (m, n)>
#mn = affine_map<(m, n) -> (n)>
func.func @w4a16_f16(%act: tensor<16x896xf16>, %wp: tensor<896x576xi8>, %scale: tensor<7x1152xf16>, %zp: tensor<7x1152xi8>,
                     %bias: tensor<1152xf16> {quantforge.int8_bias}, %y: tensor<16x1152xf16> {quantforge.int8_out}) -> tensor<16x1152xf16> {
  %w = qf.unpack %wp : tensor<896x576xi8> -> tensor<896x1152xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<896x1152xi8>, tensor<7x1152xf16>, tensor<7x1152xi8> -> tensor<896x1152xf16>
  %e = tensor.empty() : tensor<16x1152xf32>
  %c = linalg.matmul ins(%act, %wf : tensor<16x896xf16>, tensor<896x1152xf16>) outs(%e : tensor<16x1152xf32>) -> tensor<16x1152xf32>
  %r = linalg.generic {indexing_maps = [#mmn, #mn, #mmn], iterator_types = ["parallel", "parallel"], quantforge.no_tile, quantforge.no_vectorize}
      ins(%c, %bias : tensor<16x1152xf32>, tensor<1152xf16>) outs(%y : tensor<16x1152xf16>) {
    ^bb0(%ci: f32, %b: f16, %o: f16):
      %bf = arith.extf %b : f16 to f32
      %v = arith.addf %ci, %bf : f32
      %h = arith.truncf %v : f32 to f16
      linalg.yield %h : f16
  } -> tensor<16x1152xf16>
  return %r : tensor<16x1152xf16>
}
