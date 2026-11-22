// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true block-tile-n=64 warp-tile-n=16 smem-stages=1 split-k=4 epilogue=atomic-add' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true block-tile-n=64 warp-tile-n=16 smem-stages=1 split-k=4 epilogue=atomic-add valid-rows=3' %s | FileCheck --check-prefix=ROWS %s

// epilogue=atomic-add (with split-k=S): the K reduction is cut into S slices, one grid axis, and every block adds its [16 x 64] partial result with
// red.global.add.f32 to the f32 output argument (tagged quantforge.int8_out; the caller initialises it). The partial buffer P[S, M, N], its device
// allocation and the kernel that adds the slices are gone: ONE kernel, nothing to allocate per call (so it can be captured in a CUDA graph).

// CHECK:     gpu.binary @w4a16_gemm_kernel
// CHECK-NOT: gpu.binary @w4a16_gemm_kernel_0
// CHECK-NOT: gpu.alloc
// CHECK:     red.global.add.f32
// ROWS:      red.global.add.f32
func.func @w4a16_gemm(%act: tensor<16x1024xf16>, %wp: tensor<1024x128xi8>,
                      %scale: tensor<8x256xf16>, %zp: tensor<8x256xi8>,
                      %c: tensor<16x256xf32> {quantforge.int8_out}) -> tensor<16x256xf32> {
  %w = qf.unpack %wp : tensor<1024x128xi8> -> tensor<1024x256xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<1024x256xi8>, tensor<8x256xf16>, tensor<8x256xi8> -> tensor<1024x256xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x1024xf16>, tensor<1024x256xf16>) outs(%c : tensor<16x256xf32>) -> tensor<16x256xf32>
  return %r : tensor<16x256xf32>
}
