// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true epilogue=silu' %s | FileCheck --check-prefix=SILU %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true epilogue=relu' %s | FileCheck --check-prefix=RELU %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true epilogue=gelu' %s | FileCheck --check-prefix=GELU %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true' %s | FileCheck --check-prefix=NONE %s

// Fused activation epilogue: act() is applied to the accumulator fragments right before the
// store to C. silu / gelu use the PTX ex2.approx instruction (no libdevice call), relu a max.
// (Numerics against torch.nn.functional: test/E2E/test_w4a16_e2e.py::test_fused_activation_epilogue.)

// SILU:     gpu.binary @w4a16_gemm_kernel
// SILU:     ex2.approx.ftz.f32
// RELU:     gpu.binary @w4a16_gemm_kernel
// RELU-NOT: ex2.approx
// RELU:     max.
// GELU:     ex2.approx.ftz.f32
// NONE:     gpu.binary @w4a16_gemm_kernel
// NONE-NOT: ex2.approx
func.func @w4a16_gemm(%act: tensor<16x512xf16>, %wp: tensor<512x64xi8>,
                      %scale: tensor<4x128xf16>, %zp: tensor<4x128xi8>,
                      %c: tensor<16x128xf32>) -> tensor<16x128xf32> {
  %w = qf.unpack %wp : tensor<512x64xi8> -> tensor<512x128xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<512x128xi8>, tensor<4x128xf16>, tensor<4x128xi8> -> tensor<512x128xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x128xf16>) outs(%c : tensor<16x128xf32>) -> tensor<16x128xf32>
  return %r : tensor<16x128xf32>
}
