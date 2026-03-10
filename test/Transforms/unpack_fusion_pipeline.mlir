// RUN: quantforge-opt --convert-linalg-to-quantforge --fuse-unpack-dequant %s | FileCheck %s

// =====================================================================
// End-to-end pipeline test: linalg.matmul → fused unpack+dequant + matmul
// =====================================================================
// Pipeline:
//   1. convert-linalg-to-quantforge: insert qf.unpack (same-shape) + qf.dequant
//   2. fuse-unpack-dequant:          fuse into one linalg.generic
//
// ConvertLinalgToQuantForge produces same-shape unpack (semantic marker),
// so the fusion pass uses the pointwise mask+dequant path — no shape
// expansion or tensor.collapse_shape.
//
// Expected output:
//   - One linalg.generic (pointwise mask + dequant): i8 → f16
//   - One linalg.matmul in FP16
//   - No qf.unpack or qf.dequant ops remaining

// CHECK-LABEL: func.func @gemv_e2e
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.generic
// CHECK:           arith.andi
// CHECK:           arith.sitofp
// CHECK:           arith.subf
// CHECK:           arith.mulf
// CHECK:           linalg.yield
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         linalg.matmul
// CHECK-SAME:        ins(%{{.*}}, %{{.*}} : tensor<1x2048xf16>, tensor<2048x4096xf16>)
// CHECK-SAME:        outs(%{{.*}} : tensor<1x4096xf16>)
func.func @gemv_e2e(%act: tensor<1x2048xf16>,
                     %weight: tensor<2048x4096xi8>)
                     -> tensor<1x4096xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x4096xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1x4096xf16>) -> tensor<1x4096xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<1x2048xf16>, tensor<2048x4096xi8>)
                          outs(%filled : tensor<1x4096xf16>) -> tensor<1x4096xf16>
  return %result : tensor<1x4096xf16>
}

// =====================================================================
// Test 2: Batch case
// =====================================================================

// CHECK-LABEL: func.func @batch_e2e
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.generic
// CHECK:           arith.andi
// CHECK:           arith.mulf
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         linalg.matmul
func.func @batch_e2e(%act: tensor<32x512xf16>,
                      %weight: tensor<512x1024xi8>)
                      -> tensor<32x1024xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<32x1024xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<32x1024xf16>) -> tensor<32x1024xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<32x512xf16>, tensor<512x1024xi8>)
                          outs(%filled : tensor<32x1024xf16>) -> tensor<32x1024xf16>
  return %result : tensor<32x1024xf16>
}

// =====================================================================
// Test 3: FP16 weight — not touched by pipeline (negative)
// =====================================================================

// CHECK-LABEL: func.func @fp16_noop_e2e
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK-NOT:     linalg.generic
// CHECK:         linalg.matmul
// CHECK-SAME:        ins(%{{.*}}, %{{.*}} : tensor<4x8xf16>, tensor<8x16xf16>)
func.func @fp16_noop_e2e(%act: tensor<4x8xf16>,
                          %weight: tensor<8x16xf16>)
                          -> tensor<4x16xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<4x16xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<4x16xf16>) -> tensor<4x16xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<4x8xf16>, tensor<8x16xf16>)
                          outs(%filled : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %result : tensor<4x16xf16>
}
