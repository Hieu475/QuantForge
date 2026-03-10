// RUN: quantforge-opt --convert-linalg-to-quantforge %s | FileCheck %s

// =====================================================================
// Test 1: Basic — linalg.matmul with INT8 weight → qf.unpack + qf.dequant
// =====================================================================
// Activation: [1, 2048] f16     (GEMV-style, batch=1)
// Weight:     [2048, 4096] i8   (INT8-quantized / INT4-packed)
// Output:     [1, 4096] f16

// CHECK-LABEL: func.func @matmul_int8_basic
// CHECK:         %[[UNPACK:.*]] = qf.unpack %{{.*}} : tensor<2048x4096xi8> -> tensor<2048x4096xi8>
// CHECK:         %[[SCALE:.*]] = arith.constant dense<1.000000e+00> : tensor<f16>
// CHECK:         %[[ZP:.*]]    = arith.constant dense<0> : tensor<i8>
// CHECK:         %[[DQ:.*]]    = qf.dequant %[[UNPACK]], %[[SCALE]], %[[ZP]]
// CHECK-SAME:        : tensor<2048x4096xi8>, tensor<f16>, tensor<i8> -> tensor<2048x4096xf16>
// CHECK:         linalg.matmul
// CHECK-SAME:        ins(%{{.*}}, %[[DQ]] : tensor<1x2048xf16>, tensor<2048x4096xf16>)
func.func @matmul_int8_basic(%act: tensor<1x2048xf16>,
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
// Test 2: Batch GEMM — larger M dimension
// =====================================================================
// Activation: [32, 512] f16
// Weight:     [512, 1024] i8
// Output:     [32, 1024] f16

// CHECK-LABEL: func.func @matmul_int8_batch
// CHECK:         qf.unpack %{{.*}} : tensor<512x1024xi8> -> tensor<512x1024xi8>
// CHECK:         qf.dequant
// CHECK-SAME:        -> tensor<512x1024xf16>
// CHECK:         linalg.matmul
// CHECK-SAME:        ins(%{{.*}}, %{{.*}} : tensor<32x512xf16>, tensor<512x1024xf16>)
// CHECK-SAME:        outs(%{{.*}} : tensor<32x1024xf16>)
func.func @matmul_int8_batch(%act: tensor<32x512xf16>,
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
// Test 3: Negative — FP16 weight should NOT be converted
// =====================================================================
// Both inputs are f16 → the pass should leave the matmul untouched.

// CHECK-LABEL: func.func @matmul_fp16_noop
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.matmul
// CHECK-SAME:        ins(%{{.*}}, %{{.*}} : tensor<4x8xf16>, tensor<8x16xf16>)
func.func @matmul_fp16_noop(%act: tensor<4x8xf16>,
                             %weight: tensor<8x16xf16>)
                             -> tensor<4x16xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<4x16xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<4x16xf16>) -> tensor<4x16xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<4x8xf16>, tensor<8x16xf16>)
                          outs(%filled : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %result : tensor<4x16xf16>
}

// =====================================================================
// Test 4: INT8 activation + INT8 weight — LHS gets arith.sitofp
// =====================================================================
// When LHS is also i8, the pass should insert arith.sitofp for LHS.

// CHECK-LABEL: func.func @matmul_int8_both
// CHECK:         qf.unpack
// CHECK:         qf.dequant
// CHECK:         arith.sitofp %{{.*}} : tensor<1x256xi8> to tensor<1x256xf16>
// CHECK:         linalg.matmul
// CHECK-SAME:        tensor<1x256xf16>, tensor<256x512xf16>
func.func @matmul_int8_both(%act: tensor<1x256xi8>,
                             %weight: tensor<256x512xi8>)
                             -> tensor<1x512xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x512xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1x512xf16>) -> tensor<1x512xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<1x256xi8>, tensor<256x512xi8>)
                          outs(%filled : tensor<1x512xf16>) -> tensor<1x512xf16>
  return %result : tensor<1x512xf16>
}

// =====================================================================
// Test 5a: attributes on weight constant are used for scale/zero_point
// =====================================================================
// CHECK-LABEL: func.func @matmul_with_attrs
// CHECK:         %[[SCALE:.*]] = arith.constant dense<2.000000e+00> : tensor<f16>
// CHECK:         %[[ZP:.*]]    = arith.constant dense<5> : tensor<i8>
// CHECK:         qf.unpack %{{.*}} : tensor<2x4xi8> -> tensor<2x4xi8>
// CHECK:         qf.dequant %{{.*}}, %[[SCALE]], %[[ZP]]
func.func @matmul_with_attrs(%act: tensor<1x2xf16>)
                              -> tensor<1x4xf16> {
  %weight = arith.constant {qf.scale = dense<2.000000e+00> : tensor<f16>, qf.zp = dense<5> : tensor<i8>} dense<0> : tensor<2x4xi8>
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x4xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1x4xf16>) -> tensor<1x4xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<1x2xf16>, tensor<2x4xi8>)
                          outs(%filled : tensor<1x4xf16>) -> tensor<1x4xf16>
  return %result : tensor<1x4xf16>
}

// =====================================================================
// Test 5b: attributes on the matmul op itself
// =====================================================================
// CHECK-LABEL: func.func @matmul_with_matmul_attrs
// CHECK:         %[[SCALE2:.*]] = arith.constant dense<2.000000e+00> : tensor<f16>
// CHECK:         %[[ZP2:.*]]    = arith.constant dense<5> : tensor<i8>
// CHECK:         qf.unpack %{{.*}} : tensor<2x4xi8> -> tensor<2x4xi8>
// CHECK:         qf.dequant %{{.*}}, %[[SCALE2]], %[[ZP2]]
func.func @matmul_with_matmul_attrs(%act: tensor<1x2xf16>)
                                       -> tensor<1x4xf16> {
  %weight = arith.constant dense<0> : tensor<2x4xi8>
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x4xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1x4xf16>) -> tensor<1x4xf16>
  %result = linalg.matmul {qf.scale = dense<2.000000e+00> : tensor<f16>,
                           qf.zp    = dense<5>   : tensor<i8>}
                          ins(%act, %weight : tensor<1x2xf16>, tensor<2x4xi8>)
                          outs(%filled : tensor<1x4xf16>) -> tensor<1x4xf16>
  return %result : tensor<1x4xf16>
}

// =====================================================================
// Test 6: verify --qf-bitwidth=4 still converts i8-packed INT4 data
// =====================================================================
// RUN: quantforge-opt -pass-pipeline='builtin.module(func.func(convert-linalg-to-quantforge{qf-bitwidth=4}))' %s | FileCheck --check-prefix=BW4 %s
// BW4-LABEL: func.func @matmul_bitwidth4
// BW4:         qf.unpack %{{.*}} : tensor<8x8xi8> -> tensor<8x8xi8>
// BW4:         qf.dequant
func.func @matmul_bitwidth4(%act: tensor<2x8xf16>,
                             %weight: tensor<8x8xi8>)
                             -> tensor<2x8xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<2x8xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<2x8xf16>, tensor<8x8xi8>)
                          outs(%filled : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %result : tensor<2x8xf16>
}
