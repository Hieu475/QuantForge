// RUN: quantforge-opt --lower-dequant-to-arith %s | FileCheck %s

// =====================================================================
// Test 1: Basic scalar scale/zp dequantization
// =====================================================================
// Input:  [4096, 4096] i8
// Scale:  scalar f16
// ZP:     scalar i8
// Output: [4096, 4096] f16

// CHECK-LABEL: func.func @dequant_basic
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.generic
// CHECK:         arith.sitofp {{.*}} : i8 to f16
// CHECK:         arith.sitofp {{.*}} : i8 to f16
// CHECK:         arith.subf {{.*}} : f16
// CHECK:         arith.mulf {{.*}} : f16
// CHECK:         linalg.yield
func.func @dequant_basic(%input: tensor<4096x4096xi8>,
                          %scale: tensor<f16>,
                          %zp: tensor<i8>) -> tensor<4096x4096xf16> {
  %output = qf.dequant %input, %scale, %zp :
      tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
  return %output : tensor<4096x4096xf16>
}

// =====================================================================
// Test 2: Small shape — verify structure
// =====================================================================

// CHECK-LABEL: func.func @dequant_small
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.generic
// CHECK-SAME:        ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME:             tensor<2x4xi8>, tensor<f16>, tensor<i8>)
// CHECK-SAME:        outs(%{{.*}} : tensor<2x4xf16>)
func.func @dequant_small(%input: tensor<2x4xi8>,
                          %scale: tensor<f16>,
                          %zp: tensor<i8>) -> tensor<2x4xf16> {
  %output = qf.dequant %input, %scale, %zp :
      tensor<2x4xi8>, tensor<f16>, tensor<i8> -> tensor<2x4xf16>
  return %output : tensor<2x4xf16>
}

// =====================================================================
// Test 3: GEMV weight shape
// =====================================================================

// CHECK-LABEL: func.func @dequant_gemv_weight
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.generic
// CHECK:         arith.sitofp
// CHECK:         arith.subf
// CHECK:         arith.mulf
func.func @dequant_gemv_weight(%input: tensor<2048x8192xi8>,
                                %scale: tensor<f16>,
                                %zp: tensor<i8>) -> tensor<2048x8192xf16> {
  %output = qf.dequant %input, %scale, %zp :
      tensor<2048x8192xi8>, tensor<f16>, tensor<i8> -> tensor<2048x8192xf16>
  return %output : tensor<2048x8192xf16>
}
