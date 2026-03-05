// RUN: quantforge-opt --convert-linalg-to-quantforge %s | FileCheck %s

// Test: linalg.matmul with INT8 weight (quantized / INT4-packed)
//
// Activation: [1, 2048] f16
// Weight:     [2048, 4096] i8   (INT8-quantized)
// Output:     [1, 4096] f16
//
// Expected after the pass:
//   qf.unpack  : [2048, 4096] i8  → [2048, 4096] i8  (semantic marker)
//   qf.dequant : [2048, 4096] i8  → [2048, 4096] f16
//   linalg.matmul [1,2048]f16 x [2048,4096]f16 → [1,4096]f16

func.func @matmul_int8_weight(%act: tensor<1x2048xf16>,
                               %weight: tensor<2048x4096xi8>)
                               -> tensor<1x4096xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x4096xf16>
  %filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1x4096xf16>) -> tensor<1x4096xf16>
  %result = linalg.matmul ins(%act, %weight : tensor<1x2048xf16>, tensor<2048x4096xi8>)
                          outs(%filled : tensor<1x4096xf16>) -> tensor<1x4096xf16>
  return %result : tensor<1x4096xf16>
}

// CHECK-LABEL: func.func @matmul_int8_weight
// CHECK:         qf.unpack
// CHECK-SAME:      tensor<2048x4096xi8> -> tensor<2048x4096xi8>
// CHECK:         qf.dequant
// CHECK-SAME:      -> tensor<2048x4096xf16>
// CHECK:         linalg.matmul
// CHECK-SAME:      tensor<1x2048xf16>, tensor<2048x4096xf16>

