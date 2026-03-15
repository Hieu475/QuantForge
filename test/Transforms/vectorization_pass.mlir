// test/Transforms/vectorization_pass.mlir
// RUN: quantforge-opt --quantforge-vectorization %s | FileCheck %s

// =====================================================================
// Task 2.3.1 — Vectorization Pass
//
// Verifies two conversion paths:
//   1) linalg.matmul  -> vector.contract (+ vector.transfer_*)
//   2) linalg.generic -> vectorized integer arithmetic (SIMD-style)
// =====================================================================

// -----------------------------------------------------------------
// Case 1: Matmul -> vector.contract
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_m16n8k16
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_read
// CHECK:         vector.contract
// CHECK:         vector.transfer_write
func.func @matmul_m16n8k16(
    %A : tensor<16x16xf16>,
    %B : tensor<16x8xf16>,
    %C : tensor<16x8xf16>) -> tensor<16x8xf16>
{
  %result = linalg.matmul
    ins(%A, %B : tensor<16x16xf16>, tensor<16x8xf16>)
    outs(%C : tensor<16x8xf16>)
    -> tensor<16x8xf16>
  return %result : tensor<16x8xf16>
}

// -----------------------------------------------------------------
// Case 2: Unpack-like linalg.generic -> vectorized SIMD arith
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @unpack_like_simd
// CHECK:         vector.transfer_read
// CHECK:         arith.shrui {{.*}} : vector<1x16xi8>
// CHECK:         arith.andi {{.*}} : vector<1x16xi8>
// CHECK:         vector.transfer_write
func.func @unpack_like_simd(
    %packed : tensor<1x16xi8>,
    %out : tensor<1x16xi8>) -> tensor<1x16xi8>
{
  %result = linalg.generic
    {indexing_maps = [
       affine_map<(d0, d1) -> (d0, d1)>,
       affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%packed : tensor<1x16xi8>)
    outs(%out : tensor<1x16xi8>)
  {
    ^bb0(%x: i8, %y: i8):
      %c4 = arith.constant 4 : i8
      %c15 = arith.constant 15 : i8
      %shift = arith.shrui %x, %c4 : i8
      %nibble = arith.andi %shift, %c15 : i8
      linalg.yield %nibble : i8
  } -> tensor<1x16xi8>
  return %result : tensor<1x16xi8>
}
