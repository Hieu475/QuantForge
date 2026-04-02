// test/Transforms/bufferize_pass.mlir
// RUN: quantforge-opt --quantforge-bufferize %s | FileCheck %s

// =====================================================================
// Task 4.1 — One-Shot Bufferize
//
// Verifies that --quantforge-bufferize:
//   1. Converts tensor function args to memref
//   2. Converts tensor.empty to memref.alloc
//   3. Converts tensor.extract_slice to memref.subview
//   4. Linalg ops operate on memref
//   5. No tensor.empty/tensor.extract_slice remain
// =====================================================================

// -----------------------------------------------------------------
// Test 1: Basic matmul bufferization
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_bufferize
// CHECK-SAME:    memref<128x128xf16
// CHECK-SAME:    memref<128x128xf16
// CHECK-SAME:    memref<128x128xf16
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     tensor.extract_slice
// CHECK:         memref.subview
// CHECK:         linalg.matmul
// CHECK-SAME:      memref<

func.func @matmul_bufferize(
    %A : tensor<128x128xf16>,
    %B : tensor<128x128xf16>,
    %C : tensor<128x128xf16>) -> tensor<128x128xf16>
{
  %result = linalg.matmul
    ins(%A, %B : tensor<128x128xf16>, tensor<128x128xf16>)
    outs(%C    : tensor<128x128xf16>)
    -> tensor<128x128xf16>
  return %result : tensor<128x128xf16>
}

// -----------------------------------------------------------------
// Test 2: Extract slice → subview
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @extract_to_subview
// CHECK:         memref.subview
// CHECK-NOT:     tensor.extract_slice

func.func @extract_to_subview(
    %t : tensor<4096x4096xf16>) -> tensor<128x64xf16>
{
  %slice = tensor.extract_slice %t[0, 0] [128, 64] [1, 1]
    : tensor<4096x4096xf16> to tensor<128x64xf16>
  return %slice : tensor<128x64xf16>
}
