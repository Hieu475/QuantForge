// test/Transforms/canonicalize_dequant_zp.mlir
// RUN: quantforge-opt --canonicalize-dequant-zp %s | FileCheck %s

// =====================================================================
// Symmetric Quantization Fast-Path Tests
//
// Verifies that --canonicalize-dequant-zp:
//   1. Eliminates arith.sitofp(zp) + arith.subf when zp == 0
//   2. Preserves full dequant when zp != 0
//   3. Handles both pre-lowering (qf.dequant) and post-lowering levels
// =====================================================================

// -----------------------------------------------------------------
// Test 1: Pre-lowering — qf.dequant with zp == 0 (symmetric)
// Expected: simplified to sitofp(input) * scale, NO subf
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @dequant_symmetric
// CHECK-NOT:     qf.dequant
// CHECK-NOT:     arith.subf
// CHECK:         linalg.generic
// CHECK:         ^bb0({{.*}}: i8, {{.*}}: f16, {{.*}}: f16):
// CHECK:           arith.sitofp {{.*}} : i8 to f16
// CHECK-NEXT:     arith.mulf
// CHECK:           linalg.yield
func.func @dequant_symmetric(%input: tensor<4096x4096xi8>,
                              %scale: tensor<f16>) -> tensor<4096x4096xf16> {
  %zp = arith.constant dense<0> : tensor<i8>
  %output = qf.dequant %input, %scale, %zp :
      tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
  return %output : tensor<4096x4096xf16>
}

// -----------------------------------------------------------------
// Test 2: Pre-lowering — qf.dequant with zp != 0 (asymmetric)
// Expected: full dequant preserved (sitofp + subf + mulf)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @dequant_asymmetric
// CHECK:         qf.dequant
func.func @dequant_asymmetric(%input: tensor<4096x4096xi8>,
                               %scale: tensor<f16>) -> tensor<4096x4096xf16> {
  %zp = arith.constant dense<8> : tensor<i8>
  %output = qf.dequant %input, %scale, %zp :
      tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
  return %output : tensor<4096x4096xf16>
}

// -----------------------------------------------------------------
// Test 3: Small shape — symmetric (zp == 0)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @dequant_symmetric_small
// CHECK-NOT:     qf.dequant
// CHECK-NOT:     arith.subf
// CHECK:         linalg.generic
// CHECK:           arith.sitofp
// CHECK:           arith.mulf
// CHECK:           linalg.yield
func.func @dequant_symmetric_small(%input: tensor<2x4xi8>,
                                    %scale: tensor<f16>) -> tensor<2x4xf16> {
  %zp = arith.constant dense<0> : tensor<i8>
  %output = qf.dequant %input, %scale, %zp :
      tensor<2x4xi8>, tensor<f16>, tensor<i8> -> tensor<2x4xf16>
  return %output : tensor<2x4xf16>
}

// -----------------------------------------------------------------
// Test 4: Post-lowering — arith.subf(x, 0.0) folding
// This tests the post-lowering pattern that catches subf with
// a constant zero RHS (e.g., from a prior pass that lowered
// dequant but didn't detect zp==0).
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @fold_subf_zero
// CHECK-NOT:     arith.subf
// CHECK:         return %{{.*}} : f16
func.func @fold_subf_zero(%x: f16) -> f16 {
  %zero = arith.constant 0.0 : f16
  %result = arith.subf %x, %zero : f16
  return %result : f16
}

// -----------------------------------------------------------------
// Test 5: Post-lowering — arith.subf with non-zero RHS preserved
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @no_fold_subf_nonzero
// CHECK:         arith.subf
func.func @no_fold_subf_nonzero(%x: f16) -> f16 {
  %nonzero = arith.constant 3.0 : f16
  %result = arith.subf %x, %nonzero : f16
  return %result : f16
}

// -----------------------------------------------------------------
// Test 6: Non-scalar zero_point that is zp == 0 (still folds)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @dequant_symmetric_non_scalar_zp
// CHECK-NOT:     qf.dequant
// CHECK-NOT:     arith.subf
// CHECK:         linalg.generic
// CHECK:           arith.sitofp
// CHECK:           arith.mulf
func.func @dequant_symmetric_non_scalar_zp(
    %input: tensor<8x16xi8>,
    %scale: tensor<f16>) -> tensor<8x16xf16> {
  %zp = arith.constant dense<0> : tensor<i8>
  %output = qf.dequant %input, %scale, %zp :
      tensor<8x16xi8>, tensor<f16>, tensor<i8> -> tensor<8x16xf16>
  return %output : tensor<8x16xf16>
}
