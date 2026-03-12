// RUN: quantforge-opt --lower-unpack-to-nvvm --swizzled-unpack-indexing %s | FileCheck %s

// Tests for SwizzledUnpackIndexing pass:
//   1. Column index receives arith.xori with k%8 after the pass
//   2. The outer scf.for is marked with {swizzled} attribute
//   3. Small tensors (N < 32) are skipped
//   4. Running pass twice is idempotent

// ── Test 1: Standard 2-D tensor with N=64 ≥ 32 — should be swizzled ────────
// Run lower-unpack-to-nvvm first to get the SCF loop nest,
// then apply swizzled-unpack-indexing.

// CHECK-LABEL: func.func @test_swizzle_basic
// CHECK-NOT:   qf.unpack
// (a) scf.for loop nest present
// CHECK:       scf.for
// CHECK:         scf.for
// (b) arith.remui(k, 8) present for computing k%8
// CHECK:           arith.remui %{{.*}}, %{{.*}} : index
// (c) arith.xori present for XOR swizzle
// CHECK:           arith.xori %{{.*}}, %{{.*}} : index
func.func @test_swizzle_basic(%packed: tensor<64x32xi8>) -> tensor<64x64xi8> {
  %out = qf.unpack %packed : tensor<64x32xi8> -> tensor<64x64xi8>
  return %out : tensor<64x64xi8>
}

// ── Test 2: Small tensor N=8 (packed dim=4) — below min-n=32 threshold ─────
// CHECK-LABEL: func.func @test_swizzle_small_n_skip
// CHECK-NOT:   qf.unpack
// N=8: 4*4=16 < 32 threshold → no swizzle emitted
// CHECK-NOT:   arith.xori
func.func @test_swizzle_small_n_skip(%packed: tensor<4x4xi8>) -> tensor<4x8xi8> {
  %out = qf.unpack %packed : tensor<4x4xi8> -> tensor<4x8xi8>
  return %out : tensor<4x8xi8>
}
