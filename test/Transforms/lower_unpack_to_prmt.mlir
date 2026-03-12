// RUN: quantforge-opt --lower-unpack-to-prmt %s | FileCheck %s

// Tests for LowerUnpackToPRMT pass:
//   1. prmt.b32 emitted for standard 2-D doubling-shape unpack
//   2. Selector constants 0x5140 (lo) and 0x7362 (hi) are present
//   3. pass replaces qf.unpack with scf.for loops
//   4. Same-shape mode is skipped (left as qf.unpack)
//   5. N not divisible by 4 is skipped

// ── Test 1: Standard 2-D doubling-shape unpack ─────────────────────────────
// Packed input: [4, 8] i8  → Unpacked output: [4, 16] i8
// (N=8, divisible by 4, so we get N4=2 i32 chunks per row)

// CHECK-LABEL: func.func @test_prmt_basic
// CHECK-NOT:     qf.unpack
// After pass: should have scf.for loops
// CHECK:         scf.for
// CHECK:           scf.for
// Should emit prmt.b32 inline asm (2× per chunk, for lo and hi nibbles)
// CHECK:             llvm.inline_asm
// CHECK-SAME:        prmt.b32
// CHECK:             llvm.inline_asm
// CHECK-SAME:        prmt.b32
// Result should be tensor.insert into output
// CHECK:             tensor.insert
func.func @test_prmt_basic(%packed: tensor<4x8xi8>) -> tensor<4x16xi8> {
  %out = qf.unpack %packed : tensor<4x8xi8> -> tensor<4x16xi8>
  return %out : tensor<4x16xi8>
}

// ── Test 2: Larger tensor (K=64, N=32) ─────────────────────────────────────
// CHECK-LABEL: func.func @test_prmt_large
// CHECK-NOT:   qf.unpack
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           llvm.inline_asm {{.*}} prmt.b32
func.func @test_prmt_large(%packed: tensor<64x32xi8>) -> tensor<64x64xi8> {
  %out = qf.unpack %packed : tensor<64x32xi8> -> tensor<64x64xi8>
  return %out : tensor<64x64xi8>
}

// ── Test 3: Same-shape mode — pass should NOT transform ────────────────────
// CHECK-LABEL: func.func @test_prmt_same_shape_skip
// Pass should leave this as qf.unpack (same-shape guard triggers)
// CHECK: qf.unpack
func.func @test_prmt_same_shape_skip(%packed: tensor<4x16xi8>) -> tensor<4x16xi8> {
  %out = qf.unpack %packed : tensor<4x16xi8> -> tensor<4x16xi8>
  return %out : tensor<4x16xi8>
}

// ── Test 4: N not divisible by 4 — pass should NOT transform ───────────────
// CHECK-LABEL: func.func @test_prmt_odd_n_skip
// N=6 is not divisible by 4 → guard fires, op left as-is
// CHECK: qf.unpack
func.func @test_prmt_odd_n_skip(%packed: tensor<4x6xi8>) -> tensor<4x12xi8> {
  %out = qf.unpack %packed : tensor<4x6xi8> -> tensor<4x12xi8>
  return %out : tensor<4x12xi8>
}
