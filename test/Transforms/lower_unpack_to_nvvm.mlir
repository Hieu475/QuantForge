// test/Transforms/lower_unpack_to_nvvm.mlir
// RUN: quantforge-opt --lower-unpack-to-nvvm %s | FileCheck %s

// =====================================================================
// Phase 2 — PTX-Ready i32-Chunk SCF Lowering
//
// Verifies that --lower-unpack-to-nvvm:
//   1. Replaces qf.unpack with nested scf.for loops
//   2. Reads 4 packed bytes individually and packs them into one i32
//      via extui + shli + ori (conceptually: one GPU i32 register load)
//   3. Extracts all 8 nibbles with CONSTANT shift immediates (0,4,8,…,28)
//      (maps to  shr.u32 %r, <imm>  +  and.b32 in PTX)
//   4. Writes 8 nibbles per inner iteration via chained tensor.insert
//   5. Does NOT contain linalg.generic / linalg.index / arith.select
// =====================================================================

// -----------------------------------------------------------------
// Test 1: Basic 2-D — 4096 × 512 packed → 4096 × 1024 unpacked
// (N = 512 = 4 × 128, satisfies N%4==0 guard)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @unpack_nvvm_basic
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     arith.select
// CHECK-NOT:     linalg.index
//
// Outer loop over rows
// CHECK:         scf.for {{.*}} = {{.*}} to {{.*}} step
// Inner loop over i32 chunks (N/4 = 128 iterations)
// CHECK:           scf.for {{.*}} = {{.*}} to {{.*}} step
//
// Read 4 bytes
// CHECK:             tensor.extract
// CHECK:             tensor.extract
// CHECK:             tensor.extract
// CHECK:             tensor.extract
//
// Pack into i32 (extui + shli + ori pattern)
// CHECK-DAG:         arith.constant 8  : i32
// CHECK-DAG:         arith.constant 16 : i32
// CHECK-DAG:         arith.constant 24 : i32
// CHECK:             arith.extui
// CHECK:             arith.shli
// CHECK:             arith.ori
//
// Extract nibbles with constant shifts
// CHECK-DAG:         arith.constant 4  : i32
// CHECK-DAG:         arith.constant 28 : i32
// CHECK-DAG:         arith.constant 15 : i32
// CHECK:             arith.shrui
// CHECK:             arith.andi
// CHECK:             arith.trunci
//
// Write 8 nibbles
// CHECK:             tensor.insert
// CHECK:             tensor.insert
// CHECK:             tensor.insert
// CHECK:             tensor.insert
// CHECK:             tensor.insert
// CHECK:             tensor.insert
// CHECK:             tensor.insert
// CHECK:             tensor.insert
func.func @unpack_nvvm_basic(%packed: tensor<4096x512xi8>)
                              -> tensor<4096x1024xi8> {
  %out = qf.unpack %packed : tensor<4096x512xi8> -> tensor<4096x1024xi8>
  return %out : tensor<4096x1024xi8>
}

// -----------------------------------------------------------------
// Test 2: Minimal shape — 2 × 4 packed → 2 × 8 unpacked
// (N = 4, exactly one i32 chunk per row)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @unpack_nvvm_small
// CHECK-NOT:   qf.unpack
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           tensor.extract
// CHECK:           arith.extui
// CHECK:           arith.ori
// CHECK:           arith.trunci
// CHECK:           tensor.insert
func.func @unpack_nvvm_small(%packed: tensor<2x4xi8>) -> tensor<2x8xi8> {
  %out = qf.unpack %packed : tensor<2x4xi8> -> tensor<2x8xi8>
  return %out : tensor<2x8xi8>
}

// -----------------------------------------------------------------
// Test 3: Same-shape must be skipped
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @unpack_nvvm_same_skip
// CHECK:         qf.unpack
func.func @unpack_nvvm_same_skip(%packed: tensor<4x8xi8>) -> tensor<4x8xi8> {
  %out = qf.unpack %packed : tensor<4x8xi8> -> tensor<4x8xi8>
  return %out : tensor<4x8xi8>
}

// -----------------------------------------------------------------
// Test 4: Negative — N not divisible by 4 must be skipped
// (Falls through to next-available pass, qf.unpack stays intact)
// N = 6 is not divisible by 4.
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @unpack_nvvm_skip_non_aligned
// CHECK:         qf.unpack
func.func @unpack_nvvm_skip_non_aligned(%packed: tensor<4x6xi8>)
                                         -> tensor<4x12xi8> {
  %out = qf.unpack %packed : tensor<4x6xi8> -> tensor<4x12xi8>
  return %out : tensor<4x12xi8>
}
