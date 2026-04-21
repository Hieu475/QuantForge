// RUN: quantforge-opt %s --quantforge-to-nvgpu | FileCheck %s

//===----------------------------------------------------------------------===//
// Unit test: LdMatrixWeightINT4Pattern (Pattern 2)
//
// Verifies that vector.transfer_read from an i8 SRAM memref is lowered to:
//   1. memref.reinterpret_cast (i8 → f16 view, halving column dimension)
//   2. nvgpu.ldmatrix on the f16 view
//   3. vector.bitcast → vector<Nxi32> for prmt processing
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @ldmatrix_int4_basic
// Verify the reinterpret_cast step
// CHECK: %[[REINTERP:.*]] = memref.reinterpret_cast %{{.*}}
// CHECK-SAME: to memref<64x32xf16, #gpu.address_space<workgroup>>

// Verify index adjustment (col / 2)
// CHECK: arith.divui

// Verify ldmatrix emission
// CHECK: %[[FRAG:.*]] = nvgpu.ldmatrix %[[REINTERP]]
// CHECK-SAME: {numTiles = 2 : i32, transpose = false}
// CHECK-SAME: -> vector<2x2xf16>

// Verify bitcast to i32 for downstream prmt
// CHECK: vector.bitcast %[[FRAG]]
// CHECK-SAME: vector<2x2xf16> to vector<2xi32>
func.func @ldmatrix_int4_basic(
    %sram: memref<64x64xi8, #gpu.address_space<workgroup>>,
    %row: index,
    %col: index) {
  %pad = arith.constant 0 : i8
  // This read represents loading a 16×8 INT4-packed weight tile
  // from shared memory. Pattern 2 should rewrite this into the
  // reinterpret → ldmatrix → bitcast sequence.
  %raw = vector.transfer_read %sram[%row, %col], %pad
      : memref<64x64xi8, #gpu.address_space<workgroup>>, vector<16x8xi8>
  return
}

// CHECK-LABEL: func.func @ldmatrix_int4_skip_f16
// Pattern should NOT match f16 SRAM reads (those are Pattern 1)
// CHECK-NOT: memref.reinterpret_cast
// CHECK: vector.transfer_read
func.func @ldmatrix_int4_skip_f16(
    %sram: memref<64x64xf16, #gpu.address_space<workgroup>>,
    %row: index,
    %col: index) {
  %pad = arith.constant 0.0 : f16
  %v = vector.transfer_read %sram[%row, %col], %pad
      : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
  return
}

// CHECK-LABEL: func.func @ldmatrix_int4_skip_hbm
// Pattern should NOT match i8 reads from HBM (non-SRAM)
// CHECK-NOT: memref.reinterpret_cast
// CHECK: vector.transfer_read
func.func @ldmatrix_int4_skip_hbm(
    %hbm: memref<64x64xi8>,
    %row: index,
    %col: index) {
  %pad = arith.constant 0 : i8
  %v = vector.transfer_read %hbm[%row, %col], %pad
      : memref<64x64xi8>, vector<16x8xi8>
  return
}
