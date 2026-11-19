// RUN: quantforge-opt --quantforge-simplify-index-math --quantforge-vectorize-global-loads --canonicalize --cse %s | FileCheck %s

// Eight adjacent scalar loads at 8*tid + j become ONE aligned 16-byte load.
// CHECK-LABEL: func.func @eight_aligned
// CHECK:         gpu.launch
// CHECK:           memref.assume_alignment %{{.*}}, 16 : memref<4x128xf16>
// CHECK:           vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x128xf16>, vector<8xf16>
// CHECK-COUNT-8:   vector.extract
// CHECK-NOT:       memref.load
func.func @eight_aligned(%s: memref<4x128xf16>, %o: memref<16xf16>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c16, %sy = %c1, %sz = %c1) {
    %n = arith.muli %tx, %c8 : index
    %acc0 = arith.constant 0.0 : f16
    %l0 = memref.load %s[%c1, %n] : memref<4x128xf16>
    %i1 = arith.addi %n, %c1 : index
    %l1 = memref.load %s[%c1, %i1] : memref<4x128xf16>
    %i2 = arith.addi %n, %c2 : index
    %l2 = memref.load %s[%c1, %i2] : memref<4x128xf16>
    %i3 = arith.addi %n, %c3 : index
    %l3 = memref.load %s[%c1, %i3] : memref<4x128xf16>
    %i4 = arith.addi %n, %c4 : index
    %l4 = memref.load %s[%c1, %i4] : memref<4x128xf16>
    %i5 = arith.addi %n, %c5 : index
    %l5 = memref.load %s[%c1, %i5] : memref<4x128xf16>
    %i6 = arith.addi %n, %c6 : index
    %l6 = memref.load %s[%c1, %i6] : memref<4x128xf16>
    %i7 = arith.addi %n, %c7 : index
    %l7 = memref.load %s[%c1, %i7] : memref<4x128xf16>
    %a0 = arith.addf %l0, %l1 : f16
    %a1 = arith.addf %a0, %l2 : f16
    %a2 = arith.addf %a1, %l3 : f16
    %a3 = arith.addf %a2, %l4 : f16
    %a4 = arith.addf %a3, %l5 : f16
    %a5 = arith.addf %a4, %l6 : f16
    %a6 = arith.addf %a5, %l7 : f16
    memref.store %a6, %o[%tx] : memref<16xf16>
    gpu.terminator
  }
  return
}

// The base is only a multiple of 4: two 4-wide vector loads, not one 8-wide.
// CHECK-LABEL: func.func @four_aligned
// CHECK:         memref.assume_alignment %{{.*}}, 8 : memref<4x128xf16>
// CHECK-NOT:     vector<8xf16>
// CHECK:         vector.load {{.*}} : memref<4x128xf16>, vector<4xf16>
// CHECK:         vector.load {{.*}} : memref<4x128xf16>, vector<4xf16>
func.func @four_aligned(%s: memref<4x128xf16>, %o: memref<16xf16>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c16, %sy = %c1, %sz = %c1) {
    %n = arith.muli %tx, %c4 : index
    %l0 = memref.load %s[%c1, %n] : memref<4x128xf16>
    %i1 = arith.addi %n, %c1 : index
    %l1 = memref.load %s[%c1, %i1] : memref<4x128xf16>
    %i2 = arith.addi %n, %c2 : index
    %l2 = memref.load %s[%c1, %i2] : memref<4x128xf16>
    %i3 = arith.addi %n, %c3 : index
    %l3 = memref.load %s[%c1, %i3] : memref<4x128xf16>
    %i4 = arith.addi %n, %c4 : index
    %l4 = memref.load %s[%c1, %i4] : memref<4x128xf16>
    %i5 = arith.addi %n, %c5 : index
    %l5 = memref.load %s[%c1, %i5] : memref<4x128xf16>
    %i6 = arith.addi %n, %c6 : index
    %l6 = memref.load %s[%c1, %i6] : memref<4x128xf16>
    %i7 = arith.addi %n, %c7 : index
    %l7 = memref.load %s[%c1, %i7] : memref<4x128xf16>
    %a0 = arith.addf %l0, %l1 : f16
    %a1 = arith.addf %a0, %l2 : f16
    %a2 = arith.addf %a1, %l3 : f16
    %a3 = arith.addf %a2, %l4 : f16
    %a4 = arith.addf %a3, %l5 : f16
    %a5 = arith.addf %a4, %l6 : f16
    %a6 = arith.addf %a5, %l7 : f16
    memref.store %a6, %o[%tx] : memref<16xf16>
    gpu.terminator
  }
  return
}

// Not provably aligned (base = tid, no multiple): untouched.
// CHECK-LABEL: func.func @unaligned
// CHECK-NOT:     vector.load
// CHECK:         memref.load
func.func @unaligned(%s: memref<4x128xf16>, %o: memref<16xf16>) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c16, %sy = %c1, %sz = %c1) {
    %l0 = memref.load %s[%c1, %tx] : memref<4x128xf16>
    %i1 = arith.addi %tx, %c1 : index
    %l1 = memref.load %s[%c1, %i1] : memref<4x128xf16>
    %a0 = arith.addf %l0, %l1 : f16
    memref.store %a0, %o[%tx] : memref<16xf16>
    gpu.terminator
  }
  return
}

// A buffer that is also written must not be touched: merging the loads could
// move one of them across the store.
// CHECK-LABEL: func.func @written_buffer
// CHECK-NOT:     vector.load
func.func @written_buffer(%s: memref<4x128xf16>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c16, %sy = %c1, %sz = %c1) {
    %n = arith.muli %tx, %c8 : index
    %l0 = memref.load %s[%c1, %n] : memref<4x128xf16>
    memref.store %l0, %s[%c2, %n] : memref<4x128xf16>
    %i1 = arith.addi %n, %c1 : index
    %l1 = memref.load %s[%c1, %i1] : memref<4x128xf16>
    memref.store %l1, %s[%c2, %i1] : memref<4x128xf16>
    gpu.terminator
  }
  return
}
