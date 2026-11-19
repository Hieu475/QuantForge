// test/Transforms/forall_to_gpu.mlir
// RUN: quantforge-opt --quantforge-forall-to-gpu %s | FileCheck %s
// RUN: quantforge-opt --quantforge-forall-to-gpu='block-dim-x=32 block-dim-y=4' %s | FileCheck --check-prefix=BLK %s

// =====================================================================
// scf.forall (memref semantics) -> gpu.launch
//   * grid = ceildiv(ub - lb, step) per dimension
//   * grid axis comes from the #gpu.block mapping (dim0 -> y, dim1 -> x)
//   * induction variables become  lb + block_id * step
// =====================================================================

// CHECK-LABEL: func.func @mapped_forall
// CHECK-NOT:     scf.forall
// CHECK:         gpu.launch blocks(%[[BX:[^,]*]], %[[BY:[^,]*]], %[[BZ:[^)]*]]) in
// dim0 (256/64 blocks) -> y, dim1 (128/32 blocks) -> x
// CHECK:           arith.muli %[[BY]]
// CHECK:           arith.addi
// CHECK:           arith.muli %[[BX]]
// CHECK:           arith.addi
// CHECK:           memref.store
// CHECK:           gpu.terminator
// BLK-LABEL: func.func @mapped_forall
// BLK:         threads(%{{[^,]*}}, %{{[^,]*}}, %{{[^)]*}}) in (%{{[^=]*}}= %c32{{[_0-9]*}}, %{{[^=]*}}= %c4{{[_0-9]*}}, %{{[^=]*}}= %c1
func.func @mapped_forall(%buf: memref<256x128xf32>, %v: f32) {
  scf.forall (%i, %j) = (0, 0) to (256, 128) step (64, 32) {
    memref.store %v, %buf[%i, %j] : memref<256x128xf32>
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return
}

// Without a mapping the last dimension maps to x.
// CHECK-LABEL: func.func @unmapped_forall
// CHECK-NOT:     scf.forall
// CHECK:         gpu.launch
func.func @unmapped_forall(%buf: memref<8x8xf32>, %v: f32) {
  scf.forall (%i, %j) in (8, 8) {
    memref.store %v, %buf[%i, %j] : memref<8x8xf32>
  }
  return
}

// A forall nested in a gpu.launch is left alone.
// CHECK-LABEL: func.func @already_launched
// CHECK:         gpu.launch
// CHECK:           scf.forall
func.func @already_launched(%buf: memref<8xf32>, %v: f32) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    scf.forall (%i) in (8) {
      memref.store %v, %buf[%i] : memref<8xf32>
    }
    gpu.terminator
  }
  return
}
