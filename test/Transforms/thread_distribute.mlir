// test/Transforms/thread_distribute.mlir
// RUN: quantforge-opt --quantforge-thread-distribute %s | FileCheck %s
// RUN: quantforge-opt --quantforge-thread-distribute='distribute-lanes=false' %s | FileCheck --check-prefix=NOLANE %s

// =====================================================================
// Mapped warp loops and the (lane_m ⊃ lane_n) tile loops are removed and
// their induction variables become thread-id arithmetic.
//   z-loop (trip 2)  -> iv = lb + tid.z * step
//   y-loop (trip 2)  -> iv = lb + tid.y * step
//   lane loops 4 x 8 -> lane = tid.x : m = lane / 8, n = lane % 8
// =====================================================================

// CHECK-LABEL: func.func @warp_and_lane
// CHECK:         gpu.launch
// CHECK-NOT:       scf.for
// CHECK:           arith.divui %{{.*}}, %c8
// CHECK:           arith.remui %{{.*}}, %c8
// CHECK:           memref.store
// NOLANE-LABEL: func.func @warp_and_lane
// NOLANE:          scf.for
// NOLANE:            scf.for
// NOLANE:              memref.store
func.func @warp_and_lane(%buf: memref<128x128xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c2, %sz = %c2) {
    scf.for %wm = %c0 to %c128 step %c64 {
      scf.for %wn = %c0 to %c128 step %c64 {
        scf.for %m = %c0 to %c64 step %c16 {
          scf.for %n = %c0 to %c64 step %c8 {
            %i = arith.addi %wm, %m : index
            %j = arith.addi %wn, %n : index
            memref.store %v, %buf[%i, %j] : memref<128x128xf32>
          } {quantforge.lane_n}
        } {quantforge.lane_m}
      } {mapping = [#gpu.thread<y>]}
    } {mapping = [#gpu.thread<z>]}
    gpu.terminator
  }
  return
}

// A loop with more iterations than threads becomes a cyclic loop
// (iv = lb + tid*step, stride dim*step); one with fewer is guarded.
// CHECK-LABEL: func.func @cyclic_and_guard
// CHECK:         gpu.launch
// CHECK:           scf.for %{{.*}} = %{{.*}} to %c256 step %{{.*}} {
// CHECK:           scf.if
func.func @cyclic_and_guard(%buf: memref<256xf32>, %v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c2, %sz = %c1) {
    // trip 4 over 2 threads along y -> cyclic
    scf.for %i = %c0 to %c256 step %c64 {
      memref.store %v, %buf[%i] : memref<256xf32>
    } {mapping = [#gpu.thread<y>]}
    // trip 1 over 2 threads along y -> guarded
    scf.for %k = %c0 to %c64 step %c64 {
      memref.store %v, %buf[%k] : memref<256xf32>
    } {mapping = [#gpu.thread<y>]}
    gpu.terminator
  }
  return
}

// Loops containing a barrier must never be distributed.
// CHECK-LABEL: func.func @barrier_blocks_distribution
// CHECK:         scf.for
// CHECK:           gpu.barrier
func.func @barrier_blocks_distribution() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c2, %sz = %c1) {
    scf.for %i = %c0 to %c2 step %c1 {
      gpu.barrier
    } {mapping = [#gpu.thread<y>]}
    gpu.terminator
  }
  return
}
