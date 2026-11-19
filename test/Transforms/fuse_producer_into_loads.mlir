// test/Transforms/fuse_producer_into_loads.mlir
// RUN: quantforge-opt --quantforge-fuse-producer-into-loads %s | FileCheck %s

// =====================================================================
// A device buffer written by ONE elementwise kernel and read through
// subview/collapse_shape chains: every load is replaced by a recomputation
// of the producer (here: packed i8 -> two f16 per byte), then the producer
// kernel and the buffer are deleted.
//   * collapse_shape [[0],[1,2]] is delinearised:  j = c / 2, p = c % 2
//   * the per-tensor scalar (scale) is loaded once per consuming launch
// =====================================================================

// CHECK-LABEL: func.func @fuse_dequant
// CHECK-NOT:     gpu.alloc
// CHECK-NOT:     gpu.dealloc
// the producer launch is gone: exactly one gpu.launch remains
// CHECK:         gpu.launch
// CHECK-NOT:     gpu.launch
// CHECK:           %[[SCALE:.*]] = memref.load %arg1[] : memref<f16>
// CHECK:           scf.for
// CHECK:             arith.divui
// CHECK:             arith.remui
// CHECK:             memref.load %arg0[%{{.*}}, %{{.*}}] : memref<4x2xi8>
// CHECK:             arith.mulf %{{.*}}, %[[SCALE]] : f16
// CHECK:             memref.store %{{.*}}, %arg2[%{{.*}}, %{{.*}}] : memref<4x4xf16>
func.func @fuse_dequant(%packed: memref<4x2xi8>, %scale: memref<f16>,
                        %dst: memref<4x4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %buf = gpu.alloc () : memref<4x2x2xf16>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c4, %gy = %c2, %gz = %c2)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %byte = memref.load %packed[%bx, %by] : memref<4x2xi8>
    %s = memref.load %scale[] : memref<f16>
    %f = arith.sitofp %byte : i8 to f16
    %is_lo = arith.cmpi eq, %bz, %c0 : index
    %sel = arith.select %is_lo, %f, %s : f16
    %v = arith.mulf %sel, %s : f16
    memref.store %v, %buf[%bx, %by, %bz] : memref<4x2x2xf16>
    gpu.terminator
  }
  %flat = memref.collapse_shape %buf [[0], [1, 2]] : memref<4x2x2xf16> into memref<4x4xf16>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    scf.for %i = %c0 to %c16 step %c1 {
      %r = arith.divui %i, %c4 : index
      %c = arith.remui %i, %c4 : index
      %x = memref.load %flat[%r, %c] : memref<4x4xf16>
      memref.store %x, %dst[%r, %c] : memref<4x4xf16>
    }
    gpu.terminator
  }
  gpu.dealloc %buf : memref<4x2x2xf16>
  return
}

// A buffer with two writers must not be fused.
// CHECK-LABEL: func.func @two_writers_untouched
// CHECK:         gpu.alloc
// CHECK:         memref.load
func.func @two_writers_untouched(%v: f16, %dst: memref<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %buf = gpu.alloc () : memref<4xf16>
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c4, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    memref.store %v, %buf[%bx] : memref<4xf16>
    memref.store %v, %buf[%bx] : memref<4xf16>
    gpu.terminator
  }
  %x = memref.load %buf[%c0] : memref<4xf16>
  memref.store %x, %dst[%c0] : memref<4xf16>
  return
}
