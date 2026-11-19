// test/Transforms/host_alloc_to_gpu.mlir
// RUN: quantforge-opt --quantforge-host-alloc-to-gpu %s | FileCheck %s

// Host-side memref.alloc -> gpu.alloc; non-escaping buffers are freed before
// the return.
// CHECK-LABEL: func.func @local_buffer
// CHECK-NOT:     memref.alloc
// CHECK:         %[[B:.*]] = gpu.alloc  () : memref<16xf32>
// CHECK:         gpu.dealloc  %[[B]] : memref<16xf32>
// CHECK-NEXT:    return
func.func @local_buffer(%v: f32, %i: index) {
  %b = memref.alloc() : memref<16xf32>
  memref.store %v, %b[%i] : memref<16xf32>
  return
}

// An aliasing view that is returned means the buffer escapes: no dealloc.
// CHECK-LABEL: func.func @escaping_buffer
// CHECK:         gpu.alloc
// CHECK-NOT:     gpu.dealloc
// CHECK:         return
func.func @escaping_buffer() -> memref<4x4xf32> {
  %b = memref.alloc() : memref<16xf32>
  %e = memref.expand_shape %b [[0, 1]] : memref<16xf32> into memref<4x4xf32>
  return %e : memref<4x4xf32>
}

// Allocations inside device code are not touched.
// CHECK-LABEL: func.func @device_alloc
// CHECK:         gpu.launch
// CHECK:           memref.alloc
func.func @device_alloc() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) {
    %b = memref.alloc() : memref<4xf32>
    gpu.terminator
  }
  return
}
