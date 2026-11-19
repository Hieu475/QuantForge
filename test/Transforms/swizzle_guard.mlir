// test/Transforms/swizzle_guard.mlir
// RUN: quantforge-opt --quantforge-swizzle-load %s 2>&1 | FileCheck %s

// XOR swizzling remaps only 2-D memref.load/store. If the same shared buffer
// is ALSO accessed by another op (here vector.transfer_read) only part of the
// accesses would be remapped -> silent data corruption. The pass must skip
// such buffers (and say so) while still swizzling fully-covered ones.

// CHECK: warning: shared buffer is accessed by non load/store ops
// CHECK-LABEL: func.func @mixed_access
// CHECK-NOT:     arith.xori
// CHECK-LABEL: func.func @plain_access
// CHECK:         arith.remui
// CHECK:         arith.xori
func.func @mixed_access(%v: f16, %i: index) {
  %zero = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %s = memref.alloc() : memref<16x16xf16, #gpu.address_space<workgroup>>
  memref.store %v, %s[%i, %i] : memref<16x16xf16, #gpu.address_space<workgroup>>
  %r = vector.transfer_read %s[%c0, %c0], %zero {in_bounds = [true]}
    : memref<16x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  return
}
func.func @plain_access(%v: f16, %i: index) {
  %s = memref.alloc() : memref<16x16xf16, #gpu.address_space<workgroup>>
  memref.store %v, %s[%i, %i] : memref<16x16xf16, #gpu.address_space<workgroup>>
  %x = memref.load %s[%i, %i] : memref<16x16xf16, #gpu.address_space<workgroup>>
  return
}
