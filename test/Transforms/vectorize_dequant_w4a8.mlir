// RUN: quantforge-opt %s --quantforge-vectorize-dequant --canonicalize | FileCheck %s
// W4A8 weight tile: 16 scalar trees (nibble - z) * s of the pack_k8 layout become 32-bit word arithmetic (SWAR).

func.func @w4a8_tile(%w: memref<4x64xi8>, %zm: memref<4xi8>, %sm: memref<4xi8>, %out: memref<4x16xi8>, %r: index, %c: index) {
  %c0 = arith.constant 0 : index
  %c15 = arith.constant 15 : i8
  %c4 = arith.constant 4 : i8
  %z = memref.load %zm[%r] : memref<4xi8>
  %s = memref.load %sm[%r] : memref<4xi8>
  %p = vector.load %w[%r, %c] : memref<4x64xi8>, vector<8xi8>
  %init = arith.constant dense<0> : vector<16xi8>
  %b0 = vector.extract %p[0] : i8 from vector<8xi8>
  %x0 = arith.andi %b0, %c15 : i8
  %d0 = arith.subi %x0, %z : i8
  %e0 = arith.muli %d0, %s : i8
  %v0 = vector.insert %e0, %init [0] : i8 into vector<16xi8>
  %b1 = vector.extract %p[1] : i8 from vector<8xi8>
  %x1 = arith.andi %b1, %c15 : i8
  %d1 = arith.subi %x1, %z : i8
  %e1 = arith.muli %d1, %s : i8
  %v1 = vector.insert %e1, %v0 [1] : i8 into vector<16xi8>
  %b2 = vector.extract %p[2] : i8 from vector<8xi8>
  %x2 = arith.andi %b2, %c15 : i8
  %d2 = arith.subi %x2, %z : i8
  %e2 = arith.muli %d2, %s : i8
  %v2 = vector.insert %e2, %v1 [2] : i8 into vector<16xi8>
  %b3 = vector.extract %p[3] : i8 from vector<8xi8>
  %x3 = arith.andi %b3, %c15 : i8
  %d3 = arith.subi %x3, %z : i8
  %e3 = arith.muli %d3, %s : i8
  %v3 = vector.insert %e3, %v2 [3] : i8 into vector<16xi8>
  %b4 = vector.extract %p[0] : i8 from vector<8xi8>
  %x4 = arith.shrui %b4, %c4 : i8
  %d4 = arith.subi %x4, %z : i8
  %e4 = arith.muli %d4, %s : i8
  %v4 = vector.insert %e4, %v3 [4] : i8 into vector<16xi8>
  %b5 = vector.extract %p[1] : i8 from vector<8xi8>
  %x5 = arith.shrui %b5, %c4 : i8
  %d5 = arith.subi %x5, %z : i8
  %e5 = arith.muli %d5, %s : i8
  %v5 = vector.insert %e5, %v4 [5] : i8 into vector<16xi8>
  %b6 = vector.extract %p[2] : i8 from vector<8xi8>
  %x6 = arith.shrui %b6, %c4 : i8
  %d6 = arith.subi %x6, %z : i8
  %e6 = arith.muli %d6, %s : i8
  %v6 = vector.insert %e6, %v5 [6] : i8 into vector<16xi8>
  %b7 = vector.extract %p[3] : i8 from vector<8xi8>
  %x7 = arith.shrui %b7, %c4 : i8
  %d7 = arith.subi %x7, %z : i8
  %e7 = arith.muli %d7, %s : i8
  %v7 = vector.insert %e7, %v6 [7] : i8 into vector<16xi8>
  %b8 = vector.extract %p[4] : i8 from vector<8xi8>
  %x8 = arith.andi %b8, %c15 : i8
  %d8 = arith.subi %x8, %z : i8
  %e8 = arith.muli %d8, %s : i8
  %v8 = vector.insert %e8, %v7 [8] : i8 into vector<16xi8>
  %b9 = vector.extract %p[5] : i8 from vector<8xi8>
  %x9 = arith.andi %b9, %c15 : i8
  %d9 = arith.subi %x9, %z : i8
  %e9 = arith.muli %d9, %s : i8
  %v9 = vector.insert %e9, %v8 [9] : i8 into vector<16xi8>
  %b10 = vector.extract %p[6] : i8 from vector<8xi8>
  %x10 = arith.andi %b10, %c15 : i8
  %d10 = arith.subi %x10, %z : i8
  %e10 = arith.muli %d10, %s : i8
  %v10 = vector.insert %e10, %v9 [10] : i8 into vector<16xi8>
  %b11 = vector.extract %p[7] : i8 from vector<8xi8>
  %x11 = arith.andi %b11, %c15 : i8
  %d11 = arith.subi %x11, %z : i8
  %e11 = arith.muli %d11, %s : i8
  %v11 = vector.insert %e11, %v10 [11] : i8 into vector<16xi8>
  %b12 = vector.extract %p[4] : i8 from vector<8xi8>
  %x12 = arith.shrui %b12, %c4 : i8
  %d12 = arith.subi %x12, %z : i8
  %e12 = arith.muli %d12, %s : i8
  %v12 = vector.insert %e12, %v11 [12] : i8 into vector<16xi8>
  %b13 = vector.extract %p[5] : i8 from vector<8xi8>
  %x13 = arith.shrui %b13, %c4 : i8
  %d13 = arith.subi %x13, %z : i8
  %e13 = arith.muli %d13, %s : i8
  %v13 = vector.insert %e13, %v12 [13] : i8 into vector<16xi8>
  %b14 = vector.extract %p[6] : i8 from vector<8xi8>
  %x14 = arith.shrui %b14, %c4 : i8
  %d14 = arith.subi %x14, %z : i8
  %e14 = arith.muli %d14, %s : i8
  %v14 = vector.insert %e14, %v13 [14] : i8 into vector<16xi8>
  %b15 = vector.extract %p[7] : i8 from vector<8xi8>
  %x15 = arith.shrui %b15, %c4 : i8
  %d15 = arith.subi %x15, %z : i8
  %e15 = arith.muli %d15, %s : i8
  %v15 = vector.insert %e15, %v14 [15] : i8 into vector<16xi8>
  vector.store %v15, %out[%r, %c0] : memref<4x16xi8>, vector<16xi8>
  return
}

// CHECK-LABEL: func.func @w4a8_tile
// CHECK-DAG: arith.constant 252645135 : i32
// CHECK-DAG: arith.constant -2139062144 : i32
// CHECK-NOT: arith.muli {{.*}} : i8
// CHECK: vector.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
// CHECK: arith.muli {{.*}} : i32
// CHECK: vector.bitcast {{.*}} : vector<4xi32> to vector<16xi8>
// CHECK: vector.store {{.*}} : memref<4x16xi8>, vector<16xi8>
// CHECK-NOT: arith.muli {{.*}} : i8

// A tree whose element does not follow the pack_k8 order is left alone.
func.func @w4a8_wrong_order(%p: vector<8xi8>, %z: i8, %s: i8, %out: memref<16xi8>) {
  %c15 = arith.constant 15 : i8
  %c0 = arith.constant 0 : index
  %init = arith.constant dense<0> : vector<16xi8>
  %b0 = vector.extract %p[1] : i8 from vector<8xi8>
  %x0 = arith.andi %b0, %c15 : i8
  %d0 = arith.subi %x0, %z : i8
  %e0 = arith.muli %d0, %s : i8
  %v0 = vector.insert %e0, %init [0] : i8 into vector<16xi8>
  %b1 = vector.extract %p[1] : i8 from vector<8xi8>
  %x1 = arith.andi %b1, %c15 : i8
  %d1 = arith.subi %x1, %z : i8
  %e1 = arith.muli %d1, %s : i8
  %v1 = vector.insert %e1, %v0 [1] : i8 into vector<16xi8>
  %b2 = vector.extract %p[2] : i8 from vector<8xi8>
  %x2 = arith.andi %b2, %c15 : i8
  %d2 = arith.subi %x2, %z : i8
  %e2 = arith.muli %d2, %s : i8
  %v2 = vector.insert %e2, %v1 [2] : i8 into vector<16xi8>
  %b3 = vector.extract %p[3] : i8 from vector<8xi8>
  %x3 = arith.andi %b3, %c15 : i8
  %d3 = arith.subi %x3, %z : i8
  %e3 = arith.muli %d3, %s : i8
  %v3 = vector.insert %e3, %v2 [3] : i8 into vector<16xi8>
  %b4 = vector.extract %p[0] : i8 from vector<8xi8>
  %c4_4 = arith.constant 4 : i8
  %x4 = arith.shrui %b4, %c4_4 : i8
  %d4 = arith.subi %x4, %z : i8
  %e4 = arith.muli %d4, %s : i8
  %v4 = vector.insert %e4, %v3 [4] : i8 into vector<16xi8>
  %b5 = vector.extract %p[1] : i8 from vector<8xi8>
  %c4_5 = arith.constant 4 : i8
  %x5 = arith.shrui %b5, %c4_5 : i8
  %d5 = arith.subi %x5, %z : i8
  %e5 = arith.muli %d5, %s : i8
  %v5 = vector.insert %e5, %v4 [5] : i8 into vector<16xi8>
  %b6 = vector.extract %p[2] : i8 from vector<8xi8>
  %c4_6 = arith.constant 4 : i8
  %x6 = arith.shrui %b6, %c4_6 : i8
  %d6 = arith.subi %x6, %z : i8
  %e6 = arith.muli %d6, %s : i8
  %v6 = vector.insert %e6, %v5 [6] : i8 into vector<16xi8>
  %b7 = vector.extract %p[3] : i8 from vector<8xi8>
  %c4_7 = arith.constant 4 : i8
  %x7 = arith.shrui %b7, %c4_7 : i8
  %d7 = arith.subi %x7, %z : i8
  %e7 = arith.muli %d7, %s : i8
  %v7 = vector.insert %e7, %v6 [7] : i8 into vector<16xi8>
  %b8 = vector.extract %p[4] : i8 from vector<8xi8>
  %x8 = arith.andi %b8, %c15 : i8
  %d8 = arith.subi %x8, %z : i8
  %e8 = arith.muli %d8, %s : i8
  %v8 = vector.insert %e8, %v7 [8] : i8 into vector<16xi8>
  %b9 = vector.extract %p[5] : i8 from vector<8xi8>
  %x9 = arith.andi %b9, %c15 : i8
  %d9 = arith.subi %x9, %z : i8
  %e9 = arith.muli %d9, %s : i8
  %v9 = vector.insert %e9, %v8 [9] : i8 into vector<16xi8>
  %b10 = vector.extract %p[6] : i8 from vector<8xi8>
  %x10 = arith.andi %b10, %c15 : i8
  %d10 = arith.subi %x10, %z : i8
  %e10 = arith.muli %d10, %s : i8
  %v10 = vector.insert %e10, %v9 [10] : i8 into vector<16xi8>
  %b11 = vector.extract %p[7] : i8 from vector<8xi8>
  %x11 = arith.andi %b11, %c15 : i8
  %d11 = arith.subi %x11, %z : i8
  %e11 = arith.muli %d11, %s : i8
  %v11 = vector.insert %e11, %v10 [11] : i8 into vector<16xi8>
  %b12 = vector.extract %p[4] : i8 from vector<8xi8>
  %c4_12 = arith.constant 4 : i8
  %x12 = arith.shrui %b12, %c4_12 : i8
  %d12 = arith.subi %x12, %z : i8
  %e12 = arith.muli %d12, %s : i8
  %v12 = vector.insert %e12, %v11 [12] : i8 into vector<16xi8>
  %b13 = vector.extract %p[5] : i8 from vector<8xi8>
  %c4_13 = arith.constant 4 : i8
  %x13 = arith.shrui %b13, %c4_13 : i8
  %d13 = arith.subi %x13, %z : i8
  %e13 = arith.muli %d13, %s : i8
  %v13 = vector.insert %e13, %v12 [13] : i8 into vector<16xi8>
  %b14 = vector.extract %p[6] : i8 from vector<8xi8>
  %c4_14 = arith.constant 4 : i8
  %x14 = arith.shrui %b14, %c4_14 : i8
  %d14 = arith.subi %x14, %z : i8
  %e14 = arith.muli %d14, %s : i8
  %v14 = vector.insert %e14, %v13 [14] : i8 into vector<16xi8>
  %b15 = vector.extract %p[7] : i8 from vector<8xi8>
  %c4_15 = arith.constant 4 : i8
  %x15 = arith.shrui %b15, %c4_15 : i8
  %d15 = arith.subi %x15, %z : i8
  %e15 = arith.muli %d15, %s : i8
  %v15 = vector.insert %e15, %v14 [15] : i8 into vector<16xi8>
  vector.store %v15, %out[%c0] : memref<16xi8>, vector<16xi8>
  return
}
// CHECK-LABEL: func.func @w4a8_wrong_order
// CHECK: arith.muli {{.*}} : i8
// CHECK-NOT: vector.bitcast
