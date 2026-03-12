// test/Transforms/lower_unpack_branch_free.mlir
// RUN: quantforge-opt --lower-unpack-branch-free %s | FileCheck %s

// =====================================================================
// Phase 1 — Branch-Free Vectorized Unpacking
//
// Verifies that --lower-unpack-branch-free:
//   1. Eliminates qf.unpack entirely
//   2. Emits NO linalg.index and NO arith.select
//   3. Produces two separate linalg.generic ops (low + high nibble)
//   4. Interleaves them via tensor.insert_slice with stride 2
// =====================================================================

// -----------------------------------------------------------------
// Test 1: 2-D basic — standard LLM weight shape
// -----------------------------------------------------------------
// Input:   tensor<4096 x 2048 x i8>  (packed INT4)
// Output:  tensor<4096 x 4096 x i8>  (one nibble per element)
//
// Expected layout after insert_slice:
//   out[k][2n]   = low_nibble(packed[k][n])
//   out[k][2n+1] = high_nibble(packed[k][n])

// CHECK-LABEL: func.func @unpack_2d_bf
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     linalg.index
// CHECK-NOT:     arith.select
// CHECK-DAG:     arith.constant 15 : i8
// CHECK-DAG:     arith.constant 4  : i8
// Low-nibble generic (no shrui before the andi)
// CHECK:         linalg.generic
// CHECK-SAME:        ins(%{{.*}} : tensor<4096x2048xi8>)
// CHECK-SAME:        outs(%{{.*}} : tensor<4096x2048xi8>)
// CHECK:         ^bb0(%[[B:.*]]: i8, %{{.*}}: i8):
// CHECK:           arith.andi %[[B]]
// High-nibble generic (shrui then andi)
// CHECK:         linalg.generic
// CHECK-SAME:        ins(%{{.*}} : tensor<4096x2048xi8>)
// CHECK-SAME:        outs(%{{.*}} : tensor<4096x2048xi8>)
// CHECK:         ^bb0(%[[B2:.*]]: i8, %{{.*}}: i8):
// CHECK:           arith.shrui %[[B2]]
// CHECK:           arith.andi
// Stride-2 interleave
// CHECK:         tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [4096, 2048] [1, 2]
// CHECK:         tensor.insert_slice %{{.*}} into %{{.*}}[0, 1] [4096, 2048] [1, 2]
func.func @unpack_2d_bf(%packed: tensor<4096x2048xi8>)
                         -> tensor<4096x4096xi8> {
  %out = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
  return %out : tensor<4096x4096xi8>
}

// -----------------------------------------------------------------
// Test 2: Small shape — verify numerical semantics
// -----------------------------------------------------------------
// 2 × 2 input → 2 × 4 output
// out[0][0] = low(packed[0][0]),  out[0][1] = high(packed[0][0])
// out[0][2] = low(packed[0][1]),  out[0][3] = high(packed[0][1]), …

// CHECK-LABEL: func.func @unpack_small_bf
// CHECK-NOT:   qf.unpack
// CHECK-NOT:   arith.select
// CHECK:       tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [2, 2] [1, 2]
// CHECK:       tensor.insert_slice %{{.*}} into %{{.*}}[0, 1] [2, 2] [1, 2]
func.func @unpack_small_bf(%packed: tensor<2x2xi8>) -> tensor<2x4xi8> {
  %out = qf.unpack %packed : tensor<2x2xi8> -> tensor<2x4xi8>
  return %out : tensor<2x4xi8>
}

// -----------------------------------------------------------------
// Test 3: GEMV weight — verify end-to-end pipeline compatibility
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @unpack_gemv_bf
// CHECK-NOT:   qf.unpack
// CHECK-NOT:   arith.select
// CHECK:       tensor.insert_slice {{.*}} [1, 2]
// CHECK:       tensor.insert_slice {{.*}} [1, 2]
func.func @unpack_gemv_bf(%packed: tensor<2048x4096xi8>)
                           -> tensor<2048x8192xi8> {
  %out = qf.unpack %packed : tensor<2048x4096xi8> -> tensor<2048x8192xi8>
  return %out : tensor<2048x8192xi8>
}

// -----------------------------------------------------------------
// Test 4: Same-shape (semantic marker) must NOT be rewritten
// -----------------------------------------------------------------
// The branch-free pass must NOT touch same-shape qf.unpack because
// those are semantic markers consumed by ConvertLinalgToQuantForge.

// CHECK-LABEL: func.func @unpack_same_shape_bf_skip
// CHECK:         qf.unpack
func.func @unpack_same_shape_bf_skip(%packed: tensor<4x8xi8>)
                                      -> tensor<4x8xi8> {
  %out = qf.unpack %packed : tensor<4x8xi8> -> tensor<4x8xi8>
  return %out : tensor<4x8xi8>
}

// -----------------------------------------------------------------
// Test 5: Fused branch-free — qf.unpack + qf.dequant → branch-free
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @fuse_bf
// CHECK-NOT:   qf.unpack
// CHECK-NOT:   qf.dequant
// CHECK-NOT:   arith.select
// CHECK-NOT:   linalg.index
// CHECK:       linalg.generic{{.*}}ins(%{{.*}}, %{{.*}}, %{{.*}}
// CHECK:       arith.sitofp
// CHECK:       linalg.generic{{.*}}ins(%{{.*}}, %{{.*}}, %{{.*}}
// CHECK:       arith.shrui
// CHECK:       arith.sitofp
// CHECK:       tensor.insert_slice {{.*}}[0, 0]{{.*}}[1, 2]
// CHECK:       tensor.insert_slice {{.*}}[0, 1]{{.*}}[1, 2]
func.func @fuse_bf(%packed: tensor<4096x2048xi8>,
                   %scale: tensor<f16>,
                   %zp: tensor<i8>) -> tensor<4096x4096xf16> {
  %unp = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
  %out = qf.dequant %unp, %scale, %zp :
      tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
  return %out : tensor<4096x4096xf16>
}
