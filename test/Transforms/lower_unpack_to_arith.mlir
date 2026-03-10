// RUN: quantforge-opt --lower-unpack-to-arith %s | FileCheck %s

// Verify the affine map aliases are emitted at module level.
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// =====================================================================
// Test 1: 2D tensor — basic INT4 unpacking
// =====================================================================
// Input weight is stored as [4096, 2048] i8 (each byte = two INT4 values).
// Output should be [4096, 4096] i8 (one nibble per element).

// CHECK-LABEL: func.func @unpack_2d_basic
// CHECK-NOT:     qf.unpack
// CHECK-DAG:     arith.constant 15 : i8
// CHECK-DAG:     arith.constant 4 : i8
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4096x2048x2xi8>
// CHECK:         %[[GEN:.*]] = linalg.generic
// CHECK-SAME:        indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:        ins(%{{.*}} : tensor<4096x2048xi8>)
// CHECK-SAME:        outs(%[[EMPTY]] : tensor<4096x2048x2xi8>)
// CHECK:         ^bb0(%[[E:.*]]: i8, %{{.*}}: i8):
// CHECK:           arith.andi  %[[E]], %{{.*}} : i8
// CHECK:           arith.shrui %[[E]], %{{.*}} : i8
// CHECK:           arith.andi  %{{.*}}, %{{.*}} : i8
// CHECK:           arith.select
// CHECK:           linalg.yield
// CHECK:         tensor.collapse_shape %[[GEN]] {{\[}}[0], [1, 2]{{\]}}
// CHECK-SAME:        : tensor<4096x2048x2xi8> into tensor<4096x4096xi8>
func.func @unpack_2d_basic(%packed: tensor<4096x2048xi8>)
                            -> tensor<4096x4096xi8> {
  %unpacked = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
  return %unpacked : tensor<4096x4096xi8>
}

// =====================================================================
// Test 2: Small static shape — verify numerical semantics
// =====================================================================
// [2, 2] i8 → [2, 4] i8: each byte produces low nibble (d2=0) and
// high nibble (d2=1) laid out consecutively.

// CHECK-LABEL: func.func @unpack_small
// CHECK-NOT:     qf.unpack
// CHECK:         tensor.empty() : tensor<2x2x2xi8>
// CHECK:         linalg.generic
// CHECK:         tensor.collapse_shape {{.*}} : tensor<2x2x2xi8> into tensor<2x4xi8>
func.func @unpack_small(%packed: tensor<2x2xi8>) -> tensor<2x4xi8> {
  %unpacked = qf.unpack %packed : tensor<2x2xi8> -> tensor<2x4xi8>
  return %unpacked : tensor<2x4xi8>
}

// =====================================================================
// Test 3: GEMV-style weight for the ConvertLinalgToQuantForge pipeline
// =====================================================================
// Weight matrix [2048, 4096] i8 is the RHS of a GEMV; after unpacking
// it becomes [2048, 8192] i8 — representing 4-bit weights.

// CHECK-LABEL: func.func @unpack_gemv_weight
// CHECK-NOT:     qf.unpack
// CHECK:         tensor.empty() : tensor<2048x4096x2xi8>
// CHECK:         linalg.generic
// CHECK:         tensor.collapse_shape {{.*}} : tensor<2048x4096x2xi8> into tensor<2048x8192xi8>
func.func @unpack_gemv_weight(%packed: tensor<2048x4096xi8>)
                               -> tensor<2048x8192xi8> {
  %unpacked = qf.unpack %packed : tensor<2048x4096xi8> -> tensor<2048x8192xi8>
  return %unpacked : tensor<2048x8192xi8>
}
