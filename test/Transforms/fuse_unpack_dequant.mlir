// RUN: quantforge-opt --fuse-unpack-dequant %s | FileCheck %s

// =====================================================================
// Test 1: Basic — fuse qf.unpack + qf.dequant into one linalg.generic
// =====================================================================
// packed:   [4096, 2048] i8   (INT4-packed)
// unpacked: [4096, 4096] i8
// output:   [4096, 4096] f16

// CHECK-LABEL: func.func @fuse_basic
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK-DAG:     arith.constant 15 : i8
// CHECK-DAG:     arith.constant 4 : i8
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4096x2048x2xf16>
// CHECK:         %[[GEN:.*]] = linalg.generic
// CHECK-SAME:        ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME:             tensor<4096x2048xi8>, tensor<f16>, tensor<i8>)
// CHECK-SAME:        outs(%[[EMPTY]] : tensor<4096x2048x2xf16>)
// CHECK:         ^bb0({{.*}}: i8, {{.*}}: f16, {{.*}}: i8, {{.*}}: f16):
// CHECK:           arith.andi
// CHECK:           arith.shrui
// CHECK:           arith.andi
// CHECK:           arith.select
// CHECK:           arith.sitofp {{.*}} : i8 to f16
// CHECK:           arith.sitofp {{.*}} : i8 to f16
// CHECK:           arith.subf
// CHECK:           arith.mulf
// CHECK:           linalg.yield
// CHECK:         tensor.collapse_shape %[[GEN]] {{\[}}[0], [1, 2]{{\]}}
// CHECK-SAME:        : tensor<4096x2048x2xf16> into tensor<4096x4096xf16>
func.func @fuse_basic(%packed: tensor<4096x2048xi8>,
                       %scale: tensor<f16>,
                       %zp: tensor<i8>) -> tensor<4096x4096xf16> {
  %unpacked = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
  %output = qf.dequant %unpacked, %scale, %zp :
      tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
  return %output : tensor<4096x4096xf16>
}

// =====================================================================
// Test 2: Small shape — verify numerical semantics
// =====================================================================

// CHECK-LABEL: func.func @fuse_small
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK:         tensor.empty() : tensor<2x2x2xf16>
// CHECK:         linalg.generic
// CHECK:           arith.andi
// CHECK:           arith.shrui
// CHECK:           arith.sitofp
// CHECK:           arith.subf
// CHECK:           arith.mulf
// CHECK:         tensor.collapse_shape {{.*}} : tensor<2x2x2xf16> into tensor<2x4xf16>
func.func @fuse_small(%packed: tensor<2x2xi8>,
                       %scale: tensor<f16>,
                       %zp: tensor<i8>) -> tensor<2x4xf16> {
  %unpacked = qf.unpack %packed : tensor<2x2xi8> -> tensor<2x4xi8>
  %output = qf.dequant %unpacked, %scale, %zp :
      tensor<2x4xi8>, tensor<f16>, tensor<i8> -> tensor<2x4xf16>
  return %output : tensor<2x4xf16>
}

// =====================================================================
// Test 3: Negative — standalone qf.dequant (no unpack) should not fuse
// =====================================================================

// CHECK-LABEL: func.func @no_fuse_standalone_dequant
// CHECK:         qf.dequant
func.func @no_fuse_standalone_dequant(%input: tensor<4x8xi8>,
                                       %scale: tensor<f16>,
                                       %zp: tensor<i8>) -> tensor<4x8xf16> {
  %output = qf.dequant %input, %scale, %zp :
      tensor<4x8xi8>, tensor<f16>, tensor<i8> -> tensor<4x8xf16>
  return %output : tensor<4x8xf16>
}

// =====================================================================
// Test 4: Negative — standalone qf.unpack should not be touched
// =====================================================================

// CHECK-LABEL: func.func @no_fuse_standalone_unpack
// CHECK:         qf.unpack
func.func @no_fuse_standalone_unpack(%packed: tensor<4x4xi8>) -> tensor<4x8xi8> {
  %unpacked = qf.unpack %packed : tensor<4x4xi8> -> tensor<4x8xi8>
  return %unpacked : tensor<4x8xi8>
}

// =====================================================================
// Test 5: Same-shape unpack — pointwise mask + dequant, no collapse
// =====================================================================
// packed:   [4096, 4096] i8   (semantic marker, same-shape)
// unpacked: [4096, 4096] i8
// output:   [4096, 4096] f16

// CHECK-LABEL: func.func @fuse_same_shape
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4096x4096xf16>
// CHECK:         %[[GEN:.*]] = linalg.generic
// CHECK-SAME:        ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME:             tensor<4096x4096xi8>, tensor<f16>, tensor<i8>)
// CHECK-SAME:        outs(%[[EMPTY]] : tensor<4096x4096xf16>)
// CHECK:         ^bb0({{.*}}: i8, {{.*}}: f16, {{.*}}: i8, {{.*}}: f16):
// CHECK:           arith.andi
// CHECK:           arith.sitofp {{.*}} : i8 to f16
// CHECK:           arith.sitofp {{.*}} : i8 to f16
// CHECK:           arith.subf
// CHECK:           arith.mulf
// CHECK:           linalg.yield
// CHECK-NOT:     tensor.collapse_shape
func.func @fuse_same_shape(%packed: tensor<4096x4096xi8>,
                            %scale: tensor<f16>,
                            %zp: tensor<i8>) -> tensor<4096x4096xf16> {
  %unpacked = qf.unpack %packed : tensor<4096x4096xi8> -> tensor<4096x4096xi8>
  %output = qf.dequant %unpacked, %scale, %zp :
      tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
  return %output : tensor<4096x4096xf16>
}

// =====================================================================
// Test 6: Same-shape small — verify shape preservation
// =====================================================================

// CHECK-LABEL: func.func @fuse_same_shape_small
// CHECK-NOT:     qf.unpack
// CHECK-NOT:     qf.dequant
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         tensor.empty() : tensor<2x4xf16>
// CHECK:         linalg.generic
// CHECK:           arith.andi
// CHECK:           arith.sitofp
// CHECK:           arith.subf
// CHECK:           arith.mulf
func.func @fuse_same_shape_small(%packed: tensor<2x4xi8>,
                                  %scale: tensor<f16>,
                                  %zp: tensor<i8>) -> tensor<2x4xf16> {
  %unpacked = qf.unpack %packed : tensor<2x4xi8> -> tensor<2x4xi8>
  %output = qf.dequant %unpacked, %scale, %zp :
      tensor<2x4xi8>, tensor<f16>, tensor<i8> -> tensor<2x4xf16>
  return %output : tensor<2x4xf16>
}
