// RUN: not quantforge-opt %s 2>&1 | FileCheck %s

// The group extent must divide the weight extent (64 / 5 does not).
// CHECK: scale type 'tensor<5x32xf16>' is neither rank 0 nor a block-wise layout
func.func @bad_group(%w: tensor<64x32xi8>, %s: tensor<5x32xf16>, %z: tensor<5x32xi8>) -> tensor<64x32xf16> {
  %r = qf.dequant %w, %s, %z : tensor<64x32xi8>, tensor<5x32xf16>, tensor<5x32xi8> -> tensor<64x32xf16>
  return %r : tensor<64x32xf16>
}
