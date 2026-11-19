// RUN: quantforge-opt --lower-dequant-to-arith %s | FileCheck %s
// RUN: quantforge-opt %s | FileCheck %s --check-prefix=RT

// FP8 E4M3 weights: the 7 magnitude bits shifted by 7 are an f16 that is 2^-8 times the FP8 value
// (sign bit moved to bit 15); x256 is exact, then the group scale. The zero point is not read.

// CHECK-LABEL: func.func @dequant_fp8
// CHECK-NOT:     qf.dequant
// CHECK:         linalg.generic
// CHECK:         arith.extui {{.*}} : i8 to i16
// CHECK:         arith.andi
// CHECK:         arith.shli
// CHECK:         arith.andi
// CHECK:         arith.shli
// CHECK:         arith.ori
// CHECK:         arith.bitcast {{.*}} : i16 to f16
// CHECK:         arith.mulf {{.*}} : f16
// CHECK:         arith.mulf {{.*}} : f16
// CHECK-NOT:     arith.sitofp
// CHECK:         linalg.yield
func.func @dequant_fp8(%w: tensor<256x128xi8>, %scale: tensor<2x128xf16>,
                       %zp: tensor<2x128xi8>) -> tensor<256x128xf16> {
  %o = qf.dequant %w, %scale, %zp {fp8} :
      tensor<256x128xi8>, tensor<2x128xf16>, tensor<2x128xi8> -> tensor<256x128xf16>
  return %o : tensor<256x128xf16>
}

// The attribute survives printing / parsing.
// RT-LABEL: func.func @roundtrip
// RT:         qf.dequant {{.*}} {fp8} :
func.func @roundtrip(%w: tensor<256x128xi8>, %scale: tensor<2x128xf16>,
                     %zp: tensor<2x128xi8>) -> tensor<256x128xf16> {
  %o = qf.dequant %w, %scale, %zp {fp8} :
      tensor<256x128xi8>, tensor<2x128xf16>, tensor<2x128xi8> -> tensor<256x128xf16>
  return %o : tensor<256x128xf16>
}
