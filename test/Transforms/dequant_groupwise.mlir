// RUN: quantforge-opt --lower-dequant-to-arith %s | FileCheck %s --check-prefix=LOWER
// RUN: quantforge-opt --fuse-unpack-dequant %s | FileCheck %s --check-prefix=FUSE
// RUN: quantforge-opt --fuse-unpack-dequant-branch-free %s | FileCheck %s --check-prefix=BF

// Group-wise quantization: scale / zero point [K/G, N] against a [K, N] weight.

// FUSE-DAG:   affine_map<(d0, d1, d2) -> (d0 floordiv 64, d1 * 2 + d2)>
// BF-DAG:     affine_map<(d0, d1) -> (d0 floordiv 64, d1 * 2)>
// BF-DAG:     affine_map<(d0, d1) -> (d0 floordiv 64, d1 * 2 + 1)>
// LOWER-DAG: #[[SCALE:.*]] = affine_map<(d0, d1) -> (d0 floordiv 32, d1)>
// LOWER-LABEL: func.func @lower_group
// LOWER:       linalg.generic
// LOWER-SAME:    indexing_maps = [#map, #[[SCALE]], #[[SCALE]], #map]
// FUSE-LABEL:  func.func @lower_group
// BF-LABEL:    func.func @lower_group
func.func @lower_group(%w: tensor<128x64xi8>, %s: tensor<4x64xf16>,
                       %z: tensor<4x64xi8>) -> tensor<128x64xf16> {
  %r = qf.dequant %w, %s, %z : tensor<128x64xi8>, tensor<4x64xf16>, tensor<4x64xi8> -> tensor<128x64xf16>
  return %r : tensor<128x64xf16>
}

// Fused with the INT4 unpack: loop dims (k, n/2, nibble); the column the
// scale is looked up with is 2*d1 + d2.
// FUSE-LABEL:  func.func @fuse_group
// FUSE:        linalg.generic
// FUSE-SAME:     ins(%{{[^:]*}}: tensor<128x32xi8>, tensor<2x64xf16>, tensor<2x64xi8>)
// BF-LABEL:    func.func @fuse_group
func.func @fuse_group(%p: tensor<128x32xi8>, %s: tensor<2x64xf16>,
                      %z: tensor<2x64xi8>) -> tensor<128x64xf16> {
  %u = qf.unpack %p : tensor<128x32xi8> -> tensor<128x64xi8>
  %r = qf.dequant %u, %s, %z : tensor<128x64xi8>, tensor<2x64xf16>, tensor<2x64xi8> -> tensor<128x64xf16>
  return %r : tensor<128x64xf16>
}
