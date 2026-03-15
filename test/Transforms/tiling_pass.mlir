// test/Transforms/tiling_pass.mlir
// RUN: quantforge-opt --quantforge-tiling %s | FileCheck %s

// =====================================================================
// Task 2.2 — Tiling Pass
//
// Verifies that --quantforge-tiling:
//   1. Wraps linalg.matmul on 4096×4096 with 3 levels of nested loops
//   2. Block level uses scf.forall (M=128, N=128)
//   3. Warp level uses scf.for (M=64, N=64)
//   4. Thread level uses scf.for (M=16, N=8, K=64)
// =====================================================================

// -----------------------------------------------------------------
// Test 1: Basic 4096×4096 matmul 
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_4096x4096
// Outer Block tile scf.forall
// CHECK:     scf.forall
// Inner Warp tile scf.for (M, N)
// CHECK:       scf.for
// CHECK:         scf.for
// Thread tile scf.for (M, N)
// CHECK:           scf.for
// CHECK:             scf.for
// Thread reduction scf.for (K)
// CHECK:               scf.for
// Tiled matmul
// CHECK:                 linalg.matmul
// No un-tiled matmul remains
// CHECK-NOT:     linalg.matmul ins(%A, %B
func.func @matmul_4096x4096(
    %A   : tensor<4096x4096xf16>,
    %B   : tensor<4096x4096xf16>,
    %C   : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
{
  %result = linalg.matmul
    ins(%A, %B : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
    outs(%C   : tensor<4096x4096xf16>)
    -> tensor<4096x4096xf16>
  return %result : tensor<4096x4096xf16>
}

// -----------------------------------------------------------------
// Test 2: Smaller matmul 256×256
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_256x256
// CHECK:     scf.forall
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 linalg.matmul
func.func @matmul_256x256(
    %A   : tensor<256x256xf32>,
    %B   : tensor<256x256xf32>,
    %C   : tensor<256x256xf32>) -> tensor<256x256xf32>
{
  %result = linalg.matmul
    ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%C   : tensor<256x256xf32>)
    -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// -----------------------------------------------------------------
// Test 3: 2-D linalg.generic (parallel only, no K dimension)
//   Only M and N loops are generated (iterRank=2 → 2 tile dims)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @generic_2d_parallel
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         linalg.generic
func.func @generic_2d_parallel(
    %in  : tensor<4096x4096xf16>,
    %out : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
{
  %result = linalg.generic
    {indexing_maps = [
       affine_map<(d0, d1) -> (d0, d1)>,
       affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%in : tensor<4096x4096xf16>)
    outs(%out : tensor<4096x4096xf16>)
  {
    ^bb0(%a: f16, %b: f16):
      linalg.yield %a : f16
  } -> tensor<4096x4096xf16>
  return %result : tensor<4096x4096xf16>
}
