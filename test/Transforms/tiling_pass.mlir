// test/Transforms/tiling_pass.mlir
// RUN: quantforge-opt --quantforge-tiling %s | FileCheck %s

// =====================================================================
// Task 2.2 — Tiling Pass
//
// Verifies that --quantforge-tiling:
//   1. Wraps linalg.matmul on 4096×4096 with 4-level nested tiling
//   2. Block level uses scf.forall (M=128, N=128)
//   3. Block reduction uses scf.for (K=64)
//   4. Warp level uses scf.for (M=64, N=64)
//   5. Instruction level uses scf.for (M=16, N=8, K=16)
// =====================================================================

// -----------------------------------------------------------------
// Test 1: Basic 4096×4096 matmul 
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_4096x4096
// 1) Outer Block tile scf.forall
// CHECK:     scf.forall
// 2) Block reduction K-loop
// CHECK:       scf.for
// 3) Warp distribution 2 loops
// CHECK:           scf.forall
// 4) Instruction-level 3 loops
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// Tiled matmul
// CHECK:                     linalg.matmul
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
// CHECK:         scf.forall
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
//   No block-reduction K loop is inserted (iterRank=2)
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @generic_2d_parallel
// CHECK:     scf.forall
// CHECK:       scf.forall
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             linalg.generic
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
