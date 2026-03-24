// test/Transforms/gpu_mapping_pass.mlir
// RUN: quantforge-opt --quantforge-tiling --quantforge-gpu-mapping %s | FileCheck %s

// =====================================================================
// Phase 4 — GPU Mapping Pass
//
// Verifies that --quantforge-gpu-mapping (run after --quantforge-tiling):
//   1. Sets block size function attributes
//   2. Annotates scf.forall ops with GPU mapping attributes
//   3. Inserts two gpu.barrier ops inside the K-loop
// =====================================================================

// -----------------------------------------------------------------
// Test 1: 4096x4096 matmul — full GPU mapping pipeline
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_gpu_mapped
//
// 1) Function attributes for block sizes
// CHECK-SAME: quantforge.block_size_x = 32
// CHECK-SAME: quantforge.block_size_y = 2
// CHECK-SAME: quantforge.block_size_z = 2
//
// 2) K-loop with barriers
// CHECK:         scf.forall
// CHECK:           scf.for
// CHECK-NEXT:        gpu.barrier
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   scf.for
// CHECK:                     scf.for
// CHECK:             gpu.barrier
// CHECK:             scf.yield
//
// 3) GPU mapping attribute on block forall
// CHECK:         } {mapping = [#gpu.block<y>, #gpu.block<x>]}

func.func @matmul_gpu_mapped(
    %A : tensor<4096x4096xf16>,
    %B : tensor<4096x4096xf16>,
    %C : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
{
  %result = linalg.matmul
    ins(%A, %B : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
    outs(%C    : tensor<4096x4096xf16>)
    -> tensor<4096x4096xf16>
  return %result : tensor<4096x4096xf16>
}
