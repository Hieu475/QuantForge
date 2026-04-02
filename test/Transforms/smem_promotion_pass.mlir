// test/Transforms/smem_promotion_pass.mlir
// RUN: quantforge-opt --quantforge-smem-promotion %s | FileCheck %s

// =====================================================================
// Task 4.2 + 4.4 — Shared Memory Promotion + Synchronization
//
// Verifies that --quantforge-smem-promotion:
//   1. Allocates SRAM buffers with #gpu.address_space<workgroup>
//   2. Inserts memref.copy from HBM subview to SRAM
//   3. Inserts gpu.barrier after copies (RAW hazard)
//   4. Inserts gpu.barrier before yield (WAR hazard)
//   5. Inserts memref.dealloc for SRAM liveness
//   6. Replaces uses of HBM subview with SRAM alloc
//
// NOTE: Input is already in memref form (post-bufferization).
// =====================================================================

// -----------------------------------------------------------------
// Test 1: K-loop with tile-sized subviews gets SRAM promotion
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @smem_promotion_kloop

// Tile A [128x64] promoted
// CHECK:         memref.subview {{.*}} : memref<4096x4096xf16> to memref<128x64xf16
// CHECK:         memref.alloc() : memref<128x64xf16, #gpu.address_space<workgroup>>
// CHECK:         memref.copy

// Tile B [64x128] promoted
// CHECK:         memref.subview {{.*}} : memref<4096x4096xf16> to memref<64x128xf16
// CHECK:         memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
// CHECK:         memref.copy

// RAW barrier after copies
// CHECK:         gpu.barrier

// Compute uses SRAM
// CHECK:         memref.load {{.*}} #gpu.address_space<workgroup>

// Deallocs before WAR barrier
// CHECK:         memref.dealloc
// CHECK:         memref.dealloc

// WAR barrier before yield
// CHECK:         gpu.barrier
// CHECK:         scf.yield

func.func @smem_promotion_kloop(
    %A : memref<4096x4096xf16>,
    %B : memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c4096 = arith.constant 4096 : index

  scf.forall (%bm, %bn) in (32, 32) {
    scf.for %k = %c0 to %c4096 step %c64 {
      // Tile A: 128x64
      %tileA = memref.subview %A[%bm, %k] [128, 64] [1, 1]
        : memref<4096x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
      // Tile B: 64x128
      %tileB = memref.subview %B[%k, %bn] [64, 128] [1, 1]
        : memref<4096x4096xf16> to memref<64x128xf16, strided<[4096, 1], offset: ?>>

      // Compute: load from tiles
      %v = memref.load %tileA[%c0, %c0]
        : memref<128x64xf16, strided<[4096, 1], offset: ?>>

      scf.yield
    }
    scf.forall.in_parallel {
    }
  }
  return
}
