// RUN: quantforge-opt %s --quantforge-to-nvgpu 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: K-Loop Software Pipelining (Pattern 5)
//
// Tests that the pass correctly:
//   1. Applies ldmatrix pattern within K-loops
//   2. The resulting IR has ldmatrix before mma.sync (natural ordering)
//
// Note: The full K-loop annotation (kloop_tc) requires IR construction
// from upstream passes. This test validates the structural correctness
// of the pipeline output without K-loop annotation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_k_loop_pipeline_structure
// The loop should contain ldmatrix followed by mma.sync
// CHECK: scf.for
// CHECK: nvgpu.ldmatrix
// CHECK: nvgpu.mma.sync
// CHECK: scf.yield
func.func @test_k_loop_pipeline_structure(
    %sram_act: memref<128x64xf16, #gpu.address_space<workgroup>>,
    %b_frag: vector<2x2xf16>,
    %init_accum: vector<2x2xf32>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  // K-loop: iterate over K dimension in steps of 16
  %result = scf.for %k = %c0 to %c64 step %c16
      iter_args(%accum = %init_accum) -> vector<2x2xf32> {
    // Load activation fragment — Pattern 1 should convert to ldmatrix
    %pad = arith.constant 0.0 : f16
    %a_frag = vector.transfer_read %sram_act[%c0, %k], %pad
        : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>

    // Compute with mma.sync
    %new_accum = nvgpu.mma.sync(%a_frag, %b_frag, %accum)
        {mmaShape = [16, 8, 16]}
        : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>

    scf.yield %new_accum : vector<2x2xf32>
  }

  return %result : vector<2x2xf32>
}

//===----------------------------------------------------------------------===//
// Test: Multi-tile K-loop — 2 ldmatrix + 2 mma.sync per iteration
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_k_loop_multi_tile
// CHECK: scf.for
// Pattern 1 should convert both transfer_reads to ldmatrix
// CHECK: nvgpu.ldmatrix
// CHECK: nvgpu.ldmatrix
// CHECK: nvgpu.mma.sync
// CHECK: nvgpu.mma.sync
// CHECK: scf.yield
func.func @test_k_loop_multi_tile(
    %sram_act: memref<128x64xf16, #gpu.address_space<workgroup>>,
    %b_frag0: vector<2x2xf16>,
    %b_frag1: vector<2x2xf16>,
    %init_accum: vector<2x2xf32>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  %result = scf.for %k = %c0 to %c64 step %c16
      iter_args(%accum = %init_accum) -> vector<2x2xf32> {
    %pad = arith.constant 0.0 : f16

    // Tile 0
    %a0 = vector.transfer_read %sram_act[%c0, %k], %pad
        : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>
    %acc1 = nvgpu.mma.sync(%a0, %b_frag0, %accum)
        {mmaShape = [16, 8, 16]}
        : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>

    // Tile 1
    %a1 = vector.transfer_read %sram_act[%c16, %k], %pad
        : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>
    %acc2 = nvgpu.mma.sync(%a1, %b_frag1, %acc1)
        {mmaShape = [16, 8, 16]}
        : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>

    scf.yield %acc2 : vector<2x2xf32>
  }

  return %result : vector<2x2xf32>
}
