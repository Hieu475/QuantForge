// RUN: quantforge-opt %s --quantforge-to-nvgpu 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Pattern 1 — LdMatrixActivationPattern
// vector.transfer_read from f16 SRAM → nvgpu.ldmatrix
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_ldmatrix_activation
// CHECK: nvgpu.ldmatrix
// CHECK-SAME: numTiles = 4
// CHECK-SAME: transpose = false
// CHECK-SAME: -> vector<4x2xf16>
func.func @test_ldmatrix_activation(
    %sram_act: memref<128x64xf16, #gpu.address_space<workgroup>>,
    %i: index,
    %j: index) -> vector<4x2xf16> {
  %pad = arith.constant 0.0 : f16
  %frag = vector.transfer_read %sram_act[%i, %j], %pad
      : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>
  return %frag : vector<4x2xf16>
}

//===----------------------------------------------------------------------===//
// Test: Pattern 1 negative — should NOT match non-SRAM f16 reads
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_ldmatrix_skip_hbm
// CHECK-NOT: nvgpu.ldmatrix
// CHECK: vector.transfer_read
func.func @test_ldmatrix_skip_hbm(
    %hbm_act: memref<128x64xf16>,
    %i: index,
    %j: index) -> vector<4x2xf16> {
  %pad = arith.constant 0.0 : f16
  %frag = vector.transfer_read %hbm_act[%i, %j], %pad
      : memref<128x64xf16>, vector<4x2xf16>
  return %frag : vector<4x2xf16>
}

//===----------------------------------------------------------------------===//
// Test: Direct mma.sync emission verification
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_mma_sync_direct
// CHECK: nvgpu.mma.sync
// CHECK-SAME: mmaShape = [16, 8, 16]
// CHECK-SAME: -> vector<2x2xf32>
func.func @test_mma_sync_direct(
    %a: vector<4x2xf16>,
    %b: vector<2x2xf16>,
    %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %result = nvgpu.mma.sync(%a, %b, %c)
      {mmaShape = [16, 8, 16]}
      : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
  return %result : vector<2x2xf32>
}

//===----------------------------------------------------------------------===//
// Test: Full pipeline subtest — ldmatrix → prmt → mma.sync all-in-one
//
// In the real use case, the entire pipeline (load, unpack, compute)
// happens within a gpu.func kernel body. The IR below simulates the
// post-bufferization/SMEM-promotion state.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_full_pipeline
// Verify Pattern 1: activation load → ldmatrix
// CHECK: nvgpu.ldmatrix
// CHECK-SAME: numTiles = 4
// CHECK-SAME: memref<128x64xf16, #gpu.address_space<workgroup>>
// CHECK-SAME: -> vector<4x2xf16>

// Verify mma.sync present
// CHECK: nvgpu.mma.sync
func.func @test_full_pipeline(
    %sram_act: memref<128x64xf16, #gpu.address_space<workgroup>>,
    %b_frag: vector<2x2xf16>,
    %accum: vector<2x2xf32>,
    %i: index, %j: index) -> vector<2x2xf32> {
  // Load activation fragment from SRAM (Pattern 1)
  %pad = arith.constant 0.0 : f16
  %a_frag = vector.transfer_read %sram_act[%i, %j], %pad
      : memref<128x64xf16, #gpu.address_space<workgroup>>, vector<4x2xf16>

  // Compute with mma.sync (direct, Pattern 4 would emit this)
  %result = nvgpu.mma.sync(%a_frag, %b_frag, %accum)
      {mmaShape = [16, 8, 16]}
      : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>

  return %result : vector<2x2xf32>
}
