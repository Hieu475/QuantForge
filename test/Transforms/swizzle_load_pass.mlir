// test/Transforms/swizzle_load_pass.mlir
// RUN: quantforge-opt --quantforge-swizzle-load %s | FileCheck %s

// =====================================================================
// Task 4.3 — XOR Swizzle Load for Bank Conflict Elimination
//
// Verifies that --quantforge-swizzle-load:
//   1. Inserts arith.remui %row, 8 for the swizzle phase
//   2. Inserts arith.xori %col, %row_mod for swizzled column
//   3. Replaces original column index in memref.load
//   4. Also swizzles memref.store to SRAM
//   5. Also swizzles destination index of nvgpu.device_async_copy
//   6. Marks processed ops with "swizzled" attribute
//   7. Does NOT swizzle global memory (HBM) access
//
// NOTE: Input already has SRAM allocations (post-smem-promotion).
// =====================================================================

// -----------------------------------------------------------------
// Test 1: memref.load from SRAM gets XOR swizzle
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @swizzle_sram_load
// CHECK:         %[[PHASE:.*]] = arith.constant 8 : index
// CHECK:         %[[ROW_MOD:.*]] = arith.remui %{{.*}}, %[[PHASE]]
// CHECK:         %[[SWIZ_COL:.*]] = arith.xori %{{.*}}, %[[ROW_MOD]]
// CHECK:         memref.load %{{.*}}[%{{.*}}, %[[SWIZ_COL]]]
// CHECK-SAME:      {swizzled}

func.func @swizzle_sram_load(
    %sram : memref<128x64xf16, #gpu.address_space<workgroup>>,
    %row : index, %col : index) -> f16 {
  %val = memref.load %sram[%row, %col]
    : memref<128x64xf16, #gpu.address_space<workgroup>>
  return %val : f16
}

// -----------------------------------------------------------------
// Test 2: memref.store to SRAM also gets swizzled
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @swizzle_sram_store
// CHECK:         arith.remui
// CHECK:         arith.xori
// CHECK:         memref.store
// CHECK-SAME:      {swizzled}

func.func @swizzle_sram_store(
    %sram : memref<128x64xf16, #gpu.address_space<workgroup>>,
    %val : f16, %row : index, %col : index) {
  memref.store %val, %sram[%row, %col]
    : memref<128x64xf16, #gpu.address_space<workgroup>>
  return
}

// -----------------------------------------------------------------
// Test 3: nvgpu.device_async_copy to SRAM gets swizzled on dst index
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @swizzle_async_copy_dst
// CHECK:         arith.remui
// CHECK:         arith.xori
// CHECK:         nvgpu.device_async_copy
// CHECK-SAME:      {swizzled}

func.func @swizzle_async_copy_dst(
    %src : memref<128x64xf16>,
    %dst : memref<128x64xf16, #gpu.address_space<workgroup>>,
    %row : index, %col : index) {
  %c8 = arith.constant 8 : index
  %t = nvgpu.device_async_copy %src[%row, %col], %dst[%row, %col], 8, %c8
    : memref<128x64xf16> to memref<128x64xf16, #gpu.address_space<workgroup>>
  %g = nvgpu.device_async_create_group %t
  nvgpu.device_async_wait %g {numGroups = 0 : i32}
  return
}

// -----------------------------------------------------------------
// Test 4: Global memory load is NOT swizzled
// -----------------------------------------------------------------
// CHECK-LABEL: func.func @no_swizzle_global
// CHECK-NOT:     arith.xori
// CHECK:         memref.load
// CHECK-NOT:     {swizzled}

func.func @no_swizzle_global(
    %global : memref<4096x4096xf16>,
    %row : index, %col : index) -> f16 {
  %val = memref.load %global[%row, %col] : memref<4096x4096xf16>
  return %val : f16
}
