// RUN: quantforge-opt %s --quantforge-to-nvgpu | FileCheck %s

// Memref in SRAM (memory_space = 3) dead scalar load -> nvgpu.ldmatrix.
// CHECK-LABEL: func.func @test_ldmatrix_activation
// CHECK: %[[FRAG:.*]] = nvgpu.ldmatrix %{{.*}}[%{{.*}}, %{{.*}}] {numTiles = 4 : i32, transpose = false}
// CHECK-SAME: : memref<128x64xf16, #gpu.address_space<workgroup>> -> vector<4x2xf16>
// CHECK-NOT: memref.load
// CHECK: return
func.func @test_ldmatrix_activation(
    %sram_act: memref<128x64xf16, #gpu.address_space<workgroup>>,
    %i: index,
    %j: index) {
  %dead = memref.load %sram_act[%i, %j]
      : memref<128x64xf16, #gpu.address_space<workgroup>>
  return
}
