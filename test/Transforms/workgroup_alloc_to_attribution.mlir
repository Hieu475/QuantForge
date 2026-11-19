// test/Transforms/workgroup_alloc_to_attribution.mlir
// RUN: quantforge-opt --pass-pipeline='builtin.module(gpu.module(gpu.func(quantforge-workgroup-alloc-to-attribution)))' %s | FileCheck %s

// Workgroup memref.alloc inside a gpu.func becomes a static workgroup
// attribution; the matching dealloc is dropped; global-space allocs are kept.
// CHECK-LABEL: gpu.func @kernel
// CHECK-SAME:    workgroup(%{{.*}} : memref<16x24xf16, #gpu.address_space<workgroup>>)
// CHECK-NOT:     memref.dealloc
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<16x24xf16, #gpu.address_space<workgroup>>
// CHECK:         memref.alloc() : memref<4xf32>
gpu.module @m {
  gpu.func @kernel(%v: f16, %i: index) kernel {
    %smem = memref.alloc() : memref<16x24xf16, #gpu.address_space<workgroup>>
    memref.store %v, %smem[%i, %i] : memref<16x24xf16, #gpu.address_space<workgroup>>
    memref.dealloc %smem : memref<16x24xf16, #gpu.address_space<workgroup>>
    %g = memref.alloc() : memref<4xf32>
    gpu.return
  }
}
