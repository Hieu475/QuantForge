// RUN: quantforge-opt --quantforge-lower-to-nvvm %s | FileCheck %s

// CHECK-LABEL: gpu.module @test_kernel_module
// CHECK:         llvm.func @test_kernel
// CHECK-SAME:      attributes {gpu.kernel, nvvm.kernel}
gpu.module @test_kernel_module {
  gpu.func @test_kernel(%arg0: memref<128xf32, 1>) kernel {

    // 1) Thread/Block ID -> SREG Intrinsics
    // CHECK: %{{.*}} = nvvm.read.ptx.sreg.tid.x : i32
    %tx = gpu.thread_id x

    // 2) GPU Barrier -> NVVM Barrier
    // CHECK: nvvm.barrier0
    gpu.barrier

    // 3) MemRef Store -> LLVM Store (with GEP)
    // CHECK: llvm.getelementptr
    // CHECK: llvm.store
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.0 : f32
    memref.store %cst, %arg0[%c0] : memref<128xf32, 1>

    gpu.return
  }
}
