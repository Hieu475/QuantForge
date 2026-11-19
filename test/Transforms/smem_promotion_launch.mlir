// test/Transforms/smem_promotion_launch.mlir
// RUN: quantforge-opt --quantforge-smem-promotion='copy-unroll=1 smem-vector-copy=false smem-async-copy=false' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-smem-promotion='smem-vector-copy=false smem-async-copy=false' %s | FileCheck --check-prefix=UNROLL %s
// RUN: quantforge-opt --quantforge-smem-promotion='smem-pad=8 copy-unroll=1 smem-vector-copy=false smem-async-copy=false' %s | FileCheck --check-prefix=PAD %s
// RUN: quantforge-opt --quantforge-smem-promotion='smem-pad=8 smem-async-copy=false' %s | FileCheck --check-prefix=VEC %s
// RUN: quantforge-opt --quantforge-smem-promotion='smem-pad=8' %s | FileCheck --check-prefix=ASYNC %s

// =====================================================================
// Shared-memory promotion inside a gpu.launch:
//   * the copy HBM -> SRAM is distributed over all 128 threads:
//     i = tid + t * 128 with a constant trip count (64) that is unrolled by
//     --copy-unroll (default 8) so global loads are issued back to back
//   * downstream subviews of the promoted tile are re-created with the
//     workgroup memory space (previously produced invalid IR)
//   * a tile that is WRITTEN (accumulator) is never promoted
//   * --smem-pad pads the row stride of the SRAM buffer
//   * smem-vector-copy: 16 B work items, one vector.store (st.shared.v4)
//   * smem-async-copy: cp.async (nvgpu.device_async_copy) for tiles whose
//     source is a kernel argument, followed by commit_group + wait_group 0
//     before the barrier that publishes the tile
// =====================================================================

// CHECK-LABEL: func.func @promote_in_launch
// CHECK:         gpu.launch
// CHECK:           scf.for %[[K:.*]] =
// A tile [128x64] promoted, distributed copy: i = tid + t*128, t in [0, 64)
// CHECK:             %[[A:.*]] = memref.alloc() : memref<128x64xf16, #gpu.address_space<workgroup>>
// CHECK:             scf.for %{{.*}} = %{{.*}} to %c64{{.*}} step %c1
// CHECK:               arith.divui
// CHECK:               arith.remui
// CHECK:               memref.store {{.*}} : memref<128x64xf16, #gpu.address_space<workgroup>>
// B tile promoted
// CHECK:             memref.alloc() : memref<64x128xf16, #gpu.address_space<workgroup>>
// CHECK:             gpu.barrier
// Warp-level subview of the promoted tile lives in shared memory
// CHECK:             memref.subview %[[A]]{{.*}} to memref<64x64xf16, strided<[64, 1], offset: ?>, #gpu.address_space<workgroup>>
// The written accumulator stays in global memory
// CHECK-NOT:         memref.alloc() : memref<64x64xf16
// CHECK:             vector.transfer_write {{.*}} memref<64x64xf16, strided<[128, 1], offset: ?>>
// UNROLL-LABEL: func.func @promote_in_launch
// UNROLL:         memref.alloc() : memref<128x64xf16, #gpu.address_space<workgroup>>
// UNROLL:         scf.for %{{.*}} = %{{.*}} to %c64{{.*}} step %c8
// 8 copies per iteration
// UNROLL-COUNT-8:   memref.store {{.*}} : memref<128x64xf16, #gpu.address_space<workgroup>>
// PAD-LABEL: func.func @promote_in_launch
// PAD:         memref.alloc() : memref<128x72xf16, #gpu.address_space<workgroup>>
// PAD:         memref.subview {{.*}} to memref<128x64xf16, strided<[72, 1]>, #gpu.address_space<workgroup>>
func.func @promote_in_launch(%A: memref<128x128xf16>, %B: memref<128x128xf16>,
                             %C: memref<64x64xf16, strided<[128, 1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %zero = arith.constant 0.0 : f16
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c2, %sz = %c2) {
    scf.for %k = %c0 to %c128 step %c64 {
      %tA = memref.subview %A[0, %k] [128, 64] [1, 1]
        : memref<128x128xf16> to memref<128x64xf16, strided<[128, 1], offset: ?>>
      %tB = memref.subview %B[%k, 0] [64, 128] [1, 1]
        : memref<128x128xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      gpu.barrier
      %wA = memref.subview %tA[%c0, 0] [64, 64] [1, 1]
        : memref<128x64xf16, strided<[128, 1], offset: ?>> to memref<64x64xf16, strided<[128, 1], offset: ?>>
      %a = vector.transfer_read %wA[%c0, %c0], %zero {in_bounds = [true, true]}
        : memref<64x64xf16, strided<[128, 1], offset: ?>>, vector<4x4xf16>
      // accumulator-shaped tile (64x64) that is written -> must stay global
      %acc = memref.subview %C[0, 0] [64, 64] [1, 1]
        : memref<64x64xf16, strided<[128, 1], offset: ?>> to memref<64x64xf16, strided<[128, 1], offset: ?>>
      vector.transfer_write %a, %acc[%c0, %c0] {in_bounds = [true, true]}
        : vector<4x4xf16>, memref<64x64xf16, strided<[128, 1], offset: ?>>
      gpu.barrier
    }
    gpu.terminator
  }
  return
}

// (vector copy) 8 elements per work item, packed into one 128-bit store
// VEC-LABEL: func.func @promote_in_launch
// VEC:         scf.for
// VEC:           memref.load {{.*}} : memref<128x64xf16, strided<[128, 1], offset: ?>>
// VEC:           vector.insert
// VEC:           vector.store {{.*}} : memref<128x64xf16, strided<[72, 1]>, #gpu.address_space<workgroup>>, vector<8xf16>
// (async copy) cp.async because A and B live in kernel arguments, then
// commit + wait + barrier
// ASYNC-LABEL: func.func @promote_in_launch
// ASYNC:         nvgpu.device_async_copy {{.*}}, 8 : memref<128x64xf16, strided<[128, 1], offset: ?>> to memref<128x64xf16, strided<[72, 1]>, #gpu.address_space<workgroup>>
// ASYNC:         nvgpu.device_async_copy {{.*}}, 8 : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16, strided<[136, 1]>, #gpu.address_space<workgroup>>
// ASYNC:         %[[G:.*]] = nvgpu.device_async_create_group
// ASYNC-NEXT:    nvgpu.device_async_wait %[[G]] {numGroups = 0 : i32}
// ASYNC-NEXT:    gpu.barrier

