// test/Transforms/smem_pipeline.mlir
// RUN: quantforge-opt --quantforge-smem-promotion='block-tile-m=64 block-tile-n=64 block-tile-k=32 warp-tile-m=16 warp-tile-n=16 smem-pad=8 smem-stages=2' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-smem-promotion='block-tile-m=64 block-tile-n=64 block-tile-k=32 warp-tile-m=16 warp-tile-n=16 smem-pad=8 smem-stages=1' %s | FileCheck --check-prefix=SINGLE %s
// RUN: not quantforge-opt --quantforge-smem-promotion='block-tile-m=64 block-tile-n=64 block-tile-k=32 warp-tile-m=16 warp-tile-n=16 smem-pad=8000' %s 2>&1 | FileCheck --check-prefix=BUDGET %s

// =====================================================================
// Double-buffered K loop (smem-stages=2)
//  * one [2 x R x (C+pad)] buffer per tile, allocated OUTSIDE the K loop
//  * prologue: tile 0 -> stage 0, commit + wait + barrier BEFORE the loop
//  * body: prefetch of tile i+1 into stage (i+1)%2 with a CLAMPED index
//    (arith.minui, no scf.if), compute on stage i%2, then wait and the
//    single barrier at the end of the body
//  * the pre-compute barrier of the single-buffered schedule is gone
// =====================================================================

// CHECK-LABEL: func.func @pipelined
// CHECK:         gpu.launch
// CHECK:           memref.alloc() : memref<2x64x40xf16, #gpu.address_space<workgroup>>
// CHECK:           memref.alloc() : memref<2x32x72xf16, #gpu.address_space<workgroup>>
// prologue
// CHECK:           nvgpu.device_async_copy
// CHECK:           nvgpu.device_async_wait
// CHECK-NEXT:      gpu.barrier
// loop
// CHECK:           scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step
// CHECK:             arith.remui
// CHECK:             arith.minui
// CHECK:             nvgpu.device_async_copy
// CHECK:             vector.transfer_read
// CHECK:             nvgpu.device_async_wait
// CHECK-NEXT:        gpu.barrier
// CHECK-NOT:         gpu.barrier
// CHECK:           }
// SINGLE-LABEL: func.func @pipelined
// SINGLE:         scf.for
// SINGLE:           memref.alloc() : memref<64x40xf16, #gpu.address_space<workgroup>>
// BUDGET: promoted tiles need {{[0-9]+}} bytes of shared memory
func.func @pipelined(%A: memref<64x128xf16>, %B: memref<128x64xf16>,
                     %out: memref<16x16xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %zero = arith.constant 0.0 : f16
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%sx = %c32, %sy = %c1, %sz = %c1) {
    scf.for %k = %c0 to %c128 step %c32 {
      %tA = memref.subview %A[0, %k] [64, 32] [1, 1]
        : memref<64x128xf16> to memref<64x32xf16, strided<[128, 1], offset: ?>>
      %tB = memref.subview %B[%k, 0] [32, 64] [1, 1]
        : memref<128x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
      gpu.barrier
      %a = vector.transfer_read %tA[%c0, %c0], %zero {in_bounds = [true, true]}
        : memref<64x32xf16, strided<[128, 1], offset: ?>>, vector<16x16xf16>
      %b = vector.transfer_read %tB[%c0, %c0], %zero {in_bounds = [true, true]}
        : memref<32x64xf16, strided<[64, 1], offset: ?>>, vector<16x16xf16>
      %s = arith.addf %a, %b : vector<16x16xf16>
      vector.transfer_write %s, %out[%c0, %c0] {in_bounds = [true, true]}
        : vector<16x16xf16>, memref<16x16xf16>
      gpu.barrier
    }
    gpu.terminator
  }
  return
}
