// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 block-tile-m=16 block-tile-n=64 warp-tile-m=16 warp-tile-n=16 block-tile-k=64 accumulate=false epilogue=f16-store expert-problems=6 expert-roles=reeer' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 block-tile-m=16 block-tile-n=64 warp-tile-m=16 warp-tile-n=16 block-tile-k=64 accumulate=false epilogue=f16-store' %s | FileCheck --check-prefix=PLAIN %s

// Grouped GEMM for mixture-of-experts layers (expert-problems / expert-roles): the dense kernel gets grid.z = 6 problems, an extra memref<6xi32>
// argument (the expert of each problem, negative = padding) and per-problem base pointers: the weights advance by meta[ctaid.z] slabs, the
// activations / output by ctaid.z slabs. A negative expert returns before anything is read. Without the options the kernel is the plain one.

// CHECK:      blocks in ({{.*}}) threads in
// CHECK-NOT:  quantforge.expert_layout
// CHECK-DAG:  gpu.binary @w4a16_f16_kernel
// CHECK-DAG:  %ctaid.z
// CHECK-DAG:  setp.lt.s32
// CHECK-DAG:  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// CHECK-DAG:  st.global.b32
// CHECK-DAG:  _mlir_ciface_w4a16_f16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr {quantforge.int8_out}, %arg6: !llvm.ptr)
// PLAIN-NOT:  %ctaid.z
// PLAIN:      _mlir_ciface_w4a16_f16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr {quantforge.int8_out})
#mmn = affine_map<(m, n) -> (m, n)>
func.func @w4a16_f16(%act: tensor<16x512xf16>, %wp: tensor<512x128xi8>, %scale: tensor<4x256xf16>, %zp: tensor<4x256xi8>,
                     %y: tensor<16x256xf16> {quantforge.int8_out}) -> tensor<16x256xf16> {
  %w = qf.unpack %wp : tensor<512x128xi8> -> tensor<512x256xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<512x256xi8>, tensor<4x256xf16>, tensor<4x256xi8> -> tensor<512x256xf16>
  %e = tensor.empty() : tensor<16x256xf32>
  %c = linalg.matmul ins(%act, %wf : tensor<16x512xf16>, tensor<512x256xf16>) outs(%e : tensor<16x256xf32>) -> tensor<16x256xf32>
  %r = linalg.generic {indexing_maps = [#mmn, #mmn], iterator_types = ["parallel", "parallel"], quantforge.no_tile, quantforge.no_vectorize}
      ins(%c : tensor<16x256xf32>) outs(%y : tensor<16x256xf16>) {
    ^bb0(%ci: f32, %o: f16):
      %h = arith.truncf %ci : f32 to f16
      linalg.yield %h : f16
  } -> tensor<16x256xf16>
  return %r : tensor<16x256xf16>
}
