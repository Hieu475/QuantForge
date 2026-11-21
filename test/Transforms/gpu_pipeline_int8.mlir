// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 block-tile-m=128 block-tile-n=128 block-tile-k=64 warp-tile-m=64 warp-tile-n=64 smem-stages=2 thread-tile-k=32 smem-pad=16 accumulate=false' %s | FileCheck %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='chip=sm_80 block-tile-m=128 block-tile-n=128 block-tile-k=64 warp-tile-m=64 warp-tile-n=64 smem-stages=1 smem-pad=16' %s | FileCheck --check-prefix=K16 %s

// Plain INT8 GEMM (experimental): i8 x i8 -> i32 on the Tensor Cores. The vectorised matmul is `contract(extsi a, extsi b, acc)`;
// the vectorization pass folds the sign extensions into the contraction (LLVM 18 only folds extf) so that the i8 operands reach
// the nvgpu lowering. `thread-tile-k=32` selects the m16n8k32 instruction (twice the throughput of m16n8k16 on s8).

// CHECK-NOT:  linalg.
// CHECK-DAG:  mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32
// CHECK-DAG:  ldmatrix.sync.aligned.m8n8.x4.shared.b16
// K16-DAG:    mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32
func.func @i8_gemm(%a: tensor<256x512xi8>, %bt: tensor<256x512xi8>, %c: tensor<256x256xi32>) -> tensor<256x256xi32> {
  %r = linalg.matmul_transpose_b ins(%a, %bt : tensor<256x512xi8>, tensor<256x512xi8>) outs(%c : tensor<256x256xi32>) -> tensor<256x256xi32>
  return %r : tensor<256x256xi32>
}
