// RUN: quantforge-opt --quantforge-tiling='block-tile-m=16 block-tile-n=64 block-tile-k=128 warp-tile-m=16 warp-tile-n=16 split-k=4 split-k-init=false' %s | FileCheck --check-prefix=TILING %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true block-tile-n=64 warp-tile-n=16 smem-stages=1 accumulate=false split-k=4' %s | FileCheck --check-prefix=PIPE %s
// RUN: quantforge-opt --quantforge-gpu-pipeline='decode=true' %s | FileCheck --check-prefix=OFF %s

// Split-K (experimental, off by default): the K reduction is cut into 4 chunks, one grid axis. The tiling leaves ONE
// scf.forall over (m, n, z) whose blocks write their tile of a partial-result tensor [S, M, N]; a linalg.generic adds the S
// partial results. The pipeline turns that into two kernels (GEMM + reduce); without split-k there is still exactly one.
// (Numerics + speed: test/E2E/test_split_k.py, README "Split-K".)

// TILING:     tensor.empty() : tensor<4x16x256xf32>
// TILING-NOT: linalg.fill
// TILING:     scf.forall ({{.*}}, {{.*}}, {{.*}}) = (0, 0, 0) to (16, 256, 4) step (16, 64, 1) shared_outs({{.*}}) -> (tensor<4x16x256xf32>)
// TILING:     tensor.extract_slice {{.*}} tensor<16x1024xf16> to tensor<16x256xf16>
// TILING:     tensor.parallel_insert_slice {{.*}} into {{.*}} [1, 16, 64] [1, 1, 1] : tensor<16x64xf32> into tensor<4x16x256xf32>
// TILING:     linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]

// PIPE-DAG: gpu.binary @w4a16_gemm_kernel
// PIPE-DAG: gpu.binary @w4a16_gemm_kernel_0
// PIPE-DAG: mma.sync.aligned.m16n8k16
// PIPE-NOT: llvm.call @malloc

// OFF-DAG: gpu.binary @w4a16_gemm_kernel
// OFF-NOT: gpu.binary @w4a16_gemm_kernel_0
func.func @w4a16_gemm(%act: tensor<16x1024xf16>, %wp: tensor<1024x128xi8>,
                      %scale: tensor<8x256xf16>, %zp: tensor<8x256xi8>,
                      %c: tensor<16x256xf32>) -> tensor<16x256xf32> {
  %w = qf.unpack %wp : tensor<1024x128xi8> -> tensor<1024x256xi8>
  %wf = qf.dequant %w, %scale, %zp : tensor<1024x256xi8>, tensor<8x256xf16>, tensor<8x256xi8> -> tensor<1024x256xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<16x1024xf16>, tensor<1024x256xf16>) outs(%c : tensor<16x256xf32>) -> tensor<16x256xf32>
  return %r : tensor<16x256xf32>
}
