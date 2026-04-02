//===----------------------------------------------------------------------===//
// QuantForge Transforms - Passes Header
//===----------------------------------------------------------------------===//

#ifndef QUANTFORGE_TRANSFORMS_PASSES_H
#define QUANTFORGE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace quantforge
    {

        // === Phase 2 Passes ===

        /// Create the ConvertLinalgToQuantForge pass.
        /// Rewrites linalg.matmul with INT8-packed weights into
        /// qf.unpack + qf.dequant + linalg.matmul(FP16).
        std::unique_ptr<Pass> createConvertLinalgToQuantForgePass();

        /// Register the ConvertLinalgToQuantForge pass for mlir-opt style tools.
        void registerConvertLinalgToQuantForgePass();

        // === Task 2.1: UnpackFusion Pass ===

        /// Create the LowerUnpackToArith pass.
        /// Lowers qf.unpack → linalg.generic(arith.shrui + arith.andi)
        ///   + tensor.collapse_shape.
        std::unique_ptr<Pass> createLowerUnpackToArithPass();

        /// Register the LowerUnpackToArith pass for mlir-opt style tools.
        void registerLowerUnpackToArithPass();

        /// Create the LowerDequantToArith pass.
        /// Lowers qf.dequant → linalg.generic(arith.sitofp + arith.subf + arith.mulf).
        std::unique_ptr<Pass> createLowerDequantToArithPass();

        /// Register the LowerDequantToArith pass for mlir-opt style tools.
        void registerLowerDequantToArithPass();

        /// Create the FuseUnpackDequant pass.
        /// Fuses qf.unpack + qf.dequant into one linalg.generic
        /// for on-the-fly INT4 unpacking and dequantization.
        std::unique_ptr<Pass> createFuseUnpackDequantPass();

        /// Register the FuseUnpackDequant pass for mlir-opt style tools.
        void registerFuseUnpackDequantPass();

        // === Phase 1 Passes — Vectorized (Branch-Free) Unpacking ===

        /// Create the LowerUnpackBranchFree pass.
        /// Lowers qf.unpack to two branch-free linalg.generics + stride-2
        /// tensor.insert_slice, eliminating linalg.index and arith.select.
        std::unique_ptr<Pass> createLowerUnpackBranchFreePass();

        void registerLowerUnpackBranchFreePass();

        /// Create the FuseUnpackDequantBranchFree pass.
        /// Branch-free fused lowering: two pointwise linalg.generics
        /// (nibble-extract + dequant) + stride-2 tensor.insert_slice.
        std::unique_ptr<Pass> createFuseUnpackDequantBranchFreePass();

        void registerFuseUnpackDequantBranchFreePass();

        // === Phase 2 Passes — PTX / NVVM-Ready Lowering ===

        /// Create the LowerUnpackToNVVM pass.
        /// Lowers qf.unpack to SCF loops that process one i32 chunk
        /// (4 packed bytes = 8 INT4 nibbles) per inner iteration using
        /// constant-shift extractions — PTX-ready IR.
        std::unique_ptr<Pass> createLowerUnpackToNVVMPass();

        void registerLowerUnpackToNVVMPass();

        // === Phase 3 Passes — GPU Optimization ===

        /// Create the LowerUnpackToPRMT pass.
        /// Lowers qf.unpack to SCF loops using prmt.b32 inline assembly
        /// for 2-instruction nibble extraction (Ampere/Hopper sm_80+).
        /// Reduces ALU pressure from ~16 to ~10 instructions per i32 chunk.
        std::unique_ptr<Pass> createLowerUnpackToPRMTPass();

        void registerLowerUnpackToPRMTPass();

        /// Create the RegisterLayoutAwareUnpack pass.
        /// Rewrites unpack output indices to match mma.sync fragment layout,
        /// eliminating shfl.sync warp shuffles. Requires "mma_consumer"
        /// attribute. Currently supports m16n8k16.
        std::unique_ptr<Pass> createRegisterLayoutAwareUnpackPass();

        void registerRegisterLayoutAwareUnpackPass();

        // === Canonicalization / Optimization Passes ===

        /// Create the CanonicalizeDequantZeroPoint pass.
        /// Eliminates arith.sitofp + arith.subf when zero_point == 0
        /// (symmetric quantization fast-path).
        std::unique_ptr<Pass> createCanonicalizeDequantZeroPointPass();

        void registerCanonicalizeDequantZeroPointPass();

        // === Task 2.2: Tiling Pass ===

        /// Create the TilingPass.
        /// Tiles linalg.matmul / linalg.generic operations into 128×128 (×64)
        /// tiles using scf.for loops for GPU shared-memory blocking.
        /// Configurable via --tile-m, --tile-n, --tile-k options.
        std::unique_ptr<Pass> createTilingPass();

        void registerTilingPass();

        // === Task 2.3.1: Vectorization Pass ===

        /// Create the VectorizationPass.
        /// Vectorizes innermost linalg.matmul / linalg.generic operations
        /// into vector dialect ops (vector.contract, vector.transfer_*, SIMD
        /// arithmetic), preparing IR for Tensor Core / ldmatrix lowering.
        std::unique_ptr<Pass> createVectorizationPass();

        void registerVectorizationPass();

        // === Phase 4 Passes — GPU Hardware Mapping ===

        /// Create the GPUMappingPass.
        /// Maps tiled scf.forall ops to GPU hardware: block grid → gpu.block_id,
        /// warp distribution → gpu.thread_id, inserts gpu.barrier for SRAM sync.
        std::unique_ptr<Pass> createGPUMappingPass();

        void registerGPUMappingPass();

        // === Task 3.2: LowerToNVVM Pass — PTX Translation Boundary ===

        /// Create the LowerToNVVMPass.
        /// Lowers GPU, Arith, Vector, MemRef, SCF ops inside gpu.module
        /// to LLVM + NVVM dialects (PTX-ready IR).
        std::unique_ptr<Pass> createLowerToNVVMPass();

        void registerLowerToNVVMPass();

        // === Task 4.1: Bufferization Pass ===

        /// Create the QuantForgeBufferizePass.
        /// One-Shot Bufferize: converts all tensor ops to memref ops
        /// with in-place analysis and IdentityLayoutMap at function
        /// boundaries. Runs on ModuleOp.
        std::unique_ptr<Pass> createQuantForgeBufferizePass();

        void registerQuantForgeBufferizePass();

        // === Task 4.2+4.4: Shared Memory Promotion Pass ===

        /// Create the SharedMemoryPromotionPass.
        /// Promotes tile-sized HBM memref.subview reads into Shared
        /// Memory (SRAM, memory_space=3) with gpu.barrier sync and
        /// memref.dealloc for MLIR liveness analysis.
        std::unique_ptr<Pass> createSharedMemoryPromotionPass();

        void registerSharedMemoryPromotionPass();

        // === Task 4.3: XOR Swizzle Load Pass ===

        /// Create the SwizzleLoadPass.
        /// XOR swizzle addressing for SRAM loads/stores to eliminate
        /// shared memory bank conflicts (col ^ (row % phase)).
        std::unique_ptr<Pass> createSwizzleLoadPass();

        void registerSwizzleLoadPass();

        // std::unique_ptr<Pass> createTensorCoreFusionPass();

        /// Register all QuantForge passes.
        inline void registerQuantForgePasses()
        {
            registerConvertLinalgToQuantForgePass();
            registerLowerUnpackToArithPass();
            registerLowerDequantToArithPass();
            registerFuseUnpackDequantPass();
            registerLowerUnpackBranchFreePass();
            registerFuseUnpackDequantBranchFreePass();
            registerLowerUnpackToNVVMPass();
            // Phase 3 optimization passes
            registerLowerUnpackToPRMTPass();
            registerRegisterLayoutAwareUnpackPass();
            registerCanonicalizeDequantZeroPointPass();
            // Task 2.2: Tiling pass
            registerTilingPass();
            // Task 2.3.1: Vectorization pass
            registerVectorizationPass();
            // Phase 4: GPU hardware mapping
            registerGPUMappingPass();
            // Task 3.2: PTX translation boundary
            registerLowerToNVVMPass();
            // Task 4: Bufferization & Shared Memory Management
            registerQuantForgeBufferizePass();
            registerSharedMemoryPromotionPass();
            registerSwizzleLoadPass();
        }

    } // namespace quantforge
} // namespace mlir

#endif // QUANTFORGE_TRANSFORMS_PASSES_H
