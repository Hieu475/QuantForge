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

        // === Phase 3 Passes — GPU Mapping & Tensor Core Fusion ===
        // std::unique_ptr<Pass> createGPUMappingPass();
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
        }

    } // namespace quantforge
} // namespace mlir

#endif // QUANTFORGE_TRANSFORMS_PASSES_H
