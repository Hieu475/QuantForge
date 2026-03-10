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

        // === Phase 3 Passes ===
        // std::unique_ptr<Pass> createTilingPass();
        // std::unique_ptr<Pass> createVectorizationPass();

        // === Phase 4 Passes ===
        // std::unique_ptr<Pass> createGPUMappingPass();
        // std::unique_ptr<Pass> createLowerToNVVMPass();

        /// Register all QuantForge passes.
        inline void registerQuantForgePasses()
        {
            registerConvertLinalgToQuantForgePass();
            registerLowerUnpackToArithPass();
            registerLowerDequantToArithPass();
            registerFuseUnpackDequantPass();
        }

    } // namespace quantforge
} // namespace mlir

#endif // QUANTFORGE_TRANSFORMS_PASSES_H
