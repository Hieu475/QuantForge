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
        }

    } // namespace quantforge
} // namespace mlir

#endif // QUANTFORGE_TRANSFORMS_PASSES_H
