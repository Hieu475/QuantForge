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
        // std::unique_ptr<Pass> createConvertToQuantForgePass();

        // === Phase 3 Passes ===
        // std::unique_ptr<Pass> createTilingPass();
        // std::unique_ptr<Pass> createVectorizationPass();

        // === Phase 4 Passes ===
        // std::unique_ptr<Pass> createGPUMappingPass();
        // std::unique_ptr<Pass> createLowerToNVVMPass();

    } // namespace quantforge
} // namespace mlir

#endif // QUANTFORGE_TRANSFORMS_PASSES_H
