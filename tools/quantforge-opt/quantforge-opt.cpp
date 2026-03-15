//===----------------------------------------------------------------------===//
// quantforge-opt: QuantForge MLIR Optimizer Driver
//
// This is the main entry point for running MLIR passes on QuantForge IR.
// Similar to mlir-opt but with QuantForge dialect and passes registered.
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.h"
#include "QuantForge/Transforms/Passes.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;

    // Register upstream MLIR dialects
    registry.insert<
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::linalg::LinalgDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect,
        mlir::vector::VectorDialect>(); // Vector needed by Task 2.3.1 --quantforge-vectorization

    // Register QuantForge dialect
    registry.insert<mlir::quantforge::QuantForgeDialect>();

    // Register QuantForge passes
    mlir::quantforge::registerQuantForgePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "QuantForge optimizer driver\n", registry));
}
