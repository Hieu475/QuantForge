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

#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;

    // Register upstream MLIR dialects
    registry.insert<
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::linalg::LinalgDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect>();

    // Register QuantForge dialect
    registry.insert<mlir::quantforge::QuantForgeDialect>();

    // TODO: Register QuantForge passes here
    // mlir::quantforge::registerQuantForgePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "QuantForge optimizer driver\n", registry));
}
