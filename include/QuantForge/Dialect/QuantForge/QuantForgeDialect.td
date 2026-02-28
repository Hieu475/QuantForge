//===----------------------------------------------------------------------===//
// QuantForge Dialect Definition (ODS - TableGen)
//===----------------------------------------------------------------------===//

#ifndef QUANTFORGE_DIALECT
#define QUANTFORGE_DIALECT

include "mlir/IR/OpBase.td"

def QuantForge_Dialect : Dialect {
  let name = "qf";
  let summary = "QuantForge dialect for INT4 quantized neural network inference";
  let description = [{
    The QuantForge dialect provides operations for INT4 weight unpacking
    and fused GEMV/GEMM computations. It targets NVIDIA GPUs via PTX
    code generation, performing on-the-fly dequantization in registers
    to maximize effective memory bandwidth.
  }];
  let cppNamespace = "::mlir::quantforge";
}

#endif // QUANTFORGE_DIALECT
