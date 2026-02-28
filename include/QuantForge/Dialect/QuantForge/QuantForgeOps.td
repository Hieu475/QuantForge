//===----------------------------------------------------------------------===//
// QuantForge Operations Definition (ODS - TableGen)
//===----------------------------------------------------------------------===//

#ifndef QUANTFORGE_OPS
#define QUANTFORGE_OPS

include "QuantForge/Dialect/QuantForge/QuantForgeDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class QF_Op<string mnemonic, list<Trait> traits = []> :
    Op<QuantForge_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// qf.unpack - Unpack INT4 packed weights to higher precision
//===----------------------------------------------------------------------===//
def QF_UnpackOp : QF_Op<"unpack", [Pure]> {
  let summary = "Unpack INT4 packed weights";
  let description = [{
    Unpacks a tensor of INT8 values (each containing two INT4 values)
    into a tensor of unpacked values (INT8 or FP16).

    Each input byte contains two 4-bit weights:
      - Low nibble:  bits [3:0]
      - High nibble: bits [7:4]

    The operation performs:
      low  = input & 0x0F
      high = (input >> 4) & 0x0F

    Example:
    ```mlir
    %unpacked = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
    ```
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// qf.dequant - Dequantize INT4/INT8 to FP16/FP32
//===----------------------------------------------------------------------===//
def QF_DequantOp : QF_Op<"dequant", [Pure]> {
  let summary = "Dequantize integer weights to floating point";
  let description = [{
    Applies scale and zero-point to convert quantized integer weights
    to floating point: output = (input - zero_point) * scale

    Example:
    ```mlir
    %fp = qf.dequant %int, %scale, %zp : tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
    ```
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scale,
    AnyTensor:$zero_point
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $scale `,` $zero_point attr-dict `:`
    type($input) `,` type($scale) `,` type($zero_point) `->` type($output)
  }];
}

#endif // QUANTFORGE_OPS
