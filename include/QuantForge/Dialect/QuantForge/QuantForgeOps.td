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

    `scale` / `zero_point` are either rank 0 (per-tensor) or have the same
    rank as `input` with, along every dimension, an extent that divides the
    input extent (block-wise / "group-wise" quantization): element
    `[i, j]` of a `tensor<KxN>` input with `tensor<(K/G)xN>` parameters uses
    parameter `[i / G, j]`.

    With the `fp8` unit attribute the i8 input holds FP8 E4M3 bit patterns (sign, 4 exponent
    bits with bias 7, 3 mantissa bits; no infinities, 0x7F / 0xFF are NaN and are read as 480):
    output = fp8_to_f16(input) * scale. `zero_point` is not used (it must still be passed).

    Example:
    ```mlir
    %fp = qf.dequant %int, %scale, %zp : tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
    // group size 128 along K:
    %fg = qf.dequant %int, %gs, %gz : tensor<4096x4096xi8>, tensor<32x4096xf16>, tensor<32x4096xi8> -> tensor<4096x4096xf16>
    ```
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scale,
    AnyTensor:$zero_point,
    UnitAttr:$fp8
  );
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $scale `,` $zero_point attr-dict `:`
    type($input) `,` type($scale) `,` type($zero_point) `->` type($output)
  }];
}

#endif // QUANTFORGE_OPS
