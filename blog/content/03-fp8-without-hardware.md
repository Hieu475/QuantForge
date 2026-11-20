---
title: Dequantising FP8 (E4M3) weights without hardware support: a bit trick that fits in four integer operations
date: 2026-11-20
order: 3
summary: How FP8 weights were added to the compiler although LLVM 18's NVPTX backend has no FP8 type, why the conversion is exact, how it was vectorised, and what it costs.
---

FP8 is the other common 8-bit weight format. **E4M3** has 1 sign bit, 4 exponent bits (bias 7) and 3 mantissa bits, a maximum magnitude of 448, no infinities, and a single NaN encoding (`0x7F` and `0xFF`). It keeps relative precision across magnitudes, which suits weights better than a uniform INT8 grid when the values span a wide range.

The compiler could not use the obvious route. The toolchain here is LLVM / MLIR 18: it has no E4M3 floating-point type in the NVPTX backend and does not expose the `cvt.f16x2.e4m3x2` instruction of newer PTX. Instead of waiting for a newer LLVM, I looked for a conversion that needs nothing but integer instructions.

## The observation

Take the FP8 byte `s eeee mmm` and consider the 16-bit pattern

```text
f16 bits = (s << 15) | ((byte & 0x7F) << 7)
         = s 0 0 e e e e m m m 0 0 0 0 0 0 0      (bit 15 .. bit 0)
```

Read as an IEEE half-precision number, the 4 exponent bits of the FP8 value land in the low four bits of the 5-bit f16 exponent field, the 3 mantissa bits land at the top of the 10-bit mantissa field. The exponent bias of f16 is 15 and that of E4M3 is 7, so the half-precision number is

```text
2^(e - 15) * (1 + m/8)  =  2^-8 * 2^(e - 7) * (1 + m/8)
   (the half-precision number)     (2^-8 times the FP8 value)
```

that is, **exactly 2^-8 times the FP8 value**. The same holds for subnormals: with `e = 0` the FP8 value is `m/8 * 2^-6` and the f16 pattern is a subnormal worth `(m * 128 / 1024) * 2^-14 = m/8 * 2^-14`, again 2^-8 times the FP8 value. So

```text
value(byte) = bitcast_f16( ((byte & 0x7F) << 7) | ((byte & 0x80) << 8) ) * 256
```

and the multiplication by 256 is exact (the maximum is 448, far below the f16 limit). The group scale is multiplied afterwards, which is the only rounding, so the result is identical to dequantising in higher precision and rounding once.

Cost: an AND, a shift, an AND, a shift, an OR, a bit-cast, and two multiplications, one of which is by a constant. NaN (`0x7F`) is read as 480 instead of NaN; quantised weights never contain it.

## It is verified on all 256 bytes

A unit test builds the trick in PyTorch for every byte value and compares with `torch.float8_e4m3fn` conversion: all 254 non-NaN bytes are identical, and `0x7F` reads as 480.0 as documented. The test lives next to the layout tests of the repository.

## Making it fast: packed f16x2

The first implementation was the scalar tree above (about 9 integer instructions per weight). The kernels are memory-bound at decode, so at 1 x 4096 x 4096 it already tied INT8 (97 against 100 microseconds). But inside a whole model the small layers are instruction-bound, and the scalar version decoded at 155 tokens/s against INT8's 165 on Qwen2.5-0.5B.

The existing `vectorize-dequant` pass, which recognises eight dequantisation trees feeding one vector store and emits packed arithmetic, was extended to recognise the FP8 tree. For two bytes `b0, b1` it builds a 32-bit word `x = b0 | (b1 << 16)` (each byte in its own 16-bit half), then does the AND / shift / OR on **both halves at once**:

```text
mag  = (x << 7) & 0x3F803F80
sgn  = (x << 8) & 0x80008000
half2 = mag | sgn                       // two f16 values worth 2^-8 times the FP8 values
w     = (half2 * 256.h2) * scale.h2     // two mul.f16x2
```

The packed form is **bit-identical** to the scalar form (checked by the end-to-end tests and the sanitizer cases) and brought the whole-model decode back to 165 tokens/s. A FileCheck test asserts that the decode kernel contains no 16-bit shifts (the scalar form has 128 of them), so a future change that breaks the match is caught immediately.

## What FP8 costs and buys

| | INT8 | FP8 (E4M3) | notes |
|---|--:|--:|---|
| decode kernel 1 x 4096 x 4096 | 101 us | 99 us | 8-bit weights: twice the bytes of INT4 |
| decode kernel 1 x 4096 x 11008 | 272 us | 271 us | |
| prefill 512 x 4096 x 4096 | 564 us | 584 us | 4 percent slower |
| Qwen2.5-0.5B perplexity (fp16: 19.93) | 19.91 | 20.10 | round-to-nearest, group 128 |
| Qwen2.5-0.5B decode (CUDA graph) | 165 tok/s | 165 tok/s | after vectorising |

On this model INT8 and FP8 are equivalent in speed and close in quality, both about 28 percent faster than fp16 (165 against 129 tokens/s) on a model this small. They halve the weight bytes of fp16 rather than quartering them as INT4 does, so INT4 remains the format for speed. FP8 is the format for checkpoints that already ship in it, or when the quality loss of INT4 is unacceptable.

## What it does not do

Only the *weights* are FP8. The activations stay fp16 and the multiply-accumulate runs on the fp16 Tensor Core path. Using the FP8 Tensor Core of Ada (`mma` with `e4m3` operands) to double the arithmetic rate is a different project: it needs FP8 activations, per-tensor activation scales and a way to emit those instructions from LLVM 18. It is not done.

The FP8 numbers in the table come from round-to-nearest quantisation of fp16 checkpoints. A real FP8 checkpoint (`RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic`, per-channel scales, weights only) was later loaded with `load_quantized_checkpoint(method="fp8")`: perplexity 22.543 against 22.441 for the fp16 base model on the same 16 test windows, identical to its dequantised twin to three decimals.
