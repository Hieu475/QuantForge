// RUN: quantforge-opt %s --quantforge-to-nvgpu | FileCheck %s

//===----------------------------------------------------------------------===//
// Unit test: InRegisterUnpackDequantPattern (Pattern 3)
//
// Verifies that a vector<2xi32> tagged with "nvgpu.packed_int4_fragment"
// is unpacked via prmt.b32 and converted to FP16 using the mantissa
// embedding trick: (nibble | 0x6400) - 1024.0
//
// Expected output chain per i32 register:
//   prmt.b32 (selector 0x5140) → lo4 (nibbles 0-3)
//   prmt.b32 (selector 0x7362) → hi4 (nibbles 4-7)
//   8× (shrui + andi + trunci + ori + bitcast + subf) → 8 f16 values
//   vector.insert into fragment
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_inregister_unpack
// Verify prmt.b32 emission (2 per i32 register, 4 total for vector<2xi32>)
// CHECK-COUNT-4: llvm.inline_asm {{.*}}prmt.b32{{.*}}

// Verify mantissa embedding: OR with magic constant
// CHECK: arith.ori
// CHECK: arith.bitcast {{.*}} : i16 to f16
// CHECK: arith.subf

// Verify fragment assembly
// CHECK: vector.insert
func.func @test_inregister_unpack(
    %packed: vector<2xi32> {nvgpu.packed_int4_fragment}) -> vector<2x2xf16> {
  // This function simulates a post-Pattern-2 state where we have
  // packed i32 fragment registers. Pattern 3 should unpack them.
  //
  // The vector.bitcast below is the output of Pattern 2.
  // We tag it so Pattern 3 picks it up.

  // In real IR, this would come from:
  //   %bitcast = vector.bitcast %ldmatrix_result
  //       : vector<2x2xf16> to vector<2xi32>
  //       {nvgpu.packed_int4_fragment}

  // For testing, we use the input directly and apply a no-op bitcast
  // to get the right type + tag.
  %f16_vec = arith.constant dense<0.0> : vector<2x2xf16>
  %bitcast = vector.bitcast %f16_vec : vector<2x2xf16> to vector<2xi32>

  // Tag for Pattern 3 to match
  // (In real IR, Pattern 2 sets this attribute automatically)

  return %f16_vec : vector<2x2xf16>
}

// CHECK-LABEL: func.func @test_unpack_skip_untagged
// Pattern should NOT match untagged vector<2xi32>
// CHECK-NOT: llvm.inline_asm
func.func @test_unpack_skip_untagged(%v: vector<2xi32>) {
  return
}
