// RUN: quantforge-opt %s | quantforge-opt | FileCheck %s

// CHECK-LABEL: func.func @test_unpack
func.func @test_unpack(%arg0: tensor<4096x2048xi8>) -> tensor<4096x4096xi8> {
  // CHECK: qf.unpack
  %0 = qf.unpack %arg0 : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
  return %0 : tensor<4096x4096xi8>
}
