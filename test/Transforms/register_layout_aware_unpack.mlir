// RUN: quantforge-opt --register-layout-aware-unpack %s | FileCheck %s

// Tests for RegisterLayoutAwareUnpack pass:
//   1. Emits gpu.thread_id right above the transformed outer scf.for
//   2. Computes lane/fragment mapping via dynamic constants (divui/muli)
//   3. Leaves loops without mma_consumer untouched

// CHECK-LABEL: func.func @layout_aware_transform
// CHECK:           %[[TID:.*]] = gpu.thread_id
// CHECK:           %[[LANE:.*]] = arith.remui %[[TID]], %{{.*}} : index
// CHECK-NEXT:      %[[OUTER:.*]] = scf.for
// CHECK-SAME:        layout_aware
// CHECK-NOT:         mma_consumer
// CHECK:               %[[LPR:.*]] = arith.constant 4 : index
// CHECK:               %[[CPG:.*]] = arith.constant 2 : index
// CHECK:               %[[ROW0:.*]] = arith.divui %[[LANE]], %[[LPR]] : index
// CHECK:               %[[LMOD:.*]] = arith.remui %[[LANE]], %[[LPR]] : index
// CHECK:               %[[COL0:.*]] = arith.muli %[[LMOD]], %[[CPG]] : index
func.func @layout_aware_transform() -> tensor<16x256xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<16x256xi8>

  %result = "scf.for"(%c0, %c1, %c1, %init) ({
  ^bb0(%k: index, %acc0: tensor<16x256xi8>):
    %inner = scf.for %ch = %c0 to %c1 step %c1 iter_args(%acc = %acc0) -> (tensor<16x256xi8>) {
      %v = arith.constant 0 : i8
      %t0 = tensor.insert %v into %acc[%c0, %c0] : tensor<16x256xi8>
      %t1 = tensor.insert %v into %t0[%c0, %c0] : tensor<16x256xi8>
      %t2 = tensor.insert %v into %t1[%c0, %c0] : tensor<16x256xi8>
      %t3 = tensor.insert %v into %t2[%c0, %c0] : tensor<16x256xi8>
      %t4 = tensor.insert %v into %t3[%c0, %c0] : tensor<16x256xi8>
      %t5 = tensor.insert %v into %t4[%c0, %c0] : tensor<16x256xi8>
      %t6 = tensor.insert %v into %t5[%c0, %c0] : tensor<16x256xi8>
      %t7 = tensor.insert %v into %t6[%c0, %c0] : tensor<16x256xi8>
      scf.yield %t7 : tensor<16x256xi8>
    }
    "scf.yield"(%inner) : (tensor<16x256xi8>) -> ()
  }) {mma_consumer, mma_m = 16 : i64, mma_n = 8 : i64, mma_k = 16 : i64} : (index, index, index, tensor<16x256xi8>) -> tensor<16x256xi8>

  return %result : tensor<16x256xi8>
}

// CHECK-LABEL: func.func @layout_aware_no_attr
// CHECK-NOT:   gpu.thread_id
func.func @layout_aware_no_attr() -> tensor<16x256xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<16x256xi8>
  %result = scf.for %k = %c0 to %c1 step %c1 iter_args(%acc = %init) -> (tensor<16x256xi8>) {
    scf.yield %acc : tensor<16x256xi8>
  }
  return %result : tensor<16x256xi8>
}
