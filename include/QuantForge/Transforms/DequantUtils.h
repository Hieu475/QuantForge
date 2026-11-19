//===----------------------------------------------------------------------===//
// Helpers shared by the passes that expand qf.dequant into linalg.generic.
//
// qf.dequant takes `scale` / `zero_point` operands in one of two layouts:
//   * rank 0           per-tensor: one value for every element;
//   * rank == rank(x)  block-wise ("group-wise"): along every dim d the
//                      parameter extent P[d] divides the input extent X[d]
//                      (or is 1); element i uses parameter i[d] / (X[d]/P[d]).
//                      The usual W4 layout is x : [K, N], scale/zp : [K/G, N].
//===----------------------------------------------------------------------===//

#ifndef QUANTFORGE_TRANSFORMS_DEQUANTUTILS_H
#define QUANTFORGE_TRANSFORMS_DEQUANTUTILS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::quantforge {

/// True if `paramTy` is a valid scale / zero-point layout for an input of
/// type `inputTy` (see the file comment).
inline bool isValidDequantParamType(RankedTensorType inputTy,
                                    RankedTensorType paramTy) {
  if (paramTy.getRank() == 0)
    return true;
  if (paramTy.getRank() != inputTy.getRank())
    return false;
  for (int64_t d = 0; d < inputTy.getRank(); ++d) {
    int64_t x = inputTy.getDimSize(d), p = paramTy.getDimSize(d);
    if (x == p)
      continue;
    if (ShapedType::isDynamic(x) || ShapedType::isDynamic(p) || p <= 0 ||
        x % p != 0)
      return false;
  }
  return true;
}

/// Indexing map (generic loop dims -> scale/zp element) for a linalg.generic
/// iterating over `numLoopDims` loops, where `elemExprs[d]` is the affine
/// expression (over the loop dims) of the logical input index along dim d.
inline FailureOr<AffineMap>
getDequantParamMap(RankedTensorType inputTy, RankedTensorType paramTy,
                   ArrayRef<AffineExpr> elemExprs, unsigned numLoopDims) {
  MLIRContext *ctx = inputTy.getContext();
  if (!isValidDequantParamType(inputTy, paramTy))
    return failure();
  if (paramTy.getRank() == 0)
    return AffineMap::get(numLoopDims, 0, {}, ctx);
  SmallVector<AffineExpr> results;
  for (int64_t d = 0; d < inputTy.getRank(); ++d) {
    int64_t x = inputTy.getDimSize(d), p = paramTy.getDimSize(d);
    if (x == p)
      results.push_back(elemExprs[d]);
    else if (p == 1)
      results.push_back(getAffineConstantExpr(0, ctx));
    else
      results.push_back(elemExprs[d].floorDiv(x / p));
  }
  return AffineMap::get(numLoopDims, 0, results, ctx);
}

/// Convenience: logical index == loop dim d for every dim (pointwise generic).
inline FailureOr<AffineMap> getDequantParamMapIdentity(RankedTensorType inputTy,
                                                       RankedTensorType paramTy) {
  SmallVector<AffineExpr> exprs;
  for (int64_t d = 0; d < inputTy.getRank(); ++d)
    exprs.push_back(getAffineDimExpr(d, inputTy.getContext()));
  return getDequantParamMap(inputTy, paramTy, exprs, inputTy.getRank());
}

} // namespace mlir::quantforge

#endif // QUANTFORGE_TRANSFORMS_DEQUANTUTILS_H
