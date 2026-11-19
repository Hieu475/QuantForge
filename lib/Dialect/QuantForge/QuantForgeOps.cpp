//===----------------------------------------------------------------------===//
// QuantForge Operations Implementation
//===----------------------------------------------------------------------===//

#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

#include "QuantForge/Transforms/DequantUtils.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantforge;

// TableGen-generated operation definitions
#define GET_OP_CLASSES
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.cpp.inc"

LogicalResult DequantOp::verify() {
  auto inTy = dyn_cast<RankedTensorType>(getInput().getType());
  auto outTy = dyn_cast<RankedTensorType>(getOutput().getType());
  auto sTy = dyn_cast<RankedTensorType>(getScale().getType());
  auto zTy = dyn_cast<RankedTensorType>(getZeroPoint().getType());
  if (!inTy || !outTy || !sTy || !zTy)
    return success(); // unranked: nothing to check
  if (inTy.getShape() != outTy.getShape())
    return emitOpError("output shape ") << outTy << " must equal input shape "
                                        << inTy;
  if (!isValidDequantParamType(inTy, sTy))
    return emitOpError("scale type ")
           << sTy << " is neither rank 0 nor a block-wise layout of " << inTy
           << " (every extent must divide the input extent)";
  if (!isValidDequantParamType(inTy, zTy))
    return emitOpError("zero_point type ")
           << zTy << " is neither rank 0 nor a block-wise layout of " << inTy;
  if (sTy.getShape() != zTy.getShape() && sTy.getRank() != 0 &&
      zTy.getRank() != 0)
    return emitOpError("scale ") << sTy << " and zero_point " << zTy
                                 << " must have the same shape";
  return success();
}
