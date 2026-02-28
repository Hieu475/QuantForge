//===----------------------------------------------------------------------===//
// QuantForge Operations Implementation
//===----------------------------------------------------------------------===//

#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantforge;

// TableGen-generated operation definitions
#define GET_OP_CLASSES
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.cpp.inc"
