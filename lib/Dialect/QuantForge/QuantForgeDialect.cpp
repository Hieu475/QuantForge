//===----------------------------------------------------------------------===//
// QuantForge Dialect Implementation
//===----------------------------------------------------------------------===//

#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.h"
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h"

using namespace mlir;
using namespace mlir::quantforge;

// TableGen-generated dialect definitions
#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// QuantForge Dialect initialize
//===----------------------------------------------------------------------===//
void QuantForgeDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.cpp.inc"
        >();
}
