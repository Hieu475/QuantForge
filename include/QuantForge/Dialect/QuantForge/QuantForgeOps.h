//===----------------------------------------------------------------------===//
// QuantForge Operations - C++ Header
//===----------------------------------------------------------------------===//

#ifndef QUANTFORGE_DIALECT_QUANTFORGE_QUANTFORGEOPS_H
#define QUANTFORGE_DIALECT_QUANTFORGE_QUANTFORGEOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include dialect header (which already has the .h.inc)
#include "QuantForge/Dialect/QuantForge/QuantForgeDialect.h"

// TableGen-generated operation declarations
#define GET_OP_CLASSES
#include "QuantForge/Dialect/QuantForge/QuantForgeOps.h.inc"

#endif // QUANTFORGE_DIALECT_QUANTFORGE_QUANTFORGEOPS_H
