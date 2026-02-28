# --- Dialect TableGen (from QuantForgeDialect.td) ---
set(LLVM_TARGET_DEFINITIONS QuantForgeDialect.td)
mlir_tablegen(QuantForgeDialect.h.inc -gen-dialect-decls)
mlir_tablegen(QuantForgeDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(QuantForgeDialectIncGen)

# --- Ops TableGen (from QuantForgeOps.td) ---
set(LLVM_TARGET_DEFINITIONS QuantForgeOps.td)
mlir_tablegen(QuantForgeOps.h.inc -gen-op-decls)
mlir_tablegen(QuantForgeOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(QuantForgeOpsIncGen)
