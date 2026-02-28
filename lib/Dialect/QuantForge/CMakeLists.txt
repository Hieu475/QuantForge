add_mlir_dialect_library(QuantForgeDialect
  QuantForgeDialect.cpp
  QuantForgeOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QuantForge/Dialect/QuantForge

  DEPENDS
  QuantForgeDialectIncGen
  QuantForgeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
