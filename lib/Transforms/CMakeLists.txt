# Transforms — lowering/optimization passes (Phase 2-4)

add_mlir_library(QuantForgeTransforms
  ConvertLinalgToQuantForge.cpp
  LowerUnpackToArith.cpp
  LowerDequantToArith.cpp
  FuseUnpackDequant.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QuantForge/Transforms

  DEPENDS
  QuantForgeDialectIncGen
  QuantForgeOpsIncGen

  LINK_LIBS PUBLIC
  QuantForgeDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
)
