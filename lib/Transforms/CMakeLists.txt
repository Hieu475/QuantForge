# Transforms — lowering/optimization passes (Phase 2-4)

add_mlir_library(QuantForgeTransforms
  ConvertLinalgToQuantForge.cpp
  LowerUnpackToArith.cpp
  LowerDequantToArith.cpp
  FuseUnpackDequant.cpp
  LowerUnpackBranchFree.cpp   # Phase 1: branch-free vectorized unpacking
  LowerUnpackToNVVM.cpp       # Phase 2: PTX-ready i32-chunk SCF lowering

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
  MLIRSCFDialect   # Phase 2: scf.for loops
)
