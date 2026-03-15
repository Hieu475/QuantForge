# Transforms — lowering/optimization passes (Phase 2-4)

add_mlir_library(QuantForgeTransforms
  ConvertLinalgToQuantForge.cpp
  LowerUnpackToArith.cpp
  LowerDequantToArith.cpp
  FuseUnpackDequant.cpp
  LowerUnpackBranchFree.cpp            # Phase 1: branch-free vectorized unpacking
  LowerUnpackToNVVM.cpp                # Phase 2: PTX-ready i32-chunk SCF lowering
  CanonicalizeDequantZeroPoint.cpp     # Opt 4: symmetric quant fast-path (zp=0)
  LowerUnpackToPRMT.cpp                # Opt 1: prmt.b32 intrinsic (Ampere/Hopper)
  SwizzledUnpackIndexing.cpp           # Opt 3: shared memory bank-conflict prevention
  RegisterLayoutAwareUnpack.cpp        # Opt 2: mma.sync fragment layout (skeleton)
  TilingPass.cpp                       # Task 2.2: 128×128 tile tiling pass
  VectorizationPass.cpp                # Task 2.3.1: linalg -> vector dialect

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
  MLIRSCFDialect      # Phase 2: scf.for loops
  MLIRLLVMDialect     # Opt 1: LLVM::InlineAsmOp for prmt.b32
  MLIRGPUDialect      # Opt 2: gpu::ThreadIdOp for register layout
  MLIRLinalgTransforms # Task 2.2: scf::tileUsingSCFForOp
  MLIRTilingInterface  # Task 2.2: TilingInterface used by tiling
  MLIRVectorDialect    # Task 2.3.1: vector.contract / vector.transfer_*
  MLIRVectorTransforms # Task 2.3.1: vector lowering/vectorization patterns
)
