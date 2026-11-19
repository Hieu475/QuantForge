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
  RegisterLayoutAwareUnpack.cpp        # Opt 2: mma.sync fragment layout (skeleton)
  TilingPass.cpp                       # Task 2.2: 128×128 tile tiling pass
  VectorizationPass.cpp                # Task 2.3.1: linalg -> vector dialect
  GPUMappingPass.cpp                   # Phase 4: GPU hardware mapping
  LowerToNVVMPass.cpp                  # Task 3.2: PTX translation boundary
  QuantForgeBufferizePass.cpp          # Task 4.1: One-Shot Bufferize
  SharedMemoryPromotionPass.cpp        # Task 4.2+4.4: HBM → SRAM promotion
  SwizzleLoadPass.cpp                  # Task 4.3: XOR bank conflict prevention
  GPUPipeline.cpp                      # --quantforge-gpu-pipeline (qf -> PTX)
  ThreadDistributePass.cpp             # warp/lane loops -> threadIdx
  FuseProducerIntoLoadsPass.cpp        # dequant recomputed at the smem staging loads
  PrefetchWeightLoadsPass.cpp          # multi-tile look-ahead of the weight loads (iter_args)
  OverwriteOutputPass.cpp              # C = A @ W (accumulators start from zero)
  FuseEpiloguePass.cpp                 # relu / silu / gelu on the accumulator before the C store
  SinkDequantStoresPass.cpp            # overlap the weight-load latency with the MMAs
  VectorizeDequantPass.cpp             # packed f16x2 dequantisation (magic-number i->f16)
  VectorizeGlobalLoadsPass.cpp         # simplify index math, merge adjacent global loads
  GPULaunchLowering.cpp                # scf.forall -> gpu.launch, host alloc -> gpu.alloc

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/QuantForge/Transforms

  DEPENDS
  QuantForgeDialectIncGen
  QuantForgeOpsIncGen

  LINK_LIBS
  QuantForgeDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRSCFDialect      # Phase 2: scf.for loops
  MLIRLLVMDialect     # Opt 1: LLVM::InlineAsmOp for prmt.b32
  MLIRGPUDialect      # Opt 2: gpu::ThreadIdOp for register layout
  MLIRGPUTransformOps  # Phase 4: GPUBlockMappingAttr / GPUThreadMappingAttr
  MLIRLinalgTransforms # Task 2.2: scf::tileUsingSCFForOp
  MLIRTilingInterface  # Task 2.2: TilingInterface used by tiling
  MLIRVectorDialect    # Task 2.3.1: vector.contract / vector.transfer_*
  MLIRVectorTransforms # Task 2.3.1: vector lowering/vectorization patterns
  # Task 3.2: GPU -> NVVM + LLVM conversion libraries
  MLIRNVVMDialect
  MLIRNVGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRVectorToLLVM
  MLIRSCFToControlFlow
  # Task 4.1: Bufferization infrastructure
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRMemRefDialect
  MLIRMemRefTransforms
  # Task 4.1: Dialect-specific bufferization interfaces
  MLIRArithTransforms            # arith::registerBufferizableOpInterface
  MLIRLinalgTransformOps         # linalg bufferization interface (external models)
  MLIRSCFUtils
  MLIRSCFTransforms              # scf::registerBufferizableOpInterface
  MLIRTensorTransforms           # tensor::registerBufferizableOpInterface
)
