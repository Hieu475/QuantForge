get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${translation_libs}
  MLIRNVVMTarget
  QuantForgeDialect
  QuantForgeTransforms
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

add_llvm_executable(quantforge-opt
  quantforge-opt.cpp
)

llvm_update_compile_flags(quantforge-opt)

target_link_libraries(quantforge-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(quantforge-opt)
