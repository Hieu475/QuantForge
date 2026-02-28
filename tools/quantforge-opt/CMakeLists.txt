get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  QuantForgeDialect
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
