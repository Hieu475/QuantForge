# ==============================================================================
# QuantForge tests
#
#   cmake --build <build> --target check-quantforge
#       FileCheck/lit tests of every pass and of the pipeline (CPU only)
#   ctest -L lit      the same, through CTest
#   ctest -L gpu      end-to-end tests: compile to PTX, run on the GPU and compare
#                     with PyTorch (needs an NVIDIA GPU, torch, the CUDA toolkit)
# ==============================================================================

set(QUANTFORGE_TOOLS_DIR "${CMAKE_BINARY_DIR}/tools/quantforge-opt")
set(QUANTFORGE_LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")

configure_file(lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py @ONLY)

# --- lit -----------------------------------------------------------------------
find_program(QUANTFORGE_LIT NAMES llvm-lit lit lit.py
             HINTS "${LLVM_TOOLS_BINARY_DIR}")
if(QUANTFORGE_LIT)
  message(STATUS "lit: ${QUANTFORGE_LIT}")
  add_custom_target(check-quantforge
    COMMAND ${QUANTFORGE_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS quantforge-opt
    COMMENT "Running QuantForge lit tests"
    USES_TERMINAL)
  add_test(NAME quantforge-lit
           COMMAND ${QUANTFORGE_LIT} -sv ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(quantforge-lit PROPERTIES LABELS "lit")
else()
  message(STATUS "lit not found: `pip install lit` to enable `check-quantforge` / ctest -L lit")
endif()

# --- GPU end-to-end ------------------------------------------------------------
option(QUANTFORGE_ENABLE_E2E_TESTS
       "Register the GPU end-to-end tests with CTest (label: gpu)" ON)
if(QUANTFORGE_ENABLE_E2E_TESTS)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME quantforge-e2e
             COMMAND ${Python3_EXECUTABLE} -m pytest -v
                     ${CMAKE_CURRENT_SOURCE_DIR}/E2E
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(quantforge-e2e PROPERTIES
      LABELS "gpu"
      ENVIRONMENT "QUANTFORGE_BUILD_DIR=${CMAKE_BINARY_DIR};QUANTFORGE_LLVM_BIN=${LLVM_TOOLS_BINARY_DIR}"
      TIMEOUT 3600)
  endif()
endif()
