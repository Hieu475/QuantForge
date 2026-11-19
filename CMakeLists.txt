cmake_minimum_required(VERSION 3.20)
project(QuantForge LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Find LLVM & MLIR
# ==============================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# ------------------------------------------------------------------------------
# Supported toolchain. The MLIR C++ API changes every release; QuantForge is
# developed and tested against LLVM/MLIR 18.1.x (Ubuntu 24.04 packages
# llvm-18-dev, libmlir-18-dev, mlir-18-tools). Anything else is at your own risk.
# ------------------------------------------------------------------------------
set(QUANTFORGE_SUPPORTED_LLVM_MAJOR 18)
option(QUANTFORGE_ALLOW_UNSUPPORTED_LLVM
       "Only warn (do not fail) when LLVM is not ${QUANTFORGE_SUPPORTED_LLVM_MAJOR}.x" OFF)
if(NOT LLVM_VERSION_MAJOR EQUAL QUANTFORGE_SUPPORTED_LLVM_MAJOR)
  set(_msg "QuantForge supports LLVM/MLIR ${QUANTFORGE_SUPPORTED_LLVM_MAJOR}.x, found ${LLVM_PACKAGE_VERSION}")
  if(QUANTFORGE_ALLOW_UNSUPPORTED_LLVM)
    message(WARNING "${_msg} (continuing: QUANTFORGE_ALLOW_UNSUPPORTED_LLVM=ON)")
  else()
    message(FATAL_ERROR "${_msg}. Point -DMLIR_DIR/-DLLVM_DIR at an ${QUANTFORGE_SUPPORTED_LLVM_MAJOR}.x install "
                        "or pass -DQUANTFORGE_ALLOW_UNSUPPORTED_LLVM=ON.")
  endif()
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# ==============================================================================
# Project-wide include paths
# ==============================================================================
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Project source includes
include_directories(${PROJECT_SOURCE_DIR}/include)
# TableGen-generated includes
include_directories(${PROJECT_BINARY_DIR}/include)

# ==============================================================================
# LLVM definitions
# ==============================================================================
add_definitions(${LLVM_DEFINITIONS})

# ==============================================================================
# Subdirectories
# ==============================================================================
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# ==============================================================================
# Tests
# ==============================================================================
enable_testing()
add_subdirectory(test)
