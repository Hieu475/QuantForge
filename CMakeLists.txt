cmake_minimum_required(VERSION 3.20)
project(QuantForge LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# Find LLVM & MLIR
# ==============================================================================
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

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
