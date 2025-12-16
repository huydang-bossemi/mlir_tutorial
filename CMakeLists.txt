cmake_minimum_required(VERSION 3.20)
project(mlir_tutorial LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find MLIR and LLVM
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Build toy dialect
add_subdirectory(toy)

# Build passes
add_subdirectory(passes)

# Build tools
add_subdirectory(tools)

add_custom_target(mlir_tutorial ALL DEPENDS toy-dialect toy-passes toy-opt)

# Install targets
install(TARGETS toy-dialect toy-passes DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
