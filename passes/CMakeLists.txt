add_mlir_library(toy-passes SHARED
  ShapeInferencePass.cpp
  CanonicalizePass.cpp
  DEPENDS
  ToyOpsIncGen
  LINK_LIBS PUBLIC
  toy-dialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

target_include_directories(toy-passes PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/toy
)