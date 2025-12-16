//===- toy-opt.cpp - Toy optimizer tool ----------------------------------===//
//
// This is the main driver for the Toy optimizer tool.
//
//===----------------------------------------------------------------------===//

#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"
#include "toy/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  // Register Toy passes
  toy::registerToyPasses();
  toy::registerToyPassPipelines();

  DialectRegistry registry;
  
  // Register core MLIR dialects
  registry.insert<BuiltinDialect>();
  registry.insert<func::FuncDialect>();
  
  // Register Toy dialect
  toy::registerToyDialect(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Toy optimizer\n", registry));
}
