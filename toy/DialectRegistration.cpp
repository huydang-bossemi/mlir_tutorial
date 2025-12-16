//===- DialectRegistration.cpp - Toy dialect registration ----------------===//
//
// Registers the Toy dialect with MLIR.
//
//===----------------------------------------------------------------------===//

#include "toy/ToyDialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace toy;

void toy::registerToyDialect(DialectRegistry &registry) {
  registry.insert<ToyDialect>();
}
