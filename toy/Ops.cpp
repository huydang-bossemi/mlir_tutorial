//===- Ops.cpp - Toy dialect operations ----------------------------------===//
//
// Implementation of Toy dialect operations.
//
//===----------------------------------------------------------------------===//

#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace toy;

//===----------------------------------------------------------------------===//
// Toy Dialect
//===----------------------------------------------------------------------===//

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ToyOps.cpp.inc"

// Include dialect definitions
#include "ToyDialect.cpp.inc"
