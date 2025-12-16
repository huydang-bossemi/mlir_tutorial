//===- ToyDialect.td - Toy dialect definition --------------*- tablegen -*-===//
//
// Defines the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_TD
#define TOY_DIALECT_TD

include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A simple toy dialect for MLIR tutorial";
  let description = [{
    This dialect provides basic arithmetic operations on tensors
    for educational purposes.
  }];
  let cppNamespace = "::toy";
}

#endif // TOY_DIALECT_TD
