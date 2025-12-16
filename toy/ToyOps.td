//===- ToyOps.td - Toy dialect ops -------------------------*- tablegen -*-===//
//
// Defines operations for the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "ToyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

// Base class for Toy dialect ops
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def Toy_AddOp : Toy_Op<"add", [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    
    Example:
      %result = toy.add %lhs, %rhs : tensor<?xf64>
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def Toy_MulOp : Toy_Op<"mul", [Pure]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two tensors.
    
    Example:
      %result = toy.mul %lhs, %rhs : tensor<?xf64>
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

#endif // TOY_OPS_TD
