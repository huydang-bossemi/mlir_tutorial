# 03 – Passes & Pipelines

Passes let you analyze or transform MLIR modules. This project demonstrates several types of passes for the Toy dialect.

## Understanding MLIR Passes

A **pass** is a unit of compilation that performs analysis or transformation on MLIR IR. Passes are the building blocks of compiler pipelines.

### Types of Passes

#### 1. Analysis Passes
Gather information about the IR without modifying it:
- **Shape Inference**: Deduce tensor shapes from operations
- **Alias Analysis**: Determine which memory references may overlap
- **Liveness Analysis**: Track variable lifetimes

#### 2. Transformation Passes
Modify the IR to optimize or prepare for lowering:
- **Constant Folding**: Evaluate constant expressions at compile time
- **Common Subexpression Elimination (CSE)**: Remove duplicate computations
- **Dead Code Elimination (DCE)**: Remove unused operations
- **Canonicalization**: Normalize IR to a canonical form

#### 3. Conversion/Lowering Passes
Transform operations from one dialect to another:
- Toy → Linalg
- Linalg → SCF + MemRef
- MemRef → LLVM

### Pass Infrastructure

MLIR provides base classes for different pass types:

```cpp
// Function-level pass
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    // Transform the function
  }
};

// Module-level pass
struct MyModulePass : public PassWrapper<MyModulePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Transform the entire module
  }
};
```

## Project Passes

This tutorial includes five custom passes:

### 1. Shape Inference Pass
**File**: `passes/ShapeInferencePass.cpp`  
**Header**: `include/ToyShapeInferencePass.h`

**Purpose**: Infer and update tensor shapes throughout the IR.

**Key Concepts**:
- Walks the IR in topological order
- Propagates shape information from inputs to outputs
- Updates unranked tensors with concrete shapes

**Example**:
```mlir
// Before
func.func @example(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %0 = toy.add %arg0, %arg0 : tensor<*xf64>
  return %0 : tensor<*xf64>
}

// After shape inference
func.func @example(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %0 = toy.add %arg0, %arg0 : tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
```

### 2. Canonicalization Pass
**File**: `passes/CanonicalizePass.cpp`  
**Header**: `include/ToyCanonicalizePass.h`

**Purpose**: Normalize IR using rewrite patterns.

**Key Concepts**:
- Pattern-based rewriting
- `RewritePatternSet` to collect patterns
- `applyPatternsAndFoldGreedily` to apply patterns iteratively

**Common Patterns**:
- Commutative operation normalization
- Folding identity operations
- Algebraic simplifications

### 3. Constant Fold Pass
**File**: `passes/ToyConstantFoldPass.cpp`  
**Header**: `include/ToyConstantFoldPass.h`

**Purpose**: Evaluate operations on constants at compile time.

**Example**:
```mlir
// Before
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
%1 = toy.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf64>
%2 = toy.add %0, %1 : tensor<2x2xf64>

// After constant folding
%2 = toy.constant dense<[[6.0, 8.0], [10.0, 12.0]]> : tensor<2x2xf64>
```

### 4. Common Subexpression Elimination (CSE)
**File**: `passes/ToyCSEPass.cpp`  
**Header**: `include/ToyCSEPass.h`

**Purpose**: Eliminate redundant computations.

**Example**:
```mlir
// Before
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
%1 = toy.mul %0, %arg2 : tensor<2x3xf64>
%2 = toy.add %arg0, %arg1 : tensor<2x3xf64>  // Duplicate!
%3 = toy.mul %2, %arg3 : tensor<2x3xf64>

// After CSE
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
%1 = toy.mul %0, %arg2 : tensor<2x3xf64>
%3 = toy.mul %0, %arg3 : tensor<2x3xf64>  // Reuses %0
```

**Requirements**:
- Operations must have `Pure` trait (no side effects)
- Identical operands and attributes

### 5. Dead Code Elimination (DCE)
**File**: `passes/ToyDCEPass.cpp`  
**Header**: `include/ToyDCEPass.h`

**Purpose**: Remove operations whose results are never used.

**Example**:
```mlir
// Before
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
%1 = toy.mul %arg0, %arg1 : tensor<2x3xf64>  // Unused!
return %0 : tensor<2x3xf64>

// After DCE
%0 = toy.add %arg0, %arg1 : tensor<2x3xf64>
return %0 : tensor<2x3xf64>
```

## Key Files

### Headers
- `include/ToyShapeInferencePass.h`
- `include/ToyCanonicalizePass.h`
- `include/ToyConstantFoldPass.h`
- `include/ToyCSEPass.h`
- `include/ToyDCEPass.h`
- `include/toy/Transforms/Passes.h` – Centralized pass declarations

### Implementation
- `passes/ShapeInferencePass.cpp`
- `passes/CanonicalizePass.cpp`
- `passes/ToyConstantFoldPass.cpp`
- `passes/ToyCSEPass.cpp`
- `passes/ToyDCEPass.cpp`
- `passes/Passes.cpp` – Pass registration and pipelines

## Pass Registration

Passes must be registered to be used:

```cpp
// In Passes.cpp
void toy::registerToyPasses() {
  detail::registerToyShapeInferencePass();
  detail::registerToyCanonicalizePass();
  detail::registerToyConstantFoldPass();
  detail::registerToyCSEPass();
  detail::registerToyDCEPass();
}
```

## Pass Pipelines

A **pipeline** is an ordered sequence of passes:

```cpp
void registerToyPassPipelines() {
  PassPipelineRegistration<>("toy-full",
    "Complete Toy optimization pipeline",
    [](OpPassManager &pm) {
      pm.addPass(createToyShapeInferencePass());
      pm.addPass(createToyCanonicalizePass());
      pm.addPass(createToyConstantFoldPass());
      pm.addPass(createToyCSEPass());
      pm.addPass(createToyDCEPass());
    });
}
```

### Running Pipelines

```bash
# Run individual pass
./build/tools/toy-opt examples/intro.toy --toy-shape-inference

# Run complete pipeline
./examples/pipeline.sh
```

## Writing a New Pass

### Step 1: Create Header File

Create `include/MyNewPass.h`:
```cpp
#ifndef TOY_MY_NEW_PASS_H
#define TOY_MY_NEW_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {
std::unique_ptr<mlir::Pass> createMyNewPass();

namespace detail {
void registerMyNewPass();
}
}

#endif
```

### Step 2: Implement the Pass

Create `passes/MyNewPass.cpp`:
```cpp
#include "MyNewPass.h"
#include "toy/ToyOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace toy;

namespace {
struct MyNewPass : public PassWrapper<MyNewPass, 
                                      OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyNewPass)
  
  StringRef getArgument() const final { return "my-new-pass"; }
  StringRef getDescription() const final { 
    return "Description of my pass"; 
  }
  
  void runOnOperation() override {
    auto func = getOperation();
    
    // Walk all operations in the function
    func.walk([&](Operation *op) {
      // Your transformation logic here
    });
  }
};
}

std::unique_ptr<Pass> toy::createMyNewPass() {
  return std::make_unique<MyNewPass>();
}

void toy::detail::registerMyNewPass() {
  PassRegistration<MyNewPass>();
}
```

### Step 3: Register the Pass

Update `passes/Passes.cpp`:
```cpp
#include "MyNewPass.h"

void toy::registerToyPasses() {
  // ... existing passes ...
  detail::registerMyNewPass();
}
```

### Step 4: Update CMakeLists.txt

Add your new file to `passes/CMakeLists.txt`:
```cmake
add_mlir_library(toy-passes
  # ... existing files ...
  MyNewPass.cpp
)
```

### Step 5: Build and Test

```bash
cd build
ninja
./tools/toy-opt --help | grep my-new-pass
```

## Pattern Rewriting Example

For canonicalization and optimization passes:

```cpp
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
// Pattern to fold mul by 1.0
struct FoldMulByOne : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MulOp op, 
                                PatternRewriter &rewriter) const override {
    // Check if one operand is constant 1.0
    auto lhsConst = op.getLhs().getDefiningOp<ConstantOp>();
    if (lhsConst && isOne(lhsConst)) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (rhsConst && isOne(rhsConst)) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    
    return failure();
  }
};

void runOnOperation() override {
  RewritePatternSet patterns(&getContext());
  patterns.add<FoldMulByOne>(&getContext());
  
  if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                          std::move(patterns)))) {
    signalPassFailure();
  }
}
}
```

## TODO Exercises

- [ ] Implement `runOnOperation()` in ShapeInferencePass with shape propagation logic
- [ ] Add canonicalization patterns for commutative operations
- [ ] Implement constant folding for Add and Mul operations
- [ ] Test CSE with duplicate operations
- [ ] Verify DCE removes unused operations

## Testing Your Passes

### Create Test Files

Create `test/my-pass.mlir`:
```mlir
// RUN: toy-opt %s --my-new-pass | FileCheck %s

func.func @test_pass(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
  // CHECK: toy.add
  %0 = toy.add %arg0, %arg0 : tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
```

### Run Tests

```bash
./build/tools/toy-opt test/my-pass.mlir --my-new-pass
```

## Debugging Passes

### Print IR Before/After

```cpp
void runOnOperation() override {
  auto func = getOperation();
  
  llvm::outs() << "Before transformation:\n";
  func.dump();
  
  // Your transformation
  
  llvm::outs() << "After transformation:\n";
  func.dump();
}
```

### Use `-mlir-print-ir-*` Flags

```bash
toy-opt input.mlir --pass-pipeline='toy-full' \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all
```

### Enable Diagnostics

```cpp
if (someConditionFails) {
  return op->emitError("Detailed error message here");
}
```

## References

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Diagnostic Infrastructure](https://mlir.llvm.org/docs/Diagnostics/)
