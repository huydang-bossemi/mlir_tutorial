#include "toy/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

void toy::registerToyPasses() {
  detail::registerToyShapeInferencePass();
  detail::registerToyCanonicalizePass();
  detail::registerToyDCEPass();
  detail::registerToyCSEPass();
  detail::registerToyConstantFoldPass();
}

void toy::registerToyPassPipelines() {
  mlir::PassPipelineRegistration<>(
      "toy-full",
      "Toy tutorial pipeline: shape inference then canonicalization",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createToyShapeInferencePass());
        pm.addPass(createToyCanonicalizePass());
        pm.addNestedPass<mlir::func::FuncOp>(createToyDCEPass());
        pm.addNestedPass<mlir::func::FuncOp>(createToyCSEPass());
        pm.addNestedPass<mlir::func::FuncOp>(createToyConstantFoldPass());
      });
}
