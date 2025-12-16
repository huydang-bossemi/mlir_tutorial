//===- Passes.h - Toy pass declarations ------------------------*- C++ -*-===//
//
// Pass declarations for the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TRANSFORMS_PASSES_H
#define TOY_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace toy {

// Pass factory functions
std::unique_ptr<mlir::Pass> createToyShapeInferencePass();
std::unique_ptr<mlir::Pass> createToyCanonicalizePass();

// Register all passes
void registerToyPasses();

// Register pass pipelines
void registerToyPassPipelines();

// Detail namespace for registration helpers
namespace detail {
void registerToyShapeInferencePass();
void registerToyCanonicalizePass();
} // namespace detail

} // namespace toy

#endif // TOY_TRANSFORMS_PASSES_H
