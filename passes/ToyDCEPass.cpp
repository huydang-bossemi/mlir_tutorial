#include "ToyDCEPass.h"

#include "toy/ToyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace toy;

namespace {

struct ToyDCEPass
    : public mlir::PassWrapper<ToyDCEPass,
                              mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyDCEPass)

  mlir::StringRef getArgument() const final { return "toy-dce"; }
  mlir::StringRef getDescription() const final {
    return "Skeleton pass for Toy dead code elimination";
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    (void)funcOp;

    // TODO: Traverse every operation inside the function to analyze SSA uses.
    // TODO: Detect operations with no users via `op->use_empty()`.
    // TODO: Skip terminators because they are required to maintain block
    // structure and control flow.
    // TODO: Collect dead operations first, then erase them safely after the
    // traversal to avoid invalidating iterators.
  }
};

} // namespace

std::unique_ptr<mlir::Pass> toy::createToyDCEPass() {
  return std::make_unique<ToyDCEPass>();
}

void toy::detail::registerToyDCEPass() {
  static mlir::PassRegistration<ToyDCEPass> pass;
  (void)pass;
}
