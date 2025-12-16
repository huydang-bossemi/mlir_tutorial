# mlir_tutorial

A tiny project skeleton that demonstrates how to build an MLIR-based toy dialect, write passes, and run a minimal lowering pipeline. Everything is intentionally stubbed with TODO markers so learners can complete each step while following the docs under `docs/`.

## Project Layout

- `toy/` – Dialect sources, TableGen definitions, and registration hooks.
- `passes/` – Example pass stubs (shape inference & canonicalization).
- `examples/` – Toy source snippet plus helper script for the pipeline.
- `docs/` – Tutorial chapters explaining each stage of the journey.
- `docker/` – Self-contained build environment powered by LLVM/MLIR.
- `CMakeLists.txt` – Configures the project against an existing MLIR build.

## Prerequisites

- LLVM/MLIR build (provide `LLVM_DIR`/`MLIR_DIR` via CMake cache or environment).
- CMake ≥ 3.20 and Ninja (already bundled in the docker image).

## Configure & Build

There is an available Docker image on Wormhole server so user just need to run:

```bash
./scripts/docker_run.sh
```

And a container with name **mlir-tutorial-<USER_NAME>** will be ready to used.

Commands to build the repo:

```bash
apt-get update
apt-get install -y zlib1g-dev libtinfo-dev libedit-dev libxml2-dev
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/opt/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
ninja
```

The configuration step automatically adds the Toy dialect library (`toy-dialect`) and pass library (`toy-passes`).

## Run the Example Pipeline

1. Build the project (generates `build/tools/toy-opt`).
2. (Optional) Source your MLIR environment for extra tooling like `mlir-translate`.
3. Execute the helper script:
   ```bash
   ./examples/pipeline.sh
   ```
  The script feeds `examples/intro.toy` into `toy-opt`, executes the Toy pipeline (`toy-shape-inference`, `toy-canonicalize`, ...), and prints each IR stage.

## Learning Objectives

- Understand how MLIR dialects are defined via TableGen and C++ scaffolding.
- Implement operation builders, printers, and verifiers.
- Write analysis / transformation passes and chain them into pipelines.
- Use MLIR tooling (`mlir-opt`, `mlir-translate`, pass pipelines) for debugging.

Dive into `docs/00_overview.md` to get started!
