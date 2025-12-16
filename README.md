# MLIR Toy Tutorial

A hands-on MLIR tutorial workspace for learning dialect creation, TableGen usage, pass infrastructure, and pipeline execution through practical examples.

## Overview

This project provides a minimal but complete MLIR development environment featuring:
- A simple "Toy" dialect with basic tensor operations (`toy.add`, `toy.mul`)
- Shape inference and canonicalization pass stubs ready for implementation
- Pass pipeline infrastructure with the `toy-full` pipeline pre-configured
- Docker-based development workflow for team collaboration
- Structured documentation in `docs/` with guided exercises

## Learning Objectives

- **Dialect Design**: Define custom MLIR dialects using TableGen
- **Operation Definition**: Create operations with TableGen's declarative syntax
- **Pass Development**: Write transformation and analysis passes
- **Pipeline Composition**: Chain multiple passes into reusable pipelines
- **MLIR Infrastructure**: Understand MLIR's build system and project structure

## Prerequisites

### For Team Members (Docker Workflow)

No installation required! Just use the pre-built Docker image on the host PC.

### For Administrators (Optional - Docker Image Build)

The Docker image already contains LLVM/MLIR, so you only need Docker:

```bash
# Ubuntu/Debian
cd docker/
./docker_build.sh
```

## Project Structure

```
mlir_tutorial/
├── README.md                    # This file
├── CMakeLists.txt              # Root build configuration
├── toy/                        # Toy dialect implementation
│   ├── CMakeLists.txt
│   ├── ToyDialect.td           # Dialect TableGen definition
│   ├── ToyOps.td               # Operations TableGen definition
│   ├── Ops.cpp                 # Operation implementations
│   └── DialectRegistration.cpp
├── include/toy/
│   ├── ToyDialect.h            # Dialect header
│   ├── ToyOps.h                # Operations header
│   └── Transforms/
│       └── Passes.h            # Pass declarations
├── passes/                     # Transformation passes
│   ├── CMakeLists.txt
│   ├── ShapeInferencePass.cpp
│   └── CanonicalizePass.cpp
├── tools/                      # Command-line tools
│   ├── CMakeLists.txt
│   └── toy-opt.cpp             # Optimizer tool
├── examples/                   # Example Toy programs
│   ├── intro.toy
│   └── pipeline.sh             # Run script
├── docs/                       # Documentation
└── docker/                     # Docker setup
```

## Quick Start (Recommended)

### 1. Run Docker Container

```bash
# Clone or navigate to the project
cd /<path_to_your_mlir_tutorial>/

# Start your personal container
./scripts/docker_run.sh
```

This creates a persistent container named `mlir_tutorial_${USER}` with LLVM/MLIR pre-installed.

### 2. Build Inside Container

```bash
# Inside the container at /workspace
apt-get update
apt-get install -y zlib1g-dev libtinfo-dev libedit-dev libxml2-dev
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=/opt/llvm/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
ninja
```

### 3. Run Examples

```bash
cd /workspace/examples
./pipeline.sh
```

This runs the full Toy optimization pipeline with:
- `toy-shape-inference`: Infers tensor shapes through the IR
- `toy-canonicalize`: Applies canonicalization patterns
- `toy-full`: Complete pipeline (shape inference → canonicalize)

### 4. Restart Your Container

```bash
# Outside container - to restart later
docker start -i mlir_tutorial_${USER}

# To stop
docker stop mlir_tutorial_${USER}
```

## Manual Execution (Advanced)

After building, you can run passes individually:

```bash
cd /workspace

# View original IR
./build/tools/toy-opt examples/intro.toy

# Run shape inference only
./build/tools/toy-opt examples/intro.toy \
  --pass-pipeline='builtin.module(toy-shape-inference)'

# Run full pipeline with IR printing
./build/tools/toy-opt examples/intro.toy \
  --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(toy-full)' \
  -mlir-print-ir-after-all
```

## Development Guide

### Getting Started

1. **Read the documentation**: Start with `docs/00_overview.md` for a structured learning path
2. **Understand the structure**: Review `docs/01_dialect.md` and `docs/02_ops.md`
3. **Implement passes**: Follow exercises in `docs/03_passes.md` and `docs/06_exercises.md`
4. **Extend functionality**: Add new operations or implement lowering passes

### Key Implementation Files

- **Dialect Definition**: `toy/ToyDialect.td` - Define dialect metadata
- **Operations**: `toy/ToyOps.td` - Define operations using TableGen
- **Pass Logic**: `passes/ShapeInferencePass.cpp`, `passes/CanonicalizePass.cpp`
- **Pipeline Registration**: See `registerToyPassPipelines()` in `ShapeInferencePass.cpp`

After modifying `.td` files, rebuild with `ninja` to regenerate C++ headers.

### Adding a New Operation

1. Define in `toy/ToyOps.td`:
```tablegen
def Toy_SubOp : Toy_Op<"sub", [Pure]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}
```

2. Rebuild:
```bash
cd build && ninja
```

3. The operation is now available as `toy.sub`

## Acknowledgments

This tutorial workspace was inspired by and references concepts from:
- [Official MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) - The canonical MLIR learning resource
- [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial) - Excellent community tutorial with detailed walkthroughs

Our implementation focuses on a streamlined Docker workflow suitable for team-based learning environments.

## Resources

- [MLIR Documentation](https://mlir.llvm.org/)
- [MLIR Toy Tutorial (Official)](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [TableGen Language Reference](https://llvm.org/docs/TableGen/)
- [Jeremy Kun's MLIR Tutorial Series](https://github.com/j2kun/mlir-tutorial)

## License

This tutorial is provided as-is for educational purposes. MLIR and LLVM are licensed under the Apache License 2.0 with LLVM Exceptions.
