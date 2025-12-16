#!/bin/bash

set -e

# Find toy-opt binary
if [ -f "build/tools/toy-opt" ]; then
    TOY_OPT="build/tools/toy-opt"
else
    echo "Error: toy-opt not found in build/tools/"
    echo "Please build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake -G Ninja .. -DMLIR_DIR=<path> -DLLVM_DIR=<path>"
    echo "  ninja"
    exit 1
fi

# Get input file
INPUT="${1:-intro.toy}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found"
    exit 1
fi

echo "========================================"
echo "Running Toy optimization pipeline"
echo "========================================"
echo "Tool:  $TOY_OPT"
echo "Input: $INPUT"
echo "========================================"
echo ""

# Run the full pipeline
"${TOY_OPT}" "${INPUT}" \
  --allow-unregistered-dialect \
  --pass-pipeline='builtin.module(toy-full)' \
  -mlir-print-ir-after-all

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
