#!/usr/bin/env bash
set -e

# verify-parity.sh - Wrapper for the APR QA Matrix Falsification Suite
#
# Usage: ./verify-parity.sh [OPTIONS]
# Options:
#   --class <quantized|full-precision|all>  (Default: quantized)
#   --backend <cpu|gpu>                     Force specific backend
#   --trace                                 Enable tracing
#   --with-ollama                           Compare against Ollama baseline

echo "🧪 Running Popperian Parity Verification (Matrix Mode)..."
echo "Target: Qwen2.5-Coder-1.5B (Canonical)"
echo "Docs:   docs/specifications/qwen2.5-coder-showcase-demo.md"
echo ""

# Ensure we are in the project root
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Must be run from project root"
    exit 1
fi

# Run the matrix test using the qa_run example
# This example handles the downloading of canonical models automatically
cargo run --release --quiet --example qa_run -- --matrix "$@"
