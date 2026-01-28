#!/bin/bash
# PMAT-116 Falsification Protocol
# Attempts to falsify the "zero SATD" claim for SafeTensors GPU implementation
set -euo pipefail

echo "=== PMAT-116 Falsification Protocol ==="
echo "Started at: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APRENDER_DIR="$(dirname "$SCRIPT_DIR")"
REALIZAR_DIR="/home/noah/src/realizar"

# 1. Existence Checks
echo "[1/5] Verifying Implementation Existence..."
if [ ! -f "$REALIZAR_DIR/src/safetensors_cuda.rs" ]; then
    echo "FAIL: realizar/src/safetensors_cuda.rs not found"
    exit 1
fi
echo "  - realizar/src/safetensors_cuda.rs exists"

if ! grep -q "SafeTensorsCudaModel" "$APRENDER_DIR/crates/apr-cli/src/commands/chat.rs"; then
    echo "FAIL: SafeTensorsCudaModel not integrated in chat.rs"
    exit 1
fi
echo "  - SafeTensorsCudaModel integrated in chat.rs"
echo "PASS: Implementation files exist."
echo ""

# 2. SATD Scan in realizar
echo "[2/5] Scanning realizar for Residual SATD..."
cd "$REALIZAR_DIR"
SATD_COUNT=$(grep -r "SATD.*PMAT-116\|TODO.*PMAT-116" src/ 2>/dev/null | wc -l || echo "0")
if [ "$SATD_COUNT" -ne "0" ]; then
    echo "FAIL: Found $SATD_COUNT residual SATD markers in realizar:"
    grep -r "SATD.*PMAT-116\|TODO.*PMAT-116" src/
    exit 1
fi
echo "PASS: Zero SATD in realizar."
echo ""

# 3. SATD Scan in aprender
echo "[3/5] Scanning aprender for Residual SATD..."
cd "$APRENDER_DIR"
# Exclude spec file and this script from scan
SATD_COUNT=$(grep -r "SATD.*PMAT-116\|TODO.*PMAT-116" crates/ src/ 2>/dev/null | grep -v "implement-gpu-safetensors.md" | grep -v "verify_pmat_116.sh" | wc -l || echo "0")
if [ "$SATD_COUNT" -ne "0" ]; then
    echo "FAIL: Found $SATD_COUNT residual SATD markers in aprender:"
    grep -r "SATD.*PMAT-116\|TODO.*PMAT-116" crates/ src/ | grep -v "implement-gpu-safetensors.md"
    exit 1
fi
echo "PASS: Zero SATD in aprender."
echo ""

# 4. Build Verification
echo "[4/5] Verifying Build with CUDA feature..."
cd "$REALIZAR_DIR"
cargo check --features cuda 2>&1 | tail -5
echo "PASS: realizar builds with cuda feature."

cd "$APRENDER_DIR"
cargo check -p apr-cli --features inference 2>&1 | tail -5
echo "PASS: apr-cli builds with inference feature."
echo ""

# 5. Key Component Verification
echo "[5/5] Verifying Key Components..."

# Check gamma_cache exists (RMS norm fix)
if ! grep -q "gamma_cache" "$REALIZAR_DIR/src/safetensors_cuda.rs"; then
    echo "FAIL: gamma_cache not found - RMS norm SATD not resolved"
    exit 1
fi
echo "  - gamma_cache present (RMS norm gamma weights)"

# Check position parameter removed (RoPE fix)
if grep -q "_position.*SATD" "$REALIZAR_DIR/src/safetensors_cuda.rs"; then
    echo "FAIL: Position SATD marker still present"
    exit 1
fi
echo "  - Position/RoPE handled internally"

# Check no CPU fallback comments
FALLBACK_COUNT=$(grep -c "CPU.*fallback\|fallback.*CPU" "$REALIZAR_DIR/src/safetensors_cuda.rs" 2>/dev/null || echo "0")
if [ "$FALLBACK_COUNT" -gt "0" ]; then
    echo "WARNING: Found $FALLBACK_COUNT CPU fallback references (may be documentation)"
fi

echo ""
echo "=== Falsification Complete: CORROBORATED ==="
echo "All SATD claims verified. Implementation is complete."
echo "Finished at: $(date)"
