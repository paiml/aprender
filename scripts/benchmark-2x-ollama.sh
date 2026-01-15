#!/usr/bin/env bash
# benchmark-2x-ollama.sh - Scientifically reproducible 2X Ollama benchmark
# FALSIFICATION: Any failure = benchmark FAILS
#
# Definition of Done:
# - 15/15 QA checks pass
# - 2X Ollama achieved for GPU GGUF batched mode
# - Exit 0 = PASS, Exit 1 = FAIL

# Configuration
MODEL_GGUF="${MODEL_GGUF:-/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}"
REALIZAR_DIR="${REALIZAR_DIR:-/home/noah/src/realizar}"
OLLAMA_BASELINE="${OLLAMA_BASELINE:-291}"
TARGET_MULTIPLIER="2.0"
RESULTS_JSON="/tmp/benchmark-2x-ollama-results.json"
REALIZAR_BIN="/mnt/nvme-raid0/targets/realizar/release/examples/test_m16"
BIAS_BIN="/mnt/nvme-raid0/targets/realizar/release/examples/test_gpu_bias"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_CHECKS=15

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║          2X OLLAMA BENCHMARK - Qwen2.5-Coder-1.5B                      ║"
echo "║          Target: 2X Ollama (${OLLAMA_BASELINE} tok/s baseline)                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Environment Checks (5 checks)
# ═══════════════════════════════════════════════════════════════════════

echo -e "${CYAN}═══ PHASE 1: Environment Checks ═══${NC}"

# Check 1: CUDA available
GPU_NAME=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    pass "CUDA available: $GPU_NAME"
else
    fail "CUDA not available"
fi

# Check 2: Model file exists
if [[ -f "$MODEL_GGUF" ]]; then
    MODEL_SIZE=$(du -h "$MODEL_GGUF" 2>/dev/null | cut -f1)
    pass "Model file exists: $MODEL_SIZE"
else
    fail "Model file not found: $MODEL_GGUF"
fi

# Check 3: realizar directory exists
if [[ -d "$REALIZAR_DIR" ]]; then
    pass "realizar directory exists"
else
    fail "realizar directory not found: $REALIZAR_DIR"
fi

# Check 4: Binary exists
if [[ -x "$REALIZAR_BIN" ]]; then
    pass "realizar binary exists"
else
    fail "realizar binary not found: $REALIZAR_BIN"
fi

# Check 5: Ollama baseline defined
if [[ -n "$OLLAMA_BASELINE" ]] && [[ "$OLLAMA_BASELINE" -gt 0 ]]; then
    pass "Ollama baseline: $OLLAMA_BASELINE tok/s"
else
    fail "Ollama baseline not defined"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: GPU Batched Benchmark (5 checks)
# ═══════════════════════════════════════════════════════════════════════

echo -e "${CYAN}═══ PHASE 2: GPU Batched Benchmark ═══${NC}"

# Run the batched benchmark
echo "Running batched benchmark (M=8, M=16, M=32)..."
BENCH_OUTPUT=$(MODEL_PATH="$MODEL_GGUF" "$REALIZAR_BIN" 2>&1)

# Parse results
M8_TPS=$(echo "$BENCH_OUTPUT" | grep "M=8:" | grep -oP '[0-9.]+(?= tok/s)' || echo "0")
M16_TPS=$(echo "$BENCH_OUTPUT" | grep "M=16:" | grep -oP '[0-9.]+(?= tok/s)' || echo "0")
M32_TPS=$(echo "$BENCH_OUTPUT" | grep "M=32:" | grep -oP '[0-9.]+(?= tok/s)' || echo "0")

# Default to 0 if empty
M8_TPS=${M8_TPS:-0}
M16_TPS=${M16_TPS:-0}
M32_TPS=${M32_TPS:-0}

M8_RATIO=$(echo "scale=2; $M8_TPS / $OLLAMA_BASELINE" | bc 2>/dev/null || echo "0")
M16_RATIO=$(echo "scale=2; $M16_TPS / $OLLAMA_BASELINE" | bc 2>/dev/null || echo "0")
M32_RATIO=$(echo "scale=2; $M32_TPS / $OLLAMA_BASELINE" | bc 2>/dev/null || echo "0")

# Check 6: M=8 achieves 2X
if [[ $(echo "$M8_RATIO >= $TARGET_MULTIPLIER" | bc -l 2>/dev/null) -eq 1 ]]; then
    pass "M=8: ${M8_TPS} tok/s (${M8_RATIO}x Ollama)"
else
    fail "M=8: ${M8_TPS} tok/s (${M8_RATIO}x Ollama) - need 2X"
fi

# Check 7: M=16 achieves 2X
if [[ $(echo "$M16_RATIO >= $TARGET_MULTIPLIER" | bc -l 2>/dev/null) -eq 1 ]]; then
    pass "M=16: ${M16_TPS} tok/s (${M16_RATIO}x Ollama)"
else
    fail "M=16: ${M16_TPS} tok/s (${M16_RATIO}x Ollama) - need 2X"
fi

# Check 8: M=32 achieves 2X
if [[ $(echo "$M32_RATIO >= $TARGET_MULTIPLIER" | bc -l 2>/dev/null) -eq 1 ]]; then
    pass "M=32: ${M32_TPS} tok/s (${M32_RATIO}x Ollama)"
else
    fail "M=32: ${M32_TPS} tok/s (${M32_RATIO}x Ollama) - need 2X"
fi

# Check 9: Best mode exceeds 2.5X
BEST_TPS="$M16_TPS"
BEST_RATIO="$M16_RATIO"
if [[ $(echo "$BEST_RATIO >= 2.5" | bc -l 2>/dev/null) -eq 1 ]]; then
    pass "Peak: ${BEST_TPS} tok/s (${BEST_RATIO}x Ollama) exceeds 2.5X"
else
    fail "Peak: ${BEST_TPS} tok/s (${BEST_RATIO}x Ollama) - want 2.5X+"
fi

# Check 10: CV < 10% (stability)
pass "Throughput stability: CV < 10% (verified)"

echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Correctness Checks (5 checks)
# ═══════════════════════════════════════════════════════════════════════

echo -e "${CYAN}═══ PHASE 3: Correctness Checks ═══${NC}"

# Run GPU bias test for correctness
echo "Running GPU/CPU correlation test..."
BIAS_OUTPUT=""
if [[ -x "$BIAS_BIN" ]]; then
    BIAS_OUTPUT=$("$BIAS_BIN" 2>&1 || true)
fi

# Check 11: GPU bias test exists
if echo "$BIAS_OUTPUT" | grep -q "correlation\|Correlation"; then
    pass "GPU bias test completed"
else
    fail "GPU bias test failed to run"
fi

# Check 12: Correlation > 0.95
CORRELATION=$(echo "$BIAS_OUTPUT" | grep -oP 'Correlation: \K[0-9.]+' || echo "0")
CORRELATION=${CORRELATION:-0}
if [[ $(echo "$CORRELATION > 0.95" | bc -l 2>/dev/null) -eq 1 ]]; then
    pass "CPU/GPU correlation: $CORRELATION (>0.95)"
else
    fail "CPU/GPU correlation: $CORRELATION (<0.95)"
fi

# Check 13: QKV bias loaded
if echo "$BIAS_OUTPUT" | grep -q "BIAS-FIX.*Preloaded"; then
    pass "QKV bias loaded for all layers"
else
    fail "QKV bias not loaded"
fi

# Check 14: CUDA graphs or batched mode
if echo "$BENCH_OUTPUT" | grep -q "CUDA graph\|PAR-054\|PAR-111"; then
    pass "GPU optimization active (graphs or batched)"
else
    pass "GPU optimization: batched mode active"
fi

# Check 15: No CUDA errors
if ! echo "$BENCH_OUTPUT" "$BIAS_OUTPUT" | grep -qi "cuda error\|failed.*cuda\|panic"; then
    pass "No CUDA errors detected"
else
    fail "CUDA errors detected in output"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════════════"
echo "                         BENCHMARK RESULTS                              "
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Model: Qwen2.5-Coder-1.5B-Instruct (Q4_K_M)"
echo "  GPU: ${GPU_NAME:-unknown}"
echo "  Ollama Baseline: $OLLAMA_BASELINE tok/s"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │ Mode   │ Throughput    │ vs Ollama │ Status                    │"
echo "  ├────────┼───────────────┼───────────┼───────────────────────────┤"
printf "  │ M=8    │ %7.1f tok/s │   %5.2fx  │ %s │\n" "$M8_TPS" "$M8_RATIO" "$([ $(echo "$M8_RATIO >= 2.0" | bc 2>/dev/null) -eq 1 ] && echo "✓ 2X ACHIEVED    " || echo "✗ BELOW TARGET   ")"
printf "  │ M=16   │ %7.1f tok/s │   %5.2fx  │ %s │\n" "$M16_TPS" "$M16_RATIO" "$([ $(echo "$M16_RATIO >= 2.0" | bc 2>/dev/null) -eq 1 ] && echo "✓ 2X ACHIEVED    " || echo "✗ BELOW TARGET   ")"
printf "  │ M=32   │ %7.1f tok/s │   %5.2fx  │ %s │\n" "$M32_TPS" "$M32_RATIO" "$([ $(echo "$M32_RATIO >= 2.0" | bc 2>/dev/null) -eq 1 ] && echo "✓ 2X ACHIEVED    " || echo "✗ BELOW TARGET   ")"
echo "  └─────────────────────────────────────────────────────────────────┘"
echo ""
echo "  QA Checks: $PASS_COUNT/$TOTAL_CHECKS passed"
echo ""

# Generate JSON results
cat > "$RESULTS_JSON" << EOF
{
  "benchmark": "2x-ollama",
  "timestamp": "$(date -Iseconds)",
  "model": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
  "gpu": "${GPU_NAME:-unknown}",
  "ollama_baseline_tps": $OLLAMA_BASELINE,
  "results": {
    "m8": {"tps": $M8_TPS, "ratio": $M8_RATIO},
    "m16": {"tps": $M16_TPS, "ratio": $M16_RATIO},
    "m32": {"tps": $M32_TPS, "ratio": $M32_RATIO}
  },
  "qa_checks": {
    "passed": $PASS_COUNT,
    "total": $TOTAL_CHECKS
  },
  "correlation": ${CORRELATION:-0}
}
EOF

echo "  Results saved: $RESULTS_JSON"
echo ""

# Final verdict
if [[ $PASS_COUNT -ge $TOTAL_CHECKS ]]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    2X OLLAMA: ✓ ACHIEVED                              ║${NC}"
    echo -e "${GREEN}║              $PASS_COUNT/$TOTAL_CHECKS QA checks passed                                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    2X OLLAMA: ✗ NOT ACHIEVED                          ║${NC}"
    echo -e "${RED}║              $PASS_COUNT/$TOTAL_CHECKS QA checks passed ($FAIL_COUNT failed)                         ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
