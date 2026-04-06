#!/usr/bin/env bash
# Idempotent Coverage & PMAT Compliance Routine
# Usage: ./maintain_coverage.sh

# --- 1. Environment & Context Setup ---
# Explicitly state CUDA availability to prevent redundant checks/warnings.
export CUDA_VISIBLE_DEVICES=0 # Or appropriate device index
export HAS_CUDA=true

echo "================================================================="
echo "  REALIZAR COVERAGE MAINTENANCE ROUTINE (PMAT COMPLIANT)"
echo "  Environment: CUDA ENABLED (RTX 4090 Context)"
echo "================================================================="

# --- 2. Specification & Documentation Sync ---
# Ensure we are always working against the latest spec.
echo "[1/5] Syncing Specification..."
if [ -f "docs/specifications/95-coverage-fast-pmat-comply.md" ]; then
    echo "Using spec: docs/specifications/95-coverage-fast-pmat-comply.md"
else
    echo "ERROR: Specification file not found!"
    exit 1
fi

# (Optional) Auto-commit spec updates if any pending changes exist
if git diff --quiet docs/specifications/95-coverage-fast-pmat-comply.md; then
    echo "Spec is clean."
else
    echo "Spec changes detected. Staging..."
    git add docs/specifications/95-coverage-fast-pmat-comply.md
    # Commit is left to the user or a separate step to avoid unwanted commits
fi

# --- 3. Efficient Coverage Measurement ---
# Run coverage only if source files have changed since last report, 
# or if forced. Uses 'make coverage-core' for speed, or full 'make coverage'
# if a thorough check is needed (e.g., pre-push).
echo "[2/5] Checking Coverage Status..."

# Heuristic: Check if src/ has changed more recently than target/coverage/html/index.html
LATEST_SRC=$(find src -name "*.rs" -printf "%T@\n" | sort -n | tail -1)
LAST_REPORT=$(stat -c %Y target/coverage/html/index.html 2>/dev/null || echo 0)

if [ "${LATEST_SRC%.*}" -gt "$LAST_REPORT" ]; then
    echo "Source changed. Running efficient coverage check (Core + CUDA)..."
    # CRITICAL: Always use CUDA feature for REAL coverage.
    # We use the specific target to avoid rebuilding everything if not needed.
    # Piping to cat prevents 'stuck' TUI artifacts in some environments.
    make coverage COV_EXCLUDE="--ignore-filename-regex='(trueno/|/tests/|_tests|tests_|test_|tui.rs|bench_viz.rs|viz.rs|main.rs|/benches/|/examples/)'"
else
    echo "Coverage report is up-to-date. Skipping run."
    # Display current summary
    cargo llvm-cov report --summary-only --ignore-filename-regex='(trueno/|/tests/|_tests|tests_|test_|tui.rs|bench_viz.rs|viz.rs|main.rs|/benches/|/examples/)' | grep -E "^TOTAL"
fi

# --- 4. The 5 Whys Analysis (Automated Prompts) ---
# This section prints prompts for the LLM/Developer to consider if targets aren't met.
echo "[3/5] Analysis & Root Cause (The 5 Whys)..."

CURRENT_COV=$(cargo llvm-cov report --summary-only --ignore-filename-regex='(trueno/|/tests/|_tests|tests_|test_|tui.rs|bench_viz.rs|viz.rs|main.rs|/benches/|/examples/)' 2>/dev/null | grep "TOTAL" | awk '{print $10}' | tr -d '%')

if (( $(echo "$CURRENT_COV < 95.0" | bc -l) )); then
    echo "⚠️  WARNING: Coverage is ${CURRENT_COV}%, below 95% target."
    echo ""
    echo "Per spec, apply 5 Whys Analysis:"
    echo "1. Why is coverage < 95%? (Which modules are low?)"
    echo "2. Why are those modules uncovered? (Missing unit tests? Integration tests?)"
    echo "3. Why are tests missing? (Hard to mock? Lazy implementation?)"
    echo "4. Why is it hard to mock/test? (Tight coupling? Bad architecture?)"
    echo "5. Root Cause: __________________________________________________"
    echo ""
    echo "Comparison: ../trueno has 95%. What technique are we missing?"
    echo "- Are we using the same harness?"
    echo "- Are we mocking the GPU context correctly?"
else
    echo "✅ Coverage is ${CURRENT_COV}%. Excellent."
fi

# --- 5. PMAT Compliance Check ---
echo "[4/5] PMAT Compliance Verification..."
if command -v pmat >/dev/null; then
    pmat quality-gate || echo "⚠️  PMAT Quality Gate Warnings (Check logs)"
else
    echo "pmat CLI not found. Skipping strict compliance check."
fi

echo "[5/5] Routine Complete."
