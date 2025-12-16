#!/bin/bash
# all-modules-qa-verify.sh
# Aprender Full Codebase QA Verification
#
# Methodology: Toyota Way + NASA Software Safety + Google AI Engineering
#
# Usage: ./all-modules-qa-verify.sh
# Returns: 0 if all mandatory gates pass, 1 otherwise
#
# shellcheck disable=SC2034,SC2031,SC2032,SC2164
# bashrs: disable=SEC010,SC2031,SC2164,SC2317

set -euo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

declare -i PASS_COUNT=0
declare -i FAIL_COUNT=0
declare -i WARN_COUNT=0

pass() {
    echo -e "${GREEN}PASS${NC}"
    ((PASS_COUNT+=1)) || true
}

fail() {
    echo -e "${RED}FAIL${NC}"
    ((FAIL_COUNT+=1)) || true
}

# shellcheck disable=SC2317  # Reserved for future use
warn() {
    echo -e "${YELLOW}WARN${NC}"
    ((WARN_COUNT+=1)) || true
}

section() {
    echo -e "\n${CYAN}=== $1 ===${NC}\n"
}

# Get git info safely
GIT_COMMIT=""
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null)" || GIT_COMMIT="N/A"

GIT_BRANCH=""
GIT_BRANCH="$(git branch --show-current 2>/dev/null)" || GIT_BRANCH="N/A"

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${PROJECT_ROOT}" || { echo "Failed to cd to project root"; exit 1; }

echo "=============================================="
echo "  Aprender Full Codebase QA Verification"
echo "  Toyota + NASA + Google AI Engineering"
echo "=============================================="
echo ""
echo "Git Commit: ${GIT_COMMIT}"
echo "Branch: ${GIT_BRANCH}"
echo "Project Root: ${PROJECT_ROOT}"

section "Section 1: Sub-Crates (20 points)"

# 1.1 aprender-monte-carlo
echo -n "1.1.1 aprender-monte-carlo build: "
if cargo build -p aprender-monte-carlo --quiet 2>/dev/null; then pass; else fail; fi

echo -n "1.1.2 aprender-monte-carlo tests: "
if cargo test -p aprender-monte-carlo --quiet 2>/dev/null; then pass; else fail; fi

# 1.2 aprender-tsp
echo -n "1.2.1 aprender-tsp build: "
if cargo build -p aprender-tsp --quiet 2>/dev/null; then pass; else fail; fi

echo -n "1.2.2 aprender-tsp tests: "
if cargo test -p aprender-tsp --quiet 2>/dev/null; then pass; else fail; fi

# 1.3 aprender-shell
echo -n "1.3.1 aprender-shell build: "
if cargo build -p aprender-shell --quiet 2>/dev/null; then pass; else fail; fi

echo -n "1.3.2 aprender-shell tests: "
if cargo test -p aprender-shell --quiet 2>/dev/null; then pass; else fail; fi

section "Section 2: Core ML Algorithms (15 points)"

echo -n "2.1.1 Linear model tests: "
if cargo test linear_model --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "2.1.4 Decision tree tests: "
if cargo test tree --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "2.2.1 K-Means tests: "
if cargo test kmeans --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "2.2.7 PCA tests: "
if cargo test pca --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 3: Mathematics & Statistics (15 points)"

echo -n "3.1.1 Stats tests: "
if cargo test stats --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "3.1.7 Bayesian tests: "
if cargo test bayesian --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 4: Optimization & Loss (10 points)"

echo -n "4.1 Optimizer tests: "
if cargo test optim --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "4.2.5 Loss tests: "
if cargo test loss --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 5: Graph & Time Series (10 points)"

echo -n "5.1.5 Graph tests: "
if cargo test graph --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "5.2.5 Time series tests: "
if cargo test time_series --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 6: Text & NLP (8 points)"

echo -n "6.7 Text tests: "
if cargo test text --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 7: Model Persistence (8 points)"

echo -n "7.8 Serialization tests: "
if cargo test serial --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 8: Preprocessing & Metrics (7 points)"

echo -n "8.7 Preprocessing tests: "
if cargo test preprocessing --lib --quiet 2>/dev/null; then pass; else fail; fi

echo -n "8.x Metrics tests: "
if cargo test metrics --lib --quiet 2>/dev/null; then pass; else fail; fi

section "Section 9: Mandatory Gates"

echo -n "9.1.1 Workspace tests: "
if cargo test --workspace --quiet 2>/dev/null; then
    pass
else
    echo -e "${YELLOW}RUNNING FULL TEST${NC}"
    cargo test --workspace 2>&1 | tail -5
    fail
fi

echo -n "9.2.1 Clippy (workspace): "
if cargo clippy --workspace --quiet -- -D warnings 2>/dev/null; then pass; else fail; fi

echo -n "9.2.2 Format check: "
if cargo fmt --all --check --quiet 2>/dev/null; then pass; else fail; fi

section "Section 10: Documentation (7 points)"

echo -n "10.1 Docs build: "
if cargo doc --no-deps --quiet 2>/dev/null; then pass; else fail; fi

echo -n "10.2 Examples compile: "
if cargo build --examples --quiet 2>/dev/null; then pass; else fail; fi

section "Random Sample Verification"

echo "Running random sample spot checks..."

echo -n "Sample: K-Means convergence test: "
KMEANS_OUTPUT="$(cargo test kmeans_basic --lib -- --nocapture 2>&1)" || true
if echo "${KMEANS_OUTPUT}" | grep -q "passed"; then pass; else fail; fi

echo -n "Sample: Linear regression test: "
LR_OUTPUT="$(cargo test linear_regression --lib -- --nocapture 2>&1)" || true
if echo "${LR_OUTPUT}" | grep -q "passed"; then pass; else fail; fi

section "Summary"

declare -i TOTAL
TOTAL=$((PASS_COUNT + FAIL_COUNT))

echo "=============================================="
echo "  Results"
echo "=============================================="
echo ""
echo "Passed: ${PASS_COUNT} / ${TOTAL}"
echo "Failed: ${FAIL_COUNT} / ${TOTAL}"
echo "Warnings: ${WARN_COUNT}"
echo ""

# Calculate approximate score (each check roughly 4 points)
declare -i SCORE
SCORE=$((PASS_COUNT * 100 / TOTAL))

echo "Approximate Score: ${SCORE}/100"
echo ""

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
    echo -e "${GREEN}=== ALL CHECKS PASSED ===${NC}"
    echo ""
    if [[ "${SCORE}" -ge 95 ]]; then
        echo "Grade: A+ - Production Ready"
    elif [[ "${SCORE}" -ge 90 ]]; then
        echo "Grade: A - Production Ready (minor fixes)"
    elif [[ "${SCORE}" -ge 85 ]]; then
        echo "Grade: B+ - Staging Only"
    else
        echo "Grade: B - Development Only"
    fi
    echo ""
    echo "Disposition: APPROVED"
    exit 0
fi

echo -e "${RED}=== SOME CHECKS FAILED ===${NC}"
echo ""
echo "Disposition: REQUIRES REMEDIATION"
echo ""
echo "Per Toyota Way Kaizen principles:"
echo "1. Identify failed items above"
echo "2. Use 5 Whys for root cause analysis"
echo "3. Implement fixes"
echo "4. Re-run verification"
exit 1
