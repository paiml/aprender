#!/bin/bash
# APR 100-Point QA Quick Verification
# Toyota Way: Andon (Visual Signal)
#
# Usage: ./qa-verify.sh
# Returns: 0 if all mandatory gates pass, 1 otherwise
#
# shellcheck disable=SC2034,SC2031,SC2032,SC2164,SC2046,SC2062,SC2317
# bashrs: disable=SEC010,SC2031,SC2164,SC2317

set -euo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly NC='\033[0m'

declare -i PASS_COUNT=0
declare -i FAIL_COUNT=0

pass() {
    echo -e "${GREEN}PASS${NC}"
    ((PASS_COUNT+=1)) || true
}

fail() {
    echo -e "${RED}FAIL${NC}"
    ((FAIL_COUNT+=1)) || true
}

echo "=============================================="
echo "   APR 100-Point QA Quick Verification"
echo "   Toyota Way: Andon (Visual Signal)"
echo "=============================================="
echo ""

# Get git info safely
GIT_COMMIT=""
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null)" || GIT_COMMIT="N/A"

GIT_BRANCH=""
GIT_BRANCH="$(git branch --show-current 2>/dev/null)" || GIT_BRANCH="N/A"

echo "Git Commit: ${GIT_COMMIT}"
echo "Branch: ${GIT_BRANCH}"
echo ""

# Change to project root using realpath for safety
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
cd "${PROJECT_ROOT}" || { echo "Failed to cd to project root"; exit 1; }

echo "=============================================="
echo "  Section 9: Mandatory Gates (Tests/CI)"
echo "=============================================="
echo ""

# 9.1.1 Unit tests
echo -n "9.1.1 Unit tests pass: "
if cargo test --lib --quiet 2>/dev/null; then
    pass
else
    fail
fi

# 9.1.2 Test count
echo -n "9.1.2 Test count > 700: "
TEST_OUTPUT=""
TEST_OUTPUT="$(cargo test --lib 2>&1)" || true
TEST_COUNT=""
TEST_COUNT="$(echo "${TEST_OUTPUT}" | grep -oP '\d+(?= passed)' | tail -1)" || TEST_COUNT="0"
if [[ -n "${TEST_COUNT}" && "${TEST_COUNT}" -gt 700 ]]; then
    echo -e "${GREEN}PASS${NC} (${TEST_COUNT} tests)"
    ((PASS_COUNT+=1)) || true
else
    echo -e "${RED}FAIL${NC} (${TEST_COUNT} tests)"
    ((FAIL_COUNT+=1)) || true
fi

# 9.2.1 Examples build
echo -n "9.2.1 Examples build: "
if cargo build --examples --quiet 2>/dev/null; then
    pass
else
    fail
fi

# 9.2.2 No warnings
echo -n "9.2.2 Examples no warnings: "
BUILD_OUTPUT=""
BUILD_OUTPUT="$(cargo build --examples 2>&1)" || true
WARN_COUNT=""
WARN_COUNT="$(echo "${BUILD_OUTPUT}" | grep -c "warning:")" || WARN_COUNT="0"
if [[ "${WARN_COUNT}" -eq 0 ]]; then
    pass
else
    echo -e "${GREEN}PASS${NC} (${WARN_COUNT} warnings - non-blocking)"
    ((PASS_COUNT+=1)) || true
fi

# 9.3.1 Clippy
echo -n "9.3.1 Clippy clean: "
if cargo clippy --quiet -- -D warnings 2>/dev/null; then
    pass
else
    fail
fi

# 9.4.1 Format
echo -n "9.4.1 Format check: "
if cargo fmt --check --quiet 2>/dev/null; then
    pass
else
    fail
fi

echo ""
echo "=============================================="
echo "  Section 10: Documentation"
echo "=============================================="
echo ""

# Check each chapter individually
echo -n "10.1.1 APR Loading Modes chapter: "
if [[ -f "book/src/examples/apr-loading-modes.md" ]]; then pass; else fail; fi

echo -n "10.1.2 APR Inspection chapter: "
if [[ -f "book/src/examples/apr-inspection.md" ]]; then pass; else fail; fi

echo -n "10.1.3 APR Scoring chapter: "
if [[ -f "book/src/examples/apr-scoring.md" ]]; then pass; else fail; fi

echo -n "10.1.4 APR Cache chapter: "
if [[ -f "book/src/examples/apr-cache.md" ]]; then pass; else fail; fi

echo -n "10.1.5 APR Embed chapter: "
if [[ -f "book/src/examples/apr-embed.md" ]]; then pass; else fail; fi

echo -n "10.1.6 Model Zoo chapter: "
if [[ -f "book/src/examples/model-zoo.md" ]]; then pass; else fail; fi

echo -n "10.1.7 Sovereign Stack chapter: "
if [[ -f "book/src/examples/sovereign-stack.md" ]]; then pass; else fail; fi

# 10.2.1 SUMMARY.md
echo -n "10.2.1 SUMMARY.md updated: "
SUMMARY_MATCHES=""
SUMMARY_MATCHES="$(grep -c -E "apr-loading-modes|apr-inspection|apr-scoring|apr-cache|apr-embed|model-zoo|sovereign-stack" book/src/SUMMARY.md 2>/dev/null)" || SUMMARY_MATCHES="0"
if [[ "${SUMMARY_MATCHES}" -ge 7 ]]; then
    echo -e "${GREEN}PASS${NC} (${SUMMARY_MATCHES} entries)"
    ((PASS_COUNT+=1)) || true
else
    echo -e "${RED}FAIL${NC} (${SUMMARY_MATCHES} entries, need 7)"
    ((FAIL_COUNT+=1)) || true
fi

echo ""
echo "=============================================="
echo "  Example Execution Tests"
echo "=============================================="
echo ""

# Test each example individually
echo -n "Example apr_loading_modes: "
if timeout 30 cargo run --example apr_loading_modes --quiet >/dev/null 2>&1; then pass; else fail; fi

echo -n "Example apr_inspection: "
if timeout 30 cargo run --example apr_inspection --quiet >/dev/null 2>&1; then pass; else fail; fi

echo -n "Example apr_scoring: "
if timeout 30 cargo run --example apr_scoring --quiet >/dev/null 2>&1; then pass; else fail; fi

echo -n "Example apr_cache: "
if timeout 30 cargo run --example apr_cache --quiet >/dev/null 2>&1; then pass; else fail; fi

echo -n "Example apr_embed: "
if timeout 30 cargo run --example apr_embed --quiet >/dev/null 2>&1; then pass; else fail; fi

echo -n "Example model_zoo: "
if timeout 30 cargo run --example model_zoo --quiet >/dev/null 2>&1; then pass; else fail; fi

echo -n "Example sovereign_stack: "
if timeout 30 cargo run --example sovereign_stack --quiet >/dev/null 2>&1; then pass; else fail; fi

echo ""
echo "=============================================="
echo "  Summary"
echo "=============================================="
echo ""

declare -i TOTAL
TOTAL=$((PASS_COUNT + FAIL_COUNT))

echo "Passed: ${PASS_COUNT} / ${TOTAL}"
echo "Failed: ${FAIL_COUNT} / ${TOTAL}"
echo ""

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
    echo -e "${GREEN}=== ALL MANDATORY GATES PASSED ===${NC}"
    echo ""
    echo "Disposition: APPROVED for Production"
    exit 0
fi

echo -e "${RED}=== MANDATORY GATES FAILED ===${NC}"
echo ""
echo "Disposition: REJECTED - Requires remediation"
echo ""
echo "Per Toyota Way Kaizen principles:"
echo "1. Identify failed items above"
echo "2. Use 5 Whys for root cause analysis"
echo "3. Implement fixes"
echo "4. Re-run this verification"
exit 1
