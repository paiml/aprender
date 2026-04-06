#!/bin/bash
# verify_wasm_spec.sh
# Automates key checks from PROBAR-SPEC-WASM-001
#
# Per Appendix A of docs/specifications/wasm-threaded-testing-mock-runtime.md

set -e
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_NC='\033[0m'

echo "Verifying PROBAR-SPEC-WASM-001 Compliance..."
echo ""

check_file() {
    if [ -f "$1" ]; then
        echo -e "${COLOR_GREEN}[PASS] File exists: $1${COLOR_NC}"
        return 0
    else
        echo -e "${COLOR_RED}[FAIL] File missing: $1${COLOR_NC}"
        return 1
    fi
}

check_content() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo -e "${COLOR_GREEN}[PASS] Found '$2' in $1${COLOR_NC}"
        return 0
    else
        echo -e "${COLOR_RED}[FAIL] Missing '$2' in $1${COLOR_NC}"
        return 1
    fi
}

EXIT_CODE=0

# Section A: Static Analysis (1-20)
echo "--- Section A: Static Analysis ---"
check_file "crates/probar/src/lint/mod.rs" || EXIT_CODE=1
check_file "crates/probar/src/lint/state_sync.rs" || EXIT_CODE=1
if [ -f "crates/probar/src/lint/state_sync.rs" ]; then
    check_content "crates/probar/src/lint/state_sync.rs" "StateSyncLinter" || EXIT_CODE=1
    check_content "crates/probar/src/lint/state_sync.rs" "WASM-SS-001" || EXIT_CODE=1
    check_content "crates/probar/src/lint/state_sync.rs" "WASM-SS-002" || EXIT_CODE=1
    check_content "crates/probar/src/lint/state_sync.rs" "WASM-SS-005" || EXIT_CODE=1
    check_content "crates/probar/src/lint/state_sync.rs" "LintSeverity" || EXIT_CODE=1
    check_content "crates/probar/src/lint/state_sync.rs" "LintError" || EXIT_CODE=1
fi
echo ""

# Section B: Mock Runtime (21-40)
echo "--- Section B: Mock Runtime ---"
check_file "crates/probar/src/mock/mod.rs" || EXIT_CODE=1
check_file "crates/probar/src/mock/wasm_runtime.rs" || EXIT_CODE=1
if [ -f "crates/probar/src/mock/wasm_runtime.rs" ]; then
    check_content "crates/probar/src/mock/wasm_runtime.rs" "MockWasmRuntime" || EXIT_CODE=1
    check_content "crates/probar/src/mock/wasm_runtime.rs" "MockMessage" || EXIT_CODE=1
    check_content "crates/probar/src/mock/wasm_runtime.rs" "MockableWorker" || EXIT_CODE=1
    check_content "crates/probar/src/mock/wasm_runtime.rs" "receive_message" || EXIT_CODE=1
    check_content "crates/probar/src/mock/wasm_runtime.rs" "post_message" || EXIT_CODE=1
    check_content "crates/probar/src/mock/wasm_runtime.rs" "on_message" || EXIT_CODE=1
fi
echo ""

# Section B: Test Harness (41-60)
echo "--- Section B.2: Test Harness ---"
check_file "crates/probar/src/mock/test_harness.rs" || EXIT_CODE=1
if [ -f "crates/probar/src/mock/test_harness.rs" ]; then
    check_content "crates/probar/src/mock/test_harness.rs" "WasmCallbackTestHarness" || EXIT_CODE=1
    check_content "crates/probar/src/mock/test_harness.rs" "TestStep" || EXIT_CODE=1
    check_content "crates/probar/src/mock/test_harness.rs" "StateAssertion" || EXIT_CODE=1
    check_content "crates/probar/src/mock/test_harness.rs" "assert_state_synced" || EXIT_CODE=1
    check_content "crates/probar/src/mock/test_harness.rs" "WAPR-QA-REGRESSION-005" || EXIT_CODE=1
fi
echo ""

# Section C: Property Testing (61-80)
echo "--- Section C: Property Testing ---"
check_file "crates/probar/src/mock/strategies.rs" || EXIT_CODE=1
if [ -f "crates/probar/src/mock/strategies.rs" ]; then
    check_content "crates/probar/src/mock/strategies.rs" "any_mock_message" || EXIT_CODE=1
    check_content "crates/probar/src/mock/strategies.rs" "valid_message_sequence" || EXIT_CODE=1
    check_content "crates/probar/src/mock/strategies.rs" "realistic_lifecycle" || EXIT_CODE=1
    check_content "crates/probar/src/mock/strategies.rs" "proptest" || EXIT_CODE=1
    check_content "crates/probar/src/mock/strategies.rs" "Iron Lotus" || EXIT_CODE=1
fi
echo ""

# Section D: Compliance (81-100)
echo "--- Section D: Compliance ---"
check_file "crates/probar/src/comply/mod.rs" || EXIT_CODE=1
check_file "crates/probar/src/comply/wasm_threading.rs" || EXIT_CODE=1
if [ -f "crates/probar/src/comply/wasm_threading.rs" ]; then
    check_content "crates/probar/src/comply/wasm_threading.rs" "WasmThreadingCompliance" || EXIT_CODE=1
    check_content "crates/probar/src/comply/wasm_threading.rs" "ComplianceResult" || EXIT_CODE=1
    check_content "crates/probar/src/comply/wasm_threading.rs" "ComplianceStatus" || EXIT_CODE=1
    check_content "crates/probar/src/comply/wasm_threading.rs" "WASM-COMPLY-001" || EXIT_CODE=1
    check_content "crates/probar/src/comply/wasm_threading.rs" "WASM-COMPLY-002" || EXIT_CODE=1
    check_content "crates/probar/src/comply/wasm_threading.rs" "WASM-COMPLY-003" || EXIT_CODE=1
    check_content "crates/probar/src/comply/wasm_threading.rs" "WASM-COMPLY-004" || EXIT_CODE=1
fi
echo ""

# Section E: Tests Pass
echo "--- Section E: Test Verification ---"
echo "Running lint module tests..."
if cargo test lint::state_sync --lib -q 2>/dev/null; then
    echo -e "${COLOR_GREEN}[PASS] Lint tests pass${COLOR_NC}"
else
    echo -e "${COLOR_RED}[FAIL] Lint tests fail${COLOR_NC}"
    EXIT_CODE=1
fi

echo "Running mock module tests..."
if cargo test mock:: --lib -q 2>/dev/null; then
    echo -e "${COLOR_GREEN}[PASS] Mock tests pass${COLOR_NC}"
else
    echo -e "${COLOR_RED}[FAIL] Mock tests fail${COLOR_NC}"
    EXIT_CODE=1
fi

echo "Running comply module tests..."
if cargo test comply:: --lib -q 2>/dev/null; then
    echo -e "${COLOR_GREEN}[PASS] Comply tests pass${COLOR_NC}"
else
    echo -e "${COLOR_RED}[FAIL] Comply tests fail${COLOR_NC}"
    EXIT_CODE=1
fi
echo ""

# Summary
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${COLOR_GREEN}PROBAR-SPEC-WASM-001 COMPLIANCE: PASS${COLOR_NC}"
else
    echo -e "${COLOR_RED}PROBAR-SPEC-WASM-001 COMPLIANCE: FAIL${COLOR_NC}"
fi
echo "============================================="

exit $EXIT_CODE
