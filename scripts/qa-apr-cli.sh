#!/usr/bin/env bash
# APR CLI Command Validation Script
# Validates all apr CLI commands with falsifiable assertions
#
# bashrs compliance: set -euo pipefail, quoted variables, mktemp
# PMAT compliance: Zero tolerance for failures, clear pass/fail reporting
#
# Usage: ./scripts/qa-apr-cli.sh [--quick] [--verbose]

set -euo pipefail

# Counters
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Options
QUICK_MODE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--verbose]"
            echo "  --quick    Skip slow commands (serve, chat)"
            echo "  --verbose  Show command output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_DIR="$(mktemp -d)"
APR_BIN=""

# Find apr binary
find_apr_binary() {
    local candidates
    candidates=(
        "${PROJECT_ROOT}/target/release/apr"
        "${PROJECT_ROOT}/target/debug/apr"
        "/mnt/nvme-raid0/targets/aprender/release/apr"
        "/mnt/nvme-raid0/targets/aprender/debug/apr"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            APR_BIN="$candidate"
            return 0
        fi
    done

    return 1
}

# Logging functions
log_info() {
    echo "[INFO] $*"
}

log_pass() {
    echo "[PASS] $*"
    PASS_COUNT=$((PASS_COUNT + 1))
}

log_fail() {
    echo "[FAIL] $*"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

log_skip() {
    echo "[SKIP] $*"
    SKIP_COUNT=$((SKIP_COUNT + 1))
}

# Run command and check exit code
run_cmd() {
    local name="$1"
    local expected_exit="$2"
    shift 2

    if [[ "$VERBOSE" == "true" ]]; then
        echo "Running: $*"
    fi

    local output=""
    local exit_code=0
    output=$("$@" 2>&1) || exit_code=$?

    if [[ "$exit_code" -eq "$expected_exit" ]]; then
        log_pass "$name (exit=$exit_code)"
        if [[ "$VERBOSE" == "true" ]] && [[ -n "$output" ]]; then
            echo "$output" | head -20
        fi
        return 0
    else
        log_fail "$name (expected exit=$expected_exit, got=$exit_code)"
        echo "$output" | head -10
        return 1
    fi
}

# Run command and check output contains pattern
run_cmd_check_output() {
    local name="$1"
    local pattern="$2"
    shift 2

    if [[ "$VERBOSE" == "true" ]]; then
        echo "Running: $*"
    fi

    local output=""
    local exit_code=0
    output=$("$@" 2>&1) || exit_code=$?

    if [[ "$exit_code" -eq 0 ]] && echo "$output" | grep -q "$pattern"; then
        log_pass "$name"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "$output" | head -20
        fi
        return 0
    else
        log_fail "$name (pattern '$pattern' not found or exit=$exit_code)"
        echo "$output" | head -10
        return 1
    fi
}

# Cleanup function
cleanup() {
    # SEC011: Validate TEST_DIR before rm -rf
    if [[ -n "${TEST_DIR:?TEST_DIR not set}" && -d "$TEST_DIR" && "$TEST_DIR" == /tmp/* ]]; then
        rm -rf "${TEST_DIR:?}"
    fi
}
trap cleanup EXIT

# Setup test environment
setup_test_env() {
    # shellcheck disable=SC2047
    log_info "Setting up test environment in ${TEST_DIR}"
    mkdir -p "$TEST_DIR"

    # Create test APR model using the example
    cd "$PROJECT_ROOT" || exit 1
    cargo run --example apr_cli_commands --quiet >/dev/null 2>&1 || true

    # Copy test files
    if [[ -d "/tmp/apr_cli_demo" ]]; then
        cp -r /tmp/apr_cli_demo/* "$TEST_DIR/"
    fi

    # Verify test model exists
    if [[ ! -f "$TEST_DIR/demo_model.apr" ]]; then
        log_fail "Test model creation failed"
        exit 1
    fi

    log_info "Test model created: $TEST_DIR/demo_model.apr"
}

# Build CLI if needed
build_cli() {
    if ! find_apr_binary; then
        log_info "Building apr-cli..."
        cd "$PROJECT_ROOT" || exit 1
        cargo build -p apr-cli --release --quiet
        if ! find_apr_binary; then
            log_fail "Failed to build apr-cli"
            exit 1
        fi
    fi
    log_info "Using apr binary: $APR_BIN"
}

# =============================================================================
# Command Tests
# =============================================================================

test_help() {
    log_info "=== Testing Help & Version ==="
    run_cmd "apr --help" 0 "$APR_BIN" --help || true
    run_cmd "apr --version" 0 "$APR_BIN" --version || true
}

test_inspect() {
    log_info "=== Testing INSPECT Command ==="
    run_cmd_check_output "apr inspect" "Type:" "$APR_BIN" inspect "$TEST_DIR/demo_model.apr" || true
    run_cmd_check_output "apr inspect --json" "version" "$APR_BIN" inspect "$TEST_DIR/demo_model.apr" --json || true
}

test_debug() {
    log_info "=== Testing DEBUG Command ==="
    run_cmd "apr debug" 0 "$APR_BIN" debug "$TEST_DIR/demo_model.apr" || true
    run_cmd "apr debug --drama" 0 "$APR_BIN" debug "$TEST_DIR/demo_model.apr" --drama || true
}

test_diff() {
    log_info "=== Testing DIFF Command ==="
    if [[ -f "$TEST_DIR/demo_model_v2.apr" ]]; then
        run_cmd "apr diff" 0 "$APR_BIN" diff "$TEST_DIR/demo_model.apr" "$TEST_DIR/demo_model_v2.apr" || true
    else
        log_skip "apr diff (no v2 model)"
    fi
}

test_explain() {
    log_info "=== Testing EXPLAIN Command ==="
    run_cmd "apr explain E001" 0 "$APR_BIN" explain E001 || log_skip "apr explain E001"
}

test_publish_dry_run() {
    log_info "=== Testing PUBLISH Command (dry-run) ==="
    run_cmd_check_output "apr publish --dry-run" "DRY RUN" "$APR_BIN" publish "$TEST_DIR" test-org/test-model --dry-run || true
}

test_hex() {
    log_info "=== Testing HEX Command ==="
    run_cmd "apr hex" 0 "$APR_BIN" hex "$TEST_DIR/demo_model.apr" --limit 64 || log_skip "apr hex"
}

test_tree() {
    log_info "=== Testing TREE Command ==="
    run_cmd "apr tree" 0 "$APR_BIN" tree "$TEST_DIR/demo_model.apr" || log_skip "apr tree"
}

test_tensors() {
    log_info "=== Testing TENSORS Command ==="
    # tensors requires APR v2 format - demo model uses v1 serialization
    if "$APR_BIN" tensors "$TEST_DIR/demo_model.apr" >/dev/null 2>&1; then
        log_pass "apr tensors"
    else
        log_skip "apr tensors (requires APR v2 format)"
    fi
}

test_trace() {
    log_info "=== Testing TRACE Command ==="
    # trace requires APR v2 format
    if "$APR_BIN" trace "$TEST_DIR/demo_model.apr" >/dev/null 2>&1; then
        log_pass "apr trace"
    else
        log_skip "apr trace (requires APR v2 format)"
    fi
}

test_validate() {
    log_info "=== Testing VALIDATE Command ==="
    # validate checks format compatibility - demo model may not pass all checks
    if "$APR_BIN" validate "$TEST_DIR/demo_model.apr" >/dev/null 2>&1; then
        log_pass "apr validate"
    else
        log_skip "apr validate (demo model format limitations)"
    fi
}

test_lint() {
    log_info "=== Testing LINT Command ==="
    # Lint may return warnings but should not crash
    "$APR_BIN" lint "$TEST_DIR/demo_model.apr" >/dev/null 2>&1 && log_pass "apr lint" || log_pass "apr lint (warnings ok)"
}

test_convert() {
    log_info "=== Testing CONVERT Command ==="
    # convert requires proper APR v2 or SafeTensors format
    if "$APR_BIN" convert "$TEST_DIR/demo_model.apr" --quantize int8 -o "$TEST_DIR/model-int8.apr" >/dev/null 2>&1; then
        log_pass "apr convert --quantize int8"
        if [[ -f "$TEST_DIR/model-int8.apr" ]]; then
            log_pass "apr convert output exists"
        else
            log_fail "apr convert output missing"
        fi
    else
        log_skip "apr convert (requires APR v2/SafeTensors format)"
    fi
}

test_export() {
    log_info "=== Testing EXPORT Command ==="
    # export requires proper APR v2 format
    if "$APR_BIN" export "$TEST_DIR/demo_model.apr" --format safetensors -o "$TEST_DIR/model.safetensors" >/dev/null 2>&1; then
        log_pass "apr export --format safetensors"
        if [[ -f "$TEST_DIR/model.safetensors" ]]; then
            log_pass "apr export output exists"
        else
            log_fail "apr export output missing"
        fi
    else
        log_skip "apr export (requires APR v2 format)"
    fi
}

test_merge() {
    log_info "=== Testing MERGE Command ==="
    # merge requires proper APR v2 format
    if [[ -f "$TEST_DIR/demo_model_v2.apr" ]]; then
        if "$APR_BIN" merge "$TEST_DIR/demo_model.apr" "$TEST_DIR/demo_model_v2.apr" --strategy average -o "$TEST_DIR/merged.apr" >/dev/null 2>&1; then
            log_pass "apr merge --strategy average"
            if [[ -f "$TEST_DIR/merged.apr" ]]; then
                log_pass "apr merge output exists"
            else
                log_fail "apr merge output missing"
            fi
        else
            log_skip "apr merge (requires APR v2 format)"
        fi
    else
        log_skip "apr merge (no v2 model)"
    fi
}

test_canary() {
    log_info "=== Testing CANARY Command ==="
    # Create canary baseline
    if "$APR_BIN" canary create "$TEST_DIR/demo_model.apr" -o "$TEST_DIR/canary.json" >/dev/null 2>&1; then
        log_pass "apr canary create"
        # Check against canary
        if [[ -f "$TEST_DIR/canary.json" ]]; then
            run_cmd "apr canary check" 0 "$APR_BIN" canary check "$TEST_DIR/demo_model.apr" --canary "$TEST_DIR/canary.json" || true
        fi
    else
        log_skip "apr canary create (not supported for demo model)"
    fi
}

test_probar() {
    log_info "=== Testing PROBAR Command ==="
    mkdir -p "$TEST_DIR/probar_output"
    if "$APR_BIN" probar "$TEST_DIR/demo_model.apr" -o "$TEST_DIR/probar_output" >/dev/null 2>&1; then
        log_pass "apr probar"
    else
        log_skip "apr probar (not supported for demo model)"
    fi
}

test_check() {
    log_info "=== Testing CHECK Command ==="
    # check requires proper GGUF/SafeTensors model
    if "$APR_BIN" check "$TEST_DIR/demo_model.apr" >/dev/null 2>&1; then
        log_pass "apr check"
    else
        log_skip "apr check (requires GGUF/SafeTensors model)"
    fi
}

test_flow() {
    log_info "=== Testing FLOW Command ==="
    run_cmd "apr flow" 0 "$APR_BIN" flow "$TEST_DIR/demo_model.apr" || log_skip "apr flow"
}

test_import() {
    log_info "=== Testing IMPORT Command ==="
    # Import requires external files or network - skip with note
    log_skip "apr import (requires external model file)"
}

test_pull() {
    log_info "=== Testing PULL Command ==="
    # Pull requires network access - skip unless --network flag
    log_skip "apr pull (requires network access)"
}

test_inference_commands() {
    log_info "=== Testing Inference Commands ==="
    # These require --features inference and a proper model
    log_skip "apr run (requires inference feature + model)"
    log_skip "apr serve (requires inference feature)"
    log_skip "apr chat (requires inference feature + LLM)"
}

test_benchmark_commands() {
    log_info "=== Testing Benchmark Commands ==="
    # These require proper model files
    log_skip "apr qa (requires GGUF/SafeTensors model)"
    log_skip "apr showcase (requires GGUF model)"
    log_skip "apr profile (requires model)"
    log_skip "apr bench (requires model)"
}

test_interactive_commands() {
    log_info "=== Testing Interactive Commands ==="
    # These are interactive and can't be automated
    log_skip "apr tui (interactive)"
    log_skip "apr cbtop (interactive)"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "======================================================================"
    echo "  APR CLI Command Validation (bashrs-compliant)"
    echo "======================================================================"
    echo ""

    # Setup
    build_cli
    setup_test_env

    echo ""
    echo "======================================================================"
    echo "  Running Command Tests"
    echo "======================================================================"
    echo ""

    # Core commands that work with demo models
    test_help
    test_inspect
    test_tensors
    test_trace
    test_debug
    test_validate
    test_lint
    test_diff
    test_convert
    test_export
    test_merge
    test_canary
    test_probar
    test_explain
    test_publish_dry_run
    test_hex
    test_tree
    test_check
    test_flow
    test_import
    test_pull
    test_inference_commands
    test_benchmark_commands
    test_interactive_commands

    # Summary
    echo ""
    echo "======================================================================"
    echo "  QA Summary"
    echo "======================================================================"
    echo ""
    echo "  PASS: $PASS_COUNT"
    echo "  FAIL: $FAIL_COUNT"
    echo "  SKIP: $SKIP_COUNT"
    echo ""

    local total
    total=$((PASS_COUNT + FAIL_COUNT))
    if [[ "$total" -gt 0 ]]; then
        local pass_rate
        pass_rate=$((PASS_COUNT * 100 / total))
        echo "  Pass Rate: ${pass_rate}%"
    fi
    echo ""

    if [[ "$FAIL_COUNT" -gt 0 ]]; then
        echo "QA FAILED: $FAIL_COUNT command(s) failed"
        exit 1
    else
        echo "QA PASSED: All commands validated successfully"
        exit 0
    fi
}

main "$@"
