#!/bin/bash
# Unit tests for scientific-swap-benchmark.sh
# Run with: bashrs test scripts/scientific-swap-benchmark.test.sh
#           OR: ./scripts/scientific-swap-benchmark.test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the script in test mode (skip main execution and trap)
export TEST_MODE=true
# shellcheck source=scientific-swap-benchmark.sh
source "${SCRIPT_DIR}/scientific-swap-benchmark.sh"

#═══════════════════════════════════════════════════════════════════════════════
# Test Assertions
#═══════════════════════════════════════════════════════════════════════════════

_test_count=0
_test_passed=0
_test_failed=0

assert_eq() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-assertion}"
    ((_test_count++)) || true
    if [[ "$expected" == "$actual" ]]; then
        ((_test_passed++)) || true
        return 0
    else
        ((_test_failed++)) || true
        echo "  FAIL: $msg (expected '$expected', got '$actual')" >&2
        return 1
    fi
}

assert_ne() {
    local unexpected="$1"
    local actual="$2"
    local msg="${3:-assertion}"
    ((_test_count++)) || true
    if [[ "$unexpected" != "$actual" ]]; then
        ((_test_passed++)) || true
        return 0
    else
        ((_test_failed++)) || true
        echo "  FAIL: $msg (should not be '$unexpected')" >&2
        return 1
    fi
}

assert_defined() {
    local var_name="$1"
    local msg="${2:-$var_name should be defined}"
    ((_test_count++)) || true
    if declare -p "$var_name" &>/dev/null; then
        ((_test_passed++)) || true
        return 0
    else
        ((_test_failed++)) || true
        echo "  FAIL: $msg" >&2
        return 1
    fi
}

assert_fn_exists() {
    local fn_name="$1"
    local msg="${2:-function $fn_name should exist}"
    ((_test_count++)) || true
    if declare -f "$fn_name" >/dev/null 2>&1; then
        ((_test_passed++)) || true
        return 0
    else
        ((_test_failed++)) || true
        echo "  FAIL: $msg" >&2
        return 1
    fi
}

assert_cmd_exists() {
    local cmd="$1"
    local msg="${2:-command $cmd should exist}"
    ((_test_count++)) || true
    if command -v "$cmd" &>/dev/null; then
        ((_test_passed++)) || true
        return 0
    else
        ((_test_failed++)) || true
        echo "  FAIL: $msg" >&2
        return 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Platform Detection
#═══════════════════════════════════════════════════════════════════════════════

test_platform_detection() {
    echo "test_platform_detection"
    assert_defined OS_TYPE "OS_TYPE variable"
    assert_defined IS_LINUX "IS_LINUX variable"
    assert_defined IS_MACOS "IS_MACOS variable"
    assert_defined GPU_BACKEND "GPU_BACKEND variable"

    # Verify platform flags are boolean strings
    if [[ "$IS_LINUX" != "true" && "$IS_LINUX" != "false" ]]; then
        echo "  FAIL: IS_LINUX should be 'true' or 'false', got '$IS_LINUX'" >&2
        return 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Configuration Defaults
#═══════════════════════════════════════════════════════════════════════════════

test_configuration_defaults() {
    echo "test_configuration_defaults"
    assert_eq "8" "$SWAP_SIZE_GB" "SWAP_SIZE_GB"
    assert_eq "3" "$RUNS_PER_TEST" "RUNS_PER_TEST"
    assert_eq "30" "$TEST_RUNTIME" "TEST_RUNTIME"
    assert_eq "false" "$QUICK_MODE" "QUICK_MODE"
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Project Structure
#═══════════════════════════════════════════════════════════════════════════════

test_project_structure() {
    echo "test_project_structure"
    assert_defined PROJECT_ROOT "PROJECT_ROOT"
    assert_defined SCRIPT_DIR "SCRIPT_DIR (from test)"

    if [[ ! -d "$PROJECT_ROOT" ]]; then
        echo "  FAIL: PROJECT_ROOT '$PROJECT_ROOT' is not a directory" >&2
        return 1
    fi

    if [[ ! -f "$PROJECT_ROOT/Cargo.toml" ]]; then
        echo "  FAIL: Cargo.toml not found in PROJECT_ROOT" >&2
        return 1
    fi
    ((_test_count++)) || true
    ((_test_passed++)) || true
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Logging Functions
#═══════════════════════════════════════════════════════════════════════════════

test_logging_functions() {
    echo "test_logging_functions"
    assert_fn_exists log_info
    assert_fn_exists log_pass
    assert_fn_exists log_fail
    assert_fn_exists log_warn
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Setup Functions
#═══════════════════════════════════════════════════════════════════════════════

test_setup_functions() {
    echo "test_setup_functions"
    assert_fn_exists setup_regular_swap
    assert_fn_exists setup_kernel_zram
    assert_fn_exists setup_trueno_zram
    assert_fn_exists teardown_swap
    assert_fn_exists cleanup_environment
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Benchmark Functions
#═══════════════════════════════════════════════════════════════════════════════

test_benchmark_functions() {
    echo "test_benchmark_functions"
    assert_fn_exists run_fio_benchmark
    assert_fn_exists run_full_benchmark
    assert_fn_exists run_compression_benchmark
    assert_fn_exists extract_metrics
    assert_fn_exists generate_report
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: Required Commands
#═══════════════════════════════════════════════════════════════════════════════

test_required_commands() {
    echo "test_required_commands"
    assert_cmd_exists bc "bc (calculator)"
    assert_cmd_exists jq "jq (JSON processor)"
    # fio is optional for unit tests
}

#═══════════════════════════════════════════════════════════════════════════════
# Test: GPU Backend Values
#═══════════════════════════════════════════════════════════════════════════════

test_gpu_backend_values() {
    echo "test_gpu_backend_values"
    ((_test_count++)) || true
    case "$GPU_BACKEND" in
        cuda|wgpu|none)
            ((_test_passed++)) || true
            ;;
        *)
            ((_test_failed++)) || true
            echo "  FAIL: GPU_BACKEND should be cuda/wgpu/none, got '$GPU_BACKEND'" >&2
            return 1
            ;;
    esac
}

#═══════════════════════════════════════════════════════════════════════════════
# Test Runner
#═══════════════════════════════════════════════════════════════════════════════

run_tests() {
    local tests=(
        test_platform_detection
        test_configuration_defaults
        test_project_structure
        test_logging_functions
        test_setup_functions
        test_benchmark_functions
        test_required_commands
        test_gpu_backend_values
    )

    echo "═══════════════════════════════════════════════════════════════"
    echo "  scientific-swap-benchmark.sh - Unit Tests"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    local test_failures=0
    for test_fn in "${tests[@]}"; do
        if $test_fn; then
            echo "  [PASS] $test_fn"
        else
            echo "  [FAIL] $test_fn"
            ((test_failures++)) || true
        fi
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Assertions: $_test_passed/$_test_count passed"
    echo "  Tests: $((${#tests[@]} - test_failures))/${#tests[@]} passed"
    echo "═══════════════════════════════════════════════════════════════"

    return $test_failures
}

# Run tests when executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_tests
fi
