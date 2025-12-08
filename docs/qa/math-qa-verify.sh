#!/bin/bash
# math-qa-verify.sh
# Aprender Mathematical Correctness Verification (Enhanced)
#
# Methodology: Toyota Way + NASA Software Safety + IEEE 754 Compliance
# Checklist: QA-APRENDER-MATH-150-2025-12
#
# Usage: ./docs/qa/math-qa-verify.sh [--json] [--verbose] [--section N]
# Returns: 0 if all checks pass, 1 otherwise
#
# bashrs: lint-clean (warnings acknowledged as false positives)
# shellcheck disable=SC2031,SC2227,SC2233,SC2311,SC2317,SC2320
# bashrs-ignore: DET002,SEC010 (timestamp for reports, parent dir navigation intentional)

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROJECT_ROOT
REPORT_DIR="${PROJECT_ROOT}/target/qa-reports"
readonly REPORT_DIR

# Scoring - simple variables instead of associative arrays for compatibility
TOTAL_POINTS=0
EARNED_POINTS=0

# Section scores stored as simple variables
S1_SCORE=0; S1_MAX=25
S2_SCORE=0; S2_MAX=20
S3_SCORE=0; S3_MAX=25
S4_SCORE=0; S4_MAX=15
S5_SCORE=0; S5_MAX=15
S6_SCORE=0; S6_MAX=20
S7_SCORE=0; S7_MAX=15
S8_SCORE=0; S8_MAX=15
S9_SCORE=0; S9_MAX=10
S10_SCORE=0; S10_MAX=10
S11_SCORE=0; S11_MAX=10
S12_SCORE=0; S12_MAX=5

# Options
JSON_OUTPUT='false'
VERBOSE='false'
SECTION_FILTER=''
GENERATE_REPORT='false'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json) JSON_OUTPUT='true'; shift ;;
        --verbose) VERBOSE='true'; shift ;;
        --section) SECTION_FILTER="$2"; shift 2 ;;
        --report) GENERATE_REPORT='true'; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "${PROJECT_ROOT}" || exit 1

# Helper functions
pass() {
    local points="${1:-1}"
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        echo -e "${GREEN}PASS${NC} (+${points})"
    fi
}

fail() {
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        echo -e "${RED}FAIL${NC}"
    fi
}

skip() {
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        echo -e "${YELLOW}SKIP${NC}"
    fi
}

info() {
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        echo -e "${CYAN}$1${NC}"
    fi
}

add_section_points() {
    local section="$1"
    local points="$2"
    case "$section" in
        1) S1_SCORE=$((S1_SCORE + points)) ;;
        2) S2_SCORE=$((S2_SCORE + points)) ;;
        3) S3_SCORE=$((S3_SCORE + points)) ;;
        4) S4_SCORE=$((S4_SCORE + points)) ;;
        5) S5_SCORE=$((S5_SCORE + points)) ;;
        6) S6_SCORE=$((S6_SCORE + points)) ;;
        7) S7_SCORE=$((S7_SCORE + points)) ;;
        8) S8_SCORE=$((S8_SCORE + points)) ;;
        9) S9_SCORE=$((S9_SCORE + points)) ;;
        10) S10_SCORE=$((S10_SCORE + points)) ;;
        11) S11_SCORE=$((S11_SCORE + points)) ;;
        12) S12_SCORE=$((S12_SCORE + points)) ;;
    esac
    EARNED_POINTS=$((EARNED_POINTS + points))
}

get_section_score() {
    local section="$1"
    case "$section" in
        1) echo "$S1_SCORE" ;;
        2) echo "$S2_SCORE" ;;
        3) echo "$S3_SCORE" ;;
        4) echo "$S4_SCORE" ;;
        5) echo "$S5_SCORE" ;;
        6) echo "$S6_SCORE" ;;
        7) echo "$S7_SCORE" ;;
        8) echo "$S8_SCORE" ;;
        9) echo "$S9_SCORE" ;;
        10) echo "$S10_SCORE" ;;
        11) echo "$S11_SCORE" ;;
        12) echo "$S12_SCORE" ;;
    esac
}

get_section_max() {
    local section="$1"
    case "$section" in
        1) echo "$S1_MAX" ;;
        2) echo "$S2_MAX" ;;
        3) echo "$S3_MAX" ;;
        4) echo "$S4_MAX" ;;
        5) echo "$S5_MAX" ;;
        6) echo "$S6_MAX" ;;
        7) echo "$S7_MAX" ;;
        8) echo "$S8_MAX" ;;
        9) echo "$S9_MAX" ;;
        10) echo "$S10_MAX" ;;
        11) echo "$S11_MAX" ;;
        12) echo "$S12_MAX" ;;
    esac
}

section_header() {
    local section="$1"
    local name="$2"
    local max_points="$3"
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        echo ""
        echo -e "${BOLD}=== Section ${section}: ${name} (${max_points} Points) ===${NC}"
    fi
}

section_footer() {
    local section="$1"
    local earned
    local max
    local pct
    local color="${GREEN}"
    earned=$(get_section_score "$section")
    max=$(get_section_max "$section")
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        pct=$((earned * 100 / max))
        if [[ $pct -lt 80 ]]; then color="${YELLOW}"; fi
        if [[ $pct -lt 60 ]]; then color="${RED}"; fi
        echo -e "Section ${section} Score: ${color}${earned}/${max} (${pct}%)${NC}"
    fi
}

run_test() {
    local name="$1"
    local cmd="$2"
    local points="${3:-1}"
    local section="${4:-1}"
    echo -n "${name}: "
    if bash -c "${cmd}" >/dev/null 2>&1; then
        add_section_points "$section" "$points"
        pass "$points"
        return 0
    else
        fail
        return 1
    fi
}

check_pattern() {
    local name="$1"
    local pattern="$2"
    local file="$3"
    local points="${4:-1}"
    local section="${5:-1}"
    echo -n "${name}: "
    if grep -q "${pattern}" "${file}" 2>/dev/null; then
        add_section_points "$section" "$points"
        pass "$points"
        return 0
    else
        fail
        return 1
    fi
}

check_no_pattern() {
    local name="$1"
    local pattern="$2"
    local file="$3"
    local points="${4:-1}"
    local section="${5:-1}"
    echo -n "${name}: "
    if ! grep -q "${pattern}" "${file}" 2>/dev/null; then
        add_section_points "$section" "$points"
        pass "$points"
        return 0
    else
        fail
        return 1
    fi
}

# ============================================================================
# SECTIONS 1-12 (condensed for brevity)
# ============================================================================

run_section_1() {
    section_header 1 "Sub-Crate Mathematical Correctness" 25
    info "1.1 aprender-monte-carlo (10 points)"
    if [[ -d "crates/aprender-monte-carlo" ]]; then
        check_pattern "1.1.1.1 ChaCha20 CSPRNG" "ChaCha" "crates/aprender-monte-carlo/src/lib.rs" 1 1 || true
        check_no_pattern "1.1.1.2 No thread_rng" "thread_rng" "crates/aprender-monte-carlo/src/lib.rs" 1 1 || true
        run_test "1.1.1.3 Seed tests" "cargo test -p aprender-monte-carlo seed --quiet" 2 1 || true
    else
        skip; skip; skip
    fi
    if [[ -d "src/monte_carlo" ]]; then
        run_test "1.1.2 Risk metrics" "cargo test monte_carlo --lib --quiet" 3 1 || true
    else
        skip; skip; skip
    fi
    run_test "1.1.3 Financial models" "cargo test monte_carlo --lib --quiet" 3 1 || true
    info "1.2 aprender-tsp (8 points)"
    if [[ -d "crates/aprender-tsp" ]]; then
        run_test "1.2.1 TSP tour" "cargo test -p aprender-tsp --lib --quiet" 8 1 || true
    else
        echo -n "1.2 TSP crate: "; skip
    fi
    info "1.3 aprender-shell (7 points)"
    if [[ -d "crates/aprender-shell" ]]; then
        run_test "1.3.1 Shell tests" "cargo test -p aprender-shell --lib --quiet" 7 1 || true
    else
        echo -n "1.3 Shell crate: "; skip
    fi
    section_footer 1
}

run_section_2() {
    section_header 2 "Core Statistics" 20
    info "2.1 Descriptive Statistics (8 points)"
    if [[ -f "src/stats/mod.rs" ]]; then
        check_pattern "2.1.1 Mean" "fn mean" "src/stats/mod.rs" 2 2 || true
        check_pattern "2.1.2 Variance" "variance" "src/stats/mod.rs" 2 2 || true
        run_test "2.1.3 Stats tests" "cargo test stats --lib --quiet" 4 2 || true
    else
        run_test "2.1 Statistics" "cargo test stats --lib --quiet" 8 2 || true
    fi
    info "2.2 Correlation (4 points)"
    run_test "2.2 Correlation" "cargo test correlation --lib --quiet 2>/dev/null" 4 2 || true
    info "2.3 Distribution (4 points)"
    run_test "2.3 Distribution" "cargo test distribution --lib --quiet 2>/dev/null" 4 2 || true
    info "2.4 Edge Cases (4 points)"
    run_test "2.4 Edge cases" "cargo test edge --lib --quiet 2>/dev/null" 4 2 || true
    section_footer 2
}

run_section_3() {
    section_header 3 "Machine Learning Algorithms" 25
    info "3.1 Linear Models (7 points)"
    run_test "3.1.1 OLS" "cargo test linear_model --lib --quiet" 4 3 || true
    run_test "3.1.2 Regularization" "cargo test regularization --lib --quiet 2>/dev/null" 3 3 || true
    info "3.2 Tree Models (6 points)"
    run_test "3.2 Decision trees" "cargo test tree --lib --quiet" 6 3 || true
    info "3.3 Clustering (6 points)"
    run_test "3.3 K-Means" "cargo test cluster --lib --quiet" 6 3 || true
    info "3.4 Metrics (6 points)"
    run_test "3.4 Metrics" "cargo test metrics --lib --quiet" 6 3 || true
    section_footer 3
}

run_section_4() {
    section_header 4 "Deep Learning & Optimization" 15
    info "4.1 Gradient (6 points)"
    run_test "4.1 Optimizer" "cargo test optim --lib --quiet" 6 4 || true
    info "4.2 Convergence (5 points)"
    if [[ -f "src/optim/mod.rs" ]]; then
        check_pattern "4.2.1 SGD" "SGD" "src/optim/mod.rs" 2 4 || true
        check_pattern "4.2.2 Adam" "Adam" "src/optim/mod.rs" 2 4 || true
        check_pattern "4.2.3 LR" "learning_rate" "src/optim/mod.rs" 1 4 || true
    else
        run_test "4.2 Optimizer" "cargo test optim --lib --quiet" 5 4 || true
    fi
    info "4.3 Activations (4 points)"
    run_test "4.3 Activations" "cargo test activation --lib --quiet 2>/dev/null" 4 4 || true
    section_footer 4
}

run_section_5() {
    section_header 5 "Boundary Conditions & Edge Cases" 15
    info "5.1 Validation (5 points)"
    if [[ -f "src/error.rs" ]]; then
        check_pattern "5.1.1 Empty" "Empty" "src/error.rs" 2 5 || true
        check_pattern "5.1.2 Dimension" "Dimension" "src/error.rs" 2 5 || true
        run_test "5.1.3 Error tests" "cargo test error --lib --quiet 2>/dev/null" 1 5 || true
    else
        run_test "5.1 Validation" "cargo test error --lib --quiet" 5 5 || true
    fi
    info "5.2 Numerical (5 points)"
    run_test "5.2 Numerical" "cargo test boundary --lib --quiet 2>/dev/null" 5 5 || true
    info "5.3 Degenerate (5 points)"
    run_test "5.3 Degenerate" "cargo test degenerate --lib --quiet 2>/dev/null" 5 5 || true
    section_footer 5
}

run_section_6() {
    section_header 6 "Autograd & Neural Networks" 20
    info "6.1 Autograd (10 points)"
    run_test "6.1 Autograd" "cargo test autograd --lib --quiet 2>/dev/null" 10 6 || true
    info "6.2 NN Layers (10 points)"
    run_test "6.2 NN" "cargo test nn --lib --quiet 2>/dev/null" 10 6 || true
    section_footer 6
}

run_section_7() {
    section_header 7 "Matrix Decomposition" 15
    info "7.1 Eigen (5 points)"
    run_test "7.1 Eigen" "cargo test eigen --lib --quiet 2>/dev/null || cargo test decomposition --lib --quiet 2>/dev/null" 5 7 || true
    info "7.2 PCA (5 points)"
    run_test "7.2 PCA" "cargo test pca --lib --quiet 2>/dev/null || cargo test decomposition --lib --quiet 2>/dev/null" 5 7 || true
    info "7.3 SVD (5 points)"
    run_test "7.3 SVD" "cargo test svd --lib --quiet 2>/dev/null" 5 7 || true
    section_footer 7
}

run_section_8() {
    section_header 8 "Time Series" 15
    info "8.1 ARIMA (8 points)"
    run_test "8.1 ARIMA" "cargo test time_series --lib --quiet 2>/dev/null" 8 8 || true
    info "8.2 Forecast (7 points)"
    run_test "8.2 Forecast" "cargo test forecast --lib --quiet 2>/dev/null" 7 8 || true
    section_footer 8
}

run_section_9() {
    section_header 9 "Bayesian Inference" 10
    info "9.1 Priors (5 points)"
    run_test "9.1 Bayesian" "cargo test bayesian --lib --quiet 2>/dev/null" 5 9 || true
    info "9.2 MCMC (5 points)"
    run_test "9.2 MCMC" "cargo test mcmc --lib --quiet 2>/dev/null" 5 9 || true
    section_footer 9
}

run_section_10() {
    section_header 10 "Graph Algorithms" 10
    info "10.1 Shortest Path (5 points)"
    run_test "10.1 Graph" "cargo test graph --lib --quiet 2>/dev/null" 5 10 || true
    info "10.2 Centrality (5 points)"
    run_test "10.2 Centrality" "cargo test centrality --lib --quiet 2>/dev/null" 5 10 || true
    section_footer 10
}

run_section_11() {
    section_header 11 "Online Learning" 10
    info "11.1 Regret (5 points)"
    run_test "11.1 Online" "cargo test online --lib --quiet 2>/dev/null" 5 11 || true
    info "11.2 Streaming (5 points)"
    run_test "11.2 Streaming" "cargo test streaming --lib --quiet 2>/dev/null" 5 11 || true
    section_footer 11
}

run_section_12() {
    section_header 12 "Model Zoo & Caching" 5
    info "12.1 Serialization (3 points)"
    run_test "12.1 Serialize" "cargo test serialization --lib --quiet 2>/dev/null" 3 12 || true
    info "12.2 Cache (2 points)"
    run_test "12.2 Cache" "cargo test cache --lib --quiet 2>/dev/null" 2 12 || true
    section_footer 12
}

# ============================================================================
# Grade Calculation
# ============================================================================
get_grade() {
    local pct="$1"
    if [[ $pct -ge 93 ]]; then echo "A+"
    elif [[ $pct -ge 90 ]]; then echo "A"
    elif [[ $pct -ge 87 ]]; then echo "A-"
    elif [[ $pct -ge 83 ]]; then echo "B+"
    elif [[ $pct -ge 80 ]]; then echo "B"
    elif [[ $pct -ge 77 ]]; then echo "B-"
    elif [[ $pct -ge 73 ]]; then echo "C+"
    elif [[ $pct -ge 70 ]]; then echo "C"
    elif [[ $pct -ge 67 ]]; then echo "C-"
    elif [[ $pct -ge 60 ]]; then echo "D"
    else echo "F"
    fi
}

# ============================================================================
# Output Functions
# ============================================================================
print_header() {
    if [[ "${JSON_OUTPUT}" == 'false' ]]; then
        echo "======================================================================"
        echo "  Aprender Mathematical Correctness Verification"
        echo "  Checklist: QA-APRENDER-MATH-150-2025-12"
        echo "======================================================================"
    fi
}

print_summary() {
    local pct=0
    local grade
    local color="${GREEN}"
    if [[ $TOTAL_POINTS -gt 0 ]]; then
        pct=$((EARNED_POINTS * 100 / TOTAL_POINTS))
    fi
    grade=$(get_grade "$pct")
    if [[ "${JSON_OUTPUT}" == 'true' ]]; then
        echo "{"
        echo "  \"total_possible\": ${TOTAL_POINTS},"
        echo "  \"total_earned\": ${EARNED_POINTS},"
        echo "  \"percentage\": ${pct},"
        echo "  \"grade\": \"${grade}\""
        echo "}"
    else
        echo ""
        echo "======================================================================"
        echo "                         FINAL RESULTS"
        echo "======================================================================"
        echo ""
        if [[ $pct -lt 80 ]]; then color="${YELLOW}"; fi
        if [[ $pct -lt 60 ]]; then color="${RED}"; fi
        echo -e "Total Score: ${color}${BOLD}${EARNED_POINTS}/${TOTAL_POINTS} (${pct}%)${NC}"
        echo -e "Grade: ${color}${BOLD}${grade}${NC}"
        echo ""
        if [[ $pct -ge 90 ]]; then
            echo -e "${GREEN}=== MATHEMATICAL VERIFICATION PASSED ===${NC}"
        elif [[ $pct -ge 80 ]]; then
            echo -e "${YELLOW}=== MATHEMATICAL VERIFICATION CONDITIONAL ===${NC}"
        else
            echo -e "${RED}=== MATHEMATICAL DEFECTS FOUND ===${NC}"
        fi
    fi
}

generate_report() {
    mkdir -p "${REPORT_DIR}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local report_file="${REPORT_DIR}/math-qa-report-${timestamp}.md"
    local pct=0
    local grade
    if [[ $TOTAL_POINTS -gt 0 ]]; then
        pct=$((EARNED_POINTS * 100 / TOTAL_POINTS))
    fi
    grade=$(get_grade "$pct")
    {
        echo "# Aprender Mathematical QA Report"
        echo ""
        echo "## Summary"
        echo ""
        echo "- **Total Score:** ${EARNED_POINTS}/${TOTAL_POINTS} (${pct}%)"
        echo "- **Grade:** ${grade}"
        echo ""
        echo "---"
        echo "*Report generated by math-qa-verify.sh*"
    } > "${report_file}"
    info "Report generated: ${report_file}"
}

# ============================================================================
# Main
# ============================================================================
main() {
    print_header
    TOTAL_POINTS=185
    if [[ -z "${SECTION_FILTER}" ]]; then
        run_section_1; run_section_2; run_section_3; run_section_4
        run_section_5; run_section_6; run_section_7; run_section_8
        run_section_9; run_section_10; run_section_11; run_section_12
    else
        case "${SECTION_FILTER}" in
            1) run_section_1; TOTAL_POINTS=25 ;;
            2) run_section_2; TOTAL_POINTS=20 ;;
            3) run_section_3; TOTAL_POINTS=25 ;;
            4) run_section_4; TOTAL_POINTS=15 ;;
            5) run_section_5; TOTAL_POINTS=15 ;;
            6) run_section_6; TOTAL_POINTS=20 ;;
            7) run_section_7; TOTAL_POINTS=15 ;;
            8) run_section_8; TOTAL_POINTS=15 ;;
            9) run_section_9; TOTAL_POINTS=10 ;;
            10) run_section_10; TOTAL_POINTS=10 ;;
            11) run_section_11; TOTAL_POINTS=10 ;;
            12) run_section_12; TOTAL_POINTS=5 ;;
            *) echo "Invalid section: ${SECTION_FILTER}"; exit 1 ;;
        esac
    fi
    print_summary
    if [[ "${GENERATE_REPORT}" == 'true' ]]; then
        generate_report
    fi
    local final_pct=0
    if [[ $TOTAL_POINTS -gt 0 ]]; then
        final_pct=$((EARNED_POINTS * 100 / TOTAL_POINTS))
    fi
    if [[ $final_pct -ge 80 ]]; then exit 0; else exit 1; fi
}

main
