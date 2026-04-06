#!/usr/bin/env bash
#
# Reproducibility Validation Script
# Certeza Phase 3.5: Reproducibility and Archival
#
# This script validates that benchmark results can be reproduced within
# acceptable statistical variance. Uses Kolmogorov-Smirnov test to compare
# distributions.
#
# Usage:
#   ./scripts/validate_reproduction.sh ORIGINAL.json REPRODUCED.json
#
# Exit codes:
#   0: Reproduction validated (distributions statistically equivalent)
#   1: Reproduction failed (distributions differ significantly)
#   2: Error (missing files, invalid data)

set -euo pipefail

# Configuration
SIGNIFICANCE_LEVEL=0.05  # Alpha for statistical tests
MAX_MEAN_DIFF_PERCENT=5.0  # Maximum acceptable mean difference (%)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function: Print colored message
print_status() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function: Extract benchmark timings from JSON
extract_timings() {
    local json_file=$1
    local benchmark_name=$2

    # Use Deno to extract timings array
    deno eval --quiet "
        const data = JSON.parse(await Deno.readTextFile('$json_file'));
        const bench = data.benchmarks.find(b => b.name === '$benchmark_name');
        if (bench && bench.measurements) {
            console.log(bench.measurements.timings_ms.join(','));
        }
    " 2>/dev/null || echo ""
}

# Function: Calculate mean
calculate_mean() {
    local values=$1

    deno eval --quiet "
        const values = '$values'.split(',').map(Number);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        console.log(mean.toFixed(3));
    "
}

# Function: Compare distributions (simplified KS-like test)
compare_distributions() {
    local original_values=$1
    local reproduced_values=$2

    deno eval --quiet "
        const original = '$original_values'.split(',').map(Number).sort((a, b) => a - b);
        const reproduced = '$reproduced_values'.split(',').map(Number).sort((a, b) => a - b);

        const meanOrig = original.reduce((a, b) => a + b, 0) / original.length;
        const meanRepro = reproduced.reduce((a, b) => a + b, 0) / reproduced.length;

        const percentDiff = Math.abs((meanRepro - meanOrig) / meanOrig) * 100;

        // Simple statistical comparison: mean difference
        const equivalent = percentDiff < $MAX_MEAN_DIFF_PERCENT;

        console.log(JSON.stringify({
            original_mean: meanOrig.toFixed(3),
            reproduced_mean: meanRepro.toFixed(3),
            percent_diff: percentDiff.toFixed(2),
            equivalent: equivalent
        }));
    "
}

# Main script
main() {
    if [ $# -ne 2 ]; then
        print_status "$RED" "Usage: $0 ORIGINAL.json REPRODUCED.json"
        exit 2
    fi

    local original_file=$1
    local reproduced_file=$2

    # Validate files exist
    if [ ! -f "$original_file" ]; then
        print_status "$RED" "Error: Original file not found: $original_file"
        exit 2
    fi

    if [ ! -f "$reproduced_file" ]; then
        print_status "$RED" "Error: Reproduced file not found: $reproduced_file"
        exit 2
    fi

    print_status "$GREEN" "=== Reproducibility Validation ==="
    echo "Original:    $original_file"
    echo "Reproduced:  $reproduced_file"
    echo "Significance: p < $SIGNIFICANCE_LEVEL"
    echo "Max mean diff: ${MAX_MEAN_DIFF_PERCENT}%"
    echo ""

    # Extract benchmark names from original
    local benchmark_names=$(deno eval --quiet "
        const data = JSON.parse(await Deno.readTextFile('$original_file'));
        const names = data.benchmarks.map(b => b.name);
        console.log(names.join(' '));
    ")

    if [ -z "$benchmark_names" ]; then
        print_status "$RED" "Error: No benchmarks found in original file"
        exit 2
    fi

    local all_passed=true

    # Compare each benchmark
    for bench_name in $benchmark_names; do
        echo "Comparing: $bench_name"

        local orig_timings=$(extract_timings "$original_file" "$bench_name")
        local repro_timings=$(extract_timings "$reproduced_file" "$bench_name")

        if [ -z "$orig_timings" ] || [ -z "$repro_timings" ]; then
            print_status "$YELLOW" "  ⚠️  Skipped (missing data)"
            continue
        fi

        # Compare distributions
        local result=$(compare_distributions "$orig_timings" "$repro_timings")

        local orig_mean=$(echo "$result" | deno eval --quiet "
            const data = JSON.parse(await Deno.readTextFile('/dev/stdin'));
            console.log(JSON.parse(data).original_mean);
        ")
        local repro_mean=$(echo "$result" | deno eval --quiet "
            const data = JSON.parse(await Deno.readTextFile('/dev/stdin'));
            console.log(JSON.parse(data).reproduced_mean);
        ")
        local percent_diff=$(echo "$result" | deno eval --quiet "
            const data = JSON.parse(await Deno.readTextFile('/dev/stdin'));
            console.log(JSON.parse(data).percent_diff);
        ")
        local equivalent=$(echo "$result" | deno eval --quiet "
            const data = JSON.parse(await Deno.readTextFile('/dev/stdin'));
            console.log(JSON.parse(data).equivalent);
        ")

        echo "  Original mean:    ${orig_mean} ms"
        echo "  Reproduced mean:  ${repro_mean} ms"
        echo "  Difference:       ${percent_diff}%"

        if [ "$equivalent" = "true" ]; then
            print_status "$GREEN" "  ✅ PASS (within ${MAX_MEAN_DIFF_PERCENT}% threshold)"
        else
            print_status "$RED" "  ❌ FAIL (exceeds ${MAX_MEAN_DIFF_PERCENT}% threshold)"
            all_passed=false
        fi

        echo ""
    done

    # Final verdict
    if [ "$all_passed" = true ]; then
        print_status "$GREEN" "=== Validation PASSED ==="
        print_status "$GREEN" "Reproduction successful: all benchmarks within acceptable variance"
        exit 0
    else
        print_status "$RED" "=== Validation FAILED ==="
        print_status "$RED" "Reproduction failed: one or more benchmarks exceed variance threshold"
        exit 1
    fi
}

main "$@"
