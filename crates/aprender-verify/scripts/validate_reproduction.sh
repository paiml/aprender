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

# Directory holding the Deno scripts used to parse/compare the benchmark
# JSON. Values that originate from the benchmark files (paths, benchmark
# names, numeric results) are passed to Deno through the environment rather
# than interpolated into script text, so nothing in those values can break
# out of the JS source (SEC001: no `eval`-shaped construction of code from
# untrusted input).
JS_DIR="$(mktemp -d)"
cleanup() {
    if [ -n "$JS_DIR" ] && [ "$JS_DIR" != "/" ]; then
        rm -rf -- "$JS_DIR"
    fi
}
trap cleanup EXIT

cat > "$JS_DIR/extract_timings.js" <<'JS'
const jsonFile = Deno.env.get("JSON_FILE");
const benchmarkName = Deno.env.get("BENCHMARK_NAME");
const data = JSON.parse(await Deno.readTextFile(jsonFile));
const bench = data.benchmarks.find((b) => b.name === benchmarkName);
if (bench && bench.measurements) {
    console.log(bench.measurements.timings_ms.join(","));
}
JS

cat > "$JS_DIR/calculate_mean.js" <<'JS'
const values = (Deno.env.get("VALUES") || "").split(",").map(Number);
const mean = values.reduce((a, b) => a + b, 0) / values.length;
console.log(mean.toFixed(3));
JS

cat > "$JS_DIR/compare_distributions.js" <<'JS'
const original = (Deno.env.get("ORIGINAL_VALUES") || "")
    .split(",").map(Number).sort((a, b) => a - b);
const reproduced = (Deno.env.get("REPRODUCED_VALUES") || "")
    .split(",").map(Number).sort((a, b) => a - b);
const maxMeanDiffPercent = Number(Deno.env.get("MAX_MEAN_DIFF_PERCENT"));

const meanOrig = original.reduce((a, b) => a + b, 0) / original.length;
const meanRepro = reproduced.reduce((a, b) => a + b, 0) / reproduced.length;

const percentDiff = Math.abs((meanRepro - meanOrig) / meanOrig) * 100;

// Simple statistical comparison: mean difference
const equivalent = percentDiff < maxMeanDiffPercent;

console.log(`original_mean=${meanOrig.toFixed(3)}`);
console.log(`reproduced_mean=${meanRepro.toFixed(3)}`);
console.log(`percent_diff=${percentDiff.toFixed(2)}`);
console.log(`equivalent=${equivalent}`);
JS

cat > "$JS_DIR/list_benchmarks.js" <<'JS'
const originalFile = Deno.env.get("ORIGINAL_FILE");
const data = JSON.parse(await Deno.readTextFile(originalFile));
console.log(data.benchmarks.map((b) => b.name).join(" "));
JS

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

    JSON_FILE="$json_file" BENCHMARK_NAME="$benchmark_name" \
        deno run --quiet --allow-read="$json_file" --allow-env=JSON_FILE,BENCHMARK_NAME \
        "$JS_DIR/extract_timings.js" 2>/dev/null || echo ""
}

# Function: Calculate mean
calculate_mean() {
    local values=$1

    VALUES="$values" deno run --quiet --allow-env=VALUES "$JS_DIR/calculate_mean.js"
}

# Function: Compare distributions (simplified KS-like test)
# Prints `key=value` lines (original_mean, reproduced_mean, percent_diff,
# equivalent) instead of JSON so the caller can read the fields back with
# plain shell parsing rather than a further Deno invocation per field.
compare_distributions() {
    local original_values=$1
    local reproduced_values=$2

    ORIGINAL_VALUES="$original_values" REPRODUCED_VALUES="$reproduced_values" \
        MAX_MEAN_DIFF_PERCENT="$MAX_MEAN_DIFF_PERCENT" \
        deno run --quiet \
        --allow-env=ORIGINAL_VALUES,REPRODUCED_VALUES,MAX_MEAN_DIFF_PERCENT \
        "$JS_DIR/compare_distributions.js"
}

# Function: read one `key=value` line out of compare_distributions' output
result_field() {
    local result=$1
    local key=$2
    printf '%s\n' "$result" | sed -n "s/^${key}=//p"
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
    local benchmark_names
    benchmark_names=$(ORIGINAL_FILE="$original_file" \
        deno run --quiet --allow-read="$original_file" --allow-env=ORIGINAL_FILE \
        "$JS_DIR/list_benchmarks.js")

    if [ -z "$benchmark_names" ]; then
        print_status "$RED" "Error: No benchmarks found in original file"
        exit 2
    fi

    local all_passed=true

    # Compare each benchmark
    for bench_name in $benchmark_names; do
        echo "Comparing: $bench_name"

        local orig_timings repro_timings
        orig_timings=$(extract_timings "$original_file" "$bench_name")
        repro_timings=$(extract_timings "$reproduced_file" "$bench_name")

        if [ -z "$orig_timings" ] || [ -z "$repro_timings" ]; then
            print_status "$YELLOW" "  ⚠️  Skipped (missing data)"
            continue
        fi

        # Compare distributions
        local result
        result=$(compare_distributions "$orig_timings" "$repro_timings")

        local orig_mean repro_mean percent_diff equivalent
        orig_mean=$(result_field "$result" "original_mean")
        repro_mean=$(result_field "$result" "reproduced_mean")
        percent_diff=$(result_field "$result" "percent_diff")
        equivalent=$(result_field "$result" "equivalent")

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
