#!/usr/bin/env bash
# Check coverage threshold (excludes wasm.rs)
# Usage: ./scripts/check-coverage.sh <threshold>
# Example: ./scripts/check-coverage.sh 95

set -euo pipefail

THRESHOLD="${1:-95}"

# Get coverage report (match .rs files, exclude platform-specific code and header/separator lines)
# Exclusions match Makefile COVERAGE_EXCLUDE: wasm.rs, app.rs, gpu_amd.rs, gpu_apple.rs,
# battery.rs, battery_sensors_simd.rs, kernels.rs
if ! cargo llvm-cov report 2>/dev/null | grep '\.rs ' | grep -Ev 'wasm\.rs|app\.rs|gpu_amd\.rs|gpu_apple\.rs|battery\.rs|battery_sensors_simd\.rs|kernels\.rs' > /tmp/cov_lines.txt; then
    echo "❌ No coverage data found"
    echo "   Run: make coverage"
    exit 1
fi

if [[ ! -s /tmp/cov_lines.txt ]]; then
    echo "❌ No coverage data found for source files"
    exit 1
fi

# Sum total and uncovered lines (columns 8 and 9 in llvm-cov report)
# Format: Filename Regions MissedRegions Cover% Functions MissedFunctions Exec% Lines MissedLines Cover% ...
total=$(awk '{sum += $8} END {print sum}' /tmp/cov_lines.txt)
uncovered=$(awk '{sum += $9} END {print sum}' /tmp/cov_lines.txt)

rm -f /tmp/cov_lines.txt

if [[ "$total" -eq 0 ]]; then
    echo "❌ No lines found"
    exit 1
fi

# Calculate coverage
coverage=$(awk "BEGIN {printf \"%.2f\", 100 * ($total - $uncovered) / $total}")

echo "Coverage: ${coverage}% (excluding platform-specific code)"

# Check threshold
if awk "BEGIN {exit !($coverage < $THRESHOLD)}"; then
    echo "❌ FAIL: Coverage ${coverage}% < ${THRESHOLD}%"
    exit 1
fi

echo "✅ Coverage threshold met (≥${THRESHOLD}%)"
