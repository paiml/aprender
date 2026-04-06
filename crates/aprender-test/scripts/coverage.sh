#!/bin/bash
# Simple GUI coverage reporter
# Usage: ./scripts/coverage.sh [threshold]
#
# Examples:
#   ./scripts/coverage.sh        # Show coverage
#   ./scripts/coverage.sh 95     # Fail if below 95%

set -e

THRESHOLD=${1:-0}
PACKAGE=${2:-showcase-calculator}

echo "📊 Coverage Report"
echo "=================="

# Run coverage and extract the percentage
RESULT=$(cargo llvm-cov --package "$PACKAGE" --features tui --summary-only 2>/dev/null | grep TOTAL)
COVERAGE=$(echo "$RESULT" | awk '{print $10}' | tr -d '%')

echo ""
echo "  Package:  $PACKAGE"
echo "  Coverage: ${COVERAGE}%"
echo ""

# Check threshold
if [ "$THRESHOLD" -gt 0 ]; then
    if (( $(echo "$COVERAGE >= $THRESHOLD" | bc -l) )); then
        echo "✅ PASS (threshold: ${THRESHOLD}%)"
        exit 0
    else
        echo "❌ FAIL (threshold: ${THRESHOLD}%, got: ${COVERAGE}%)"
        exit 1
    fi
fi
