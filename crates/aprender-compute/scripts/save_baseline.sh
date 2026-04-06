#!/bin/bash
# Save Performance Baseline
#
# Saves current benchmark results as the active baseline for regression detection.
#
# Usage:
#   ./scripts/save_baseline.sh [benchmark_file]
#
# If no file provided, runs benchmarks automatically.

set -e

BASELINE_DIR=".performance-baselines"
BASELINE_FILE="$BASELINE_DIR/baseline-current.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Trueno Performance Baseline Management"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Ensure directory exists
mkdir -p "$BASELINE_DIR"

# Get commit info
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE=$(date +%Y%m%d-%H%M%S)
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

echo "📊 Current State:"
echo "   Commit:  $COMMIT"
echo "   Branch:  $BRANCH"
echo "   Date:    $DATE"
echo

# Check if baseline already exists
if [ -f "$BASELINE_FILE" ]; then
    echo -e "${YELLOW}⚠️  Existing baseline found${NC}"

    # Archive old baseline
    ARCHIVE_NAME="$BASELINE_DIR/baseline-$(head -1 $BASELINE_FILE | grep -oP 'Commit: \K[a-f0-9]+' || echo 'unknown')-archived.txt"
    echo "   Archiving old baseline to: $(basename $ARCHIVE_NAME)"
    cp "$BASELINE_FILE" "$ARCHIVE_NAME"
    echo
fi

# Run benchmarks or use provided file
if [ -n "$1" ]; then
    BENCH_FILE="$1"

    if [ ! -f "$BENCH_FILE" ]; then
        echo -e "${RED}❌ Error: Benchmark file not found: $BENCH_FILE${NC}"
        exit 1
    fi

    echo "📁 Using existing benchmark file: $BENCH_FILE"
    cp "$BENCH_FILE" "$BASELINE_FILE.tmp"
else
    echo "🔨 Running benchmarks (this may take 5-10 minutes)..."
    echo "   Command: cargo bench --bench vector_ops -- --sample-size 20"
    echo

    # Run benchmarks
    cargo bench --bench vector_ops -- --sample-size 20 2>&1 | tee "$BASELINE_FILE.tmp"
fi

# Add metadata header
{
    echo "# Trueno Performance Baseline"
    echo "# Commit: $COMMIT"
    echo "# Branch: $BRANCH"
    echo "# Date: $DATE"
    echo "# CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs || echo 'unknown')"
    echo "#"
    cat "$BASELINE_FILE.tmp"
} > "$BASELINE_FILE"

rm "$BASELINE_FILE.tmp"

# Count benchmarks
BENCH_COUNT=$(grep -c 'time:.*\[' "$BASELINE_FILE" || echo 0)

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Baseline saved successfully${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "📍 Location: $BASELINE_FILE"
echo "📈 Benchmarks: $BENCH_COUNT"
echo "📦 Size: $(du -h $BASELINE_FILE | cut -f1)"
echo
echo "Next steps:"
echo "  • Commit baseline: git add $BASELINE_FILE && git commit -m 'Update performance baseline'"
echo "  • Check for regressions: cargo bench | python3 scripts/check_regression.py -b $BASELINE_FILE"
echo
