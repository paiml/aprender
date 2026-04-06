#!/usr/bin/env bash
# Pixel diff testing for cbtop/system_dashboard
# Uses probar's visual regression testing via direct comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BASELINE_DIR="$PROJECT_DIR/__pixel_baselines__"
DIFF_DIR="$PROJECT_DIR/__pixel_diffs__"
SCREENSHOTS_DIR="$HOME/Pictures/Screenshots"

usage() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  capture <name>     - Capture latest screenshot as baseline"
    echo "  compare <name>     - Compare latest screenshot against baseline"
    echo "  update <name>      - Update baseline with latest screenshot"
    echo "  list               - List available baselines"
    echo ""
    echo "Example:"
    echo "  $0 capture system_dashboard   # Save baseline"
    echo "  $0 compare system_dashboard   # Compare against baseline"
}

get_latest_screenshot() {
    ls -t "$SCREENSHOTS_DIR"/Screenshot*.png 2>/dev/null | head -1
}

capture() {
    local name="$1"
    if [ -z "$name" ]; then
        echo "Error: name required"
        exit 1
    fi

    mkdir -p "$BASELINE_DIR"

    local latest=$(get_latest_screenshot)
    if [ -z "$latest" ]; then
        echo "Error: No screenshots found in $SCREENSHOTS_DIR"
        exit 1
    fi

    cp "$latest" "$BASELINE_DIR/${name}.png"
    echo "Baseline captured: $BASELINE_DIR/${name}.png"
    echo "Source: $latest"
}

compare() {
    local name="$1"
    if [ -z "$name" ]; then
        echo "Error: name required"
        exit 1
    fi

    local baseline="$BASELINE_DIR/${name}.png"
    if [ ! -f "$baseline" ]; then
        echo "Error: Baseline not found: $baseline"
        echo "Run: $0 capture $name"
        exit 1
    fi

    local latest=$(get_latest_screenshot)
    if [ -z "$latest" ]; then
        echo "Error: No screenshots found in $SCREENSHOTS_DIR"
        exit 1
    fi

    mkdir -p "$DIFF_DIR"

    # Use ImageMagick compare for pixel diff
    local diff_output="$DIFF_DIR/${name}_diff.png"
    local metric_output="$DIFF_DIR/${name}_metrics.txt"

    echo "Comparing:"
    echo "  Baseline: $baseline"
    echo "  Current:  $latest"
    echo ""

    # Compare using ImageMagick (AE = Absolute Error count)
    local diff_pixels=$(compare -metric AE "$baseline" "$latest" "$diff_output" 2>&1 || true)

    echo "Differing pixels: $diff_pixels"
    echo "Diff image: $diff_output"

    # Also generate metrics
    compare -metric RMSE "$baseline" "$latest" null: 2>&1 > "$metric_output" || true
    echo "RMSE: $(cat "$metric_output" 2>/dev/null || echo 'N/A')"

    if [ "$diff_pixels" = "0" ]; then
        echo ""
        echo "PASS: Images are identical"
        return 0
    else
        echo ""
        echo "FAIL: Images differ by $diff_pixels pixels"
        echo "View diff: feh $diff_output"
        return 1
    fi
}

update() {
    local name="$1"
    capture "$name"
    echo "Baseline updated"
}

list_baselines() {
    echo "Baselines in $BASELINE_DIR:"
    if [ -d "$BASELINE_DIR" ]; then
        ls -la "$BASELINE_DIR"/*.png 2>/dev/null || echo "  (none)"
    else
        echo "  (directory not created yet)"
    fi
}

case "${1:-}" in
    capture) capture "$2" ;;
    compare) compare "$2" ;;
    update)  update "$2" ;;
    list)    list_baselines ;;
    *)       usage ;;
esac
