#!/bin/bash
# Benchmark runner that updates README.md with results
# Usage: ./scripts/bench-update-readme.sh [ollama|vllm|llama-cpp] [URL]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/benches/external"
README="$PROJECT_DIR/README.md"

# Default values
RUNTIME="${1:-ollama}"
URL="${2:-http://localhost:11434}"
ITERATIONS=10

# Create results directory
mkdir -p "$RESULTS_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Realizar External Runtime Benchmark ===${NC}"
echo "Runtime: $RUNTIME"
echo "URL: $URL"
echo ""

# Detect available models based on runtime
detect_models() {
    case "$RUNTIME" in
        ollama)
            curl -s "$URL/api/tags" 2>/dev/null | jq -r '.models[].name' 2>/dev/null || echo ""
            ;;
        vllm|llama-cpp)
            # For OpenAI-compatible APIs, we'd need to query /v1/models
            curl -s "$URL/v1/models" 2>/dev/null | jq -r '.data[].id' 2>/dev/null || echo "default"
            ;;
    esac
}

# Run benchmark for a single model
run_benchmark() {
    local model="$1"
    local output_file="$RESULTS_DIR/${RUNTIME}_${model//\//_}_$(date +%Y%m%d_%H%M%S).json"

    echo -e "${GREEN}Benchmarking: $model${NC}"

    "$PROJECT_DIR/target/release/realizar" bench \
        --runtime "$RUNTIME" \
        --url "$URL" \
        --model "$model" \
        -o "$output_file" 2>&1 | grep -E "(TTFT|Results|Mean|p50|p99|Throughput)" || true

    echo "$output_file"
}

# Generate markdown table from results
generate_table() {
    echo ""
    echo "## External Runtime Benchmarks"
    echo ""
    echo "Real HTTP benchmarks against external model servers. **NO MOCK DATA** - actual network latency + inference timing."
    echo ""
    echo "| Runtime | Model | Mean (ms) | p50 (ms) | p99 (ms) | Throughput (tok/s) |"
    echo "|---------|-------|-----------|----------|----------|-------------------|"

    for result_file in "$RESULTS_DIR"/*.json; do
        if [ -f "$result_file" ]; then
            runtime=$(jq -r '.runtime' "$result_file")
            model=$(jq -r '.model' "$result_file")
            mean=$(jq -r '.latency_ms.mean | floor' "$result_file")
            p50=$(jq -r '.latency_ms.p50 | floor' "$result_file")
            p99=$(jq -r '.latency_ms.p99 | floor' "$result_file")
            tps=$(jq -r '.throughput_tokens_per_sec | floor' "$result_file")

            echo "| $runtime | $model | $mean | $p50 | $p99 | $tps |"
        fi
    done

    echo ""
    echo "_Benchmarks run on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')_"
    echo ""
}

# Update README with benchmark table
update_readme() {
    local table_content="$1"
    local start_marker="<!-- BENCHMARK_TABLE_START -->"
    local end_marker="<!-- BENCHMARK_TABLE_END -->"

    # Check if markers exist
    if grep -q "$start_marker" "$README"; then
        # Replace existing table
        awk -v start="$start_marker" -v end="$end_marker" -v table="$table_content" '
            $0 ~ start { print; print table; skip=1; next }
            $0 ~ end { skip=0 }
            !skip { print }
        ' "$README" > "$README.tmp" && mv "$README.tmp" "$README"
        echo -e "${GREEN}Updated existing benchmark table in README.md${NC}"
    else
        echo -e "${BLUE}Benchmark table markers not found in README.md${NC}"
        echo "Add these markers to README.md where you want the table:"
        echo "  $start_marker"
        echo "  $end_marker"
        echo ""
        echo "Generated table:"
        echo "$table_content"
    fi
}

# Main execution
main() {
    # Build release binary if needed
    if [ ! -f "$PROJECT_DIR/target/release/realizar" ]; then
        echo "Building release binary..."
        cd "$PROJECT_DIR" && cargo build --release --features bench-http --bin realizar
    fi

    # Detect models
    MODELS=$(detect_models)

    if [ -z "$MODELS" ]; then
        echo "No models detected at $URL"
        echo "Running with default model..."
        MODELS="default"
    fi

    # Run benchmarks for each model
    for model in $MODELS; do
        run_benchmark "$model"
        echo ""
    done

    # Generate and update table
    TABLE=$(generate_table)
    update_readme "$TABLE"

    echo -e "${GREEN}=== Benchmark Complete ===${NC}"
    echo "Results saved to: $RESULTS_DIR"
}

main "$@"
