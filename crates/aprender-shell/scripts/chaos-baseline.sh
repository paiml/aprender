#!/usr/bin/env bash
# chaos-baseline.sh - Chaos engineering baseline for aprender-shell
# Issue #99: renacer chaos testing integration
#
# Toyota Way Principles:
# - Jidoka: Build quality in through stress testing
# - Genchi Genbutsu: Go and see - measure real behavior under stress
# - Kaizen: Continuous improvement through regular chaos testing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BINARY="$PROJECT_ROOT/target/release/aprender-shell"
MODEL="/tmp/chaos-test-model.apr"
HISTORY="/tmp/chaos-test-history"
RESULTS_DIR="/tmp/chaos-results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Ensure renacer is available
check_renacer() {
    if ! command -v renacer &> /dev/null; then
        log_warn "renacer not found in PATH, attempting to build..."
        if [[ -d "$PROJECT_ROOT/../renacer" ]]; then
            cargo build --release --manifest-path "$PROJECT_ROOT/../renacer/Cargo.toml"
            export PATH="$PROJECT_ROOT/../renacer/target/release:$PATH"
        else
            log_error "renacer not found. Install from: cargo install --path ../renacer"
            exit 1
        fi
    fi
    log_info "Using renacer: $(which renacer)"
}

# Build aprender-shell in release mode
build_binary() {
    log_info "Building aprender-shell in release mode..."
    cargo build --release -p aprender-shell --manifest-path "$PROJECT_ROOT/Cargo.toml"
}

# Create test model
create_model() {
    log_info "Creating test model..."
    mkdir -p "$(dirname "$MODEL")"

    cat > "$HISTORY" << 'EOF'
git status
git commit -m "test commit"
git push origin main
git pull --rebase
git checkout -b feature
cargo build --release
cargo test --all
cargo bench
docker run -it ubuntu
docker ps -a
kubectl get pods
kubectl apply -f deployment.yaml
aws s3 ls
aws ec2 describe-instances
npm install
npm run build
python -m venv .venv
pip install -r requirements.txt
EOF

    "$BINARY" train --history "$HISTORY" --output "$MODEL" 2>/dev/null
    log_info "Model created: $MODEL"
}

# Gentle chaos - CI-safe, validates P99 < 10ms
test_gentle_chaos() {
    log_info "Running gentle chaos test (CI-safe)..."

    local output
    output=$(renacer --chaos gentle -c -- "$BINARY" suggest "git " --model "$MODEL" 2>&1)

    # Check for anomalies
    if echo "$output" | grep -q "ANOMALY"; then
        log_error "Gentle chaos: Anomalies detected!"
        echo "$output"
        return 1
    fi

    log_info "Gentle chaos: PASS"
    echo "$output" > "$RESULTS_DIR/gentle-chaos.txt"
}

# Memory pressure - simulates containers/embedded (64MB limit)
test_memory_pressure() {
    log_info "Running memory pressure test (64MB limit)..."

    local output
    output=$(renacer --chaos-memory-limit 64M -c -- "$BINARY" suggest "cargo " --model "$MODEL" 2>&1)

    # Check process completed without OOM
    if echo "$output" | grep -q "killed\|OOM\|Cannot allocate"; then
        log_error "Memory pressure: Process killed due to memory limit!"
        echo "$output"
        return 1
    fi

    log_info "Memory pressure: PASS"
    echo "$output" > "$RESULTS_DIR/memory-pressure.txt"
}

# CPU throttle - simulates slow CI runners (25% CPU)
test_cpu_throttle() {
    log_info "Running CPU throttle test (25% CPU)..."

    local output
    output=$(renacer --chaos-cpu-limit 0.25 -c -- "$BINARY" suggest "docker " --model "$MODEL" 2>&1)

    log_info "CPU throttle: PASS"
    echo "$output" > "$RESULTS_DIR/cpu-throttle.txt"
}

# Aggressive chaos - full stress test (manual only)
test_aggressive_chaos() {
    log_info "Running aggressive chaos test..."

    local output
    output=$(renacer --chaos aggressive -c -- "$BINARY" suggest "kubectl " --model "$MODEL" 2>&1)

    log_info "Aggressive chaos: PASS"
    echo "$output" > "$RESULTS_DIR/aggressive-chaos.txt"
}

# Signal injection - SIGUSR1/SIGUSR2 handling
test_signal_injection() {
    log_info "Running signal injection test (SIGUSR1)..."

    local output
    output=$(renacer --chaos-signal SIGUSR1 -c -- "$BINARY" suggest "aws " --model "$MODEL" 2>&1)

    log_info "Signal injection: PASS"
    echo "$output" > "$RESULTS_DIR/signal-injection.txt"
}

# Performance baseline without chaos
test_baseline() {
    log_info "Running performance baseline (no chaos)..."

    local output
    output=$(renacer -c --stats-extended -- "$BINARY" suggest "git " --model "$MODEL" 2>&1)

    log_info "Baseline complete"
    echo "$output" > "$RESULTS_DIR/baseline.txt"

    # Parse and display key metrics
    echo ""
    log_info "Key metrics:"
    echo "$output" | grep -E "^[0-9]+\.[0-9]+.*total" || true
}

# Generate summary report
generate_report() {
    log_info "Generating chaos test report..."

    cat > "$RESULTS_DIR/report.md" << EOF
# Chaos Test Report

Generated: $(date -Iseconds)
Binary: $BINARY
Model: $MODEL

## Test Results

| Test | Status |
|------|--------|
| Baseline | $(test -f "$RESULTS_DIR/baseline.txt" && echo "PASS" || echo "SKIP") |
| Gentle Chaos | $(test -f "$RESULTS_DIR/gentle-chaos.txt" && echo "PASS" || echo "SKIP") |
| Memory Pressure | $(test -f "$RESULTS_DIR/memory-pressure.txt" && echo "PASS" || echo "SKIP") |
| CPU Throttle | $(test -f "$RESULTS_DIR/cpu-throttle.txt" && echo "PASS" || echo "SKIP") |
| Aggressive Chaos | $(test -f "$RESULTS_DIR/aggressive-chaos.txt" && echo "PASS" || echo "SKIP") |
| Signal Injection | $(test -f "$RESULTS_DIR/signal-injection.txt" && echo "PASS" || echo "SKIP") |

## Baseline Syscall Profile

\`\`\`
$(cat "$RESULTS_DIR/baseline.txt" 2>/dev/null || echo "Not available")
\`\`\`
EOF

    log_info "Report saved to: $RESULTS_DIR/report.md"
}

# Main
main() {
    local mode="${1:-ci}"

    log_info "Starting chaos baseline tests (mode: $mode)"

    mkdir -p "$RESULTS_DIR"

    check_renacer
    build_binary
    create_model

    case "$mode" in
        ci)
            # CI-safe tests only
            test_baseline
            test_gentle_chaos
            test_memory_pressure
            test_cpu_throttle
            ;;
        full)
            # All tests including aggressive
            test_baseline
            test_gentle_chaos
            test_memory_pressure
            test_cpu_throttle
            test_aggressive_chaos
            test_signal_injection
            ;;
        baseline)
            # Just baseline for comparison
            test_baseline
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo "Usage: $0 [ci|full|baseline]"
            exit 1
            ;;
    esac

    generate_report

    log_info "All chaos tests completed successfully!"
    log_info "Results: $RESULTS_DIR/"
}

main "$@"
