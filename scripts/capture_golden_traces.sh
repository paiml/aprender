#!/bin/bash
# Golden Trace Capture Script for repartir
#
# Captures syscall traces for repartir (distributed computing primitives) examples using Renacer.
# Generates 3 formats: JSON, summary statistics, and source-correlated traces.
#
# Usage: ./scripts/capture_golden_traces.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TRACES_DIR='golden_traces'

# Validate paths (prevent path traversal)
if [[ "$TRACES_DIR" == *..* ]] || [[ "$TRACES_DIR" == /* ]]; then
    echo "Error: Invalid TRACES_DIR path" >&2
    exit 1
fi

# Ensure renacer is installed
if ! command -v renacer &> /dev/null; then
    echo -e "${YELLOW}Renacer not found. Installing from crates.io...${NC}"
    cargo install renacer --version 0.6.2
fi

# Build examples
echo -e "${YELLOW}Building release examples...${NC}"
cargo build --release --example hello_repartir --example pubsub_example --example v1_1_showcase --features full

# Create traces directory
# Using literal path to satisfy bashrs SEC010 check (TRACES_DIR='golden_traces' is validated above)
mkdir -p golden_traces

echo -e "${BLUE}=== Capturing Golden Traces for repartir ===${NC}"
echo -e "Examples: ./target/release/examples/"
echo -e "Output: $TRACES_DIR/"
echo ""

# ==============================================================================
# Trace 1: hello_repartir (CPU work-stealing scheduler)
# ==============================================================================
echo -e "${GREEN}[1/3]${NC} Capturing: hello_repartir"
BINARY_PATH="./target/release/examples/hello_repartir"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🎯\|^Pool\|^Task\|^  \|^Output\|^✅\|^━" | \
    head -1 > "$TRACES_DIR/hello_repartir.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/hello_repartir.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/hello_repartir_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🎯\|^Pool\|^Task\|^  \|^Output\|^✅\|^━" | \
    head -1 > "$TRACES_DIR/hello_repartir_source.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/hello_repartir_source.json"

# ==============================================================================
# Trace 2: pubsub_example (PUB/SUB messaging patterns)
# ==============================================================================
echo -e "${GREEN}[2/3]${NC} Capturing: pubsub_example"
BINARY_PATH="./target/release/examples/pubsub_example"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^📡\|^Creating\|^Subscriber\|^Publisher\|^Received\|^  \|^✅\|^━" | \
    head -1 > "$TRACES_DIR/pubsub_example.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/pubsub_example.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/pubsub_example_summary.txt"

# ==============================================================================
# Trace 3: v1_1_showcase (comprehensive distributed computing showcase)
# ==============================================================================
echo -e "${GREEN}[3/3]${NC} Capturing: v1_1_showcase"
BINARY_PATH="./target/release/examples/v1_1_showcase"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🚀\|^═\|^CPU\|^GPU\|^TLS\|^Priority\|^Parallel\|^  \|^✅\|^❌\|^━\|^Repartir\|^──\|^Workers\|^Backend" | \
    head -1 > "$TRACES_DIR/v1_1_showcase.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/v1_1_showcase.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/v1_1_showcase_summary.txt"

# ==============================================================================
# Generate Analysis Report
# ==============================================================================
echo ""
echo -e "${GREEN}Generating analysis report...${NC}"

# Using literal path to satisfy bashrs SEC010 check
cat > golden_traces/ANALYSIS.md << 'EOF'
# Golden Trace Analysis Report - repartir

## Overview

This directory contains golden traces captured from repartir (distributed computing primitives for Rust) examples.

## Trace Files

| File | Description | Format |
|------|-------------|--------|
| `hello_repartir.json` | CPU work-stealing scheduler | JSON |
| `hello_repartir_summary.txt` | Hello repartir syscall summary | Text |
| `hello_repartir_source.json` | Hello repartir with source locations | JSON |
| `pubsub_example.json` | PUB/SUB messaging patterns | JSON |
| `pubsub_example_summary.txt` | PUB/SUB syscall summary | Text |
| `v1_1_showcase.json` | Comprehensive distributed computing showcase | JSON |
| `v1_1_showcase_summary.txt` | Showcase syscall summary | Text |

## How to Use These Traces

### 1. Regression Testing

Compare new builds against golden traces:

```bash
# Capture new trace
renacer --format json -- ./target/release/examples/hello_repartir > new_trace.json

# Compare with golden
diff golden_traces/hello_repartir.json new_trace.json

# Or use semantic equivalence validator (in test suite)
cargo test --test golden_trace_validation
```

### 2. Performance Budgeting

Check if new build meets performance requirements:

```bash
# Run with assertions
cargo test --test performance_assertions

# Or manually check against summary
cat golden_traces/hello_repartir_summary.txt
```

### 3. CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate Distributed Computing Performance
  run: |
    renacer --format json -- ./target/release/examples/hello_repartir > trace.json
    # Compare against golden trace or run assertions
    cargo test --test golden_trace_validation
```

## Trace Interpretation Guide

### JSON Trace Format

```json
{
  "version": "0.6.2",
  "format": "renacer-json-v1",
  "syscalls": [
    {
      "name": "write",
      "args": [["fd", "1"], ["buf", "Results: [...]"], ["count", "25"]],
      "result": 25
    }
  ]
}
```

### Summary Statistics Format

```
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 19.27    0.000137          10        13           mmap
 14.35    0.000102          17         6           write
...
```

**Key metrics:**
- `% time`: Percentage of total runtime spent in this syscall
- `usecs/call`: Average latency per call (microseconds)
- `calls`: Total number of invocations
- `errors`: Number of failed calls

## Baseline Performance Metrics

From initial golden trace capture:

| Operation | Runtime | Syscalls | Notes |
|-----------|---------|----------|-------|
| `hello_repartir` | TBD | TBD | CPU work-stealing scheduler |
| `pubsub_example` | TBD | TBD | PUB/SUB messaging |
| `v1_1_showcase` | TBD | TBD | Comprehensive showcase (CPU + GPU + TLS) |

## Distributed Computing Performance Characteristics

### Expected Syscall Patterns

**CPU Work-Stealing**:
- Process/thread spawning for worker pool
- Futex operations for task queue synchronization
- Memory allocation for task data
- Minimal I/O overhead (compute-bound)

**PUB/SUB Messaging**:
- Async channel operations (tokio runtime)
- Write syscalls for message delivery
- Poll operations for async coordination
- Memory allocation for message buffers

**Comprehensive Showcase**:
- GPU detection (file I/O for device enumeration)
- TLS setup (certificate loading)
- Worker pool initialization (clone/futex)
- Multi-backend coordination (CPU + GPU + Remote)

### Anti-Pattern Detection

Renacer can detect:

1. **Tight Loop**:
   - Symptom: Excessive loop iterations without I/O
   - Solution: Optimize work-stealing algorithm or add backoff

2. **God Process**:
   - Symptom: Single process doing too much
   - Solution: Delegate work to worker pool

## Next Steps

1. **Set performance baselines** using these golden traces
2. **Add assertions** in `renacer.toml` for automated checking
3. **Integrate with CI** to prevent regressions
4. **Compare work-stealing** efficiency across different pool sizes
5. **Monitor async runtime** overhead (tokio)

Renacer Version: 0.6.2
repartir Version: 1.0.0
EOF

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo -e "${BLUE}=== Golden Trace Capture Complete ===${NC}"
echo ""
echo "Traces saved to: $TRACES_DIR/"
echo ""
echo "Files generated:"
ls -lh "$TRACES_DIR"/*.json "$TRACES_DIR"/*.txt 2>/dev/null | sort | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review traces: cat golden_traces/hello_repartir_summary.txt"
echo "  2. View JSON: jq . golden_traces/hello_repartir.json | less"
echo "  3. Run tests: cargo test --test golden_trace_validation"
echo "  4. Update baselines in ANALYSIS.md with actual metrics"
