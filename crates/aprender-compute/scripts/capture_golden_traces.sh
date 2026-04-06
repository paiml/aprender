#!/bin/bash
# Golden Trace Capture Script for Trueno
#
# Captures syscall traces for Trueno SIMD/GPU compute examples using Renacer.
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
TRACES_DIR="golden_traces"

# Ensure renacer is installed
if ! command -v renacer &> /dev/null; then
    echo -e "${YELLOW}Renacer not found. Installing from crates.io...${NC}"
    cargo install renacer --version 0.6.2
fi

# Build examples
echo -e "${YELLOW}Building release examples...${NC}"
cargo build --release --examples

# Create traces directory
mkdir -p "$TRACES_DIR"

echo -e "${BLUE}=== Capturing Golden Traces for Trueno ===${NC}"
echo -e "Examples: ./target/release/examples/"
echo -e "Output: $TRACES_DIR/"
echo ""

# ==============================================================================
# Trace 1: backend_detection (SIMD backend detection)
# ==============================================================================
echo -e "${GREEN}[1/5]${NC} Capturing: backend_detection"
BINARY_PATH="./target/release/examples/backend_detection"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Trueno\|^Backend\|^CPU\|^SIMD\|^-\|^  " | \
    head -1 > "$TRACES_DIR/backend_detection.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/backend_detection.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/backend_detection_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Trueno\|^Backend\|^CPU\|^SIMD\|^-\|^  " | \
    head -1 > "$TRACES_DIR/backend_detection_source.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/backend_detection_source.json"

# ==============================================================================
# Trace 2: matrix_operations (matrix multiply, transpose)
# ==============================================================================
echo -e "${GREEN}[2/5]${NC} Capturing: matrix_operations"
BINARY_PATH="./target/release/examples/matrix_operations"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Matrix\|^Result\|^Transpose\|^Multiply\|^  \|^-\|^\[" | \
    head -1 > "$TRACES_DIR/matrix_operations.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/matrix_operations.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/matrix_operations_summary.txt"

# ==============================================================================
# Trace 3: activation_functions (ML activation functions - ReLU, sigmoid)
# ==============================================================================
echo -e "${GREEN}[3/5]${NC} Capturing: activation_functions"
BINARY_PATH="./target/release/examples/activation_functions"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Activation\|^Input\|^ReLU\|^Sigmoid\|^Tanh\|^  \|^-\|^\[" | \
    head -1 > "$TRACES_DIR/activation_functions.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/activation_functions.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/activation_functions_summary.txt"

# ==============================================================================
# Trace 4: performance_demo (comprehensive performance demonstration)
# ==============================================================================
echo -e "${GREEN}[4/5]${NC} Capturing: performance_demo"
BINARY_PATH="./target/release/examples/performance_demo"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Trueno\|^Testing\|^Vector\|^Matrix\|^Performance\|^Backend\|^  \|^-\|^×" | \
    head -1 > "$TRACES_DIR/performance_demo.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/performance_demo.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/performance_demo_summary.txt"

# ==============================================================================
# Trace 5: ml_similarity (ML similarity operations)
# ==============================================================================
echo -e "${GREEN}[5/5]${NC} Capturing: ml_similarity"
BINARY_PATH="./target/release/examples/ml_similarity"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^ML\|^Vector\|^Cosine\|^Euclidean\|^Manhattan\|^  \|^-\|^✓" | \
    head -1 > "$TRACES_DIR/ml_similarity.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/ml_similarity.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/ml_similarity_summary.txt"

# ==============================================================================
# Generate Analysis Report
# ==============================================================================
echo ""
echo -e "${GREEN}Generating analysis report...${NC}"

cat > "$TRACES_DIR/ANALYSIS.md" << 'EOF'
# Golden Trace Analysis Report - Trueno

## Overview

This directory contains golden traces captured from Trueno SIMD/GPU compute examples.

## Trace Files

| File | Description | Format |
|------|-------------|--------|
| `backend_detection.json` | SIMD backend detection | JSON |
| `backend_detection_summary.txt` | Backend detection syscall summary | Text |
| `backend_detection_source.json` | Backend detection with source locations | JSON |
| `matrix_operations.json` | Matrix multiply/transpose trace | JSON |
| `matrix_operations_summary.txt` | Matrix ops syscall summary | Text |
| `activation_functions.json` | ML activation functions trace | JSON |
| `activation_functions_summary.txt` | Activation syscall summary | Text |
| `performance_demo.json` | Comprehensive performance demo | JSON |
| `performance_demo_summary.txt` | Performance demo syscall summary | Text |
| `ml_similarity.json` | ML similarity operations trace | JSON |
| `ml_similarity_summary.txt` | ML similarity syscall summary | Text |

## How to Use These Traces

### 1. Regression Testing

Compare new builds against golden traces:

```bash
# Capture new trace
renacer --format json -- ./target/release/examples/backend_detection > new_trace.json

# Compare with golden
diff golden_traces/backend_detection.json new_trace.json

# Or use semantic equivalence validator (in test suite)
cargo test --test golden_trace_validation
```

### 2. Performance Budgeting

Check if new build meets performance requirements:

```bash
# Run with assertions
cargo test --test performance_assertions

# Or manually check against summary
cat golden_traces/backend_detection_summary.txt
```

### 3. CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate Performance
  run: |
    renacer --format json -- ./target/release/examples/backend_detection > trace.json
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
      "args": [["fd", "1"], ["buf", "Backend: AVX2\n"], ["count", "15"]],
      "result": 15
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
| `backend_detection` | TBD | TBD | SIMD backend selection |
| `matrix_operations` | TBD | TBD | Matrix multiply/transpose |
| `activation_functions` | TBD | TBD | ReLU, sigmoid, tanh |
| `performance_demo` | TBD | TBD | Comprehensive performance test |
| `ml_similarity` | TBD | TBD | Cosine/Euclidean similarity |

## SIMD/GPU Performance Characteristics

### Expected Syscall Patterns

**SIMD Operations (CPU-bound)**:
- Minimal syscalls (mostly memory allocation via `brk`/`mmap`)
- Fast execution (<10ms for small examples)
- Low I/O overhead

**GPU Operations (GPU-enabled builds)**:
- Additional syscalls for GPU initialization (`ioctl`, device opens)
- Higher memory allocation (GPU buffer management)
- Potential PCIe bottleneck for small workloads

### Anti-Pattern Detection

Renacer can detect **PCIe Bottleneck** anti-pattern:
- Excessive GPU memory transfers relative to compute time
- Symptom: Many `write`/`read` syscalls to GPU device
- Solution: Batch operations, keep data on GPU

## Next Steps

1. **Set performance baselines** using these golden traces
2. **Add assertions** in `renacer.toml` for automated checking
3. **Integrate with CI** to prevent regressions
4. **Compare SIMD backends** (SSE2 vs AVX2 vs AVX-512) syscall patterns
5. **Monitor GPU workloads** for PCIe bottlenecks

Generated: $(date)
Renacer Version: 0.6.2
Trueno Version: 0.7.0
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
ls -lh "$TRACES_DIR"/*.json "$TRACES_DIR"/*.txt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review traces: cat golden_traces/backend_detection_summary.txt"
echo "  2. View JSON: jq . golden_traces/backend_detection.json | less"
echo "  3. Run tests: cargo test --test golden_trace_validation"
echo "  4. Update baselines in ANALYSIS.md with actual metrics"
