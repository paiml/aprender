#!/bin/bash
# Golden Trace Capture Script for trueno-db
#
# Captures syscall traces for trueno-db (GPU-first analytics database) examples using Renacer.
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
cargo build --release --example basic_usage --example simd_acceleration --example sql_query_interface

# Create traces directory
mkdir -p "$TRACES_DIR"

echo -e "${BLUE}=== Capturing Golden Traces for trueno-db ===${NC}"
echo -e "Examples: ./target/release/examples/"
echo -e "Output: $TRACES_DIR/"
echo ""

# ==============================================================================
# Trace 1: basic_usage (simple analytics query)
# ==============================================================================
echo -e "${GREEN}[1/3]${NC} Capturing: basic_usage"
BINARY_PATH="./target/release/examples/basic_usage"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^trueno-db\|^Creating\|^Sample\|^Query\|^Results\|^  \|^-\|^✓\|^Backend" | \
    head -1 > "$TRACES_DIR/basic_usage.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/basic_usage.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/basic_usage_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^trueno-db\|^Creating\|^Sample\|^Query\|^Results\|^  \|^-\|^✓\|^Backend" | \
    head -1 > "$TRACES_DIR/basic_usage_source.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/basic_usage_source.json"

# ==============================================================================
# Trace 2: simd_acceleration (SIMD-accelerated aggregations)
# ==============================================================================
echo -e "${GREEN}[2/3]${NC} Capturing: simd_acceleration"
BINARY_PATH="./target/release/examples/simd_acceleration"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^trueno-db\|^Testing\|^Scalar\|^SIMD\|^Speedup\|^Results\|^  \|^-\|^✓" | \
    head -1 > "$TRACES_DIR/simd_acceleration.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/simd_acceleration.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/simd_acceleration_summary.txt"

# ==============================================================================
# Trace 3: sql_query_interface (SQL query execution)
# ==============================================================================
echo -e "${GREEN}[3/3]${NC} Capturing: sql_query_interface"
BINARY_PATH="./target/release/examples/sql_query_interface"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^trueno-db\|^Creating\|^SQL\|^SELECT\|^WHERE\|^Results\|^Query\|^  \|^-\|^✓\|^│" | \
    head -1 > "$TRACES_DIR/sql_query_interface.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/sql_query_interface.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/sql_query_interface_summary.txt"

# ==============================================================================
# Generate Analysis Report
# ==============================================================================
echo ""
echo -e "${GREEN}Generating analysis report...${NC}"

cat > "$TRACES_DIR/ANALYSIS.md" << 'EOF'
# Golden Trace Analysis Report - trueno-db

## Overview

This directory contains golden traces captured from trueno-db (GPU-first embedded analytics database) examples.

## Trace Files

| File | Description | Format |
|------|-------------|--------|
| `basic_usage.json` | Simple analytics query | JSON |
| `basic_usage_summary.txt` | Basic usage syscall summary | Text |
| `basic_usage_source.json` | Basic usage with source locations | JSON |
| `simd_acceleration.json` | SIMD-accelerated aggregations | JSON |
| `simd_acceleration_summary.txt` | SIMD acceleration syscall summary | Text |
| `sql_query_interface.json` | SQL query execution | JSON |
| `sql_query_interface_summary.txt` | SQL query syscall summary | Text |

## How to Use These Traces

### 1. Regression Testing

Compare new builds against golden traces:

```bash
# Capture new trace
renacer --format json -- ./target/release/examples/basic_usage > new_trace.json

# Compare with golden
diff golden_traces/basic_usage.json new_trace.json

# Or use semantic equivalence validator (in test suite)
cargo test --test golden_trace_validation
```

### 2. Performance Budgeting

Check if new build meets performance requirements:

```bash
# Run with assertions
cargo test --test performance_assertions

# Or manually check against summary
cat golden_traces/basic_usage_summary.txt
```

### 3. CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate Performance
  run: |
    renacer --format json -- ./target/release/examples/basic_usage > trace.json
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
| `basic_usage` | TBD | TBD | Simple analytics query |
| `simd_acceleration` | TBD | TBD | SIMD aggregations |
| `sql_query_interface` | TBD | TBD | SQL query execution |

## Analytics Database Performance Characteristics

### Expected Syscall Patterns

**Columnar Data Loading**:
- Memory allocation (`brk`, `mmap`) for Arrow columnar structures
- Possible file I/O for Parquet loading

**Query Execution (SIMD-accelerated)**:
- CPU-intensive (minimal syscalls during SIMD operations)
- Write syscalls for result output

**SQL Query Interface**:
- SQL parsing overhead (minimal)
- Query execution syscalls similar to basic analytics
- Higher memory allocation for query plan structures

**GPU Operations (when GPU feature enabled)**:
- Additional syscalls for GPU initialization (`ioctl`, device opens)
- PCIe transfers for large datasets
- Potential bottleneck: small datasets don't benefit from GPU

### Anti-Pattern Detection

Renacer can detect:

1. **PCIe Bottleneck** (GPU builds only):
   - Symptom: GPU transfer time > compute time
   - Solution: Use SIMD backend for small datasets (auto-selected)

2. **God Process**:
   - Symptom: Single process doing too much
   - Solution: Separate data loading from query execution

## Next Steps

1. **Set performance baselines** using these golden traces
2. **Add assertions** in `renacer.toml` for automated checking
3. **Integrate with CI** to prevent regressions
4. **Compare SIMD vs GPU** traces (when GPU feature enabled)
5. **Monitor Arrow/Parquet I/O** patterns for optimization opportunities

Generated: $(date)
Renacer Version: 0.6.2
trueno-db Version: 0.3.1
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
echo "  1. Review traces: cat golden_traces/basic_usage_summary.txt"
echo "  2. View JSON: jq . golden_traces/basic_usage.json | less"
echo "  3. Run tests: cargo test --test golden_trace_validation"
echo "  4. Update baselines in ANALYSIS.md with actual metrics"
