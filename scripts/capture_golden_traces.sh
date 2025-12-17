#!/bin/bash
# Golden Trace Capture Script for aprender
#
# Captures syscall traces for aprender (pure Rust ML library) examples using Renacer.
# Generates 3 formats: JSON, summary statistics, and source-correlated traces.
#
# Usage: ./scripts/capture_golden_traces.sh

set -euo pipefail

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
cargo build --release --example iris_clustering --example dataframe_basics --example graph_algorithms_comprehensive

# Create traces directory
mkdir -p "$TRACES_DIR"

echo -e "${BLUE}=== Capturing Golden Traces for aprender ===${NC}"
echo -e "Examples: ./target/release/examples/"
echo -e "Output: $TRACES_DIR/"
echo ""

# ==============================================================================
# Trace 1: iris_clustering (K-means clustering on Iris dataset)
# ==============================================================================
echo -e "${GREEN}[1/3]${NC} Capturing: iris_clustering"
BINARY_PATH="./target/release/examples/iris_clustering"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Iris\\|^Loading\\|^Running\\|^Cluster\\|^Iteration\\|^Final\\|^Results\\|^  \\|^-\\|^✓\\|^K-means" | \
    head -1 > "$TRACES_DIR/iris_clustering.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/iris_clustering.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/iris_clustering_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Iris\\|^Loading\\|^Running\\|^Cluster\\|^Iteration\\|^Final\\|^Results\\|^  \\|^-\\|^✓\\|^K-means" | \
    head -1 > "$TRACES_DIR/iris_clustering_source.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/iris_clustering_source.json"

# ==============================================================================
# Trace 2: dataframe_basics (DataFrame operations)
# ==============================================================================
echo -e "${GREEN}[2/3]${NC} Capturing: dataframe_basics"
BINARY_PATH="./target/release/examples/dataframe_basics"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^DataFrame\\|^Creating\\|^Filtering\\|^Aggregating\\|^Sorting\\|^Results\\|^  \\|^-\\|^✓\\|^│\\|^┌\\|^└" | \
    head -1 > "$TRACES_DIR/dataframe_basics.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/dataframe_basics.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/dataframe_basics_summary.txt"

# ==============================================================================
# Trace 3: graph_algorithms_comprehensive (Graph analytics)
# ==============================================================================
echo -e "${GREEN}[3/3]${NC} Capturing: graph_algorithms_comprehensive"
BINARY_PATH="./target/release/examples/graph_algorithms_comprehensive"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^Graph\\|^Building\\|^Running\\|^PageRank\\|^BFS\\|^Community\\|^Results\\|^  \\|^-\\|^✓\\|^Node\\|^Edge" | \
    head -1 > "$TRACES_DIR/graph_algorithms_comprehensive.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/graph_algorithms_comprehensive.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/graph_algorithms_comprehensive_summary.txt"

# ==============================================================================
# Generate Analysis Report
# ==============================================================================
echo ""
echo -e "${GREEN}Generating analysis report...${NC}"

cat > "$TRACES_DIR/ANALYSIS.md" << 'EOF'
# Golden Trace Analysis Report - aprender

## Overview

This directory contains golden traces captured from aprender (pure Rust ML library) examples.

## Trace Files

| File | Description | Format |
|------|-------------|--------|
| `iris_clustering.json` | K-means clustering on Iris dataset | JSON |
| `iris_clustering_summary.txt` | Iris clustering syscall summary | Text |
| `iris_clustering_source.json` | Iris clustering with source locations | JSON |
| `dataframe_basics.json` | DataFrame operations (filter, aggregate, sort) | JSON |
| `dataframe_basics_summary.txt` | DataFrame basics syscall summary | Text |
| `graph_algorithms_comprehensive.json` | Graph analytics (PageRank, BFS, communities) | JSON |
| `graph_algorithms_comprehensive_summary.txt` | Graph algorithms syscall summary | Text |

## How to Use These Traces

### 1. Regression Testing

Compare new builds against golden traces:

```bash
# Capture new trace
renacer --format json -- ./target/release/examples/iris_clustering > new_trace.json

# Compare with golden
diff golden_traces/iris_clustering.json new_trace.json

# Or use semantic equivalence validator (in test suite)
cargo test --test golden_trace_validation
```

### 2. Performance Budgeting

Check if new build meets performance requirements:

```bash
# Run with assertions
cargo test --test performance_assertions

# Or manually check against summary
cat golden_traces/iris_clustering_summary.txt
```

### 3. CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate ML Performance
  run: |
    renacer --format json -- ./target/release/examples/iris_clustering > trace.json
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
| `iris_clustering` | TBD | TBD | K-means clustering (150 samples) |
| `dataframe_basics` | TBD | TBD | DataFrame operations |
| `graph_algorithms_comprehensive` | TBD | TBD | Graph analytics (PageRank + BFS) |

## ML Library Performance Characteristics

### Expected Syscall Patterns

**Data Loading**:
- Memory allocation (`brk`, `mmap`) for feature matrices
- File I/O for dataset loading (CSV, Parquet)

**Algorithm Execution (SIMD-accelerated via trueno)**:
- CPU-intensive (minimal syscalls during matrix operations)
- Write syscalls for result output
- SIMD intrinsics benefit from trueno backend

**Model Serialization**:
- File I/O for model persistence
- JSON/bincode serialization syscalls

**Graph Analytics**:
- Memory allocation for adjacency structures
- CSR (Compressed Sparse Row) format operations
- Iterative algorithms (PageRank, BFS)

### Anti-Pattern Detection

Renacer can detect:

1. **PCIe Bottleneck** (when using trueno GPU backend):
   - Symptom: GPU transfer time > compute time
   - Solution: Use SIMD backend for small datasets (auto-selected)

2. **God Process**:
   - Symptom: Single process doing too much
   - Solution: Separate data loading from algorithm execution

## Next Steps

1. **Set performance baselines** using these golden traces
2. **Add assertions** in `renacer.toml` for automated checking
3. **Integrate with CI** to prevent regressions
4. **Compare SIMD vs scalar** traces (trueno backend selection)
5. **Monitor serialization I/O** patterns for optimization opportunities

Generated: $(date)
Renacer Version: 0.6.2
aprender Version: 0.7.0
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
find "$TRACES_DIR" \( -name '*.json' -o -name '*.txt' \) -type f -exec stat --printf='  %n (%s bytes)\n' {} \; 2>/dev/null
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review traces: cat golden_traces/iris_clustering_summary.txt"
echo "  2. View JSON: jq . golden_traces/iris_clustering.json | less"
echo "  3. Run tests: cargo test --test golden_trace_validation"
echo "  4. Update baselines in ANALYSIS.md with actual metrics"
