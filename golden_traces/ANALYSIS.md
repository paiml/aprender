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
| `iris_clustering` | **0.842ms** | **97** | K-means clustering (150 samples) ✅ |
| `dataframe_basics` | **0.855ms** | **96** | **Fastest syscalls!** DataFrame operations ✅ |
| `graph_algorithms_comprehensive` | **1.734ms** | **191** | 11 graph algorithms (BFS, Dijkstra, A*, DFS, PageRank, etc.) ✅ |

**Performance Budget Compliance:**
- ✅ All examples complete in <2ms (well under 1000ms budget)
- ✅ DataFrame operations exceptionally fast at 0.855ms
- ✅ K-means clustering with minimal overhead: 0.842ms
- ✅ Comprehensive graph algorithms demo: 1.734ms for 11 algorithms
- ✅ Excellent ML library performance for embedded use cases

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
