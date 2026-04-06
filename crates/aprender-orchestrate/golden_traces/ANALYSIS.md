# Golden Trace Analysis Report - batuta

## Overview

This directory contains golden traces captured from batuta (orchestration framework for converting projects to Rust) examples.

## Trace Files

| File | Description | Format |
|------|-------------|--------|
| `backend_selection.json` | GPU/SIMD backend selection logic | JSON |
| `backend_selection_summary.txt` | Backend selection syscall summary | Text |
| `backend_selection_source.json` | Backend selection with source locations | JSON |
| `full_transpilation.json` | Full transpilation pipeline | JSON |
| `full_transpilation_summary.txt` | Full transpilation syscall summary | Text |
| `pipeline_demo.json` | Pipeline execution demonstration | JSON |
| `pipeline_demo_summary.txt` | Pipeline demo syscall summary | Text |
| `oracle_demo.json` | Oracle Mode intelligent query interface | JSON |
| `oracle_demo_summary.txt` | Oracle demo syscall summary | Text |
| `oracle_demo_source.json` | Oracle demo with source locations | JSON |

## How to Use These Traces

### 1. Regression Testing

Compare new builds against golden traces:

```bash
# Capture new trace
renacer --format json -- ./target/release/examples/backend_selection > new_trace.json

# Compare with golden
diff golden_traces/backend_selection.json new_trace.json

# Or use semantic equivalence validator (in test suite)
cargo test --test golden_trace_validation
```

### 2. Performance Budgeting

Check if new build meets performance requirements:

```bash
# Run with assertions
cargo test --test performance_assertions

# Or manually check against summary
cat golden_traces/backend_selection_summary.txt
```

### 3. CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate Orchestration Performance
  run: |
    renacer --format json -- ./target/release/examples/backend_selection > trace.json
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
| `backend_selection` | TBD | TBD | GPU/SIMD backend selection |
| `full_transpilation` | TBD | TBD | Full transpilation pipeline |
| `pipeline_demo` | TBD | TBD | Pipeline execution |
| `oracle_demo` | TBD | TBD | Oracle Mode query interface |

## Orchestration Framework Performance Characteristics

### Expected Syscall Patterns

**Backend Selection**:
- CPU-intensive compute/transfer ratio calculations
- Minimal syscalls during decision logic
- Write syscalls for output

**Transpilation Pipeline**:
- Tool detection (file I/O for finding binaries)
- Process spawning for external tools
- File I/O for reading/writing transpiled code
- Memory allocation for AST structures

**Pipeline Execution**:
- Multi-stage pipeline coordination
- Process spawning for each stage
- File I/O for intermediate results
- Error handling and rollback operations

**Oracle Mode**:
- Knowledge graph initialization (memory allocations)
- Query parsing (minimal CPU, string operations)
- Component lookups (hash map operations)
- Backend selection calculations (CPU-bound)
- Code example generation (string formatting)

### Anti-Pattern Detection

Renacer can detect:

1. **Tight Loop**:
   - Symptom: Excessive loop iterations without I/O
   - Solution: Optimize orchestration logic or batch operations

2. **God Process**:
   - Symptom: Single process doing too much
   - Solution: Delegate work to specialized tools

## Next Steps

1. **Set performance baselines** using these golden traces
2. **Add assertions** in `renacer.toml` for automated checking
3. **Integrate with CI** to prevent regressions
4. **Monitor tool spawning** for process overhead
5. **Optimize pipeline coordination** based on trace analysis

Generated: $(date)
Renacer Version: 0.6.5
batuta Version: 0.1.0
