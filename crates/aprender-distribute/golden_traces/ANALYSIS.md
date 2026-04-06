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
| `hello_repartir` | 382.138ms | 600 | CPU work-stealing scheduler (4 workers, 10 parallel tasks) |
| `pubsub_example` | 5.210ms | 516 | PUB/SUB messaging (3 scenarios, 3 topics) |
| `v1_1_showcase` | 1011.226ms | 18169 | Comprehensive showcase (CPU + GPU + TLS + Priority + Fault Tolerance) |

**Key Insights:**
- **hello_repartir** (382.138ms): Dominated by futex (98.58%) for work-stealing synchronization. 48 clone3 calls for worker pool. Achieved 3.82× parallel speedup with 4 workers.
- **pubsub_example** (5.210ms): Extremely fast async messaging. Futex (21.69%) and clone3 (17.66%) for tokio runtime coordination. 48 clone3 calls for subscribers.
- **v1_1_showcase** (1.011s): Complex multi-backend demo. Futex (60.23%) dominates for synchronization. 4157 read calls for GPU device enumeration and TLS certificate loading. 1912 close calls. Achieved 3.85× parallel speedup with 4 workers on 20 tasks.

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

Generated: 2025-11-23
Renacer Version: 0.6.2
repartir Version: 1.0.0
