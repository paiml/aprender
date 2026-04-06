# Golden Trace Analysis Report - entrenar

## Overview

This directory contains golden traces captured from entrenar
(training & optimization library with autograd, LoRA, quantization) examples.

## Trace Files

| File | Description | Format |
|------|-------------|--------|
| `training_loop.json` | Basic training loop (autograd + optimizer) | JSON |
| `training_loop_summary.txt` | Training loop syscall summary | Text |
| `training_loop_source.json` | Training loop with source locations | JSON |
| `model_io.json` | Model save/load operations | JSON |
| `model_io_summary.txt` | Model I/O syscall summary | Text |
| `llama2_finetune_lora.json` | LoRA parameter-efficient fine-tuning | JSON |
| `llama2_finetune_lora_summary.txt` | LoRA fine-tuning syscall summary | Text |

## How to Use These Traces

### 1. Regression Testing

Compare new builds against golden traces:

```bash
# Capture new trace
renacer --format json -- ./target/release/examples/training_loop > new_trace.json

# Compare with golden
diff golden_traces/training_loop.json new_trace.json

# Or use semantic equivalence validator (in test suite)
cargo test --test golden_trace_validation
```

### 2. Performance Budgeting

Check if new build meets performance requirements:

```bash
# Run with assertions
cargo test --test performance_assertions

# Or manually check against summary
cat golden_traces/training_loop_summary.txt
```

### 3. CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate Training Performance
  run: |
    renacer --format json -- ./target/release/examples/training_loop > trace.json
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
| `training_loop` | 0.720ms | 92 | Basic autograd + optimizer (10 epochs, 30 steps) |
| `model_io` | 1.191ms | 126 | Model save/load (JSON/YAML, 4 parameters) |
| `llama2_finetune_lora` | 144.728ms | 10866 | LoRA fine-tuning (162.4M params → 1.2M trainable, 3 epochs, 300 steps) |

**Key Insights:**
- **training_loop** (0.720ms): Extremely fast compute-bound training. Dominated by write (26.39%) for metrics output.
  Minimal syscall overhead (92 total).
- **model_io** (1.191ms): Fast serialization. Write (25.69%) for JSON/YAML output, read (7.39%) for loading. 3 unlink
  calls for cleanup.
- **llama2_finetune_lora** (144.728ms): Dominated by getrandom (53.45%, 9914 calls) for random initialization. Munmap
  (37.48%, 137 calls) for memory management. Mremap (6.89%, 511 calls) for ndarray reallocation. Training 162.4M
  parameter model with only 1.2M trainable (99.3% reduction).

## Training & Optimization Performance Characteristics

### Expected Syscall Patterns

**Training Loop**:
- Compute-intensive autograd operations
- Minimal syscalls (memory-bound)
- Write syscalls for loss/metrics output
- Memory allocation for gradients

**Model I/O**:
- File write operations (model serialization)
- File read operations (model loading)
- JSON/YAML parsing overhead
- Temporary file creation

**LoRA Fine-Tuning**:
- Similar to training loop but with adapter matrices
- Reduced memory allocation (99.75% fewer parameters)
- Gradient computation only for low-rank adapters
- Model checkpoint saving

### Anti-Pattern Detection

Renacer can detect:

1. **Tight Loop**:
   - Symptom: Excessive loop iterations without I/O
   - Solution: Optimize gradient computation or vectorize operations

2. **God Process**:
   - Symptom: Single process doing too much
   - Solution: Distribute training across workers

## Next Steps

1. **Set performance baselines** using these golden traces
2. **Add assertions** in `renacer.toml` for automated checking
3. **Integrate with CI** to prevent regressions
4. **Compare autograd** performance across different optimizers
5. **Monitor memory** allocation patterns for large models

Generated: 2025-11-23
Renacer Version: 0.6.2
entrenar Version: 0.1.0
