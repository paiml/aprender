#!/bin/bash
# Golden Trace Capture Script for entrenar
#
# Captures syscall traces for entrenar (training & optimization library) examples using Renacer.
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
cargo build --release --example training_loop --example model_io --example llama2-finetune-lora

# Create traces directory
mkdir -p "$TRACES_DIR"

echo -e "${BLUE}=== Capturing Golden Traces for entrenar ===${NC}"
echo -e "Examples: ./target/release/examples/"
echo -e "Output: $TRACES_DIR/"
echo ""

# ==============================================================================
# Trace 1: training_loop (basic autograd + optimizer)
# ==============================================================================
echo -e "${GREEN}[1/3]${NC} Capturing: training_loop"
BINARY_PATH="./target/release/examples/training_loop"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^📊\|^Epoch\|^  \|^Training\|^Loss\|^✅\|^━\|^Final" | \
    head -1 > "$TRACES_DIR/training_loop.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/training_loop.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/training_loop_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^📊\|^Epoch\|^  \|^Training\|^Loss\|^✅\|^━\|^Final" | \
    head -1 > "$TRACES_DIR/training_loop_source.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/training_loop_source.json"

# ==============================================================================
# Trace 2: model_io (model save/load operations)
# ==============================================================================
echo -e "${GREEN}[2/3]${NC} Capturing: model_io"
BINARY_PATH="./target/release/examples/model_io"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^💾\|^Creating\|^Saving\|^Loading\|^  \|^✅\|^━\|^Model" | \
    head -1 > "$TRACES_DIR/model_io.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/model_io.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/model_io_summary.txt"

# ==============================================================================
# Trace 3: llama2-finetune-lora (LoRA parameter-efficient fine-tuning)
# ==============================================================================
echo -e "${GREEN}[3/3]${NC} Capturing: llama2-finetune-lora"
BINARY_PATH="./target/release/examples/llama2-finetune-lora"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🦙\|^🎯\|^LoRA\|^Parameters\|^Memory\|^Training\|^Epoch\|^  \|^✅\|^━" | \
    head -1 > "$TRACES_DIR/llama2_finetune_lora.json" 2>/dev/null || \
    echo '{"version":"0.6.2","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/llama2_finetune_lora.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/llama2_finetune_lora_summary.txt"

# ==============================================================================
# Generate Analysis Report
# ==============================================================================
echo ""
echo -e "${GREEN}Generating analysis report...${NC}"

cat > "$TRACES_DIR/ANALYSIS.md" << 'EOF'
# Golden Trace Analysis Report - entrenar

## Overview

This directory contains golden traces captured from entrenar (training & optimization library with autograd, LoRA, quantization) examples.

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
| `training_loop` | TBD | TBD | Basic autograd + optimizer (SGD) |
| `model_io` | TBD | TBD | Model save/load (JSON/YAML formats) |
| `llama2_finetune_lora` | TBD | TBD | LoRA fine-tuning (99.75% param reduction) |

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

Renacer Version: 0.6.2
entrenar Version: 0.1.0
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
echo "  1. Review traces: cat golden_traces/training_loop_summary.txt"
echo "  2. View JSON: jq . golden_traces/training_loop.json | less"
echo "  3. Run tests: cargo test --test golden_trace_validation"
echo "  4. Update baselines in ANALYSIS.md with actual metrics"
