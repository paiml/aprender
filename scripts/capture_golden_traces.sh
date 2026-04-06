#!/bin/bash
# Golden Trace Capture Script for batuta
#
# Captures syscall traces for batuta (orchestration framework) examples using Renacer.
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
    cargo install renacer --version 0.6.5
fi

# Build examples
echo -e "${YELLOW}Building release examples...${NC}"
cargo build --release --example backend_selection --example pipeline_demo --example full_transpilation --example oracle_demo

# Create traces directory
mkdir -p "$TRACES_DIR"

echo -e "${BLUE}=== Capturing Golden Traces for batuta ===${NC}"
echo -e "Examples: ./target/release/examples/"
echo -e "Output: $TRACES_DIR/"
echo ""

# ==============================================================================
# Trace 1: backend_selection (GPU/SIMD selection logic)
# ==============================================================================
echo -e "${GREEN}[1/4]${NC} Capturing: backend_selection"
BINARY_PATH="./target/release/examples/backend_selection"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🎯\\|^Based\\|^Example\\|^  \\|^Rationale\\|^Data\\|^FLOPs\\|^Compute" | \
    head -1 > "$TRACES_DIR/backend_selection.json" 2>/dev/null || \
    echo '{"version":"0.6.5","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/backend_selection.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/backend_selection_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🎯\\|^Based\\|^Example\\|^  \\|^Rationale\\|^Data\\|^FLOPs\\|^Compute" | \
    head -1 > "$TRACES_DIR/backend_selection_source.json" 2>/dev/null || \
    echo '{"version":"0.6.5","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/backend_selection_source.json"

# ==============================================================================
# Trace 2: full_transpilation (transpilation pipeline)
# ==============================================================================
echo -e "${GREEN}[2/4]${NC} Capturing: full_transpilation"
BINARY_PATH="./target/release/examples/full_transpilation"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🔧\\|^📋\\|^   \\|^     \\|^──\\|^  ✅\\|^📊\\|^🎯\\|^─" | \
    head -1 > "$TRACES_DIR/full_transpilation.json" 2>/dev/null || \
    echo '{"version":"0.6.5","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/full_transpilation.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/full_transpilation_summary.txt"

# ==============================================================================
# Trace 3: pipeline_demo (pipeline execution)
# ==============================================================================
echo -e "${GREEN}[3/4]${NC} Capturing: pipeline_demo"
BINARY_PATH="./target/release/examples/pipeline_demo"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🚀\\|^Based\\|^Input\\|^Output\\|^Running\\|^❌" | \
    head -1 > "$TRACES_DIR/pipeline_demo.json" 2>/dev/null || \
    echo '{"version":"0.6.5","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/pipeline_demo.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/pipeline_demo_summary.txt"

# ==============================================================================
# Trace 4: oracle_demo (Oracle Mode intelligent query interface)
# ==============================================================================
echo -e "${GREEN}[4/4]${NC} Capturing: oracle_demo"
BINARY_PATH="./target/release/examples/oracle_demo"

renacer --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🔮\\|^━\\|^📚\\|^🔍\\|^🧠\\|^📝\\|^🎯\\|^💡\\|^🔗\\|^💻\\|^✅\\|^  " | \
    head -1 > "$TRACES_DIR/oracle_demo.json" 2>/dev/null || \
    echo '{"version":"0.6.5","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/oracle_demo.json"

renacer --summary --timing -- "$BINARY_PATH" 2>&1 | \
    tail -n +2 > "$TRACES_DIR/oracle_demo_summary.txt"

renacer -s --format json -- "$BINARY_PATH" 2>&1 | \
    grep -v "^🔮\\|^━\\|^📚\\|^🔍\\|^🧠\\|^📝\\|^🎯\\|^💡\\|^🔗\\|^💻\\|^✅\\|^  " | \
    head -1 > "$TRACES_DIR/oracle_demo_source.json" 2>/dev/null || \
    echo '{"version":"0.6.5","format":"renacer-json-v1","syscalls":[]}' > "$TRACES_DIR/oracle_demo_source.json"

# ==============================================================================
# Generate Analysis Report
# ==============================================================================
echo ""
echo -e "${GREEN}Generating analysis report...${NC}"

cat > "$TRACES_DIR/ANALYSIS.md" << 'EOF'
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
echo "  1. Review traces: cat golden_traces/backend_selection_summary.txt"
echo "  2. View JSON: jq . golden_traces/backend_selection.json | less"
echo "  3. Run tests: cargo test --test golden_trace_validation"
echo "  4. Update baselines in ANALYSIS.md with actual metrics"
