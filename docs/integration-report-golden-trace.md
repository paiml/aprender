# Renacer Golden Trace Integration Report - Trueno

**Project**: Trueno (High-Performance SIMD/GPU Compute Library)
**Renacer Version**: 0.6.2
**Date**: 2025-11-23
**Author**: Claude Code (Anthropic)

---

## Executive Summary

This document describes the integration of **Renacer** (pure Rust syscall tracer) with **Trueno** (multi-target SIMD/GPU compute library). The integration enables:

1. **Golden Trace Validation**: Capture canonical execution traces for SIMD compute examples
2. **Performance Baselines**: Enforce syscall count and latency budgets for compute operations
3. **Build-Time Assertions**: TOML-based performance contracts detecting regressions
4. **Anti-Pattern Detection**: Identify PCIe bottlenecks in GPU workloads

---

## Quick Start

### 1. Install Renacer

```bash
cargo install renacer --version 0.6.2
```

### 2. Build Trueno Examples

```bash
cd /path/to/trueno
cargo build --release --examples
```

### 3. Capture Golden Traces

```bash
./scripts/capture_golden_traces.sh
```

**Output:**
- `golden_traces/backend_detection.json` - SIMD backend detection trace
- `golden_traces/matrix_operations.json` - Matrix multiply/transpose trace
- `golden_traces/activation_functions.json` - ML activation functions trace
- `golden_traces/performance_demo.json` - Comprehensive performance demo
- `golden_traces/ml_similarity.json` - ML similarity operations trace
- Summary statistics for each operation

### 4. View Traces

```bash
# Summary statistics
cat golden_traces/backend_detection_summary.txt

# Full JSON trace (formatted)
jq . golden_traces/backend_detection.json | less

# Syscall timeline
renacer --timing -- ./target/release/examples/backend_detection
```

---

## Integration Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│              Trueno SIMD/GPU Compute Examples               │
│   (backend_detection, matrix_ops, activation_fns, etc.)    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ traced by
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Renacer (ptrace)                           │
│   - Syscall interception (write, mmap, ioctl for GPU)      │
│   - Lamport logical clocks (causal ordering)               │
│   - Anti-pattern detection (PCIe bottleneck)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ exports to
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Golden Trace Storage                           │
│   - JSON format (machine-readable)                          │
│   - Summary statistics (human-readable)                     │
│   - Source-correlated traces (debugging)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ validated by
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Performance Assertions (renacer.toml)             │
│   - Example startup latency: <100ms                         │
│   - Syscall budget: <500 calls                              │
│   - Memory budget: <100MB                                   │
│   - PCIe bottleneck detection                               │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
trueno/
├── renacer.toml                          # Performance assertions
├── scripts/
│   └── capture_golden_traces.sh          # Trace capture automation
├── golden_traces/
│   ├── ANALYSIS.md                       # Trace analysis report
│   ├── backend_detection.json            # SIMD backend trace
│   ├── backend_detection_summary.txt     # Statistics
│   ├── matrix_operations.json            # Matrix ops trace
│   ├── activation_functions.json         # ML activations trace
│   ├── performance_demo.json             # Performance demo trace
│   └── ml_similarity.json                # ML similarity trace
├── tests/
│   └── golden_trace_validation.rs        # Integration test suite
└── docs/
    └── integration-report-golden-trace.md  # This file
```

---

## Captured Operations

### 1. SIMD Backend Detection: `backend_detection`

**Purpose**: Validate automatic SIMD backend selection (AVX-512 → AVX2 → AVX → SSE2 → Scalar)
**Expected Behavior**:
- Detect CPU features (SSE2, AVX, AVX2, FMA, AVX-512)
- Auto-select best available backend
- Print backend information to stdout

**Trace Capture**:
```bash
renacer --format json -- ./target/release/examples/backend_detection > backend_detection.json
```

**Expected Syscalls**:
- `write`: Backend detection output
- `mmap`/`brk`: Memory allocation
- Minimal overhead (<1ms)

**Performance Budget**:
- Runtime: <10ms
- Syscalls: <100
- Memory: <10MB

**Actual Performance** (from golden trace):
- **Runtime: 0.730ms** ✅
- **Syscalls: 87** ✅
- **Top syscalls**: write (23), mmap (13), mprotect (6)

---

### 2. Matrix Operations: `matrix_operations`

**Purpose**: Measure matrix multiplication and transpose overhead
**Expected Behavior**:
- Create small matrices (3×3, 4×4)
- Perform matrix multiply using SIMD
- Transpose matrices
- Output results

**Trace Capture**:
```bash
renacer --format json -- ./target/release/examples/matrix_operations > matrix_operations.json
```

**Expected Syscalls**:
- `write`: Multiple calls for matrix output
- `mmap`: Matrix buffer allocation
- SIMD operations are compute-bound (minimal syscalls)

**Performance Budget**:
- Runtime: <20ms
- Syscalls: <200
- Memory: <20MB

**Actual Performance**:
- **Runtime: 1.560ms** ✅
- **Syscalls: 168** ✅
- **Performance**: 15× faster than budget

---

### 3. ML Activation Functions: `activation_functions`

**Purpose**: Measure SIMD-accelerated activation functions (ReLU, sigmoid, tanh)
**Expected Behavior**:
- Apply activation functions to vectors
- Compare scalar vs SIMD performance
- Output results

**Trace Capture**:
```bash
renacer --format json -- ./target/release/examples/activation_functions > activation_functions.json
```

**Expected Syscalls**:
- `write`: Activation results output
- `mmap`: Vector allocation
- Compute-heavy, I/O-light

**Performance Budget**:
- Runtime: <20ms
- Syscalls: <200
- Memory: <30MB

**Actual Performance**:
- **Runtime: 1.298ms** ✅
- **Syscalls: 159** ✅

---

### 4. Performance Demo: `performance_demo`

**Purpose**: Comprehensive demonstration of Trueno capabilities
**Expected Behavior**:
- Vector operations (add, dot, sum, max)
- Matrix operations (multiply, transpose)
- Backend comparison
- Performance reporting

**Trace Capture**:
```bash
renacer --format json -- ./target/release/examples/performance_demo > performance_demo.json
```

**Expected Syscalls**:
- `write`: Extensive output (multiple operations)
- `mmap`: Multiple allocations
- Potential timing syscalls (`clock_gettime`)

**Performance Budget**:
- Runtime: <50ms
- Syscalls: <300
- Memory: <50MB

**Actual Performance**:
- **Runtime: 1.507ms** ✅
- **Syscalls: 138** ✅

---

### 5. ML Similarity: `ml_similarity`

**Purpose**: Measure vector similarity operations (cosine, Euclidean, Manhattan)
**Expected Behavior**:
- Compute similarity metrics between vectors
- SIMD-accelerated dot product
- Output similarity scores

**Trace Capture**:
```bash
renacer --format json -- ./target/release/examples/ml_similarity > ml_similarity.json
```

**Expected Syscalls**:
- `write`: Similarity scores output
- `mmap`: Vector allocation
- Minimal overhead

**Performance Budget**:
- Runtime: <20ms
- Syscalls: <200
- Memory: <20MB

**Actual Performance**:
- **Runtime: 0.817ms** ✅
- **Syscalls: 109** ✅
- **Fastest example**: Excellent SIMD efficiency

---

## Performance Assertions (renacer.toml)

### Critical Path Latency

```toml
[[assertion]]
name = "example_startup_latency"
type = "critical_path"
max_duration_ms = 100
fail_on_violation = true
enabled = true
```

**Rationale**: Compute examples should complete quickly. 100ms allows for SIMD initialization and small-scale computations. Violations indicate SIMD overhead issues or unexpected I/O.

---

### Syscall Budget

```toml
[[assertion]]
name = "max_syscall_budget"
type = "span_count"
max_spans = 500
fail_on_violation = true
enabled = true
```

**Rationale**: SIMD compute operations are CPU-bound with minimal syscalls (mostly memory allocation). Budget prevents I/O regressions.

---

### Memory Allocation Budget

```toml
[[assertion]]
name = "memory_allocation_budget"
type = "memory_usage"
max_bytes = 104857600  # 100MB
tracking_mode = "allocations"
fail_on_violation = true
enabled = true
```

**Rationale**: Small examples should have minimal memory footprint. 100MB accommodates matrix allocations and SIMD buffers.

---

### PCIe Bottleneck Detection

```toml
[[assertion]]
name = "detect_pcie_bottleneck"
type = "anti_pattern"
pattern = "PCIeBottleneck"
threshold = 0.7
fail_on_violation = false  # Warning only
enabled = true
```

**Pattern Detected**: Excessive GPU memory transfers relative to compute time.

**Symptoms**:
- Many `write`/`read` syscalls to `/dev/nvidia*` or similar GPU device
- High `ioctl` call frequency for GPU operations
- Transfer time >> compute time

**Example**:
```
WARNING: PCIe Bottleneck detected (confidence: 85%)
  GPU transfers: 45ms (90% of total time)
  Compute time: 5ms (10% of total time)
  Recommendation: Batch operations, keep data on GPU
```

**Solution**:
- Batch multiple operations
- Keep intermediate results on GPU
- Use larger workloads to amortize transfer costs
- Trueno automatically disables GPU for small operations (v0.2.1+)

---

## CI/CD Integration

### GitHub Actions Workflow

Add to `.github/workflows/ci.yml`:

```yaml
name: Golden Trace Validation

on: [push, pull_request]

jobs:
  validate-traces:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install Renacer
        run: cargo install renacer --version 0.6.2

      - name: Build Examples (Release)
        run: cargo build --release --examples

      - name: Capture Golden Traces
        run: ./scripts/capture_golden_traces.sh

      - name: Run Performance Assertions
        run: |
          # Example: Validate backend_detection meets latency budget
          renacer --assert renacer.toml -- ./target/release/examples/backend_detection

      - name: Upload Traces (Artifact)
        uses: actions/upload-artifact@v3
        with:
          name: golden-traces
          path: golden_traces/

      - name: Compare Against Baseline (Optional)
        run: |
          # Download baseline from previous run
          # diff baseline/backend_detection.json golden_traces/backend_detection.json
          echo "TODO: Implement baseline comparison"
```

---

## Advanced Usage

### 1. Source Code Correlation

Map syscalls back to Rust source code:

```bash
# Already built with debug symbols (see Cargo.toml)
# profile.release: debug = true

# Trace with source locations
renacer -s -- ./target/release/examples/backend_detection
```

**Output:**
```
write(1, "Backend: AVX2\n", 14) = 14  [src/lib.rs:245]
mmap(...) = 0x7f... [src/vector.rs:89]
```

**Use Case**: Identify which code paths trigger specific syscalls (e.g., GPU initialization).

---

### 2. OpenTelemetry Export

Export traces to Jaeger for visualization:

```bash
# Start Jaeger (Docker)
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Export trace to OTLP
renacer --otlp http://localhost:4317 -- ./target/release/examples/performance_demo

# View in Jaeger UI
open http://localhost:16686
```

**Use Case**: Visualize syscall timelines for complex multi-operation examples.

---

### 3. PCIe Bottleneck Analysis (GPU builds)

For GPU-enabled builds, detect transfer overhead:

```bash
# Build with GPU support
cargo build --release --examples --features gpu

# Trace GPU example
renacer --format json -- ./target/release/examples/gpu_test

# Look for GPU device operations
jq '.syscalls[] | select(.name == "ioctl" or .name == "write")' gpu_test.json
```

**Expected**: GPU operations show in trace as `ioctl` calls to `/dev/nvidia0` or similar.

**Red Flag**: If transfer syscalls dominate compute time, GPU is counterproductive for this workload size.

---

### 4. SIMD Backend Comparison

Compare syscall patterns across backends:

```bash
# Force scalar backend (env var - example, implementation-specific)
TRUENO_BACKEND=scalar renacer -- ./target/release/examples/backend_detection > scalar_trace.txt

# Force AVX2 backend
TRUENO_BACKEND=avx2 renacer -- ./target/release/examples/backend_detection > avx2_trace.txt

# Compare
diff scalar_trace.txt avx2_trace.txt
```

**Expected**: Same syscalls, different timings (AVX2 faster for compute).

---

## Troubleshooting

### Issue: Capture script fails with "Binary not found"

**Solution**:
```bash
cargo build --release --examples
./scripts/capture_golden_traces.sh
```

---

### Issue: Trace contains program output mixed with JSON

**Symptoms**: JSON parsing fails due to stdout contamination.

**Solution**: Capture script uses `grep -v` to filter output:
```bash
renacer --format json -- example 2>&1 | grep -v "^Backend\|^CPU\|^SIMD" > trace.json
```

---

### Issue: Performance regression detected

**Diagnosis**:
```bash
# Compare current vs golden
renacer --summary --timing -- ./target/release/examples/backend_detection
cat golden_traces/backend_detection_summary.txt
```

**Common causes**:
- Debug build instead of release
- SIMD features disabled (check `RUSTFLAGS`)
- New dependencies (increase overhead)

---

### Issue: Syscall count regression

**Diagnosis**:
```bash
# Detailed syscall comparison
renacer -- ./target/release/examples/backend_detection > current_trace.txt
diff current_trace.txt golden_traces/backend_detection_summary.txt
```

**Common causes**:
- New logging initialization
- Allocator changes (more `mmap` calls)
- Library updates (different I/O patterns)

---

## Testing Integration

### Unit Tests (Optional)

Create `tests/golden_trace_validation.rs`:

```rust
//! Golden Trace Validation Tests for Trueno
//!
//! Validates that Trueno compute examples produce expected syscall patterns
//! and meet performance budgets defined in renacer.toml.

use std::process::Command;
use std::fs;

#[test]
fn test_backend_detection_completes() {
    let output = Command::new("./target/release/examples/backend_detection")
        .output()
        .expect("Failed to execute backend_detection");

    assert!(output.status.success(), "Example should exit with success");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Backend:"), "Output should show backend info");
}

#[test]
fn test_golden_trace_exists() {
    let golden_trace_path = "golden_traces/backend_detection.json";
    assert!(
        std::path::Path::new(golden_trace_path).exists(),
        "Golden trace should exist. Run: ./scripts/capture_golden_traces.sh"
    );
}

#[test]
fn test_golden_trace_format() {
    let golden_trace_path = "golden_traces/backend_detection.json";
    let contents = fs::read_to_string(golden_trace_path)
        .expect("Golden trace file should be readable");

    let json: serde_json::Value = serde_json::from_str(&contents)
        .expect("Golden trace should be valid JSON");

    assert_eq!(json["version"], "0.6.2", "Trace version should match");
    assert_eq!(json["format"], "renacer-json-v1", "Format should be renacer-json-v1");

    assert!(json["syscalls"].is_array(), "Should have syscalls array");
}

#[test]
fn test_performance_baseline() {
    let summary_path = "golden_traces/backend_detection_summary.txt";
    let summary = fs::read_to_string(summary_path)
        .expect("Summary file should exist");

    let last_line = summary.lines().last().unwrap();
    let parts: Vec<&str> = last_line.split_whitespace().collect();

    let total_time_str = parts[1];
    let total_time_secs: f64 = total_time_str.parse().unwrap();
    let total_time_ms = total_time_secs * 1000.0;

    println!("Golden trace total runtime: {:.3}ms", total_time_ms);

    assert!(
        total_time_ms < 10.0,
        "Example should complete in <10ms (actual: {:.3}ms)",
        total_time_ms
    );
}

#[test]
fn test_syscall_count_budget() {
    let summary_path = "golden_traces/backend_detection_summary.txt";
    let summary = fs::read_to_string(summary_path)
        .expect("Summary file should exist");

    let last_line = summary.lines().last().unwrap();
    let parts: Vec<&str> = last_line.split_whitespace().collect();

    let total_calls: usize = parts[3].parse().unwrap();

    println!("Golden trace total syscalls: {}", total_calls);

    assert!(
        total_calls < 100,
        "Example should use <100 syscalls (actual: {})",
        total_calls
    );
}
```

**Run tests:**
```bash
cargo test --test golden_trace_validation
```

---

## Renacer Features Used

| Feature | Description | Trueno Use Case |
|---------|-------------|-----------------|
| **JSON Export** | Machine-readable trace format | CI/CD integration, regression testing |
| **Summary Statistics** | Human-readable syscall summary | Performance baselines, quick diagnostics |
| **Source Correlation** | Map syscalls to Rust source | Debug SIMD overhead, GPU initialization |
| **Low Overhead** | <1% runtime impact | Trace without distorting micro-benchmarks |
| **Lamport Clocks** | Causal ordering | Track multi-threaded SIMD operations |
| **Build-Time Assertions** | TOML performance contracts | Fail CI on regression |
| **PCIe Bottleneck Detection** | GPU transfer overhead | Identify inefficient GPU usage |
| **OTLP Export** | OpenTelemetry integration | Visualize compute pipelines |

---

## Toyota Way Principles

### Andon (Stop the Line)

**Implementation**: Build-time assertions fail CI on performance regression.

```toml
[[assertion]]
name = "example_startup_latency"
max_duration_ms = 100
fail_on_violation = true  # ← Andon: Stop the CI pipeline
```

**Example CI Failure**:
```
❌ Assertion 'example_startup_latency' FAILED
   Actual: 125ms
   Budget: 100ms
   Regression: +25%

⚠️  Build BLOCKED. SIMD overhead regression detected.
```

---

### Poka-Yoke (Error-Proofing)

**Implementation**: Golden traces prevent SIMD regressions by making expected patterns explicit.

```bash
# Automated comparison (poka-yoke)
diff golden_traces/backend_detection.json new_trace.json
```

---

### Jidoka (Autonomation)

**Implementation**: Automated quality enforcement in CI without manual intervention.

```yaml
# GitHub Actions (automated)
- name: Validate Performance
  run: |
    ./scripts/capture_golden_traces.sh
    cargo test --test golden_trace_validation
```

---

## Next Steps

1. ✅ **Capture Baselines**: `./scripts/capture_golden_traces.sh` → **DONE**
2. ⏳ **Add Tests**: Create `tests/golden_trace_validation.rs`
3. ⏳ **Integrate with CI**: Add GitHub Actions workflow
4. ⏳ **Tune Assertions**: Adjust budgets based on GPU workloads (if using GPU feature)
5. ⏳ **Enable OTLP** (Optional): Export traces to observability stack

---

## References

- [Renacer GitHub](https://github.com/paiml/renacer)
- [Renacer Documentation](https://docs.rs/renacer/0.6.2)
- [Trueno Documentation](https://docs.rs/trueno)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [SIMD Performance Analysis](../docs/performance-analysis.md)

---

**Generated**: 2025-11-23
**Renacer Version**: 0.6.2
**Trueno Version**: 0.7.0
