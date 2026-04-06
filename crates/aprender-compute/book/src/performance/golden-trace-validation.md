# Golden Trace Validation

**Status**: Integrated (v0.7.0)
**Tool**: [Renacer](https://github.com/paiml/renacer) 0.6.2
**Purpose**: Performance regression detection via syscall tracing

---

## Overview

Golden trace validation uses **Renacer** (pure Rust syscall tracer) to capture canonical execution traces for Trueno compute examples. These traces serve as performance baselines, enabling:

1. **Regression Detection**: Detect performance degradation via syscall count/latency budgets
2. **PCIe Bottleneck Analysis**: Identify inefficient GPU memory transfers
3. **Build-Time Assertions**: Enforce performance contracts in CI/CD
4. **Root Cause Analysis**: Correlate syscalls to Rust source code

---

## Quick Start

### 1. Install Renacer

```bash
cargo install renacer --version 0.6.2
```

### 2. Capture Golden Traces

```bash
cd /path/to/trueno
./scripts/capture_golden_traces.sh
```

**Output:**
```
✅ Captured: golden_traces/backend_detection.json (0.73ms, 87 syscalls)
✅ Captured: golden_traces/matrix_operations.json (1.56ms, 168 syscalls)
✅ Captured: golden_traces/activation_functions.json (1.30ms, 159 syscalls)
✅ Captured: golden_traces/performance_demo.json (1.51ms, 138 syscalls)
✅ Captured: golden_traces/ml_similarity.json (0.82ms, 109 syscalls)
```

### 3. View Trace Summary

```bash
cat golden_traces/backend_detection_summary.txt
```

**Example Output:**
```
Syscall Summary:
write:     23 calls  (0.15ms total)
mmap:      13 calls  (0.21ms total)
mprotect:   6 calls  (0.08ms total)
munmap:     5 calls  (0.04ms total)
...
TOTAL:     87 calls  (0.73ms total)
```

---

## Traced Operations

### 1. Backend Detection (`backend_detection`)

**Purpose**: Validate SIMD backend auto-selection (AVX-512 → AVX2 → SSE2 → Scalar)

**Performance Budget:**
- Runtime: <10ms
- Syscalls: <100
- Memory: <10MB

**Actual Performance:** ✅
- Runtime: 0.73ms (13× faster than budget)
- Syscalls: 87
- Top syscalls: `write` (23), `mmap` (13), `mprotect` (6)

**Trace Capture:**
```bash
renacer --format json -- ./target/release/examples/backend_detection > backend_detection.json
```

---

### 2. Matrix Operations (`matrix_operations`)

**Purpose**: Measure SIMD-accelerated matrix multiply and transpose overhead

**Performance Budget:**
- Runtime: <20ms
- Syscalls: <200

**Actual Performance:** ✅
- Runtime: 1.56ms (15× faster)
- Syscalls: 168

**Key Insight**: SIMD operations are compute-bound (minimal syscalls)

---

### 3. ML Activation Functions (`activation_functions`)

**Purpose**: Measure SIMD-accelerated activations (ReLU, sigmoid, tanh, GELU, swish)

**Performance Budget:**
- Runtime: <20ms
- Syscalls: <200

**Actual Performance:** ✅
- Runtime: 1.30ms
- Syscalls: 159

---

### 4. Performance Demo (`performance_demo`)

**Purpose**: Comprehensive benchmark across vector ops, matrix ops, and backend comparisons

**Performance Budget:**
- Runtime: <50ms
- Syscalls: <300

**Actual Performance:** ✅
- Runtime: 1.51ms (33× faster)
- Syscalls: 138

---

### 5. ML Similarity (`ml_similarity`)

**Purpose**: Measure vector similarity operations (cosine, Euclidean, Manhattan)

**Performance Budget:**
- Runtime: <20ms
- Syscalls: <200

**Actual Performance:** ✅ **FASTEST**
- Runtime: 0.82ms
- Syscalls: 109

**Why Fast**: Heavily optimized SIMD dot product, minimal allocations

---

## Performance Assertions (`renacer.toml`)

### Critical Path Latency

```toml
[[assertion]]
name = "example_startup_latency"
type = "critical_path"
max_duration_ms = 100
fail_on_violation = true
enabled = true
```

**Rationale**: Compute examples should complete quickly. 100ms allows for SIMD initialization and small-scale computations.

**Violation Symptoms**:
- SIMD overhead issues
- Unexpected I/O operations
- Debug build instead of release

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

**Rationale**: SIMD operations are CPU-bound with minimal syscalls (mostly `mmap` for allocation). Budget prevents I/O regressions.

**Typical Syscalls**:
- `write`: stdout output (20-50 calls)
- `mmap`: vector/matrix allocation (10-30 calls)
- `mprotect`: memory permissions (5-10 calls)

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

**Pattern Detected**: GPU transfer time >> compute time

**Symptoms**:
- Many `write`/`read` syscalls to `/dev/nvidia*`
- High `ioctl` frequency for GPU operations
- Transfer overhead dominates (>70% of total time)

**Example Warning:**
```
⚠️  PCIe Bottleneck detected (confidence: 85%)
   GPU transfers: 45ms (90% of total time)
   Compute time:   5ms (10% of total time)
   Recommendation: Batch operations, keep data on GPU
```

**Solution**:
- Batch multiple operations
- Keep intermediate results on GPU
- Use larger workloads (amortize transfer costs)
- Trueno automatically disables GPU for small ops (v0.2.1+)

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
          renacer --assert renacer.toml -- ./target/release/examples/backend_detection
          renacer --assert renacer.toml -- ./target/release/examples/matrix_operations
          renacer --assert renacer.toml -- ./target/release/examples/activation_functions

      - name: Upload Traces
        uses: actions/upload-artifact@v3
        with:
          name: golden-traces
          path: golden_traces/
```

**CI Failure Example:**
```
❌ Assertion 'example_startup_latency' FAILED
   Actual: 125ms
   Budget: 100ms
   Regression: +25%

⚠️  Build BLOCKED. SIMD overhead regression detected.
```

---

## Advanced Usage

### 1. Source Code Correlation

Map syscalls to Rust source code:

```bash
renacer -s -- ./target/release/examples/backend_detection
```

**Output:**
```
write(1, "Backend: AVX2\n", 14) = 14  [src/lib.rs:245]
mmap(...) = 0x7f... [src/vector.rs:89]
```

**Use Case**: Identify which code paths trigger GPU initialization or excessive allocations.

---

### 2. OpenTelemetry Export

Visualize traces in Jaeger:

```bash
# Start Jaeger
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Export trace
renacer --otlp http://localhost:4317 -- ./target/release/examples/performance_demo

# View in Jaeger UI
open http://localhost:16686
```

**Use Case**: Visualize syscall timelines for multi-operation pipelines.

---

### 3. Regression Analysis

Compare current execution against baseline:

```bash
# Capture current trace
renacer --format json -- ./target/release/examples/backend_detection > current.json

# Compare with golden
diff <(jq '.syscalls | length' golden_traces/backend_detection.json) \
     <(jq '.syscalls | length' current.json)
```

**Expected**: No difference in syscall count (±5% tolerance)

---

### 4. GPU Workload Analysis

For GPU-enabled builds:

```bash
# Build with GPU feature
cargo build --release --examples --features gpu

# Trace GPU example
renacer --format json -- ./target/release/examples/gpu_test > gpu_trace.json

# Filter GPU device operations
jq '.syscalls[] | select(.name == "ioctl" or .name == "write")' gpu_trace.json
```

**Expected**: GPU operations show as `ioctl` calls to `/dev/nvidia0`

**Red Flag**: If transfer syscalls dominate, GPU is inefficient for this workload size.

---

## Toyota Way Principles

### Andon (Stop the Line)

**Implementation**: Build-time assertions fail CI on regression

```toml
[[assertion]]
fail_on_violation = true  # ← Andon: Stop the CI pipeline
```

---

### Poka-Yoke (Error-Proofing)

**Implementation**: Golden traces make expected patterns explicit

```bash
# Automated comparison prevents silent regressions
diff golden_traces/backend_detection.json new_trace.json
```

---

### Jidoka (Autonomation)

**Implementation**: Automated quality enforcement without manual intervention

```yaml
# GitHub Actions runs golden trace validation automatically
- name: Validate Performance
  run: ./scripts/capture_golden_traces.sh
```

---

## Troubleshooting

### Issue: Capture script fails with "Binary not found"

**Solution:**
```bash
cargo build --release --examples
./scripts/capture_golden_traces.sh
```

---

### Issue: Performance regression detected

**Diagnosis:**
```bash
renacer --summary --timing -- ./target/release/examples/backend_detection
cat golden_traces/backend_detection_summary.txt
```

**Common Causes:**
- Debug build instead of release (`cargo build --release`)
- SIMD features disabled (check `RUSTFLAGS`)
- New dependencies (increase initialization overhead)

---

### Issue: Syscall count regression

**Diagnosis:**
```bash
renacer -- ./target/release/examples/backend_detection > current_trace.txt
diff current_trace.txt golden_traces/backend_detection_summary.txt
```

**Common Causes:**
- New logging initialization (env_logger, tracing)
- Allocator changes (jemalloc → system allocator)
- Library updates (different I/O patterns)

---

## Performance Baselines (v0.7.0)

| Example | Runtime | Syscalls | Top Syscall | Status |
|---------|---------|----------|-------------|--------|
| `backend_detection` | 0.73ms | 87 | `write` (23) | ✅ |
| `matrix_operations` | 1.56ms | 168 | `write` (45) | ✅ |
| `activation_functions` | 1.30ms | 159 | `write` (38) | ✅ |
| `performance_demo` | 1.51ms | 138 | `mmap` (25) | ✅ |
| `ml_similarity` | 0.82ms | 109 | `write` (28) | ✅ **FASTEST** |

**Platform**: x86_64 Linux, AVX2 backend, Release build

---

## References

- [Renacer GitHub](https://github.com/paiml/renacer)
- [Integration Report](../../../docs/integration-report-golden-trace.md)
- [SIMD Performance Analysis](../../../docs/performance-analysis.md)
- [Golden Trace Analysis](../../../golden_traces/ANALYSIS.md)

---

**Last Updated**: 2025-11-23
**Renacer Version**: 0.6.2
**Trueno Version**: 0.7.0
