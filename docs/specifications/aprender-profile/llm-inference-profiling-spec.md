# LLM Inference Profiling: Realizar Integration Support

**Version:** 1.0
**Date:** 2025-12-16
**Status:** Specification - Ready for Implementation
**Sprint Target:** 51-53 (LLM Inference Profiling)
**Ecosystem:** Realizar + Trueno + Renacer
**Issue Reference:** PERF-PARITY-001 (Realizar)

## Executive Summary

This specification defines **LLM inference profiling** capabilities for Renacer to support **Realizar's** performance parity journey with Ollama/llama.cpp. Based on analysis of Realizar's v7.20.0 bottlenecks, Renacer will add specialized tracing for:

1. **Batch dispatch decision tracking** - CPU vs GPU routing decisions
2. **Attention/FFN component breakdown** - Per-layer timing attribution
3. **Memory bandwidth profiling** - H2D/D2H transfer overhead detection
4. **Throughput regression validation** - tok/s golden baseline comparison

**Business Value:**
- **Empirical Batch Threshold Discovery**: Find optimal CPU↔GPU crossover point (currently ~32 tokens)
- **Bottleneck Attribution**: Identify if attention (42-85%) or FFN (3-10%) dominates
- **Performance Regression Detection**: Catch tok/s regressions in CI/CD
- **Memory Transfer Visibility**: Detect when H2D/D2H transfers dominate kernel time

**Toyota Way Principle:**
> *"Genchi Genbutsu - Go and see for yourself."* Profile actual inference to find the truth, not assumptions.

---

## Table of Contents

1. [Realizar Bottleneck Analysis](#1-realizar-bottleneck-analysis)
2. [Proposed Renacer Features](#2-proposed-renacer-features)
3. [Sprint 51: Batch Dispatch Tracing](#3-sprint-51-batch-dispatch-tracing)
4. [Sprint 52: LLM Component Attribution](#4-sprint-52-llm-component-attribution)
5. [Sprint 53: Throughput Validation](#5-sprint-53-throughput-validation)
6. [Integration Architecture](#6-integration-architecture)
7. [Testing Strategy](#7-testing-strategy)

---

## 1. Realizar Bottleneck Analysis

### 1.1 Current Performance Gap (v7.20.0)

| Runtime | Throughput | Gap | Status |
|---------|------------|-----|--------|
| Ollama (CUDA) | 240 tok/s | 1.0x (baseline) | Reference |
| llama.cpp (CUDA) | 256 tok/s | ~1.0x | Parity |
| Realizar (GPU) | 64 tok/s | 3.75x | M3 Achieved |
| Realizar (CPU+KV) | 5.25 tok/s | 45.7x | Baseline |

**Target:** M4 Parity = 192 tok/s (<1.25x gap)

### 1.2 Identified Bottlenecks (from PARITY-044 to PARITY-048)

| Component | % of Total Time | GPU Beneficial? | Root Cause |
|-----------|-----------------|-----------------|------------|
| Attention | 42-85% | Yes (seq >= 32) | Per-head MATVEC overhead |
| LM Head | 10.6% | No | Vocabulary projection |
| FFN | 3.8% | No (batch=1) | GPU 2.7x slower for m=1 |
| LayerNorm | 0.5% | No | Too small |
| Embedding | 0.2% | No | Lookup only |

### 1.3 Critical Finding: CPU/GPU Crossover

From IMP-600 and PARITY-046b:

| Batch Size | CPU FFN | GPU FFN | GPU Speedup |
|------------|---------|---------|-------------|
| 1 | 36µs | 98µs | 0.37x (slower) |
| 30 | 1080µs | 1069µs | 1.0x (crossover) |
| 32 | 1152µs | 1069µs | 1.1x |
| 64 | 2304µs | 1069µs | 2.2x |
| 128 | 4608µs | 1069µs | 4.3x |

**Key Insight:** GPU only wins at batch >= 30-32. Need empirical profiling to validate.

### 1.4 What Realizar Needs from Renacer

1. **Batch dispatch decision log** - Which component went to CPU vs GPU and why
2. **Per-component timing** - Attention, FFN, LayerNorm, Embedding breakdown
3. **Memory transfer overhead** - H2D/D2H time vs compute time
4. **tok/s regression detection** - Golden baseline for throughput

---

## 2. Proposed Renacer Features

### 2.1 Feature Matrix

| Feature | Sprint | CLI Flag | Description |
|---------|--------|----------|-------------|
| Batch Dispatch Tracing | 51 | `--trace-dispatch` | Log CPU/GPU routing decisions |
| LLM Component Attribution | 52 | `--trace-llm-components` | Per-layer timing breakdown |
| Throughput Validation | 53 | `validate --throughput` | tok/s regression detection |

### 2.2 Integration with Existing Features

| Existing Feature | Integration Point |
|------------------|-------------------|
| `--trace-compute` (Sprint 32) | Component attribution builds on compute block tracing |
| `--otlp-endpoint` (Sprint 30) | All LLM traces export to OTLP |
| `validate` (Sprint 50) | Extend to support tok/s thresholds |
| CUDA tracing (Sprint 38) | Kernel timing for GPU components |

---

## 3. Sprint 51: Batch Dispatch Tracing

### 3.1 Goal

Track every CPU vs GPU dispatch decision with context:
- What operation (attention, FFN, embedding)?
- What batch size triggered the decision?
- What was the empirical threshold?

### 3.2 CLI Interface

```bash
# Enable dispatch decision tracing
renacer --trace-dispatch --otlp-endpoint http://localhost:4317 -- realizar serve

# Combine with existing tracing
renacer --trace-compute --trace-dispatch -- realizar serve
```

### 3.3 Output Schema

```rust
/// Dispatch decision record
#[derive(Debug, Clone, Serialize)]
pub struct DispatchDecision {
    /// Timestamp (nanoseconds since epoch)
    pub timestamp_ns: u64,
    /// Component name (e.g., "attention", "ffn", "embedding")
    pub component: String,
    /// Layer index (0-31 for typical LLM)
    pub layer: u32,
    /// Batch size that triggered decision
    pub batch_size: u32,
    /// Dispatch target
    pub target: DispatchTarget,
    /// Threshold used for decision
    pub threshold: u32,
    /// Actual duration (for feedback loop)
    pub duration_us: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub enum DispatchTarget {
    Cpu,
    Gpu,
    Hybrid,  // Some heads CPU, some GPU
}
```

### 3.4 OTLP Span Attributes

```
otel.span.name: "llm.dispatch"
llm.component: "ffn"
llm.layer: 15
llm.batch_size: 32
llm.dispatch.target: "gpu"
llm.dispatch.threshold: 32
llm.dispatch.duration_us: 1069
```

### 3.5 Expected Insights

1. **Threshold validation**: Confirm batch=32 is optimal crossover
2. **Layer-specific behavior**: Some layers may have different optimal thresholds
3. **Adaptive tuning**: Historical data enables dynamic threshold adjustment

---

## 4. Sprint 52: LLM Component Attribution

### 4.1 Goal

Break down inference time into components for bottleneck identification:

```
Token generation: 15.6ms total
├── Attention: 6.4ms (41%)
│   ├── Q projection: 0.8ms
│   ├── K projection: 0.8ms
│   ├── V projection: 0.8ms
│   ├── QK^T matmul: 2.0ms
│   ├── Softmax: 0.5ms
│   └── Attn @ V: 1.5ms
├── FFN: 4.8ms (31%)
│   ├── Up projection: 2.0ms
│   ├── GELU: 0.3ms
│   └── Down projection: 2.5ms
├── LayerNorm: 0.8ms (5%)
├── LM Head: 3.5ms (22%)
└── Other: 0.1ms (1%)
```

### 4.2 CLI Interface

```bash
# Enable component-level tracing
renacer --trace-llm-components -- realizar serve

# With OTLP export
renacer --trace-llm-components --otlp-endpoint http://localhost:4317 -- realizar serve
```

### 4.3 Integration Points

Realizar will instrument its forward pass with span markers:

```rust
// In realizar's transformer forward pass
renacer::span!("llm.attention", layer = layer_idx);
renacer::span!("llm.ffn", layer = layer_idx);
renacer::span!("llm.layernorm", layer = layer_idx);
```

Renacer will aggregate these into hierarchical timing:

```rust
/// LLM component timing summary
#[derive(Debug, Clone, Serialize)]
pub struct LlmComponentSummary {
    /// Total inference time (microseconds)
    pub total_us: u64,
    /// Per-component breakdown
    pub components: HashMap<String, ComponentTiming>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComponentTiming {
    /// Component name
    pub name: String,
    /// Total time across all layers
    pub total_us: u64,
    /// Percentage of total
    pub percent: f32,
    /// Per-layer breakdown
    pub per_layer: Vec<u64>,
}
```

### 4.4 Flamegraph Export

Enable flamegraph visualization of LLM components:

```bash
# Export component timing as JSON for flamegraph tools
renacer --trace-llm-components --format json -- realizar serve > profile.json

# Convert to flamegraph
cat profile.json | jq '.llm_components' | flamegraph.pl > llm.svg
```

---

## 5. Sprint 53: Throughput Validation

### 5.1 Goal

Extend `renacer validate` to support tok/s regression detection:

```bash
# Generate throughput baseline
renacer validate --generate ./golden --throughput -- realizar bench --model phi-2

# Validate against baseline
renacer validate --baseline ./golden --throughput --tolerance 5.0 -- realizar bench --model phi-2
```

### 5.2 Throughput Metrics

```rust
/// LLM throughput baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputBaseline {
    /// Tokens per second (mean)
    pub mean_tps: f64,
    /// Standard deviation
    pub std_tps: f64,
    /// Coefficient of variation (std/mean)
    pub cv: f64,
    /// Percentiles
    pub p50_tps: f64,
    pub p95_tps: f64,
    pub p99_tps: f64,
    /// Sample count
    pub samples: usize,
    /// Time to first token (milliseconds)
    pub ttft_ms: f64,
}
```

### 5.3 Validation Logic

```rust
/// Check for throughput regression
pub fn validate_throughput(
    baseline: &ThroughputBaseline,
    actual: &ThroughputBaseline,
    tolerance_percent: f32,
) -> ValidationResult {
    let delta = (actual.mean_tps - baseline.mean_tps) / baseline.mean_tps * 100.0;

    if delta < -tolerance_percent as f64 {
        ValidationResult::Regression {
            baseline_tps: baseline.mean_tps,
            actual_tps: actual.mean_tps,
            delta_percent: delta,
        }
    } else {
        ValidationResult::Passed
    }
}
```

### 5.4 CI/CD Integration

```yaml
# GitHub Actions example
- name: Throughput Regression Check
  run: |
    renacer validate --baseline golden/ --throughput --tolerance 5.0 \
      --output junit -- realizar bench --model phi-2 > results.xml

- name: Upload Results
  uses: actions/upload-artifact@v4
  with:
    name: throughput-results
    path: results.xml
```

### 5.5 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Throughput within tolerance |
| 1 | Throughput regression detected |
| 6 | Throughput measurement failed |
| 7 | Insufficient samples (CV too high) |

---

## 6. Backend Mismatch Detection (Critical)

### 6.1 The Problem

**Root cause found via renacer tracing (2025-12-16):**

```
wgpu error: Buffer size 524288000 is greater than the maximum buffer size (268435456)
524MB > 268MB wgpu limit
```

| Path | Throughput | Issue |
|------|------------|-------|
| CLI (f32 dequant) | 1.1 tok/s | 64% futex, parallelism helps |
| Server (Q4_K CPU) | 3-4 tok/s | CPU-bound matmul |
| **Server (GPU batch)** | **FAILS** | **wgpu buffer limit 256MB** |
| Ollama (CUDA) | 250 tok/s | Direct CUDA, no wgpu limits |

**Root cause:** Realizar was routing GPU requests through wgpu (WebGPU abstraction) instead of trueno-gpu's native CudaExecutor. wgpu has a hard 256MB buffer limit that prevents large batch operations.

### 6.2 Expected vs Actual Backend

```
EXPECTED (correct):
  GPU request → trueno-gpu CudaExecutor → cuBLAS GEMM → 250 tok/s

ACTUAL (wrong):
  GPU request → wgpu backend → 256MB limit → FAIL
```

### 6.3 Renacer Detection Feature

Add `--detect-backend-mismatch` to catch this class of bug:

```bash
renacer --detect-backend-mismatch --trace-dispatch -- realizar serve --batch
```

**Detection logic:**
```rust
/// Detect when wgpu is used for operations that should use CUDA
pub fn detect_backend_mismatch(traces: &[DispatchDecision]) -> Vec<BackendMismatch> {
    traces.iter().filter_map(|d| {
        // If CUDA is available but wgpu is being used for large buffers
        if d.cuda_available && d.backend == "wgpu" && d.buffer_size > WGPU_LIMIT {
            Some(BackendMismatch {
                component: d.component.clone(),
                expected: "cuda",
                actual: "wgpu",
                buffer_size: d.buffer_size,
                limit: WGPU_LIMIT,
            })
        } else {
            None
        }
    }).collect()
}

const WGPU_LIMIT: usize = 268_435_456; // 256MB
```

**OTLP attributes for mismatch detection:**
```
llm.dispatch.backend: "wgpu"
llm.dispatch.cuda_available: true
llm.dispatch.buffer_size: 524288000
llm.dispatch.buffer_limit: 268435456
llm.dispatch.mismatch: true
```

### 6.4 Fix in Realizar

The fix is to ensure GPU batch path uses CudaExecutor directly:

```rust
// WRONG - goes through wgpu abstraction
let scheduler = HybridScheduler::new(); // Defaults to wgpu

// RIGHT - use CUDA directly when available
let scheduler = if CudaExecutor::is_available() {
    HybridScheduler::with_cuda(CudaExecutor::new()?)
} else {
    HybridScheduler::with_wgpu(WgpuBackend::new()?)
};
```

---

## 7. Integration Architecture

### 6.1 Realizar + Renacer Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      REALIZAR INFERENCE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Embedding  │───▶│  Attention  │───▶│    FFN      │───▶ Output   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘              │
│         │                  │                  │                      │
│         ▼                  ▼                  ▼                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              RENACER INSTRUMENTATION LAYER                    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │   │
│  │  │  Dispatch  │  │ Component  │  │ Throughput │               │   │
│  │  │  Decision  │  │  Timing    │  │  Metrics   │               │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘               │   │
│  └────────┼───────────────┼───────────────┼──────────────────────┘   │
│           │               │               │                          │
│           ▼               ▼               ▼                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  RENACER OTLP EXPORTER                        │   │
│  │         (Jaeger / Tempo / Prometheus / JSON)                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 API for Realizar Integration

```rust
// In renacer crate - exported for realizar to use

/// Start a component timing span
pub fn start_llm_span(component: &str, layer: u32) -> SpanGuard;

/// Record a dispatch decision
pub fn record_dispatch_decision(decision: DispatchDecision);

/// Compute throughput from token generation
pub fn compute_throughput(tokens: usize, duration: Duration) -> f64;

/// Validate throughput against baseline
pub fn validate_throughput_baseline(
    baseline_path: &Path,
    actual: ThroughputBaseline,
    tolerance: f32,
) -> Result<ValidationResult>;
```

### 6.3 Feature Flags

```toml
# In renacer's Cargo.toml
[features]
default = []
llm-profiling = ["tracing", "serde"]
cuda-tracing = ["cudarc"]
otlp = ["opentelemetry", "opentelemetry-otlp"]

# In realizar's Cargo.toml
[dependencies]
renacer = { version = "0.8", features = ["llm-profiling", "otlp"] }
```

---

## 7. Testing Strategy

### 7.1 Unit Tests (Sprint 51)

```rust
#[test]
fn test_dispatch_decision_serialization() {
    let decision = DispatchDecision {
        timestamp_ns: 1234567890,
        component: "ffn".to_string(),
        layer: 15,
        batch_size: 32,
        target: DispatchTarget::Gpu,
        threshold: 32,
        duration_us: Some(1069),
    };
    let json = serde_json::to_string(&decision).unwrap();
    assert!(json.contains("\"target\":\"Gpu\""));
}

#[test]
fn test_dispatch_threshold_crossover() {
    // Below threshold -> CPU
    let decision = decide_dispatch("ffn", 16, 32);
    assert_eq!(decision.target, DispatchTarget::Cpu);

    // At threshold -> GPU
    let decision = decide_dispatch("ffn", 32, 32);
    assert_eq!(decision.target, DispatchTarget::Gpu);
}
```

### 7.2 Integration Tests (Sprint 52)

```rust
#[test]
fn test_component_timing_aggregation() {
    let spans = vec![
        LlmSpan { component: "attention", layer: 0, duration_us: 100 },
        LlmSpan { component: "attention", layer: 1, duration_us: 120 },
        LlmSpan { component: "ffn", layer: 0, duration_us: 50 },
        LlmSpan { component: "ffn", layer: 1, duration_us: 60 },
    ];

    let summary = aggregate_components(&spans);
    assert_eq!(summary.components["attention"].total_us, 220);
    assert_eq!(summary.components["ffn"].total_us, 110);
}
```

### 7.3 Throughput Validation Tests (Sprint 53)

```rust
#[test]
fn test_throughput_regression_detected() {
    let baseline = ThroughputBaseline { mean_tps: 100.0, ..Default::default() };
    let actual = ThroughputBaseline { mean_tps: 90.0, ..Default::default() }; // 10% slower

    let result = validate_throughput(&baseline, &actual, 5.0);
    assert!(matches!(result, ValidationResult::Regression { .. }));
}

#[test]
fn test_throughput_within_tolerance() {
    let baseline = ThroughputBaseline { mean_tps: 100.0, ..Default::default() };
    let actual = ThroughputBaseline { mean_tps: 97.0, ..Default::default() }; // 3% slower

    let result = validate_throughput(&baseline, &actual, 5.0);
    assert!(matches!(result, ValidationResult::Passed));
}
```

---

## 8. Success Criteria

### 8.1 Sprint 51 Deliverables
- [ ] `--trace-dispatch` CLI flag implemented
- [ ] DispatchDecision struct with OTLP export
- [ ] 10+ unit tests passing
- [ ] Documentation in book/src/advanced/

### 8.2 Sprint 52 Deliverables
- [ ] `--trace-llm-components` CLI flag implemented
- [ ] LlmComponentSummary aggregation
- [ ] Integration with `--format json` for flamegraph export
- [ ] 15+ unit tests passing

### 8.3 Sprint 53 Deliverables
- [ ] `validate --throughput` subcommand extension
- [ ] ThroughputBaseline generation and validation
- [ ] Exit codes 6-7 for throughput-specific failures
- [ ] CI/CD integration examples
- [ ] 10+ unit tests passing

### 8.4 Realizar Integration Milestone

**Acceptance Criteria:**
```bash
# Realizar can use renacer to profile inference
renacer --trace-llm-components --trace-dispatch \
  --otlp-endpoint http://localhost:4317 \
  -- realizar serve --model phi-2

# Realizar can validate throughput in CI
renacer validate --baseline golden/ --throughput --tolerance 5.0 \
  -- realizar bench --model phi-2

# Expected: Identify that batch=32 is optimal GPU crossover empirically
```

---

## 9. References

1. Realizar Performance Parity Spec v7.20.0 - PERF-PARITY-001
2. IMP-600: GPU MATVEC vs GEMM analysis
3. PARITY-044 to PARITY-048: Single-token optimization ceiling
4. trueno-gpu Phase 8: CUDA runtime implementation
5. Renacer Sprint 50: Golden trace validation
