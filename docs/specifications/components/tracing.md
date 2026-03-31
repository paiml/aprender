# Layer Tracing and Brick Profiling

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §4
**Crates**: `renacer` (tracing), `trueno` (BrickProfiler), `realizar` (integration)

---

## 1. Overview

The stack provides two complementary observability systems:

- **Layer Tracing** (renacer): State machine tracing of the inference/training
  pipeline — what each layer computes and whether tensor statistics are sane.
- **Brick Profiling** (trueno BrickProfiler): Microsecond-resolution timing
  of individual compute kernels — where time is spent.

Both are built into every `apr` command. Enable with `--trace` and `--profile`.

---

## 2. Layer Tracing — Inference State Machine

### 2.1 The 8 Trace Steps

Inference is modeled as a state machine with 8 traced transitions:

| Step | Description | Key Metric |
|------|-------------|------------|
| TOKENIZE | Text → token IDs | Token count |
| EMBED | Token IDs → vectors | Embedding stats |
| TRANSFORMER_BLOCK | Vectors → vectors (×N layers) | Per-layer stats |
| LM_HEAD | Hidden → logits | Logit distribution |
| SAMPLE | Logits → next token ID | Temperature, top-p |
| DECODE | Token ID → text | Token string |
| KERNEL_LAUNCH | GPU kernel dispatch | Launch latency |
| BRICK_PROFILE | Compute brick measurement | µs per brick |

### 2.2 TensorStats

Each traced step records:

```rust
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub nan_count: usize,
    pub inf_count: usize,
}
```

NaN/Inf detection triggers immediate Jidoka (stop-the-line) with diagnostic
output. This catches gradient explosion, dead ReLU, and quantization errors
at the exact layer they occur.

### 2.3 Trace Levels

| Level | Detail | Use Case |
|-------|--------|----------|
| `none` | No tracing | Production |
| `basic` | Step names + durations | Quick diagnosis |
| `layer` | Per-layer TensorStats | Numerical debugging |
| `payload` | Full tensor values | Deep investigation |

### 2.4 CLI Usage

```bash
# Basic: which steps are slow?
apr run model.apr --prompt "hello" --trace

# Layer: are tensor stats sane per layer?
apr run model.apr --prompt "hello" --trace-level layer

# Payload: what are the actual values?
apr run model.apr --prompt "hello" --trace-level payload

# Filter: only trace attention and sampling
apr run model.apr --prompt "hello" --trace-steps "attention,sample"

# Export: JSON for programmatic analysis
apr run model.apr --prompt "hello" --trace-output trace.json
```

---

## 3. BrickTracer — Automatic Anomaly Escalation

### 3.1 Design (renacer)

BrickTracer is a ComputeBrick-aware tracing module that automatically
**escalates** from lightweight measurement to deep syscall tracing when
performance anomalies are detected.

**Scientific basis**:
- Mace et al. (2015) "Pivot Tracing": always-on tracing degrades throughput
- Sigelman et al. (2010) "Dapper": rate-limited sampling for production
- Curtsinger & Berger (2013) "Stabilizer": CV > 15% = unstable measurement

### 3.2 Escalation Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Unstable timing | CV > 15% | Full SyscallBreakdown |
| Budget exceeded | Efficiency < 25% | Root cause analysis |
| Rate limit | 100 traces/sec | Dapper-style sampling |

### 3.3 SyscallBreakdown

When escalation triggers, BrickTracer categorizes time spent:

| Category | What It Measures |
|----------|-----------------|
| `mmap_us` | Memory allocation (model loading, KV cache growth) |
| `futex_us` | Thread synchronization (rayon contention) |
| `ioctl_us` | CUDA driver calls (kernel launch, memory copy) |
| `read_us` | Disk I/O (model streaming) |
| `write_us` | Output I/O (logging, export) |
| `compute_us` | Actual computation (total - all syscall overhead) |

Syscall overhead percentage pinpoints the real bottleneck — if `futex_us`
dominates, the problem is thread contention, not slow kernels.

### 3.4 TracedBrickResult

```rust
pub struct TracedBrickResult<T> {
    pub result: T,
    pub duration_us: u64,
    pub syscall_breakdown: SyscallBreakdown,
    pub efficiency_score: f32,
    pub over_budget: bool,
    pub escalation_reason: Option<EscalationReason>,
    pub otlp_span_id: Option<String>,
}
```

### 3.5 W3C Trace Context + OTLP

BrickTracer emits W3C Trace Context spans (`trace_id`, `span_id`,
`parent_span_id`) compatible with Jaeger/Tempo via OTLP export:

```bash
# Export traces to Jaeger
apr serve model.apr --otlp-endpoint http://localhost:4317
```

Uses `renacer-core` SpanRecord format (Parquet-compatible) with
LamportClock for causal ordering across distributed traces.

---

## 4. Brick Profiling — Per-Kernel Timing

### 4.1 BrickProfiler (trueno)

Instruments every compute kernel with microsecond-resolution timing:

```
Per-token decode breakdown (RTX 4090, Qwen 1.5B Q4K):

| Brick           | µs/token | % of Decode |
|-----------------|----------|-------------|
| AttentionScore  | 1,891    | 17.7%       |
| GateProjection  | 1,489    | 13.9%       |
| RmsNorm         | 1,434    | 13.4%       |
| FFN Up          | 1,200    | 11.2%       |
| KV Cache Write  | 890      | 8.3%        |
| ...             | ...      | ...         |
```

### 4.2 Profiling Modes

| Mode | Method | What It Measures |
|------|--------|-----------------|
| Immediate | `cudaDeviceSynchronize()` per brick | True GPU kernel time |
| Deferred | CPU-side launch latency only | Launch overhead |

**Contract C-GDP-001**: `valid_profiling => NOT has_decode_graph`.
CUDA graph replay cannot be profiled at brick granularity — profiling
forces the eager decode path.

### 4.3 CLI Usage

```bash
# Roofline analysis with per-brick breakdown
apr profile model.apr --granular

# Flamegraph output
apr profile model.apr --format flamegraph -o profile.svg

# Focus on specific compute area
apr profile model.apr --focus attention

# Energy measurement (requires RAPL)
apr profile model.apr --energy

# CI assertion mode
apr profile model.apr --ci --assert-throughput 30 --assert-p99 50ms

# Compare against baseline
apr profile model.apr --compare-hf --perf-grade

# Detect naive implementations
apr profile model.apr --detect-naive --threshold 2x
```

---

## 5. cbtop — Live ComputeBrick Monitor

`apr cbtop` is a top-like live monitor for the compute brick pipeline:

```bash
# Live monitoring with anomaly detection
apr cbtop model.apr

# Assert minimum efficiency
apr cbtop model.apr --brick-score --assert-efficiency 0.5

# Measure batch with BrickTracer
apr cbtop model.apr --measure-batch 100
```

When anomalies are detected (CV > 15%, efficiency < 25%), cbtop
auto-escalates to full BrickTracer with syscall breakdown.

---

## 6. Integration Matrix

| `apr` Command | Layer Trace | BrickTracer | BrickProfiler | Probar |
|---------------|------------|-------------|---------------|--------|
| `run` | `--trace` | auto-escalate | `--profile` | — |
| `chat` | `--trace` | auto-escalate | `--profile` | — |
| `serve` | OTLP export | auto-escalate | `--profile` | — |
| `bench` | optional | timing | `--profile` | — |
| `train` | `--trace` | backward pass | `--profile` | — |
| `finetune` | `--trace` | backward pass | `--profile` | — |
| `qa` | golden trace | syscall check | — | golden |
| `cbtop` | — | primary | live bricks | — |
| `profile` | — | — | primary | — |
| `probar` | snapshot | — | optional | primary |

---

## 7. Conditional Compilation

When the `visualization` feature is **disabled**, renacer integration
compiles to a no-op shim (`brick_tracer_shim`) that uses `Instant::now()`
for basic timing without syscall tracing or OTLP export. Zero overhead.

When **enabled**: full BrickTracer with OTLP, SyscallBreakdown,
auto-escalation, and Jaeger/Tempo integration.

```toml
[features]
visualization = ["renacer", "trueno-viz"]  # Full tracing
```
