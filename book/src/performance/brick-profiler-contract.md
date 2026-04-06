# BrickProfiler Contract Compliance

> **Contract**: `gpu-decode-profiling-v1` v2.0.0
>
> See: `src/brick/profiler/mod.rs` — BrickProfiler, BrickStats, SyncMode

## Overview

The BrickProfiler is trueno's per-kernel instrumentation system. It records timing for each ComputeBrick (GEMV, RmsNorm, Attention, etc.) during model inference. Downstream consumers (realizar, aprender, apr-cli) depend on the profiler's data being **correct and complete**.

The `gpu-decode-profiling-v1` contract formalizes 15 invariants across the entire pipeline from trueno's `BrickStats` to the final JSON report.

## trueno's Responsibilities

trueno owns the **data collection** layer. The contract requires:

### 1. BrickStats Accuracy

Each `BrickStats` must report:

- `name` — Brick identifier (e.g., "LmHead", "GateProjection")
- `count` — Number of `start/stop` pairs recorded
- `total_ns` — Cumulative nanoseconds across all calls
- `avg_us()` — `total_ns / count / 1000.0` (per-call average in microseconds)

The `avg_us()` method is the **canonical value** for per-call timing. Downstream must use this, not derive their own average from `total_ns`.

### 2. SyncMode Propagation

```rust
pub enum SyncMode {
    Immediate,  // cudaDeviceSynchronize() after each brick
    Deferred,   // No sync — measures CPU launch latency only
}
```

The contract requires that `set_sync_mode(Immediate)` causes a device sync between every `start_brick_id()` / `stop_brick_id()` pair. Without this, timing reflects CPU-side launch latency (~50-100µs per brick) rather than actual GPU execution.

**Diagnostic invariant**: In Immediate mode, `LmHead.avg_us > 10 * RmsNorm.avg_us` (vocab GEMV is far more expensive than elementwise norm). In Deferred mode, all bricks cluster near the same value.

### 3. Token Accounting

`profiler.total_tokens` counts **brick elements** — the total number of `start/stop` pairs across all bricks. For a 28-layer Qwen model generating 32 tokens:

```
total_tokens = 32 tokens × (28 layers × ~10 bricks/layer + 1 LmHead) ≈ 9,000
```

This is NOT the number of decoded tokens. The contract explicitly forbids using `total_tokens` as a denominator for per-token metrics. Instead, `LmHead.count` equals the decoded token count (exactly 1 LmHead call per decoded token).

### 4. all\_brick\_stats() Completeness

The `all_brick_stats()` iterator must yield **every brick** that has `count > 0`. No brick may be silently omitted. The downstream JSON report must contain exactly `len(all_brick_stats())` entries.

## CUDA Graph Interaction

CUDA graphs capture kernel sequences for replay. When a graph is replayed, bricks see only the **capture-pass timing** (1 token), not the actual per-token decode time.

The contract (`C-GDP-001`) requires that consumers disable graph replay when profiling is enabled. This is enforced in realizar's `CudaExecutor`:

```rust
// forward_graphed_decode.rs
if self.should_use_eager_decode() {
    // profiler.is_enabled() → true → eager path
    return self.forward_all_layers_gpu_to_logits(...);
}
```

trueno's role: `is_enabled()` must accurately reflect whether profiling is active, so realizarcan make the correct dispatch decision.

## Verified Brick Ordering (RTX 4090)

With Immediate sync, per-call averages must respect kernel complexity:

```
LmHead (594µs) > GateProjection (53µs) > RmsNorm (25µs)
```

This ordering is a monotonicity proof obligation in the contract. Violations indicate wrong sync mode or a kernel regression.

## Related

- [Profiling](./profiling.md) — General profiling guide
- [Model-Level Inference Tracing](./model-tracing.md) — End-to-end trace
- [cbtop (Compute Block Top)](../ecosystem/cbtop.md) — TUI consumer of BrickProfiler
