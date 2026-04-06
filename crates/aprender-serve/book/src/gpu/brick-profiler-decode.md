# BrickProfiler Decode Profiling

> **Status**: Implemented (C-GDP-001)
>
> See: `src/cuda/executor/layers/forward_graphed_decode.rs` — eager path bypass
> See: `src/gguf/cuda/uses.rs` — profiling-aware graph dispatch

## Overview

The BrickProfiler (from trueno) instruments every kernel in the decode pipeline with `start_brick_id()` / `stop_brick_id()` calls, producing per-brick timing at microsecond resolution. This chapter documents the critical interaction between profiling and CUDA graph replay, and the contract that enforces correctness.

## The Graph-Profiling Conflict

CUDA graphs capture a sequence of kernel launches and replay them as a single opaque operation. This is excellent for performance (~500x reduction in launch overhead) but **incompatible with per-brick profiling**:

```
Without graphs (eager path):
  [RmsNorm] → [QkvProj] → [Attn] → [RmsNorm] → [Gate] → [Down] → ... → [LmHead]
   ▲ start    ▲ start    ▲ start   (each brick independently timed)
   ▼ stop     ▼ stop     ▼ stop

With graphs (replay path):
  [============= Single Graph Launch =============]
   ▲ start (only during capture, token 0)
   ▼ stop  (only during capture, token 0)
   Tokens 1..N: bricks see ZERO time (graph replay is opaque)
```

If profiling runs during graph replay, the profiler records timing from the **capture pass only** (1 token), producing values that are 1/N of the real decode time.

## Contract: C-GDP-001

The `gpu-decode-profiling-v1` contract (v2.0.0) enforces:

```
valid_profiling => NOT has_decode_graph
```

### Implementation

In `forward_graphed_decode.rs`, the `should_use_eager_decode()` method checks:

```rust
fn should_use_eager_decode(&self) -> bool {
    static GRAPH_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let graph_disabled = *GRAPH_DISABLED.get_or_init(|| {
        std::env::var("CUDA_GRAPH_DISABLE")
            .map(|v| v == "1")
            .unwrap_or(false)
    });
    let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
        .map(|v| v == "1")
        .unwrap_or(false);
    graph_disabled || skip_graph || self.profiler.is_enabled()
}
```

When `profiler.is_enabled()` returns true, the entire graphed path is bypassed — both capture and replay. Every decode token goes through the eager path where each brick is individually instrumented.

In `uses.rs`, the `OwnedQuantizedModelCuda` decode method adds a parallel check:

```rust
let profiling_enabled = self.executor.is_profiling_enabled();
if profiling_enabled || !self.executor.has_decode_graph() {
    // Eager path — bricks instrumented per-token
}
```

## Sync Modes

The profiler supports two sync modes that control when GPU timing is captured:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Immediate** | `cudaDeviceSynchronize()` after each brick | Real per-kernel GPU time |
| **Deferred** | No sync between bricks | CPU launch latency only |

For valid profiling, **Immediate** mode is required. In Deferred mode, all bricks show ~50-100µs (CPU-side launch cost), hiding the actual kernel execution time.

### Diagnostic: Detecting Wrong Sync Mode

```
Immediate: LmHead avg ~595µs, RmsNorm avg ~25µs (24x ratio — correct)
Deferred:  LmHead avg ~60µs,  RmsNorm avg ~55µs  (1.1x ratio — wrong)
```

The contract requires `LmHead.avg_us > 10 * RmsNorm.avg_us` when sync mode is Immediate.

## Brick Breakdown (RTX 4090, Qwen 1.5B Q4K)

Correct profiling produces this breakdown:

| Brick | Per-Call (µs) | Per-Decoded-Token (µs) | % of Decode |
|-------|--------------|----------------------|-------------|
| AttentionScore | 67.5 | 1,891 | 17.7% |
| GateProjection | 53.2 | 1,489 | 13.9% |
| RmsNorm | 25.2 | 1,434 | 13.4% |
| DownProjection | 42.1 | 1,178 | 11.0% |
| QkvProjection | 35.1 | 982 | 9.2% |
| Activation | 30.6 | 856 | 8.0% |
| Residual2 | 24.2 | 678 | 6.3% |
| LmHead | 594.2 | 594 | 5.6% |
| OutputProjection | 21.0 | 587 | 5.5% |
| RopeEmbedding | 19.6 | 549 | 5.1% |
| Residual1 | 15.9 | 446 | 4.2% |

Note: LmHead has the highest per-call time (594µs — one GEMV over 151,936 vocab) but is called once per token, while layer bricks are called 28x per token (one per layer).

## Refactoring: Complexity Reduction

The original `forward_all_layers_gpu_to_logits_graphed()` had cyclomatic complexity 37 / cognitive 53 (thresholds: 30 / 25). The C-GDP-001 fix included extracting three helper methods:

| Method | Responsibility |
|--------|---------------|
| `should_use_eager_decode()` | Consolidates graph-disable checks |
| `try_first_token_graph_capture()` | First-token capture + fallback |
| `prepare_capture_buffers()` | Buffer initialization for capture |

This brought complexity under threshold while preserving identical behavior.

## Related

- [GPU Dispatch Strategy](./dispatch-strategy.md) — How GPU/CPU routing works
- [CUDA Context Safety](./cuda-context-safety.md) — Thread safety for CUDA operations
- [provable-contracts: gpu-decode-profiling-v1](https://github.com/paiml/provable-contracts) — Formal contract
