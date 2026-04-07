# Sub-spec: ComputeBrick & Profiling

**Parent:** [trueno-spec.md](../trueno-spec.md) Section 19

---

## 1. Overview

`src/brick/` provides token-centric compute units — self-verifying blocks with budgets, assertions, and backend selection. Every kernel execution is a `ComputeBrick` with measurable pre/postconditions.

## 2. Core Types

| Type | Purpose |
|------|---------|
| `ComputeBrick` | Composable compute unit with assertions and budgets |
| `BrickLayer` | Composition of multiple bricks |
| `BrickId` | Enum identifying each brick type for O(1) profiling |
| `BrickProfiler` | Hot-path profiler (PAR-200 design) |
| `ExecutionGraph` | Full execution path with kernel checksums |
| `ModelTracer` | Model-level inference tracing |

## 3. BrickProfiler (PAR-200)

O(1) per-brick tracking via fixed-size array indexed by `BrickId`. No heap allocation in the hot path.

**Key methods:**
- `total_ns()` — wall-clock nanoseconds
- `total_tokens()` — elements processed
- `brick_stats(id)` — per-brick statistics
- `set_sync_mode(mode)` — Eager (debug) or Deferred (production)

**SyncMode:**
- `Eager` — synchronize after every brick (debug, profiling)
- `Deferred` — batch sync per layer (production, <100us overhead)

## 4. ExecutionGraph (PAR-201)

Tracks the full execution path with `ExecutionNode` types:
- `Kernel` — GPU/CPU kernel execution with arithmetic intensity
- `Transfer` — Host↔Device data movement
- `Sync` — Synchronization points

`KernelChecksum` provides per-kernel checksums for divergence detection (CORRECTNESS-011).

## 5. Quantization Ops

llama.cpp-compatible block quantization:
- `BlockQ5K`, `BlockQ6K` — 5-bit and 6-bit block formats
- `DotQ5KOp`, `DotQ6KOp` — Quantized dot product operations
- Fused transformer ops: `FusedQKVOp`, `FusedGateUpOp`

## 6. ModelTracer

5 trace types for model-level inference observability:
- Tensor stats (min/max/mean/std per layer)
- Attention weight distributions
- Logit evolution across decoding steps
- Quantization error tracking
- Layer activation tracing

## 7. Submodules

| Module | Purpose |
|--------|---------|
| `batch/` | Balance211 scheduler, batch splitting |
| `buffer/` | Watermarked buffers |
| `circuit/` | Circuit breaker pattern |
| `kv_cache/` | KV cache slots, sequential ordering |
| `memory/` | Prefetch locality, cache alignment |
| `rate_limit/` | Rate limiting |
| `simd_config/` | SIMD unroll config, AMX tiles |
| `tracing/` | Detailed tensor/attention/logit tracing |
