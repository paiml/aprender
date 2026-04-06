# ComputeBrick Architecture

> **The Oracle of Compute**
>
> `trueno` is the "Oracle" of the ComputeBrick ecosystem. specifically `trueno/src/brick.rs`.
> It defines the **`ComputeBrick` trait**, **`TokenBudget`**, and **`BrickProfiler`** logic.
> It is the central dependency that `realizar` (inference), `aprender` (algorithms), and `cbtop` (visualization) all import to mathematically verify if performance and correctness assertions are met.

## Core Concepts

A **ComputeBrick** is a self-verifying, token-centric compute unit that bundles:

1.  **Operation**: The compute operation (matmul, dot, softmax, etc.)
2.  **Assertions**: Falsifiable claims about the output (equivalence, bounds)
3.  **Budget**: Performance target in µs/token or tokens/sec
4.  **Backend**: Execution target (Scalar, AVX2, CUDA, etc.)

### The "Pure Rust" Invariant

The ComputeBrick architecture enforces a "Pure Rust" stack.
- **No FFI to C++ libraries** (like llama.cpp or ggml) for core compute.
- **Direct GPU Control**: Use `trueno-gpu` for PTX generation and `wgpu` for cross-platform support.
- **Safety**: `unsafe` is encapsulated strictly within Brick boundaries.

## TokenBudget

Performance is not measured in abstract FLOPS, but in **Tokens per Second** (tok/s) or **Microseconds per Token** (µs/token).

```rust
pub struct TokenBudget {
    /// Latency budget per token (microseconds)
    pub us_per_token: f64,
    /// Throughput target (tokens/second)
    pub tokens_per_sec: f64,
}
```

This aligns low-level compute optimization directly with high-level LLM inference goals.

## BrickProfiler

The **`BrickProfiler`** is the mechanism for "Real Profiling".

- **Real Measurements**: It measures actual execution time using `std::time::Instant`.
- **Synchronization**: For GPU operations, it mandates `cudaDeviceSynchronize()` (or equivalent) before start and after stop to ensure accurate timing.
- **Falsification**: Derived or simulated metrics are explicitly **FORBIDDEN**.

```rust
// Example of Real Profiling
profiler.start("QkvBrick");
cuda_stream.synchronize(); // Ensure pre-reqs done
// ... execute kernel ...
cuda_stream.synchronize(); // Ensure kernel done
profiler.stop("QkvBrick", num_tokens);
```

### Sovereign Stack Profiling Mandate

Every component in the Sovereign Stack MUST implement REAL `BrickProfiler` timing:

| Component | Repository | Metric | Implementation |
|-----------|------------|--------|----------------|
| **trueno** | `trueno` | SIMD Ops/sec | `Instant::now()` |
| **trueno-gpu** | `trueno` | Kernel Latency | `cudaEventRecord` |
| **trueno-zram** | `trueno` | Compression GB/s | `Instant` + Batch |
| **aprender** | `aprender` | Algorithm Latency | `BrickProfiler` |
| **realizar** | `aprender` | Inference Latency | `cudaDeviceSynchronize` |
| **presentar** | `aprender` | Frame Time | `requestAnimationFrame` |

## Integration

`trueno` provides the types.
`realizar` implements the Bricks (e.g., `QkvBrick`, `AttentionBrick`).
`aprender` uses Bricks for ML algorithms.
`cbtop` visualizes the `BrickProfiler` output.
