# GPU Dispatch Strategy

> **Status**: Implemented (Phase 4 Complete)
>
> See: `src/gpu.rs` - HybridScheduler and automatic CPU/GPU selection

## Overview

Realizar implements a **hybrid CPU/GPU dispatch strategy** that automatically selects the optimal backend based on workload size. This follows the principle of using GPU acceleration where it provides benefit while avoiding overhead for small operations.

## Key Components

### GpuCompute

The `GpuCompute` struct wraps trueno's wgpu backend and provides a unified interface:

```rust
use realizar::gpu::GpuCompute;

// Auto-detect best backend (GPU with CPU fallback)
let mut compute = GpuCompute::auto()?;

// Check which backend is active
if compute.is_gpu() {
    println!("Using GPU (wgpu)");
} else {
    println!("Using CPU fallback");
}

// Matrix multiplication (dispatched to appropriate backend)
let result = compute.matmul(&input, &weights, m, k, n)?;
```

### HybridScheduler

The `HybridScheduler` provides intelligent workload routing:

```rust
use realizar::gpu::HybridScheduler;

let scheduler = HybridScheduler::new()?;

// Automatic dispatch based on matrix dimensions
// - Small matrices (< 1000 elements): CPU (avoids GPU overhead)
// - Large matrices (>= 1000 elements): GPU (leverages parallelism)
let result = scheduler.matmul(&a, &b, m, k, n)?;
```

## Dispatch Threshold

The default threshold is **1000 elements** (approximately a 10x10x10 operation):

| Workload | Backend | Rationale |
|----------|---------|-----------|
| `m*k*n < 1000` | CPU | GPU kernel launch overhead exceeds compute benefit |
| `m*k*n >= 1000` | GPU | Parallel computation provides speedup |

This threshold is configurable:

```rust
let scheduler = HybridScheduler::with_threshold(2000)?;
```

## Backend Selection Logic

```
┌─────────────────────────────────────────────┐
│              should_use_gpu()               │
├─────────────────────────────────────────────┤
│ 1. Is GPU backend available?                │
│    NO  → CPU fallback                       │
│    YES → continue                           │
├─────────────────────────────────────────────┤
│ 2. Is workload >= threshold?                │
│    NO  → CPU (small operation)              │
│    YES → GPU (benefit from parallelism)     │
└─────────────────────────────────────────────┘
```

## Performance Results

From Phase 4 acceptance tests:

| Metric | GPU (wgpu) | CPU |
|--------|------------|-----|
| Throughput | 35.0 tok/s | 553.7 tok/s* |
| Best for | Large matrices | Small matrices |

*CPU throughput is higher for small test matrices that fit in cache.

## Buffer Pooling

For repeated operations, use pooled buffers to avoid allocation overhead:

```rust
// Get pooled buffer (reuses existing if available)
let result = scheduler.matmul_pooled(&a, &b, m, k, n)?;

// Check pool statistics
let stats = scheduler.pool_stats();
println!("Pool hits: {}, misses: {}", stats.hits, stats.misses);
```

## Async GPU Operations

For non-blocking computation:

```rust
// Start async GPU operation
let async_result = scheduler.matmul_async(&a, &b, m, k, n)?;

// Do other work...

// Wait for result when needed
let result = async_result.wait()?;
```

## Batch Processing

Process multiple matrices efficiently:

```rust
let ops: Vec<MatmulOp> = vec![
    (a1, b1, m1, k1, n1),
    (a2, b2, m2, k2, n2),
    // ...
];

let results = scheduler.matmul_batch(&ops)?;
```

## Feature Flag

GPU support requires the `gpu` feature (enabled by default):

```toml
[dependencies]
realizar = { version = "0.2", features = ["gpu"] }
```

To disable:

```toml
[dependencies]
realizar = { version = "0.2", default-features = false }
```

## See Also

- [Memory Management](./memory-management.md) - GPU buffer management
- [Trueno Backend](./trueno-backend.md) - Low-level wgpu integration
- [SIMD Optimization](./simd.md) - CPU SIMD fallback
