# GPU Memory Management

> **Status**: Implemented (Phase 4 Complete)
>
> See: `src/gpu.rs` - GpuBufferPool, AsyncGpuResult

## Overview

Efficient GPU memory management is critical for performance. Realizar provides buffer pooling and async operations to minimize allocation overhead and maximize throughput.

## Buffer Pooling

The `GpuBufferPool` reuses buffers to avoid repeated allocations:

```rust
use realizar::gpu::{HybridScheduler, GpuPoolStats};

let scheduler = HybridScheduler::new()?;

// First call: allocates new buffer
let result1 = scheduler.matmul_pooled(&a, &b, m, k, n)?;

// Second call: reuses buffer from pool (no allocation)
let result2 = scheduler.matmul_pooled(&a2, &b2, m, k, n)?;
```

### Pool Statistics

Monitor pool efficiency:

```rust
let stats: GpuPoolStats = scheduler.pool_stats();

println!("Buffer pool statistics:");
println!("  Hits: {}", stats.hits);      // Reused buffers
println!("  Misses: {}", stats.misses);  // New allocations
println!("  Hit rate: {:.1}%",
    stats.hits as f64 / (stats.hits + stats.misses) as f64 * 100.0);
```

## Buffer Pool Implementation

The pool maintains buffers keyed by size:

```
┌─────────────────────────────────────────┐
│           GpuBufferPool                 │
├─────────────────────────────────────────┤
│ buffers: HashMap<usize, Vec<GpuBuffer>> │
├─────────────────────────────────────────┤
│ get_or_create(size):                    │
│   1. Check pool for matching size       │
│   2. If found: pop and reuse (hit)      │
│   3. If not: allocate new (miss)        │
├─────────────────────────────────────────┤
│ return_buffer(buffer):                  │
│   Push buffer back to pool for reuse    │
└─────────────────────────────────────────┘
```

## Async GPU Operations

For non-blocking computation with `AsyncGpuResult`:

```rust
// Start computation without blocking
let async_result = scheduler.matmul_async(&a, &b, m, k, n)?;

// Do other work while GPU computes...
let cpu_work = process_something_else();

// Wait for GPU result when needed
let gpu_result = async_result.wait()?;

// Check if result is ready (non-blocking)
if async_result.is_ready() {
    let result = async_result.try_get()?;
}
```

### Async Result States

```
┌─────────────────┐
│ AsyncGpuResult  │
├─────────────────┤
│ Pending         │ → Computation in progress
│ Ready(Vec<f32>) │ → Result available
│ Error(String)   │ → Computation failed
└─────────────────┘
```

## Memory Layout

GPU buffers use contiguous memory for efficient transfer:

```
Host Memory (CPU)         Device Memory (GPU)
┌──────────────────┐      ┌──────────────────┐
│ Input Matrix A   │ ───► │ GPU Buffer A     │
│ [f32; M×K]       │      │ [f32; M×K]       │
└──────────────────┘      └──────────────────┘

┌──────────────────┐      ┌──────────────────┐
│ Input Matrix B   │ ───► │ GPU Buffer B     │
│ [f32; K×N]       │      │ [f32; K×N]       │
└──────────────────┘      └──────────────────┘

                          ┌──────────────────┐
                          │ Output Buffer C  │
                          │ [f32; M×N]       │
                          └──────────────────┘
                                 │
                                 ▼
┌──────────────────┐      Host Memory (Result)
│ Result           │ ◄───
│ [f32; M×N]       │
└──────────────────┘
```

## Best Practices

### 1. Reuse Buffers for Repeated Operations

```rust
// Good: Use pooled operations in loops
for batch in batches {
    let result = scheduler.matmul_pooled(&batch, &weights, m, k, n)?;
}

// Avoid: Creating new scheduler each iteration
for batch in batches {
    let scheduler = HybridScheduler::new()?;  // Wasteful!
    let result = scheduler.matmul(&batch, &weights, m, k, n)?;
}
```

### 2. Batch Operations When Possible

```rust
// Good: Batch multiple operations
let ops = inputs.iter().map(|input| (input, &weights, m, k, n)).collect();
let results = scheduler.matmul_batch(&ops)?;

// Less efficient: Individual calls
let mut results = Vec::new();
for input in &inputs {
    results.push(scheduler.matmul(input, &weights, m, k, n)?);
}
```

### 3. Use Async for Pipeline Parallelism

```rust
// Good: Overlap GPU compute with CPU work
let gpu_future = scheduler.matmul_async(&a, &b, m, k, n)?;
let cpu_result = postprocess_previous_batch(&prev_result);
let gpu_result = gpu_future.wait()?;
```

## Memory Overhead

The buffer pool adds minimal memory overhead:

| Component | Overhead |
|-----------|----------|
| Pool metadata | ~64 bytes per size class |
| Unused buffers | Configurable max pool size |
| Async handles | ~32 bytes per pending operation |

## CUDA Context Pooling

For CUDA-specific memory management, Realizar pools CUDA contexts and streams to avoid the overhead of repeated `cuCtxCreate` calls. A **sync-on-drop** mechanism ensures poisoned contexts are never returned to the pool. See [CUDA Context Safety](./cuda-context-safety.md) for details.

## See Also

- [Dispatch Strategy](./dispatch-strategy.md) - When to use GPU vs CPU
- [Trueno Backend](./trueno-backend.md) - Underlying wgpu implementation
- [CUDA Context Safety](./cuda-context-safety.md) - Context pooling and fail-fast architecture
