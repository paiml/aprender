# Trueno: Multi-target Compute

**Trueno** (Spanish: "thunder") is a Rust library providing unified, high-performance compute primitives across multiple execution targets. It serves as the foundation for numerical computation in the sovereign stack.

## Overview

Trueno delivers:
- **CPU SIMD** - x86 (SSE2/AVX/AVX2/AVX-512), ARM (NEON), WASM (SIMD128)
- **GPU** - Vulkan/Metal/DX12/WebGPU via `wgpu`
- **WebAssembly** - Portable SIMD128 for browser/edge deployment

```
┌─────────────────────────────────────────────────┐
│           Trueno Public API (Safe)              │
│  compute(), map(), reduce(), transform()        │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌──────────┐
   │  SIMD  │   │   GPU   │   │   WASM   │
   │ Backend│   │ Backend │   │  Backend │
   └────────┘   └─────────┘   └──────────┘
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌───┴─────┐
   │ Runtime │   │  wgpu   │   │ SIMD128 │
   │ Detect  │   │ Compute │   │ Portable│
   └─────────┘   └─────────┘   └─────────┘
```

## Installation

```toml
[dependencies]
trueno = "0.14"

# With GPU support
trueno = { version = "0.14", features = ["gpu"] }

# With CUDA monitoring (NVIDIA GPUs)
trueno = { version = "0.14", features = ["cuda-monitor"] }
```

## What's New in 0.14

- **Streaming Tensors**: Memory-mapped streaming for large datasets
- **Q5K/Q6K Quantization**: Extended quantization formats
- **Improved WASM**: Better WebAssembly SIMD128 support
- **LZ4/ZSTD Compression**: Built-in tensor compression for memory efficiency
- **GPU PTX Fixes**: Resolved NVIDIA PTX codegen issues
- **AVX-512 Improvements**: Better auto-vectorization
- **Simulation Framework**: Toyota-style Jidoka guards and stress testing

## Core Features

### Vector Operations

```rust
use trueno::{Vector, VectorOps};

// Create vectors
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Element-wise operations (auto-selects best SIMD backend)
let sum = a.add(&b)?;       // [6.0, 8.0, 10.0, 12.0]
let product = a.mul(&b)?;   // [5.0, 12.0, 21.0, 32.0]
let dot = a.dot(&b)?;       // 70.0

// Reductions
let total = a.sum()?;       // 10.0
let average = a.mean()?;    // 2.5
```

### Matrix Operations

```rust
use trueno::Matrix;

let a = Matrix::from_slice(2, 3, &[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
]);

let b = Matrix::from_slice(3, 2, &[
    7.0, 8.0,
    9.0, 10.0,
    11.0, 12.0,
]);

// Matrix multiplication (SIMD-accelerated)
let c = a.matmul(&b)?;  // 2x2 result

// Transpose
let at = a.transpose();

// Eigendecomposition (symmetric matrices)
let eigen = matrix.symmetric_eigen()?;
```

### Activation Functions

```rust
use trueno::activations::*;

let x = Vector::from_slice(&[-1.0, 0.0, 1.0, 2.0]);

// Neural network activations (SIMD-optimized)
let relu_out = relu(&x)?;      // [0.0, 0.0, 1.0, 2.0]
let sigmoid_out = sigmoid(&x)?;
let gelu_out = gelu(&x)?;
let swish_out = swish(&x)?;
let tanh_out = tanh_activation(&x)?;
```

## Backend Selection

Trueno automatically selects the optimal backend based on:

1. **Data size** - GPU only for large workloads (>100K elements)
2. **CPU features** - AVX-512 > AVX2 > AVX > SSE2 > NEON
3. **Operation complexity** - Complex ops benefit more from GPU

```rust
use trueno::Backend;

// Auto-select (recommended)
let result = vector.add(&other)?;

// Force specific backend
let result = vector.add_with_backend(&other, Backend::Avx2)?;
let result = vector.add_with_backend(&other, Backend::GPU)?;
```

### Backend Priority

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | GPU | Available + size > 100K |
| 2 | AVX-512 | CPU supports |
| 3 | AVX2 | CPU supports |
| 4 | AVX | CPU supports |
| 5 | SSE2 | x86_64 baseline |
| 6 | NEON | ARM64 |
| 7 | SIMD128 | WASM |
| 8 | Scalar | Fallback |

## Simulation Testing Framework (v0.8.5+)

Trueno 0.8.5 introduces a comprehensive simulation testing framework based on Toyota Production System principles.

### SimRng: Deterministic Random Number Generator

```rust
use trueno::simulation::SimRng;

// Deterministic PCG-based RNG
let mut rng = SimRng::new(42);  // Seed for reproducibility

// Generate deterministic random values
let value = rng.next_f32();           // [0.0, 1.0)
let int = rng.next_u32();             // Full u32 range
let range = rng.range(1.0, 10.0);     // Custom range
let normal = rng.normal(0.0, 1.0);    // Gaussian distribution

// Fork for parallel testing (maintains determinism)
let child_rng = rng.fork();
```

### BackendSelector: Intelligent Backend Selection

```rust
use trueno::simulation::{BackendSelector, BackendThresholds};

let thresholds = BackendThresholds {
    gpu_min_elements: 100_000,
    simd_min_elements: 32,
};

let selector = BackendSelector::new(thresholds);
let backend = selector.select(data_size, op_complexity);
```

### JidokaGuard: Stop-on-Defect Quality Checks

```rust
use trueno::simulation::JidokaGuard;

// Toyota-style quality gate - stops on first defect
let guard = JidokaGuard::new();

// Check for NaN/Inf values
guard.check_finite(&result)?;

// Custom invariant checking
guard.assert_invariant(|| value >= 0.0, "Value must be non-negative")?;
```

### BufferRenderer: Visual Regression Testing

```rust
use trueno::simulation::{BufferRenderer, ColorPalette};

let renderer = BufferRenderer::new(800, 600);
let palette = ColorPalette::viridis();

// Render data to RGBA buffer for visual comparison
let buffer = renderer.render_heatmap(&data, &palette)?;

// Compare with golden baseline
let diff = renderer.compare_buffers(&buffer, &golden)?;
assert!(diff.max_error < 1e-5);
```

### StressTestConfig: Stress Testing Infrastructure

```rust
use trueno::simulation::{StressTestConfig, StressTestResult};

let config = StressTestConfig {
    iterations: 10_000,
    data_size_range: 100..1_000_000,
    anomaly_threshold: 3.0,  // Standard deviations
};

let result = stress_test(&operation, &config)?;
assert!(result.anomaly_count == 0);
```

### BackendTolerance: Cross-Backend Comparison

```rust
use trueno::simulation::BackendTolerance;

let tolerance = BackendTolerance::relaxed();

// Get tolerance for comparing results across backends
let tol = tolerance.for_backends(Backend::GPU, Backend::Scalar);
assert!((gpu_result - scalar_result).abs() < tol);
```

## GPU Compute

### Synchronous API

```rust
use trueno::gpu::GpuDevice;

let device = GpuDevice::new()?;

// Large matrix multiplication on GPU
let result = device.matmul(&a, &b)?;

// Batch operations
let results = device.batch_add(&vectors_a, &vectors_b)?;
```

### Async API

```rust
use trueno::gpu::GpuDevice;

let device = GpuDevice::new()?;

// Non-blocking GPU operations
let future = device.matmul_async(&a, &b);
let result = future.await?;
```

## NumPy Compatibility (via Batuta)

Trueno is the target for NumPy → Rust transpilation:

| NumPy | Trueno |
|-------|--------|
| `np.array([1,2,3])` | `Vector::from_slice(&[1.0,2.0,3.0])` |
| `np.dot(a, b)` | `a.dot(&b)?` |
| `a + b` | `a.add(&b)?` |
| `a @ b` | `a.matmul(&b)?` |
| `np.sum(a)` | `a.sum()?` |
| `np.mean(a)` | `a.mean()?` |

## Performance

Expected speedups vs scalar baseline:

| Operation | Size | SSE2 | AVX2 | AVX-512 | GPU |
|-----------|------|------|------|---------|-----|
| add_f32 | 1K | 2x | 4x | 8x | - |
| add_f32 | 100K | 2x | 4x | 8x | 3x |
| add_f32 | 1M | 2x | 4x | 8x | 10x |
| add_f32 | 10M | 2x | 4x | 8x | 50x |
| dot_product | 1M | 3x | 6x | 12x | 20x |
| matmul | 1K×1K | 3x | 6x | 12x | 30x |

## Related Crates

- **trueno-gpu** - CUDA monitoring via NVML
- **trueno-db** - High-performance vector database
- **trueno-graph** - Graph analytics engine
- **trueno-viz** - GPU-accelerated visualization
- **trueno-rag** - RAG pipeline components

## References

- [trueno on crates.io](https://crates.io/crates/trueno)
- [trueno GitHub](https://github.com/paiml/trueno)
- [SIMD Programming Guide](../part2/simd.md)
- [GPU Acceleration Guide](../part2/gpu.md)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Previous: Foundation Libraries](./foundation-libs.md) | [Next: Aprender](./aprender.md)
