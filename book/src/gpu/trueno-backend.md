# Trueno Backend

Realizar uses [Trueno](https://crates.io/crates/trueno) as its compute backend, providing unified CPU SIMD and GPU acceleration through a single API.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Realizar API                      │
│         (Tensor, MatMul, Activation, etc.)          │
├─────────────────────────────────────────────────────┤
│                    Trueno Layer                      │
│         (Vector, Matrix, SIMD dispatch)             │
├───────────────┬─────────────────┬───────────────────┤
│   CPU SIMD    │   wgpu GPU      │   WASM SIMD       │
│  (AVX2/NEON)  │  (Vulkan/Metal) │  (Browser/Node)   │
└───────────────┴─────────────────┴───────────────────┘
```

## Enabling GPU Support

GPU support is enabled by default via the `gpu` feature:

```toml
[dependencies]
realizar = { version = "0.2", features = ["gpu"] }
```

For CPU-only builds:

```toml
[dependencies]
realizar = { version = "0.2", default-features = false }
```

## SIMD Backends

Trueno automatically selects the optimal SIMD backend at runtime:

| Platform | Backend | Vector Width |
|----------|---------|--------------|
| x86_64 (AVX2) | AVX2 | 256-bit |
| x86_64 (SSE2) | SSE2 | 128-bit |
| ARM64 | NEON | 128-bit |
| WASM | WASM SIMD | 128-bit |
| Fallback | Scalar | N/A |

## Vector Operations

Trueno provides high-performance vector operations used throughout Realizar:

```rust,ignore
use trueno::Vector;

// Create vectors from slices
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Element-wise operations (SIMD-accelerated)
let sum = a.add(&b);
let product = a.mul(&b);

// Reductions
let dot_product = a.dot(&b);     // 70.0
let l2_norm = a.norm_l2();       // 5.477...
let l1_norm = a.norm_l1();       // 10.0
let total = a.sum();             // 10.0
```

## Matrix Operations

Matrix multiplication is the core operation for transformer inference:

```rust,ignore
use trueno::Matrix;

// Create matrices
let weights = Matrix::from_slice(128, 256, &weight_data);
let input = Matrix::from_slice(1, 128, &input_data);

// Matrix multiplication (GPU-accelerated for large matrices)
let output = weights.matmul(&input);

// Transpose
let transposed = weights.transpose();
```

### Automatic GPU Dispatch

Trueno automatically dispatches to GPU for large matrix operations:

| Matrix Size | Backend | Typical Speedup |
|-------------|---------|-----------------|
| < 64x64 | CPU SIMD | 1x (baseline) |
| 64x64 - 512x512 | CPU SIMD | 2-4x |
| > 512x512 | GPU (wgpu) | 10-100x |

## Activation Functions

Trueno provides SIMD-accelerated activation functions:

```rust,ignore
use trueno::Vector;

let logits = Vector::from_slice(&[0.1, -0.5, 0.3, 1.2]);

// ReLU: max(0, x)
let relu = logits.relu();

// GELU: Gaussian Error Linear Unit
let gelu = logits.gelu();

// Sigmoid: 1 / (1 + exp(-x))
let sigmoid = logits.sigmoid();

// Swish: x * sigmoid(x)
let swish = logits.swish();

// Mish: x * tanh(softplus(x))
let mish = logits.mish();

// SELU: Scale Exponential Linear Unit
let selu = logits.selu();

// HardSwish: efficient Swish approximation
let hardswish = logits.hardswish();
```

## Performance Benchmarks

Trueno delivers significant speedups over naive implementations:

| Operation | Naive (ns) | Trueno SIMD (ns) | Speedup |
|-----------|------------|------------------|---------|
| vec_add (1024) | 450 | 85 | 5.3x |
| vec_mul (1024) | 460 | 90 | 5.1x |
| dot_product (1024) | 520 | 95 | 5.5x |
| norm_l2 (1024) | 580 | 110 | 5.3x |
| relu (1024) | 380 | 45 | 8.4x |
| gelu (1024) | 2100 | 190 | 11x |

## GPU Backend (wgpu)

The wgpu backend provides cross-platform GPU acceleration:

### Supported APIs

| Platform | Graphics API |
|----------|-------------|
| Windows | Vulkan, DX12, DX11 |
| macOS | Metal |
| Linux | Vulkan |
| Web | WebGPU |

### GPU Memory Management

```rust,ignore
use trueno::gpu::{GpuContext, GpuBuffer};

// Initialize GPU context (lazy)
let ctx = GpuContext::new()?;

// Create GPU buffer
let buffer = GpuBuffer::from_slice(&ctx, &data);

// Execute compute shader
let result = ctx.matmul(&a_buffer, &b_buffer);

// Read back to CPU
let output = result.to_vec();
```

### Shader Compilation

Trueno compiles WGSL shaders at runtime:

```wgsl
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = input_a[idx] + input_b[idx];
}
```

## Integration with Realizar

Realizar's tensor operations are built on Trueno primitives:

```rust,ignore
use realizar::Tensor;

// Tensor operations dispatch to Trueno
let a = Tensor::from_vec(vec![3, 3], vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]).unwrap();

// Shape and size operations
assert_eq!(a.shape(), &[3, 3]);
assert_eq!(a.ndim(), 2);
assert_eq!(a.size(), 9);
```

## WASM Support

Trueno supports WebAssembly with SIMD:

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]
```

Build for WASM:

```bash
cargo build --target wasm32-unknown-unknown --features gpu
```

## Troubleshooting

### No GPU Detected

```rust,ignore
use trueno::gpu::GpuContext;

match GpuContext::new() {
    Ok(ctx) => println!("GPU: {}", ctx.adapter_name()),
    Err(e) => println!("GPU unavailable: {}, using CPU SIMD", e),
}
```

### Forcing CPU Backend

Set environment variable to disable GPU:

```bash
export TRUENO_FORCE_CPU=1
cargo run --release
```

## Related Documentation

- [CUDA PTX Generation](./cuda-ptx.md) - Native NVIDIA CUDA support
- [SIMD Optimization](./simd.md) - CPU SIMD details
- [GPU Dispatch Strategy](./dispatch-strategy.md) - Automatic backend selection
- [Memory Management](./memory-management.md) - GPU memory handling
