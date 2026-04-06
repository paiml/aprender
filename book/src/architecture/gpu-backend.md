# GPU Backend

Trueno provides two GPU acceleration options:

1. **wgpu (Cross-platform)** - Vulkan, Metal, DX12, WebGPU via [wgpu](https://wgpu.rs/)
2. **CUDA (NVIDIA)** - Native PTX code generation via [trueno-gpu](../architecture/ptx-generation.md)

## CUDA Support (trueno-gpu)

For NVIDIA GPUs, trueno-gpu provides **pure Rust PTX code generation** without requiring LLVM, nvcc, or external toolchains.

### Quick Start with CUDA

```rust
use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};
use trueno_gpu::kernels::{GemmKernel, Kernel};

// Generate optimized GEMM kernel
let kernel = GemmKernel::tiled(1024, 1024, 1024, 32);
let ptx = kernel.emit_ptx();

// PTX can be loaded via CUDA driver API
println!("{}", ptx);
```

### Running CUDA Examples

```bash
# PTX code generation (no GPU required)
cargo run -p trueno-gpu --example ptx_quickstart
cargo run -p trueno-gpu --example gemm_kernel

# CUDA runtime examples (requires NVIDIA GPU)
cargo run -p trueno-gpu --example cuda_monitor
cargo run -p trueno-gpu --example flash_attention_cuda
```

### Pre-built CUDA Kernels

| Kernel | Description | Example |
|--------|-------------|---------|
| GEMM | Matrix multiplication (naive/tiled/tensor core) | `gemm_kernel` |
| Softmax | Numerically stable softmax | `ptx_quickstart` |
| LayerNorm | Layer normalization | `simple_attention_cuda` |
| Attention | Multi-head attention | `flash_attention_cuda` |
| Quantize | Q4_K/Q5_K/Q6_K quantization | `q4k_gemm` |

See [PTX Code Generation](./ptx-generation.md) for detailed documentation.

---

## wgpu Support (Cross-Platform)

For cross-platform GPU compute, Trueno uses [wgpu](https://wgpu.rs/), supporting Vulkan, Metal, DX12, and WebGPU.

## Overview

The wgpu backend enables massive parallelism for compute-heavy operations like matrix multiplication. It supports both native platforms (Linux, macOS, Windows) and WebAssembly (via WebGPU in browsers).

### Key Features

- **Cross-platform**: Single codebase for native and WASM
- **Async-first**: All operations have async variants for non-blocking execution
- **Sync wrappers**: Native platforms get convenient sync APIs
- **Automatic fallback**: Falls back to SIMD when GPU unavailable

## Platform Support

| Platform | Backend | Sync API | Async API |
|----------|---------|----------|-----------|
| Linux | Vulkan | ✅ | ✅ |
| macOS | Metal | ✅ | ✅ |
| Windows | DX12/Vulkan | ✅ | ✅ |
| WASM (Browser) | WebGPU | ❌ | ✅ |

**Note**: WASM cannot use sync APIs because JavaScript's single-threaded model prohibits blocking the main thread.

## Feature Flags

```toml
[dependencies]
trueno = { version = "0.7.3", features = ["gpu"] }      # Native GPU
trueno = { version = "0.7.3", features = ["gpu-wasm"] } # WASM GPU (WebGPU)
```

### Feature Differences

| Feature | `gpu` | `gpu-wasm` |
|---------|-------|------------|
| wgpu | ✅ | ✅ |
| pollster (sync runtime) | ✅ | ❌ |
| wasm-bindgen-futures | ❌ | ✅ |
| Sync methods | ✅ | ❌ |
| Async methods | ✅ | ✅ |

## API Design

### Sync API (Native Only)

```rust
use trueno::backends::gpu::GpuDevice;

// Initialize device
let device = GpuDevice::new()?;

// Check availability
if GpuDevice::is_available() {
    // Execute operations
    device.matmul(&a, &b, &mut result, m, k, n)?;
    device.relu(&input, &mut output)?;
    let dot = device.dot(&a, &b)?;
}
```

### Async API (All Platforms)

```rust
use trueno::backends::gpu::GpuDevice;

// Initialize device
let device = GpuDevice::new_async().await?;

// Check availability
if GpuDevice::is_available_async().await {
    // Execute operations
    device.matmul_async(&a, &b, &mut result, m, k, n).await?;
    device.relu_async(&input, &mut output).await?;
    let dot = device.dot_async(&a, &b).await?;
}
```

### Runtime Detection

```rust
use trueno::backends::gpu::runtime;

if runtime::sync_available() {
    // Can use sync APIs (native only)
    let device = GpuDevice::new()?;
} else {
    // Must use async APIs (WASM)
    let device = GpuDevice::new_async().await?;
}
```

## Available Operations

### Element-wise Operations

| Operation | Sync | Async | Description |
|-----------|------|-------|-------------|
| `relu` | ✅ | ✅ | max(0, x) |
| `leaky_relu` | ✅ | ✅ | max(αx, x) |
| `elu` | ✅ | ✅ | x if x>0, else α(eˣ-1) |
| `sigmoid` | ✅ | ✅ | 1/(1+e⁻ˣ) |
| `tanh` | ✅ | ✅ | tanh(x) |
| `swish` | ✅ | ✅ | x·sigmoid(x) |
| `gelu` | ✅ | ✅ | Gaussian Error Linear Unit |
| `clip` | ✅ | ✅ | clamp(x, min, max) |
| `softmax` | ✅ | ✅ | exp(x)/Σexp(x) |
| `log_softmax` | ✅ | ✅ | log(softmax(x)) |

### Vector Operations

| Operation | Sync | Async | Description |
|-----------|------|-------|-------------|
| `vec_add` | ✅ | ✅ | Element-wise addition |
| `dot` | ✅ | ✅ | Dot product with reduction |

### Matrix Operations

| Operation | Sync | Async | Description |
|-----------|------|-------|-------------|
| `matmul` | ✅ | ✅ | Matrix multiplication |
| `convolve2d` | ✅ | ✅ | 2D convolution |

## WebGPU for WASM

The `gpu-wasm` feature enables GPU compute in browsers via WebGPU. This is particularly useful for:

- **Browser-based ML inference**: Run models client-side
- **Interactive visualizations**: GPU-accelerated data processing
- **Scientific computing in browsers**: Heavy computations without server round-trips

### Example: trueno-viz

[trueno-viz](https://github.com/paiml/trueno-viz) demonstrates Trueno's WebGPU capabilities for browser-based visualization:

```rust
// In WASM context, use async API
#[wasm_bindgen]
pub async fn process_data(input: &[f32]) -> Result<Vec<f32>, JsValue> {
    let device = GpuDevice::new_async().await
        .map_err(|e| JsValue::from_str(&e))?;

    let mut output = vec![0.0; input.len()];
    device.relu_async(input, &mut output).await
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(output)
}
```

### WASM Build Configuration

```toml
# Cargo.toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
trueno = { version = "0.7.3", features = ["gpu-wasm"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
```

Build with:

```bash
wasm-pack build --target web --features gpu-wasm
```

## Batch API

For chaining multiple GPU operations, use the batch API to minimize transfer overhead:

```rust
use trueno::backends::gpu::{GpuDevice, GpuCommandBatch};

let device = GpuDevice::new()?;
let mut batch = GpuCommandBatch::new(device);

// Queue operations (no GPU execution yet)
let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
let a = batch.relu(input);
let b = batch.scale(a, 2.0);

// Execute batch in single GPU round-trip
batch.execute().await?;

// Read result
let result = batch.read(b).await?;
```

See [GPU Performance](../performance/gpu-performance.md) for detailed batch API documentation.

## Performance Considerations

### When to Use GPU

✅ **Use GPU for**:
- Matrix multiplication >500×500
- 2D convolutions with large kernels
- Batched operations (multiple ops chained)

❌ **Use SIMD instead for**:
- Vector operations (add, mul, dot)
- Small matrices (<500×500)
- Single operations (transfer overhead dominates)

### Transfer Overhead

GPU operations incur ~3.5ms fixed overhead per operation:

| Component | Time |
|-----------|------|
| Buffer creation | ~0.5ms |
| CPU→GPU transfer | ~1.5ms |
| Kernel dispatch | ~0.3ms |
| GPU→CPU readback | ~1.2ms |

This overhead makes GPU slower than SIMD for simple operations. See [GPU Performance](../performance/gpu-performance.md) for benchmarks.

## Implementation Details

### Runtime Module

The `runtime` module (`src/backends/gpu/runtime.rs`) provides platform-specific async runtime helpers:

```rust
// Native: Uses pollster for blocking
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn block_on<F: Future>(f: F) -> F::Output {
    pollster::block_on(f)
}

// Check if sync operations are available
pub const fn sync_available() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    { true }
    #[cfg(target_arch = "wasm32")]
    { false }
}

// WASM: Spawn async tasks
#[cfg(all(feature = "gpu-wasm", target_arch = "wasm32"))]
pub fn spawn_local<F: Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}
```

### Conditional Compilation

Sync methods are only available on native platforms:

```rust
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn relu(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
    runtime::block_on(self.relu_async(input, result))
}

// Async always available
pub async fn relu_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
    // Implementation
}
```

## Next Steps

- **[GPU Performance](../performance/gpu-performance.md)** - Detailed benchmarks and thresholds
- **[WASM Backend](./wasm-backend.md)** - SIMD128 for non-GPU WASM
- **[Backend Selection](./backend-selection.md)** - How Trueno chooses backends
