# Case Study: CUDA and GPU Backends

This chapter demonstrates how to configure aprender for different compute backends, including CPU SIMD, GPU (wgpu/WebGPU), and NVIDIA CUDA acceleration.

## Overview

Aprender v0.18.0 introduces flexible backend configuration through the `loading` module, supporting:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `CpuSimd` | CPU with SIMD (AVX2/AVX-512/NEON) | Default, works everywhere |
| `Gpu` | GPU via wgpu/WebGPU compute shaders | Cross-platform GPU acceleration |
| `Cuda` | NVIDIA CUDA via trueno-gpu | Maximum performance on NVIDIA hardware |
| `Wasm` | WebAssembly | Browser deployment |
| `Embedded` | Bare metal (no_std) | IoT and embedded systems |

## Cargo.toml Configuration

Enable GPU or CUDA support in your `Cargo.toml`:

```toml
[dependencies]
# Default CPU SIMD backend
aprender = "0.18"

# With GPU acceleration (wgpu/WebGPU)
aprender = { version = "0.18", features = ["gpu"] }

# With NVIDIA CUDA support
aprender = { version = "0.18", features = ["cuda"] }

# Both GPU and CUDA
aprender = { version = "0.18", features = ["gpu", "cuda"] }
```

## Backend Presets

Aprender provides preset configurations for common deployment scenarios:

### Server Deployment (CPU SIMD)

```rust
use aprender::loading::LoadConfig;

let config = LoadConfig::server();
// - Backend: CpuSimd
// - Mode: MappedDemand (memory-mapped for large models)
// - Verification: Standard
```

### GPU Deployment (wgpu/WebGPU)

```rust
use aprender::loading::LoadConfig;

let config = LoadConfig::gpu();
// - Backend: Gpu
// - Mode: MappedDemand
// - Cross-platform: Vulkan, Metal, DX12, WebGPU
```

### NVIDIA CUDA Deployment

```rust
use aprender::loading::LoadConfig;

let config = LoadConfig::cuda();
// - Backend: Cuda
// - Mode: MappedDemand
// - Requires: NVIDIA driver + `cuda` feature
```

### WASM Deployment (Browser)

```rust
use aprender::loading::LoadConfig;

let config = LoadConfig::wasm();
// - Backend: Wasm
// - Mode: Streaming (64MB memory limit)
// - For browser-based ML inference
```

### Embedded Deployment (IoT)

```rust
use aprender::loading::LoadConfig;

let config = LoadConfig::embedded(64 * 1024); // 64KB memory budget
// - Backend: Embedded
// - Mode: Eager (deterministic)
// - Verification: Paranoid (NASA Level A)
```

## Backend Properties

Each backend exposes properties to help you make runtime decisions:

```rust
use aprender::loading::Backend;

let backend = Backend::Cuda;

// Check if SIMD is available
assert!(!backend.supports_simd()); // CUDA uses GPU, not CPU SIMD

// Check if GPU accelerated
assert!(backend.is_gpu_accelerated()); // Yes!

// Check if NVIDIA driver required
assert!(backend.requires_nvidia_driver()); // Yes, CUDA needs NVIDIA

// Check if std library required
assert!(backend.requires_std()); // Yes (Embedded is no_std)
```

## Custom Configurations

Build custom configurations using the builder pattern:

```rust
use aprender::loading::{Backend, LoadConfig, LoadingMode, VerificationLevel};
use std::time::Duration;

// High-performance CUDA configuration
let cuda_config = LoadConfig::new()
    .with_backend(Backend::Cuda)
    .with_mode(LoadingMode::Eager)           // Full load for low latency
    .with_verification(VerificationLevel::Paranoid)  // NASA Level A
    .with_max_memory(4 * 1024 * 1024 * 1024) // 4GB budget
    .with_time_budget(Duration::from_millis(500));

// GPU streaming for large models
let gpu_streaming = LoadConfig::new()
    .with_backend(Backend::Gpu)
    .with_mode(LoadingMode::Streaming)
    .with_streaming(2 * 1024 * 1024);  // 2MB ring buffer
```

## Backend Comparison Matrix

| Property | CpuSimd | Gpu | Cuda | Wasm | Embedded |
|----------|---------|-----|------|------|----------|
| SIMD Support | Yes | No | No | No | No |
| GPU Accelerated | No | Yes | Yes | No | No |
| NVIDIA Required | No | No | Yes | No | No |
| Requires std | Yes | Yes | Yes | Yes | No |
| Best For | General | Cross-platform GPU | Max NVIDIA perf | Browser | IoT |

## Running the Example

```bash
# Default CPU SIMD backend
cargo run --example cuda_backend

# With GPU feature
cargo run --example cuda_backend --features gpu

# With CUDA feature (requires NVIDIA driver)
cargo run --example cuda_backend --features cuda
```

## Toyota Way: Heijunka (Level Loading)

The backend system follows Toyota Way Heijunka principles:

- **Level resource demands**: Each backend preset optimizes for its target environment
- **Jidoka (built-in quality)**: Verification levels ensure model integrity
- **Poka-yoke (error-proofing)**: Type-safe APIs prevent misconfiguration

## Integration with trueno

Aprender's GPU support is powered by [trueno](https://crates.io/crates/trueno), our SIMD-accelerated tensor library:

- **trueno**: Core SIMD operations (CPU backend)
- **trueno/gpu**: wgpu-based GPU compute shaders
- **trueno/cuda-monitor**: NVIDIA CUDA integration via [trueno-gpu](https://crates.io/crates/trueno-gpu)

The trueno-gpu crate provides:
- Pure Rust PTX generation (no LLVM, no nvcc)
- Runtime CUDA driver loading
- Device monitoring and memory metrics

## Example Output

```
=== Aprender Backend Configuration Demo ===

1. CPU SIMD Backend (Default)
   -------------------------
   Backend: CpuSimd
   Supports SIMD: true
   GPU Accelerated: false
   Requires NVIDIA: false
   Requires std: true

2. GPU Backend (wgpu/WebGPU)
   -------------------------
   Backend: Gpu
   Supports SIMD: false
   GPU Accelerated: true
   Requires NVIDIA: false

3. NVIDIA CUDA Backend
   --------------------
   Backend: Cuda
   Supports SIMD: false
   GPU Accelerated: true
   Requires NVIDIA: true

4. Backend Comparison
   ------------------
   | Backend   | SIMD | GPU Accel | NVIDIA Req | std Req |
   |-----------|------|-----------|------------|---------|
   | CpuSimd   | Yes  | No        | No         | Yes     |
   | Gpu       | No   | Yes       | No         | Yes     |
   | Cuda      | No   | Yes       | Yes        | Yes     |
   | Wasm      | No   | No        | No         | Yes     |
   | Embedded  | No   | No        | No         | No      |
```

## See Also

- [APR Loading Modes](./apr-loading-modes.md) - Memory loading strategies
- [Model Format (.apr)](./model-format.md) - The aprender model format
- [Sovereign AI Stack](./sovereign-stack.md) - Full stack integration
