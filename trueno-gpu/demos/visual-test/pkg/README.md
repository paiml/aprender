# trueno-gpu

Pure Rust PTX generation for NVIDIA CUDA - no LLVM, no nvcc, no external dependencies.

[![Crates.io](https://img.shields.io/crates/v/trueno-gpu.svg)](https://crates.io/crates/trueno-gpu)
[![Documentation](https://docs.rs/trueno-gpu/badge.svg)](https://docs.rs/trueno-gpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Philosophy

**Own the Stack** - Build everything from first principles for complete control, auditability, and reproducibility.

## Features

- **Pure Rust PTX Generation**: Generate PTX assembly directly from Rust code
- **No External Dependencies**: No LLVM, nvcc, or CUDA toolkit required for code generation
- **Builder Pattern API**: Ergonomic API for constructing PTX modules and kernels
- **Hand-Optimized Kernels**: Pre-built kernels for common ML operations

## Quick Start

```rust
use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};

// Build a vector addition kernel
let module = PtxModule::new()
    .version(8, 0)
    .target("sm_70")
    .address_size(64);

let ptx_source = module.emit();
assert!(ptx_source.contains(".version 8.0"));
```

## Available Kernels

| Kernel | Description |
|--------|-------------|
| **GEMM** | Matrix multiplication (naive, tiled, tensor core) |
| **Softmax** | Numerically stable softmax with warp shuffle |
| **LayerNorm** | Fused layer normalization |
| **Attention** | FlashAttention-style tiled attention |
| **Quantize** | Q4_K dequantization fused with matmul |

## Usage

```rust
use trueno_gpu::kernels::{GemmKernel, Kernel};

// Create a tiled GEMM kernel
let kernel = GemmKernel::tiled(1024, 1024, 1024);
let ptx = kernel.emit_ptx();

// The PTX can be loaded by CUDA driver API
println!("{}", ptx);
```

## Modules

- `ptx` - PTX code generation (builder pattern)
- `kernels` - Hand-optimized GPU kernels
- `driver` - CUDA driver API (minimal FFI, optional)
- `memory` - GPU memory management
- `backend` - Multi-backend abstraction

## Requirements

- Rust 1.70+
- For GPU execution: NVIDIA CUDA driver (optional, only needed to run generated PTX)

## License

MIT License - see [LICENSE](../../../../LICENSE) for details.

## Part of Trueno

This crate is part of the [Trueno](https://github.com/paiml/trueno) high-performance compute library.
