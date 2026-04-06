# CUDA PTX Generation

Realizar provides pure Rust CUDA PTX generation via the `trueno-gpu` crate. This enables native NVIDIA GPU acceleration without requiring LLVM, nvcc, or the CUDA toolkit for code generation.

## Philosophy

**Own the Stack** - Build everything from first principles for complete control, auditability, and reproducibility. The PTX generation is 100% Rust with no external dependencies.

## Architecture

```
+-----------------------+
|   CudaKernels API     |  <- Safe public API (realizar::cuda)
+-----------------------+
|   trueno_gpu::kernels |  <- Hand-optimized PTX kernels
+-----------------------+
|   trueno_gpu::ptx     |  <- Pure Rust PTX generation
+-----------------------+
|   CUDA Driver API     |  <- Runtime execution (optional)
+-----------------------+
```

## Enabling CUDA Support

Add the `cuda` feature to your `Cargo.toml`:

```toml
[dependencies]
realizar = { version = "0.2", features = ["cuda"] }
```

Or use the `full` feature which includes CUDA:

```toml
[dependencies]
realizar = { version = "0.2", features = ["full"] }
```

## Available Kernels

| Kernel | Description | Use Case |
|--------|-------------|----------|
| **GEMM Naive** | Simple matrix multiplication | Reference/debugging |
| **GEMM Tiled** | Shared memory tiled GEMM | Medium matrices |
| **GEMM Tensor Core** | FP16 tensor core GEMM | Maximum performance |
| **Softmax** | Numerically stable with warp shuffle | Attention scores |
| **LayerNorm** | Fused layer normalization | Transformer layers |
| **Attention** | FlashAttention-style tiled attention | Self-attention |
| **QuantizedGemm** | Q4_K dequantization fused GEMM | Quantized inference |

## Basic Usage

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

// Create the kernel generator
let kernels = CudaKernels::new();

// Generate PTX for Q4_K quantized GEMM
let ptx = kernels.generate_ptx(&KernelType::QuantizedGemm {
    m: 1024,
    n: 1024,
    k: 4096,
});

// The PTX can be loaded by CUDA driver API
println!("{}", ptx);
```

## Kernel Types

### Matrix Multiplication (GEMM)

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

let kernels = CudaKernels::new();

// Naive GEMM - simple, for reference
let ptx = kernels.generate_ptx(&KernelType::GemmNaive {
    m: 128,  // Output rows
    n: 128,  // Output columns
    k: 128,  // Inner dimension
});

// Tiled GEMM - shared memory optimization
let ptx = kernels.generate_ptx(&KernelType::GemmTiled {
    m: 1024,
    n: 1024,
    k: 1024,
    tile_size: 32,  // Shared memory tile size
});

// Tensor Core GEMM - FP16 for maximum throughput
let ptx = kernels.generate_ptx(&KernelType::GemmTensorCore {
    m: 1024,
    n: 1024,
    k: 1024,
});
```

### Softmax

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

let kernels = CudaKernels::new();

// Numerically stable softmax with warp shuffle reduction
let ptx = kernels.generate_ptx(&KernelType::Softmax {
    dim: 4096,  // Vector dimension
});
```

### Layer Normalization

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

let kernels = CudaKernels::new();

// Full LayerNorm with affine transform
let ptx = kernels.generate_ptx(&KernelType::LayerNorm {
    hidden_size: 4096,
    epsilon: 1e-5,
    affine: true,  // Include gamma/beta
});

// RMSNorm variant (no affine)
let ptx = kernels.generate_ptx(&KernelType::LayerNorm {
    hidden_size: 4096,
    epsilon: 1e-6,
    affine: false,
});
```

### FlashAttention

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

let kernels = CudaKernels::new();

// FlashAttention-style tiled attention
let ptx = kernels.generate_ptx(&KernelType::Attention {
    seq_len: 2048,   // Sequence length
    head_dim: 64,    // Head dimension
    causal: true,    // Causal masking for autoregressive
});
```

### Quantized GEMM

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

let kernels = CudaKernels::new();

// Q4_K quantized GEMM with fused dequantization
let ptx = kernels.generate_ptx(&KernelType::QuantizedGemm {
    m: 1,      // Batch size (often 1 for inference)
    n: 4096,   // Output dimension
    k: 4096,   // Input dimension (must be divisible by 32)
});
```

## LLM Inference Presets

Realizar provides pre-configured kernel settings optimized for common LLM patterns:

```rust,ignore
use realizar::cuda::{CudaKernels, presets};

let kernels = CudaKernels::new();

// Llama-style causal attention
let attention = presets::llama_attention(2048, 64);
let ptx = kernels.generate_ptx(&attention);

// Feed-forward network GEMM
let ffn = presets::ffn_gemm(32, 4096, 11008);
let ptx = kernels.generate_ptx(&ffn);

// Q4_K quantized inference
let q4k = presets::q4k_inference(1, 4096, 4096);
let ptx = kernels.generate_ptx(&q4k);

// RMSNorm (Llama-style LayerNorm)
let rmsnorm = presets::rmsnorm(4096);
let ptx = kernels.generate_ptx(&rmsnorm);
```

## CUDA Availability Detection

```rust,ignore
use realizar::cuda::CudaKernels;

// Heuristic check for NVIDIA GPU
if CudaKernels::cuda_likely_available() {
    println!("CUDA likely available - generating PTX kernels");
} else {
    println!("No CUDA detected - falling back to CPU");
}
```

This checks for:
- `/dev/nvidia0` device node
- `CUDA_VISIBLE_DEVICES` environment variable

## Kernel Names

Each kernel has a canonical name for CUDA driver API loading:

```rust,ignore
use realizar::cuda::{CudaKernels, KernelType};

let kernels = CudaKernels::new();

assert_eq!(
    kernels.kernel_name(&KernelType::GemmNaive { m: 1, n: 1, k: 1 }),
    "gemm_naive"
);
assert_eq!(
    kernels.kernel_name(&KernelType::Softmax { dim: 1 }),
    "softmax_warp"
);
assert_eq!(
    kernels.kernel_name(&KernelType::QuantizedGemm { m: 1, n: 1, k: 32 }),
    "q4k_gemm_fused"
);
```

## PTX Output Format

Generated PTX follows NVIDIA's PTX ISA:

```ptx
.version 8.0
.target sm_70
.address_size 64

.visible .entry gemm_tiled(
    .param .u64 param_A,
    .param .u64 param_B,
    .param .u64 param_C,
    .param .u32 param_M,
    .param .u32 param_N,
    .param .u32 param_K
) {
    // Shared memory declarations
    .shared .align 4 .f32 smem_A[1024];
    .shared .align 4 .f32 smem_B[1024];

    // Kernel implementation...
}
```

## Performance Characteristics

| Kernel | Compute Bound | Memory Bound | Typical Speedup |
|--------|--------------|--------------|-----------------|
| GEMM Naive | Medium | High | 1x (baseline) |
| GEMM Tiled | High | Low | 10-50x |
| GEMM Tensor Core | Very High | Very Low | 50-200x |
| Softmax | Low | Medium | 5-10x |
| LayerNorm | Low | Medium | 5-10x |
| Attention | High | Medium | 10-30x |
| Q4_K GEMM | High | Low | 20-100x |

## Test Coverage

The CUDA module has comprehensive test coverage:

```bash
cargo test --features cuda cuda::
```

Tests verify:
- Kernel creation and PTX generation
- All kernel types produce valid PTX
- Preset configurations match expected patterns
- Kernel names are correct
- Default trait implementation

## Integration with CUDA Driver API

The generated PTX can be loaded using the CUDA driver API:

```c
// C example - load PTX module
CUmodule module;
cuModuleLoadData(&module, ptx_source);

CUfunction kernel;
cuModuleGetFunction(&kernel, module, "gemm_tiled");

// Launch kernel
cuLaunchKernel(kernel, grid_x, grid_y, grid_z,
               block_x, block_y, block_z,
               shared_mem_bytes, stream,
               args, extra);
```

## Related Documentation

- [Trueno Backend](./trueno-backend.md) - wgpu-based GPU acceleration
- [SIMD Optimization](./simd.md) - CPU SIMD backends
- [GPU Dispatch Strategy](./dispatch-strategy.md) - Automatic CPU/GPU dispatch
