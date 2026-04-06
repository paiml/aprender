<h1 align="center">trueno</h1>

<p align="center">
  <strong>SIMD/GPU Compute Primitives for the Sovereign AI Stack</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/trueno">
    <img src="https://img.shields.io/crates/v/trueno.svg" alt="crates.io">
  </a>
  <a href="https://docs.rs/trueno">
    <img src="https://docs.rs/trueno/badge.svg" alt="docs.rs">
  </a>
  <a href="https://github.com/paiml/trueno/actions">
    <img src="https://github.com/paiml/trueno/actions/workflows/ci.yml/badge.svg"
         alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://www.rust-lang.org">
    <img src="https://img.shields.io/badge/rust-1.89%2B-orange.svg" alt="Rust 1.89+">
  </a>
</p>

<p align="center">
  <a href="#what-is-trueno">What is trueno?</a> |
  <a href="#installation">Installation</a> |
  <a href="#usage">Usage</a> |
  <a href="#features">Features</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#quality">Quality</a> |
  <a href="#sovereign-ai-stack">Stack</a> |
  <a href="#documentation">Docs</a>
</p>

---

## Table of Contents

- [What is trueno?](#what-is-trueno)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Quality](#quality)
- [Sovereign AI Stack](#sovereign-ai-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## What is trueno?

**trueno** (Spanish: "thunder") is a pure Rust high-performance compute
library providing unified primitives across CPU SIMD, GPU, and
WebAssembly. It is the foundation layer of the PAIML Sovereign AI Stack,
powering tensor operations, quantized inference kernels, and training
backward passes for the entire ecosystem.

trueno auto-detects the best available hardware at runtime -- AVX-512 on
server CPUs, AVX2 on desktops, NEON on ARM, SIMD128 on WASM -- and
dispatches to hand-tuned kernels without configuration. For large
matrices, wgpu GPU compute (Vulkan/Metal/DX12/WebGPU) is available as
an optional backend. For NVIDIA hardware, trueno-gpu generates CUDA PTX
kernels in pure Rust with no external toolchain.

## Installation

```toml
[dependencies]
trueno = "0.17"

# Optional: wgpu GPU support for large matrices
trueno = { version = "0.17", features = ["gpu"] }

# Optional: pure Rust CUDA PTX generation
trueno-gpu = "0.4"
```

## Usage

### Vector and Matrix Operations

```rust
use trueno::{Vector, Matrix, SymmetricEigen};

// Vector operations -- auto-selects best SIMD backend
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

let sum = a.add(&b).unwrap();           // [6.0, 8.0, 10.0, 12.0]
let dot = a.dot(&b).unwrap();           // 70.0
let activated = a.relu().unwrap();      // ReLU activation

// Matrix operations
let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
let product = m.matmul(&m).unwrap();    // Matrix multiplication
let transposed = m.transpose();          // Transpose

// Eigendecomposition (PCA, spectral analysis)
let cov = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).unwrap();
let eigen = SymmetricEigen::new(&cov).unwrap();
let eigenvalues = eigen.eigenvalues();  // [4.0, 2.0]
```

### Batched Matmul for Transformers

```rust
use trueno::Matrix;

// Q @ K^T pattern: batched 4D matmul
let batch = 2; let heads = 4; let seq = 8; let dim = 64;
let q: Vec<f32> = vec![0.1; batch * heads * seq * dim];
let kt: Vec<f32> = vec![0.1; batch * heads * dim * seq];
let attn = Matrix::batched_matmul_4d(
    &q, &kt, batch, heads, seq, dim, seq
).unwrap();
```

### GPU Training (wgpu Backward Pass)

```rust
use trueno::backends::gpu::GpuDevice;

let dev = GpuDevice::new()?;

// Backward pass: compute SiLU gradient
dev.silu_backward(&input, &grad_output, &mut grad_input)?;

// Optimizer step: AdamW update
dev.adamw_step(
    &mut params, &grads, &mut m, &mut v,
    lr, beta1, beta2, eps, weight_decay, step
)?;
```

### Pure Rust CUDA PTX (trueno-gpu)

```rust
use trueno_gpu::kernels::{GemmKernel, Kernel, SoftmaxKernel};

// Generate optimized GEMM kernel (supports sm_121 Blackwell via PTX 8.8)
let gemm = GemmKernel::tensor_core(1024, 1024, 1024);
let ptx = gemm.emit_ptx();  // Pure Rust PTX generation, no nvcc

// Generate softmax with warp shuffle reduction
let softmax = SoftmaxKernel::new(4096);
let ptx = softmax.emit_ptx();
```

## Features

### BLIS 5-Loop GEMM

Production-grade matrix multiplication implementing the BLIS
(BLAS-like Library Instantiation Software) 5-loop algorithm with
compile-time microkernel specialization via `trueno-gemm-codegen`.
Cache-oblivious tiling for L1/L2/L3 with architecture-specific
register blocking.

### SIMD Auto-Detection

Runtime detection and dispatch across four ISA families:

| ISA | Instructions | Use Case |
|-----|-------------|----------|
| AVX-512 | 512-bit vectors, VNNI | Server CPUs (Zen 4, Sapphire Rapids) |
| AVX2 | 256-bit vectors, FMA | Desktop CPUs (Haswell+) |
| NEON | 128-bit vectors | ARM (Apple Silicon, Graviton, Jetson) |
| SIMD128 | 128-bit vectors | WebAssembly |

No configuration required. trueno probes `cpuid` at startup and
selects the widest available path.

### Quantization Kernels

Fused dequantization + matmul kernels for quantized LLM inference:

| Kernel | Block Size | Bytes/Block | Description |
|--------|-----------|-------------|-------------|
| Q4K | 256 | 144 | 4-bit quantization with K-means centroids |
| Q5K | 256 | 176 | 5-bit quantization |
| Q6K | 256 | 210 | 6-bit quantization |
| Q8K | 256 | 292 | 8-bit quantization |

Each kernel has scalar, AVX2, and AVX-512 implementations with
parity tests ensuring identical output across all paths.

### wgpu GPU Compute

Cross-platform GPU via WGSL compute shaders (Vulkan, Metal, DX12,
WebGPU). Seven backward pass operations for neural network training
without CUDA:

- `silu_backward` -- SiLU activation gradient
- `gemm_backward_a` / `gemm_backward_b` -- weight and input gradients
- `rmsnorm_backward` -- RMSNorm gradient
- `rope_backward` -- rotary position embedding gradient
- `adamw_step` -- AdamW optimizer parameter update
- `nf4_dequant` -- NF4 4-bit dequantization for QLoRA

### LZ4/ZSTD Compression

SIMD-accelerated compression integrated into the tensor pipeline,
used by the APR v2 model format for compressed tensor storage.

### Pure Rust CUDA (trueno-gpu)

Generate CUDA PTX kernels without nvcc, LLVM, or external toolchains.
Supports sm_121 Blackwell via PTX 8.8 ISA. Available kernels: GEMM
(naive/tiled/tensor core), Softmax, LayerNorm, RMSNorm, Attention,
GEMV, Quantization (Q4K/Q5K/Q6K). JIT disk cache at
`~/.cache/trueno/ptx/` for instant kernel reload.

### Feature Flags

| Feature | Description |
|---------|-------------|
| `gpu` | wgpu GPU compute (Vulkan/Metal/DX12/WebGPU) |
| `gpu-wasm` | WebGPU for WASM targets |
| `parallel` | Rayon parallel iterators |
| `cuda-monitor` | NVIDIA device info via trueno-gpu |
| `cuda` | Full CUDA support via trueno-gpu |
| `ml-tuner` | ML-based kernel selection via aprender RandomForest |
| `execution-graph` | PTX-to-kernel call graph profiling |
| `tui-monitor` | Terminal monitoring dashboard |
| `hardware-detect` | Hardware capability detection and caching |
| `dhat-heap` | Heap profiling via dhat-rs |

## Architecture

```
trueno (core library)
|
|-- src/backends/
|   |-- scalar.rs        Portable reference implementations
|   |-- avx2.rs          AVX2 SIMD kernels (256-bit)
|   |-- avx512.rs        AVX-512 SIMD kernels (512-bit)
|   |-- neon.rs          ARM NEON kernels (128-bit)
|   |-- wasm_simd.rs     WASM SIMD128 kernels
|   |-- gpu/             wgpu WGSL compute shaders
|   +-- dispatch.rs      Runtime backend selection
|
|-- src/matrix.rs        Matrix ops, BLIS GEMM, batched matmul
|-- src/vector.rs        Vector ops, activations, reductions
|-- src/quant/           Q4K/Q5K/Q6K/Q8K fused kernels
|-- src/eigen.rs         Symmetric eigendecomposition (Jacobi)
|-- src/conv.rs          2D convolution, pooling
+-- src/tuner.rs         ML-based kernel selection

trueno-gpu (companion crate)
|
+-- Pure Rust PTX generation for NVIDIA GPUs (sm_50 -- sm_121)
```

## Performance

| Operation | SIMD Speedup | Notes |
|-----------|--------------|-------|
| Dot product | 6-17x | AVX-512 for compute-bound |
| Matrix multiply | 2-10x | GPU for 500x500+ |
| Reductions (sum, max, min) | 3-12x | AVX-512 optimal |
| Element-wise (add, mul) | 1-2x | Memory-bound |
| Convolution 2D | 5-8x | AVX2/AVX-512 optimized |

GPU acceleration benefits matrix multiplication. Element-wise
operations use CPU SIMD because GPU transfer overhead exceeds
compute time.

## Operations

**Vector**: add, sub, mul, div, dot, sum, min, max, argmin, argmax,
norm\_l1, norm\_l2, normalize, recip, sqrt, abs, clamp

**Activations**: relu, leaky\_relu, elu, sigmoid, tanh, gelu, swish,
softmax, log\_softmax, silu

**Matrix**: matmul, batched\_matmul, batched\_matmul\_4d, transpose,
matvec, convolve2d, pooling (max/avg), topk, gather, pad

**Statistics**: mean, variance, stddev, covariance, correlation, zscore

**Eigen**: symmetric eigendecomposition (Jacobi algorithm)

**GPU Kernels**: GEMM (naive/tiled/tensor core), Softmax, LayerNorm,
RMSNorm, Attention, GEMV, Quantization (Q4K/Q5K/Q6K)

## Quality

### Falsifiable Commitments

| Claim | Falsification Test | Status |
|-------|-------------------|--------|
| AVX-512 and AVX2 produce identical results to scalar | Parity tests across all vector/matrix ops | Passing |
| Q4K/Q5K/Q6K fused kernels match scalar reference | Per-block output comparison with epsilon tolerance | Passing |
| BLIS GEMM matches naive matmul | Property-based tests (proptest) against reference | Passing |
| wgpu backward ops produce correct gradients | 8 FALSIFY contract tests on AMD Radeon Pro W5700X | Passing |
| PTX generation produces valid NVIDIA ISA | Regex validation of emitted PTX instructions | Passing |
| Runtime SIMD detection never selects unsupported ISA | Feature probe matches cpuid at startup | Passing |

### Metrics

- 97% test coverage
- Zero clippy warnings (`-D warnings`)
- Mutation testing via cargo-mutants
- provable-contracts enforcement: 100 bindings, AllImplemented

## Sovereign AI Stack

trueno is the compute foundation for the PAIML Sovereign AI Stack.

| Layer | Crate | Relationship |
|-------|-------|-------------|
| Orchestration | [batuta](https://crates.io/crates/batuta) | Stack coordinator, uses trueno for analysis |
| ML | [aprender](https://crates.io/crates/aprender) | ML algorithms built on trueno SIMD |
| Training | [entrenar](https://crates.io/crates/entrenar) | Autograd and LoRA training on trueno tensors |
| Inference | [realizar](https://crates.io/crates/realizar) | LLM inference using trueno quantization kernels |
| Distribution | [repartir](https://crates.io/crates/repartir) | Distributed compute with trueno tensor integration |
| Database | [trueno-db](https://crates.io/crates/trueno-db) | GPU-first analytics database |
| Graph | [trueno-graph](https://crates.io/crates/trueno-graph) | Graph algorithms for code analysis |
| RAG | [trueno-rag](https://crates.io/crates/trueno-rag) | RAG pipeline (chunking, BM25+vector, RRF) |
| Visualization | [trueno-viz](https://crates.io/crates/trueno-viz) | Terminal and PNG visualization |
| Compression | [trueno-zram-core](https://crates.io/crates/trueno-zram-core) | SIMD/GPU memory compression |
| Block Device | [trueno-ublk](https://crates.io/crates/trueno-ublk) | GPU-accelerated ZRAM replacement |
| PTX Codegen | [trueno-gpu](https://crates.io/crates/trueno-gpu) | Pure Rust CUDA PTX generation |
| Contracts | [provable-contracts](https://crates.io/crates/provable-contracts) | YAML contract verification |

## Documentation

- **API Reference**: [docs.rs/trueno](https://docs.rs/trueno)
- **Examples**: [`examples/`](examples/) -- 34 runnable examples
- **Benchmarks**: [`benches/`](benches/) -- criterion benchmarks
- **trueno-gpu docs**: [docs.rs/trueno-gpu](https://docs.rs/trueno-gpu)

## Contributing

1. All tests pass: `cargo test --all-features`
2. Coverage stays above 90%: `make coverage`
3. No clippy warnings: `cargo clippy --all-features -- -D warnings`
4. Code is formatted: `cargo fmt`

## License

MIT -- see [LICENSE](LICENSE).
