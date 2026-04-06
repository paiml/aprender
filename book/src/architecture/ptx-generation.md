# PTX Code Generation (trueno-gpu)

trueno-gpu provides pure Rust PTX (Parallel Thread Execution) code generation for NVIDIA GPUs. This enables GPU kernel development without requiring LLVM, nvcc, or any external dependencies.

## Philosophy

**Own the Stack** - Build everything from first principles for complete control, auditability, and reproducibility.

## Quick Start

```rust
use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};

// Create a PTX module
let module = PtxModule::new()
    .version(8, 0)      // PTX ISA 8.0
    .target("sm_70")    // Volta+
    .address_size(64);  // 64-bit addressing

// Build a kernel with the fluent builder API
let kernel = PtxKernel::new("my_kernel")
    .param(PtxType::U64, "data_ptr")
    .param(PtxType::U32, "n")
    .build(|ctx| {
        // Generate PTX instructions
        let tid = ctx.special_reg(trueno_gpu::ptx::PtxReg::TidX);
        // ... more instructions
        ctx.ret();
    });

// Emit PTX source
let ptx_source = module.add_kernel(kernel).emit();
```

## Module Structure

A PTX module consists of:

- **Header**: Version, target architecture, address size
- **Declarations**: Register declarations, shared memory
- **Kernels**: One or more entry points

### Version and Target

```rust
// PTX ISA 8.0 for Ampere and newer
.version(8, 0)

// Target compute capability
.target("sm_70")  // Volta
.target("sm_75")  // Turing
.target("sm_80")  // Ampere
.target("sm_89")  // Ada Lovelace
.target("sm_90")  // Hopper
```

## Kernel Builder API

The `KernelBuilder` provides a fluent API for generating PTX instructions:

### Special Registers

```rust
// Thread and block IDs
ctx.special_reg(PtxReg::TidX);    // %tid.x
ctx.special_reg(PtxReg::TidY);    // %tid.y
ctx.special_reg(PtxReg::CtaIdX);  // %ctaid.x (block ID)
ctx.special_reg(PtxReg::NtidX);   // %ntid.x (block size)
```

### Arithmetic Operations

```rust
// Integer arithmetic
ctx.add_u32(a, b);
ctx.mul_wide_u32(a, b);     // 32x32 -> 64 bit
ctx.mad_lo_u32(a, b, c);    // a*b + c (low 32 bits)

// Floating point
ctx.add_f32(a, b);
ctx.mul_f32(a, b);
ctx.fma_f32(a, b, c);       // Fused multiply-add
```

### Memory Operations

```rust
// Load from global memory
let value = ctx.ld_global_f32(addr);

// Store to global memory
ctx.st_global_f32(addr, value);

// Load kernel parameters
let param = ctx.load_param_u32("param_name");
let ptr = ctx.load_param_u64("ptr_param");
```

### Control Flow

```rust
// Predicated branch
let pred = ctx.setp_ge_u32(idx, n);  // idx >= n
ctx.branch_if(pred, "exit");

// Unconditional branch
ctx.branch("loop_start");

// Labels
ctx.label("loop_start");
ctx.label("exit");

// Return
ctx.ret();
```

## Pre-built Kernels

trueno-gpu includes optimized kernel generators:

### GEMM (Matrix Multiplication)

```rust
use trueno_gpu::kernels::{GemmKernel, Kernel};

// Naive GEMM (for correctness testing)
let kernel = GemmKernel::naive(1024, 1024, 1024);

// Tiled GEMM (shared memory optimization)
let kernel = GemmKernel::tiled(1024, 1024, 1024, 32);

// Tensor Core GEMM (SM 7.0+)
let kernel = GemmKernel::tensor_core(1024, 1024, 1024);

// Generate PTX
let ptx = kernel.emit_ptx();
```

### Softmax

```rust
use trueno_gpu::kernels::{SoftmaxKernel, Kernel};

let kernel = SoftmaxKernel::new(1024);  // Vector length
let ptx = kernel.emit_ptx();
```

### Bias + Activation (Epilogue Kernel)

Fused bias addition with optional activation function, commonly used as an epilogue after GEMM:

```rust
use trueno_gpu::kernels::{BiasActivationKernel, Activation, Kernel};

// Bias only (no activation)
let kernel = BiasActivationKernel::new(4096, 256);  // n=4096, bias_size=256

// Bias + ReLU
let kernel = BiasActivationKernel::new(4096, 256).with_relu();

// Bias + GELU (Transformer default)
let kernel = BiasActivationKernel::new(4096, 256).with_gelu();

// Custom activation via builder
let kernel = BiasActivationKernel::new(4096, 256)
    .with_activation(Activation::GELU);

let ptx = kernel.emit_ptx();
```

| Activation | Formula | Use Case |
|------------|---------|----------|
| None | `x + bias` | Linear layer epilogue |
| ReLU | `max(0, x + bias)` | CNN layers |
| GELU | `(x + bias) * sigmoid(1.702 * (x + bias))` | Transformers |

**Note:** The `bias_size` is baked into the kernel at generation time for efficiency. The kernel computes `output[i] += bias[i % bias_size]`.

```bash
# Run the example
cargo run -p trueno-gpu --example bias_activation

# Run property tests and falsification tests
cargo test -p trueno-gpu bias_activation

# Run deep bug hunt (includes BiasActivation)
cargo run -p trueno-explain --example deep_bug_hunt
```

**Testing:** BiasActivationKernel includes 22 tests covering:
- Unit tests for configuration and PTX structure
- Property-based tests (proptest) for randomized validation
- Falsification tests verifying bounds checks, bias modulo, and activation correctness
- Mutation testing: 100% coverage (2 caught by tests, 4 caught by type system)

### Quantized GEMM (Q4_K, Q5_K, Q6_K)

Optimized kernels for quantized inference with GGML-compatible formats:

```rust
use trueno_gpu::kernels::{QuantizeKernel, Q5KKernel, Q6KKernel, Kernel};

// Q4_K: 4-bit quantization (144 bytes per 256 values)
let q4k = QuantizeKernel::ggml(1024, 1024, 4096);

// Q5_K: 5-bit quantization (176 bytes per 256 values) - PARITY-116
let q5k = Q5KKernel::new(1024, 1024, 4096);

// Q6_K: 6-bit quantization (210 bytes per 256 values) - PARITY-117
let q6k = Q6KKernel::new(1024, 1024, 4096);

let ptx = q5k.emit_ptx();
```

| Format | Bits | Bytes/256 | Accuracy | Use Case |
|--------|------|-----------|----------|----------|
| Q4_K | 4 | 144 | Good | Default inference |
| Q5_K | 5 | 176 | Better | Quality-sensitive |
| Q6_K | 6 | 210 | Best | Maximum accuracy |

## Memory Management

```rust
use trueno_gpu::memory::{MemoryPool, PoolConfig, GpuBuffer};

// Create memory pool
let config = PoolConfig::new(1024 * 1024 * 1024);  // 1GB
let pool = MemoryPool::new(config);

// Allocate buffer
let buffer: GpuBuffer<f32> = GpuBuffer::new(1024);
```

## Backend Detection

```rust
use trueno_gpu::backend::{detect_backend, Backend};

let backend = detect_backend();
println!("Using backend: {}", backend.name());
println!("Available: {}", backend.is_available());
```

## Running Examples

```bash
# PTX quickstart - vector addition kernel
cargo run -p trueno-gpu --example ptx_quickstart

# GEMM kernel generation
cargo run -p trueno-gpu --example gemm_kernel

# Bias + Activation epilogue kernel
cargo run -p trueno-gpu --example bias_activation

# Quantized GEMM (Q5_K/Q6_K)
cargo run -p trueno-gpu --example q5k_q6k_gemm
```

## PTX Type System

| Rust Type | PTX Type | Description |
|-----------|----------|-------------|
| `PtxType::U32` | `.u32` | 32-bit unsigned |
| `PtxType::U64` | `.u64` | 64-bit unsigned |
| `PtxType::S32` | `.s32` | 32-bit signed |
| `PtxType::F32` | `.f32` | Single precision |
| `PtxType::F64` | `.f64` | Double precision |
| `PtxType::F16` | `.f16` | Half precision |
| `PtxType::BF16` | `.bf16` | Brain float |
| `PtxType::Pred` | `.pred` | Predicate (1-bit) |

## State Spaces

| State Space | PTX | Scope | Speed |
|-------------|-----|-------|-------|
| Register | `.reg` | Per-thread | Fastest |
| Shared | `.shared` | Per-block | Fast |
| Global | `.global` | Device-wide | Slow |
| Local | `.local` | Per-thread spill | Slow |
| Constant | `.const` | Device-wide (cached) | Fast |
| Parameter | `.param` | Kernel args | - |

## Best Practices

1. **Minimize global memory access** - Use shared memory for data reuse
2. **Coalesce memory accesses** - Adjacent threads access adjacent memory
3. **Use FMA instructions** - `fma_f32` is faster than separate mul+add
4. **Avoid branch divergence** - Keep warps executing the same path
5. **Maximize occupancy** - Balance register usage vs parallelism

## Feature Flags

```toml
[dependencies]
trueno-gpu = { version = "0.1", features = ["cuda"] }
```

- `default` - PTX generation only (no CUDA runtime required)
- `cuda` - Enable CUDA driver FFI for actual execution

## Resources

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [trueno-gpu Examples](https://github.com/paiml/trueno/tree/main/trueno-gpu/examples)
