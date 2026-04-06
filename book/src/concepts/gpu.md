# GPU Batch Compression

trueno-zram supports CUDA GPU acceleration for batch compression of memory pages.

## Current Status (2026-01-06)

**Important:** GPU compression is currently blocked by NVIDIA PTX bug **F082** (Computed Address Bug). The production architecture uses:

- **Compression:** CPU SIMD (AVX-512) at 20-30 GB/s with 3.87x ratio
- **Decompression:** CPU parallel at 50+ GB/s (primary path)

GPU decompression is available but CPU parallel path is faster due to PCIe transfer overhead (~6 GB/s end-to-end vs 50+ GB/s CPU).

### NVIDIA F082 Bug (Computed Address Bug)

> **F081 (Loaded Value Bug) was FALSIFIED on 2026-01-05** - the pattern works correctly.

The **actual** bug is F082: addresses *computed from* shared memory values cause crashes:

```ptx
// F081 pattern - WORKS CORRECTLY (falsified):
ld.shared.u32 %r_val, [addr];      // Load from shared memory
st.global.u32 [dest], %r_val;      // Actually works!

// F082 pattern - CRASHES:
ld.shared.u32 %r_offset, [shared_addr];  // Load offset from shared memory
add.u64 %r_dest, %r_base, %r_offset;     // Compute destination address
st.global.u32 [%r_dest], %r_data;        // CRASH - address derived from shared load
```

Status: True root cause identified via Popperian falsification. See KF-002 and ublk-batched-gpu-compression.md for details.

## When to Use GPU

GPU decompression is beneficial when:

1. **Large batches**: 2000+ pages to decompress
2. **PCIe 5x rule satisfied**: Computation time > 5x transfer time
3. **GPU available**: CUDA-capable GPU with SM 7.0+

## Basic Usage

```rust
use trueno_zram_core::gpu::{GpuBatchCompressor, GpuBatchConfig, gpu_available};
use trueno_zram_core::{Algorithm, PAGE_SIZE};

if gpu_available() {
    let config = GpuBatchConfig {
        device_index: 0,
        algorithm: Algorithm::Lz4,
        batch_size: 1000,
        async_dma: true,
        ring_buffer_slots: 4,
    };

    let mut compressor = GpuBatchCompressor::new(config)?;

    let pages: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; 1000];
    let result = compressor.compress_batch(&pages)?;

    println!("Compression ratio: {:.2}x", result.compression_ratio());
}
```

## Configuration Options

```rust
pub struct GpuBatchConfig {
    /// CUDA device index (0 = first GPU)
    pub device_index: u32,

    /// Compression algorithm
    pub algorithm: Algorithm,

    /// Number of pages per batch
    pub batch_size: usize,

    /// Enable async DMA transfers
    pub async_dma: bool,

    /// Ring buffer slots for pipelining
    pub ring_buffer_slots: usize,
}
```

## Batch Results

The `BatchResult` provides timing breakdown:

```rust
let result = compressor.compress_batch(&pages)?;

// Timing components
println!("H2D transfer: {} ns", result.h2d_time_ns);
println!("Kernel execution: {} ns", result.kernel_time_ns);
println!("D2H transfer: {} ns", result.d2h_time_ns);
println!("Total time: {} ns", result.total_time_ns);

// Metrics
let throughput = result.throughput_bytes_per_sec(pages.len() * PAGE_SIZE);
println!("Throughput: {:.2} GB/s", throughput / 1e9);
println!("Compression ratio: {:.2}x", result.compression_ratio());
println!("PCIe 5x rule satisfied: {}", result.pcie_rule_satisfied());
```

## Backend Selection

Use `select_backend` to determine optimal backend:

```rust
use trueno_zram_core::gpu::{select_backend, gpu_available};

let batch_size = 5000;
let has_gpu = gpu_available();

let backend = select_backend(batch_size, has_gpu);
println!("Selected backend: {:?}", backend);
// Output: Gpu (for large batches with GPU available)
```

## Supported GPUs

| GPU | Architecture | SM | Optimal Batch |
|-----|--------------|-------|---------------|
| H100 | Hopper | 9.0 | 10,240 pages |
| A100 | Ampere | 8.0 | 8,192 pages |
| RTX 4090 | Ada | 8.9 | 14,745 pages |
| RTX 3090 | Ampere | 8.6 | 6,144 pages |

## Pure Rust PTX Generation

trueno-zram uses [trueno-gpu](https://crates.io/crates/trueno-gpu) for pure Rust PTX generation:

- No LLVM dependency
- No nvcc required
- Kernel code in Rust, compiled to PTX at runtime
- Warp-cooperative LZ4 compression (4 warps/block)

```rust
// The LZ4 kernel processes 4 pages per block (1 page per warp)
// Uses shared memory for hash tables and match finding
// cvta.shared.u64 for generic addressing
```
