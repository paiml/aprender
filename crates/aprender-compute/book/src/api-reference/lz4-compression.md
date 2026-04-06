# LZ4 Compression

trueno-gpu provides GPU-accelerated LZ4 compression kernels for high-throughput data compression. The implementation uses a warp-cooperative architecture optimized for ZRAM-style memory page compression.

## Overview

LZ4 is a fast compression algorithm that trades compression ratio for speed. It's widely used in:

- **ZRAM** (Linux kernel memory compression)
- **ZFS/Btrfs** (filesystem compression)
- **Database systems** (page compression)
- **Real-time streaming** (network compression)

trueno-gpu generates both NVIDIA PTX and WebGPU WGSL shaders from pure Rust, requiring no external toolchains.

## Architecture

### Warp-per-Page Strategy

Each warp (32 threads) processes one 4KB memory page cooperatively:

```
Block (128 threads = 4 warps)
┌────────┬────────┬────────┬────────┐
│ Warp 0 │ Warp 1 │ Warp 2 │ Warp 3 │
│ Page 0 │ Page 1 │ Page 2 │ Page 3 │
└────────┴────────┴────────┴────────┘
```

Each warp executes three phases:

1. **Cooperative Load**: All 32 threads load 128 bytes each (4KB total)
2. **Zero-Page Detection**: Parallel OR reduction to detect all-zero pages
3. **Hash-based Matching**: Find LZ4 match references (future enhancement)

### LZ4 Block Format

LZ4 encodes data as sequences of:

- **Literals**: Raw uncompressed bytes
- **Matches**: Back-references to previously seen data (offset + length)

Token format: `[4-bit literal length][4-bit match length]`

## API Reference

### CPU Reference Implementation

The CPU functions are useful for testing and validation:

```rust
use trueno_gpu::kernels::lz4::{
    lz4_compress_block,
    lz4_decompress_block,
    lz4_hash,
    PAGE_SIZE,
};

// Compress a 4KB page
let input = vec![0u8; PAGE_SIZE as usize];
let mut compressed = vec![0u8; PAGE_SIZE as usize];
let size = lz4_compress_block(&input, &mut compressed)?;

// Decompress back
let mut decompressed = vec![0u8; PAGE_SIZE as usize];
let decomp_size = lz4_decompress_block(&compressed[..size], &mut decompressed)?;
assert_eq!(decompressed, input);
```

### GPU Kernel Generation

Generate PTX or WGSL for GPU execution:

```rust
use trueno_gpu::kernels::{Lz4WarpCompressKernel, Kernel};

// Create kernel for batch of 1000 pages
let kernel = Lz4WarpCompressKernel::new(1000);

// Get launch configuration
let (grid_x, grid_y, grid_z) = kernel.grid_dim();   // (250, 1, 1)
let (block_x, block_y, block_z) = kernel.block_dim(); // (128, 1, 1)
let shared_mem = kernel.shared_memory_bytes();        // 49152 bytes

// Generate NVIDIA PTX
let ptx = kernel.emit_ptx();

// Generate WebGPU WGSL (cross-platform)
let wgsl = kernel.emit_wgsl();

// Analyze barrier safety (PARITY-114 prevention)
let safety = kernel.analyze_barrier_safety();
assert!(safety.is_safe);
```

### Constants

```rust
use trueno_gpu::kernels::lz4::{
    LZ4_MIN_MATCH,    // 4 - Minimum match length
    LZ4_MAX_MATCH,    // 274 - Maximum match length
    LZ4_HASH_BITS,    // 12 - Hash table index bits
    LZ4_HASH_SIZE,    // 4096 - Hash table entries
    LZ4_HASH_MULT,    // Knuth multiplicative hash constant
    LZ4_MAX_OFFSET,   // 65535 - Maximum match offset
    PAGE_SIZE,        // 4096 - Standard page size
};
```

## Compression Ratios

Typical compression ratios for different data types:

| Data Type | Input Size | Compressed | Ratio |
|-----------|------------|------------|-------|
| Zero page | 4096 bytes | ~20 bytes | 200:1 |
| Repeated pattern | 4096 bytes | ~275 bytes | 15:1 |
| Text data | 4096 bytes | ~130 bytes | 30:1 |
| Random data | 4096 bytes | ~4100 bytes | <1:1 |

## Example

Run the complete LZ4 demo:

```bash
cargo run -p trueno-gpu --example lz4_compression
```

Output:

```
╔══════════════════════════════════════════════════════════════╗
║       trueno-gpu LZ4 Compression Kernel Demo                ║
║   Pure Rust PTX/WGSL Generation - No nvcc Required          ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Part 1: CPU Reference Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Zero Page (4KB):
    Original:   4096 bytes
    Compressed: 20 bytes
    Ratio:      204.8:1
```

## Dual Backend Support

The LZ4 kernel supports both:

- **NVIDIA CUDA**: Via pure Rust PTX generation (no nvcc)
- **WebGPU**: Via WGSL generation (Vulkan/Metal/DX12/WebGPU)

This enables GPU-accelerated compression on any platform with GPU compute support.

## Use Cases

### ZRAM Memory Compression

Linux ZRAM uses LZ4 for memory page compression:

```rust
// Process 72GB of memory pages (18M pages)
let kernel = Lz4WarpCompressKernel::new(18_000_000);
let ptx = kernel.emit_ptx();
// Launch on GPU for 100+ GB/s throughput
```

### Database Page Compression

Compress database pages before writing to storage:

```rust
// Batch compress 1000 database pages
let kernel = Lz4WarpCompressKernel::new(1000);
let (grid, _, _) = kernel.grid_dim();
// 250 blocks × 4 pages/block = 1000 pages
```

### Real-time Streaming

Compress network data with minimal latency:

```rust
// Small batches for low latency
let kernel = Lz4WarpCompressKernel::new(64);
// Grid: (16, 1, 1) - fits in single SM
```

## Performance Considerations

1. **Zero-page detection**: Pages that are all zeros compress to ~20 bytes (huge savings for sparse data)
2. **Batch size**: Larger batches amortize kernel launch overhead
3. **Shared memory**: 48KB per block limits occupancy
4. **PCIe transfer**: For small batches, CPU may be faster due to transfer overhead

## Future Enhancements

- Full hash-based match finding on GPU
- Streaming compression with ring buffers
- Multi-frame dictionary support
- Hardware-accelerated entropy coding

## See Also

- [PTX Code Generation](../architecture/ptx-generation.md)
- [GPU Performance](../performance/gpu-performance.md)
- [Barrier Safety](../development/ptx-best-practices.md)
