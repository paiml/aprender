# trueno-zram: SIMD Memory Compression

**trueno-zram** provides SIMD-accelerated compression for Linux zram and general-purpose memory compression. It achieves 3+ GB/s with LZ4 and up to 13 GB/s with ZSTD on AVX-512.

## Overview

trueno-zram delivers:
- **SIMD Acceleration**: AVX2/AVX-512/NEON optimized
- **Multiple Algorithms**: LZ4 (speed) and ZSTD (ratio)
- **Adaptive Selection**: Entropy-based algorithm choice
- **Page Compression**: 4KB aligned for zram integration
- **Optional CUDA**: GPU acceleration for batch compression

```
┌─────────────────────────────────────────────────────────────┐
│                    trueno-zram                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  LZ4 SIMD   │  │ ZSTD SIMD   │  │  Adaptive Selector  │  │
│  │  (3+ GB/s)  │  │ (13 GB/s)   │  │  (entropy-based)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  AVX-512     │     AVX2      │     NEON      │   Scalar    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```toml
[dependencies]
trueno-zram-core = "0.1"

# With adaptive compression
trueno-zram-adaptive = "0.1"

# With CUDA support
trueno-zram-cuda = { version = "0.1", optional = true }
```

## Quick Start

```rust
use trueno_zram_core::{Compressor, Algorithm};

// Create compressor with LZ4 (fastest)
let compressor = Compressor::new(Algorithm::Lz4);

// Compress data
let compressed = compressor.compress(&data)?;
println!("Ratio: {:.2}x", data.len() as f64 / compressed.len() as f64);

// Decompress
let decompressed = compressor.decompress(&compressed)?;
assert_eq!(data, decompressed);
```

## Algorithm Comparison

| Algorithm | Compress | Decompress | Ratio | Use Case |
|-----------|----------|------------|-------|----------|
| LZ4 | 3+ GB/s | 4+ GB/s | 2.1x | Speed-critical |
| ZSTD-1 | 500 MB/s | 1.5 GB/s | 2.8x | Balanced |
| ZSTD-3 | 300 MB/s | 1.5 GB/s | 3.2x | Better ratio |
| ZSTD-AVX512 | 13 GB/s | 15 GB/s | 3.2x | AVX-512 systems |
| Same-Fill | N/A | N/A | 2048:1 | Zero/repeated pages |

## SIMD Backend Selection

```rust
use trueno_zram_core::{SimdBackend, detect_backend};

// Auto-detect best available backend
let backend = detect_backend();
println!("Using: {:?}", backend);

// Force specific backend
let compressor = Compressor::builder()
    .algorithm(Algorithm::Lz4)
    .backend(SimdBackend::Avx512)
    .build()?;
```

### Backend Priority

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | AVX-512 | x86_64 with avx512f |
| 2 | AVX2 | x86_64 with avx2 |
| 3 | NEON | aarch64 |
| 4 | Scalar | Fallback |

## Page Compression

Optimized for 4KB page-aligned compression:

```rust
use trueno_zram_core::{PageCompressor, PAGE_SIZE};

let compressor = PageCompressor::new();

// Compress a 4KB page
let page: [u8; PAGE_SIZE] = get_page();
let compressed = compressor.compress_page(&page)?;

// Check if page is compressible
if compressed.len() < PAGE_SIZE / 2 {
    store_compressed(compressed);
} else {
    store_uncompressed(page);  // Not worth compressing
}
```

## Adaptive Compression

Entropy-based algorithm selection:

```rust
use trueno_zram_adaptive::AdaptiveCompressor;

let compressor = AdaptiveCompressor::new();

// Automatically selects best algorithm per-page
let result = compressor.compress_adaptive(&data)?;

match result.algorithm_used {
    Algorithm::SameFill => println!("Zero/repeated page"),
    Algorithm::Lz4 => println!("High entropy, used LZ4"),
    Algorithm::Zstd { .. } => println!("Compressible, used ZSTD"),
}
```

### Decision Tree

```
Is page all zeros/same byte?
  YES → Same-Fill (2048:1 ratio)
  NO  → Check entropy
        High entropy → LZ4 (fast, low ratio)
        Low entropy  → ZSTD (slower, high ratio)
```

## Performance Benchmarks

Measured on AMD EPYC 7763 (AVX-512):

| Algorithm | Scalar | AVX2 | AVX-512 |
|-----------|--------|------|---------|
| LZ4 compress | 800 MB/s | 2.1 GB/s | 3.2 GB/s |
| LZ4 decompress | 1.2 GB/s | 3.5 GB/s | 4.5 GB/s |
| ZSTD-1 | 150 MB/s | 350 MB/s | 500 MB/s |
| ZSTD-fast | 400 MB/s | 8 GB/s | 13 GB/s |

## Running the Example

```bash
cargo run --example trueno_zram_demo
```

## Related Crates

- **trueno-ublk**: GPU-accelerated block device using trueno-zram
- **trueno**: SIMD/GPU compute primitives

## References

- [trueno-zram on crates.io](https://crates.io/crates/trueno-zram-core)
- [LZ4 Specification](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md)
- [Zstandard](https://facebook.github.io/zstd/)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Previous: whisper.apr](./whisper-apr.md) | [Next: trueno-ublk](./trueno-ublk.md)
