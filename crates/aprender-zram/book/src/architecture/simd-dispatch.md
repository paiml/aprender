# SIMD Dispatch

## Overview

trueno-zram uses runtime CPU feature detection to select the optimal SIMD implementation.

## Detection

```rust
// src/simd/detect.rs

pub fn detect() -> SimdFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        SimdFeatures {
            avx512: is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl"),
            avx2: is_x86_feature_detected!("avx2"),
            sse42: is_x86_feature_detected!("sse4.2"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        SimdFeatures {
            neon: true, // Always available on AArch64
        }
    }
}
```

## Dispatch Pattern

```rust
// src/simd/dispatch.rs

pub fn compress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let features = detect();

    #[cfg(target_arch = "x86_64")]
    {
        if features.has_avx512() {
            return unsafe { avx512::compress(input, output) };
        }
        if features.has_avx2() {
            return unsafe { avx2::compress(input, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if features.has_neon() {
            return unsafe { neon::compress(input, output) };
        }
    }

    scalar::compress(input, output)
}
```

## Implementation Structure

Each algorithm has separate implementations:

```
lz4/
├── mod.rs          # Public API, dispatch
├── compress.rs     # Core algorithm logic
├── decompress.rs   # Decompression
├── avx512.rs       # AVX-512 specialization
├── avx2.rs         # AVX2 specialization
├── neon.rs         # ARM NEON specialization
└── scalar.rs       # Fallback implementation
```

## AVX-512 Implementation

```rust
// lz4/avx512.rs

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn compress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    // 64-byte hash table lookups
    let hash_chunk = _mm512_loadu_si512(input.as_ptr());

    // Parallel match finding
    let matches = _mm512_cmpeq_epi32(hash_chunk, pattern);

    // Wide literal copies
    _mm512_storeu_si512(output.as_mut_ptr(), data);

    // ...
}
```

## AVX2 Implementation

```rust
// lz4/avx2.rs

#[target_feature(enable = "avx2")]
pub unsafe fn compress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    // 32-byte operations
    let chunk = _mm256_loadu_si256(input.as_ptr());

    // Match extension
    let cmp = _mm256_cmpeq_epi8(src, dst);
    let mask = _mm256_movemask_epi8(cmp);

    // ...
}
```

## NEON Implementation

```rust
// lz4/neon.rs

#[cfg(target_arch = "aarch64")]
pub unsafe fn compress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    use std::arch::aarch64::*;

    // 16-byte operations
    let chunk = vld1q_u8(input.as_ptr());

    // Parallel comparison
    let cmp = vceqq_u8(src, dst);

    // ...
}
```

## Benchmarking Dispatch

```rust
use trueno_zram_core::simd::{detect, SimdBackend};

fn benchmark_all_backends() {
    let features = detect();
    let input = [0xAA; PAGE_SIZE];
    let mut output = [0u8; PAGE_SIZE * 2];

    // Benchmark available backends
    if features.has_avx512() {
        bench("AVX-512", || avx512::compress(&input, &mut output));
    }
    if features.has_avx2() {
        bench("AVX2", || avx2::compress(&input, &mut output));
    }
    bench("Scalar", || scalar::compress(&input, &mut output));
}
```

## Compile-Time Optimization

For maximum performance, enable target features:

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

Or for specific features:

```toml
rustflags = ["-C", "target-feature=+avx2,+avx512f,+avx512bw"]
```
