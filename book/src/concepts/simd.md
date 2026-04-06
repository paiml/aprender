# SIMD Acceleration

trueno-zram uses runtime CPU feature detection to select the optimal SIMD implementation.

## Supported Backends

| Backend | Instruction Set | Register Width | Platforms |
|---------|-----------------|----------------|-----------|
| AVX-512 | AVX-512F/BW/VL | 512-bit | Skylake-X, Ice Lake, Zen 4 |
| AVX2 | AVX2 + FMA | 256-bit | Haswell+, Zen 1+ |
| NEON | ARM NEON | 128-bit | ARMv8-A (AArch64) |
| Scalar | None | 64-bit | All platforms |

## Runtime Detection

```rust
use trueno_zram_core::simd::{detect, SimdFeatures};

let features = detect();

println!("AVX-512: {}", features.has_avx512());
println!("AVX2: {}", features.has_avx2());
println!("SSE4.2: {}", features.has_sse42());
```

## Automatic Dispatch

The compressor automatically uses the best available backend:

```rust
use trueno_zram_core::CompressorBuilder;

let compressor = CompressorBuilder::new().build()?;

// Check which backend was selected
println!("Backend: {:?}", compressor.backend());
```

## Performance by Backend

### LZ4 Compression

| Backend | Throughput | Relative |
|---------|------------|----------|
| AVX-512 | 4.4 GB/s | 1.45x |
| AVX2 | 3.2 GB/s | 1.05x |
| Scalar | 3.0 GB/s | 1.0x |

### ZSTD Compression

| Backend | Throughput | Relative |
|---------|------------|----------|
| AVX-512 | 11.2 GB/s | 1.40x |
| AVX2 | 8.5 GB/s | 1.06x |
| Scalar | 8.0 GB/s | 1.0x |

## SIMD Optimizations

### Hash Table Lookups (LZ4)

AVX-512 enables parallel hash probing for match finding:

```
// Scalar: Sequential probe
for offset in 0..16 {
    if hash_table[hash + offset] == pattern { ... }
}

// AVX-512: Parallel probe (16 comparisons at once)
let matches = _mm512_cmpeq_epi32(hash_values, pattern_broadcast);
```

### Literal Copying

Wide vector moves for copying uncompressed literals:

```
// AVX-512: 64 bytes per iteration
_mm512_storeu_si512(dst, _mm512_loadu_si512(src));

// AVX2: 32 bytes per iteration
_mm256_storeu_si256(dst, _mm256_loadu_si256(src));
```

### Match Extension

SIMD comparison for extending matches:

```rust
// Compare 64 bytes at once with AVX-512
let cmp = _mm512_cmpeq_epi8(src_chunk, dst_chunk);
let mask = _mm512_movepi8_mask(cmp);
let match_len = mask.trailing_ones();
```

## Forcing a Backend

For testing or benchmarking, you can force a specific backend:

```rust
use trueno_zram_core::{CompressorBuilder, SimdBackend};

// Force scalar (no SIMD)
let scalar = CompressorBuilder::new()
    .prefer_backend(SimdBackend::Scalar)
    .build()?;

// Force AVX2 (will fail if not available)
let avx2 = CompressorBuilder::new()
    .prefer_backend(SimdBackend::Avx2)
    .build()?;
```
