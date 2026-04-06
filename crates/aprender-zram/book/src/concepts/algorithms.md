# Compression Algorithms

trueno-zram supports multiple compression algorithms optimized for memory page compression.

## LZ4

LZ4 is a lossless compression algorithm focused on compression and decompression speed.

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;
```

### Characteristics

| Metric | Value |
|--------|-------|
| Compression speed | 4.4 GB/s (AVX-512) |
| Decompression speed | 5.4 GB/s (AVX-512) |
| Typical ratio | 2-4x for compressible data |
| Best for | Speed-critical workloads |

### When to Use LZ4

- Real-time compression requirements
- High-throughput memory compression
- When CPU overhead must be minimal
- Mixed workloads with varying compressibility

## ZSTD (Zstandard)

ZSTD provides better compression ratios while maintaining good speed.

```rust
// Fast mode (level 1)
let fast = CompressorBuilder::new()
    .algorithm(Algorithm::Zstd { level: 1 })
    .build()?;

// Better compression (level 3)
let better = CompressorBuilder::new()
    .algorithm(Algorithm::Zstd { level: 3 })
    .build()?;
```

### Compression Levels

| Level | Compression | Decompression | Ratio |
|-------|-------------|---------------|-------|
| 1 | 11.2 GB/s | 46 GB/s | Better than LZ4 |
| 3 | 8.5 GB/s | 45 GB/s | Best |

### When to Use ZSTD

- Memory-constrained systems
- Highly compressible data (text, logs)
- When compression ratio matters more than speed
- Cold/archived memory pages

## Adaptive Selection

The adaptive algorithm automatically selects based on page entropy:

```rust
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Adaptive)
    .build()?;
```

### Selection Logic

1. **Same-fill detection**: Pages with repeated values use 2048:1 encoding
2. **Entropy analysis**: Shannon entropy determines compressibility
3. **Algorithm selection**:
   - Low entropy (< 4 bits): ZSTD for best ratio
   - Medium entropy (4-7 bits): LZ4 for balance
   - High entropy (> 7 bits): Pass-through (incompressible)

## Same-Fill Optimization

Pages filled with the same byte value get special 2048:1 compression:

```rust
use trueno_zram_core::samefill::{detect_same_fill, CompactSameFill};

let zero_page = [0u8; 4096];

if let Some(fill) = detect_same_fill(&zero_page) {
    let compact = CompactSameFill::new(fill);
    // Only 2 bytes needed to represent 4096-byte page!
    assert_eq!(compact.to_bytes().len(), 2);
}
```

### Same-Fill Statistics

- Zero pages: ~30-40% of typical memory
- Same-fill pages: ~35-45% total
- Compression ratio: 2048:1

## Algorithm Comparison

| Algorithm | Compress | Decompress | Ratio | Use Case |
|-----------|----------|------------|-------|----------|
| LZ4 | 4.4 GB/s | 5.4 GB/s | 2-4x | General |
| ZSTD-1 | 11.2 GB/s | 46 GB/s | 3-5x | Balanced |
| ZSTD-3 | 8.5 GB/s | 45 GB/s | 4-6x | Best ratio |
| Same-fill | N/A | N/A | 2048x | Zero/repeated |
| Adaptive | Varies | Varies | Optimal | Automatic |
