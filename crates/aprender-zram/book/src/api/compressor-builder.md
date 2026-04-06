# CompressorBuilder API

The `CompressorBuilder` provides a fluent API for configuring compression.

## Basic Usage

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;
```

## Builder Methods

### `algorithm(algo: Algorithm)`

Sets the compression algorithm:

```rust
// LZ4 (fastest)
.algorithm(Algorithm::Lz4)

// ZSTD with compression level
.algorithm(Algorithm::Zstd { level: 1 })
.algorithm(Algorithm::Zstd { level: 3 })

// Adaptive (auto-select based on entropy)
.algorithm(Algorithm::Adaptive)
```

### `prefer_backend(backend: SimdBackend)`

Forces a specific SIMD backend:

```rust
use trueno_zram_core::SimdBackend;

// Force scalar (no SIMD)
.prefer_backend(SimdBackend::Scalar)

// Force AVX2
.prefer_backend(SimdBackend::Avx2)

// Force AVX-512
.prefer_backend(SimdBackend::Avx512)
```

### `build()`

Creates the compressor:

```rust
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?; // Returns Result<Compressor, Error>
```

## Compressor Methods

### `compress(&self, page: &[u8; PAGE_SIZE]) -> Result<CompressedPage>`

Compresses a single page:

```rust
let page = [0u8; PAGE_SIZE];
let compressed = compressor.compress(&page)?;

println!("Size: {} bytes", compressed.data.len());
println!("Ratio: {:.2}x", compressed.ratio());
```

### `decompress(&self, page: &CompressedPage) -> Result<[u8; PAGE_SIZE]>`

Decompresses a page:

```rust
let decompressed = compressor.decompress(&compressed)?;
assert_eq!(page, decompressed);
```

### `stats(&self) -> CompressionStats`

Returns compression statistics:

```rust
let stats = compressor.stats();

println!("Pages: {}", stats.pages);
println!("Bytes in: {}", stats.bytes_in);
println!("Bytes out: {}", stats.bytes_out);
println!("Ratio: {:.2}x", stats.ratio());
println!("Compress time: {} ns", stats.compress_time_ns);
println!("Decompress time: {} ns", stats.decompress_time_ns);
println!("Throughput: {:.2} GB/s", stats.throughput_gbps());
```

### `reset_stats(&mut self)`

Resets statistics counters:

```rust
compressor.reset_stats();
```

### `backend(&self) -> SimdBackend`

Returns the active SIMD backend:

```rust
println!("Backend: {:?}", compressor.backend());
// Output: Avx512, Avx2, Neon, or Scalar
```

## CompressedPage

The compressed page structure:

```rust
pub struct CompressedPage {
    /// Compressed data
    pub data: Vec<u8>,

    /// Original size (always PAGE_SIZE)
    pub original_size: usize,

    /// Algorithm used
    pub algorithm: Algorithm,
}

impl CompressedPage {
    /// Compression ratio
    pub fn ratio(&self) -> f64;

    /// Bytes saved
    pub fn bytes_saved(&self) -> usize;

    /// Check if actually compressed
    pub fn is_compressed(&self) -> bool;
}
```

## Error Handling

```rust
use trueno_zram_core::Error;

match compressor.compress(&page) {
    Ok(compressed) => { /* success */ }
    Err(Error::BufferTooSmall(msg)) => { /* buffer issue */ }
    Err(Error::CorruptedData(msg)) => { /* corrupt input */ }
    Err(Error::InvalidInput(msg)) => { /* invalid params */ }
    Err(e) => { /* other error */ }
}
```
