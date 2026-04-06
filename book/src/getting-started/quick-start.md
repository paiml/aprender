# Quick Start

## Basic Compression

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm, PAGE_SIZE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a compressor with LZ4 algorithm
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()?;

    // Create a test page (4KB)
    let mut page = [0u8; PAGE_SIZE];
    page[0..100].copy_from_slice(&[0xAA; 100]);

    // Compress
    let compressed = compressor.compress(&page)?;
    println!("Original size: {} bytes", PAGE_SIZE);
    println!("Compressed size: {} bytes", compressed.data.len());
    println!("Ratio: {:.2}x", compressed.ratio());

    // Decompress
    let decompressed = compressor.decompress(&compressed)?;
    assert_eq!(page, decompressed);
    println!("Decompression verified!");

    Ok(())
}
```

## Choosing an Algorithm

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

// LZ4: Fastest compression, good for most workloads
let lz4 = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// ZSTD Level 1: Better ratio, still fast
let zstd_fast = CompressorBuilder::new()
    .algorithm(Algorithm::Zstd { level: 1 })
    .build()?;

// ZSTD Level 3: Best ratio for compressible data
let zstd_best = CompressorBuilder::new()
    .algorithm(Algorithm::Zstd { level: 3 })
    .build()?;

// Adaptive: Automatically selects based on entropy
let adaptive = CompressorBuilder::new()
    .algorithm(Algorithm::Adaptive)
    .build()?;
```

## Compression Statistics

```rust
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// Compress some pages
for _ in 0..100 {
    let page = [0u8; PAGE_SIZE];
    let _ = compressor.compress(&page)?;
}

// Get statistics
let stats = compressor.stats();
println!("Pages compressed: {}", stats.pages);
println!("Total input: {} bytes", stats.bytes_in);
println!("Total output: {} bytes", stats.bytes_out);
println!("Overall ratio: {:.2}x", stats.ratio());
println!("Throughput: {:.2} GB/s", stats.throughput_gbps());
```

## Error Handling

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm, Error};

fn compress_page(data: &[u8; PAGE_SIZE]) -> Result<Vec<u8>, Error> {
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()?;

    let compressed = compressor.compress(data)?;
    Ok(compressed.data)
}

fn main() {
    let page = [0u8; PAGE_SIZE];

    match compress_page(&page) {
        Ok(data) => println!("Compressed to {} bytes", data.len()),
        Err(Error::BufferTooSmall(msg)) => eprintln!("Buffer error: {msg}"),
        Err(Error::CorruptedData(msg)) => eprintln!("Corrupt data: {msg}"),
        Err(e) => eprintln!("Other error: {e}"),
    }
}
```

## Next Steps

- Learn about [GPU Batch Compression](../concepts/gpu.md)
- Explore [SIMD Acceleration](../concepts/simd.md)
- See more [Examples](./examples.md)
