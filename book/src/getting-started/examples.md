# Examples

## Running Examples

trueno-zram includes several examples to demonstrate its features:

```bash
# GPU information and backend selection
cargo run -p trueno-zram-core --example gpu_info
cargo run -p trueno-zram-core --example gpu_info --features cuda

# Compression benchmarks
cargo run -p trueno-zram-core --example compress_benchmark --release
cargo run -p trueno-zram-core --example compress_benchmark --release --features cuda
```

## GPU Info Example

Shows GPU detection, backend selection logic, and PCIe 5x rule evaluation:

```
trueno-zram GPU Information
============================

1. GPU Availability
   ----------------
   GPU available: true

   CUDA Device Information:
     Device: NVIDIA GeForce RTX 4090
     Compute Capability: SM 8.9
     Memory: 24564 MB
     L2 Cache: 73728 KB
     Optimal batch: 14745 pages
     Supported: true

2. Backend Selection Logic
   -----------------------
   GPU_MIN_BATCH_SIZE: 1000 pages
   PAGE_SIZE: 4096 bytes

      Batch          No GPU        With GPU
   ------------------------------------------
         1          Scalar          Scalar
        10            Simd            Simd
       100            Simd            Simd
       500            Simd            Simd
      1000            Simd             Gpu
      5000            Simd             Gpu
     10000            Simd             Gpu

3. PCIe 5x Rule Evaluation
   -----------------------
   GPU offload beneficial when: T_cpu > 5 * (T_transfer + T_gpu)

   1K pages, PCIe 4.0, 100 GB/s GPU (4 MB): CPU preferred
   10K pages, PCIe 4.0, 100 GB/s GPU (40 MB): GPU beneficial
   100K pages, PCIe 5.0, 500 GB/s GPU (400 MB): GPU beneficial
```

## Compression Benchmark

Measures throughput across different data patterns:

```
trueno-zram Compression Benchmark
=================================

Pattern: Zeros (compressible)
----------------------------------------------------------------------
   Pages  Algorithm     Compress   Decompress      Ratio    Backend
     100        Lz4      22.01 GB/s      46.02 GB/s   2048.00x Avx512
    1000        Lz4      21.87 GB/s      45.91 GB/s   2048.00x Avx512

Pattern: Text (compressible)
----------------------------------------------------------------------
   Pages  Algorithm     Compress   Decompress      Ratio    Backend
     100        Lz4       4.57 GB/s       5.42 GB/s      3.21x Avx512
    1000        Lz4       4.44 GB/s       5.37 GB/s      3.21x Avx512

Pattern: Random (incompressible)
----------------------------------------------------------------------
   Pages  Algorithm     Compress   Decompress      Ratio    Backend
     100        Lz4       1.87 GB/s      43.78 GB/s      1.00x Avx512
    1000        Lz4       1.61 GB/s      31.64 GB/s      1.00x Avx512
```

## Custom Example: Batch Processing

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm, PAGE_SIZE};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()?;

    // Generate test pages
    let pages: Vec<[u8; PAGE_SIZE]> = (0..1000)
        .map(|i| {
            let mut page = [0u8; PAGE_SIZE];
            // Create compressible pattern
            for (j, byte) in page.iter_mut().enumerate() {
                *byte = ((i + j) % 256) as u8;
            }
            page
        })
        .collect();

    // Benchmark compression
    let start = Instant::now();
    let mut total_compressed = 0;

    for page in &pages {
        let compressed = compressor.compress(page)?;
        total_compressed += compressed.data.len();
    }

    let elapsed = start.elapsed();
    let input_bytes = pages.len() * PAGE_SIZE;
    let throughput = input_bytes as f64 / elapsed.as_secs_f64() / 1e9;
    let ratio = input_bytes as f64 / total_compressed as f64;

    println!("Compressed {} pages in {:?}", pages.len(), elapsed);
    println!("Throughput: {:.2} GB/s", throughput);
    println!("Compression ratio: {:.2}x", ratio);

    Ok(())
}
```

## Custom Example: GPU Batch Compression

```rust
use trueno_zram_core::gpu::{GpuBatchCompressor, GpuBatchConfig, gpu_available};
use trueno_zram_core::{Algorithm, PAGE_SIZE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !gpu_available() {
        println!("No GPU available, skipping GPU example");
        return Ok(());
    }

    let config = GpuBatchConfig {
        device_index: 0,
        algorithm: Algorithm::Lz4,
        batch_size: 1000,
        async_dma: true,
        ring_buffer_slots: 4,
    };

    let mut compressor = GpuBatchCompressor::new(config)?;

    // Create batch of pages
    let pages: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; 1000];

    // Compress batch
    let result = compressor.compress_batch(&pages)?;

    println!("Batch Results:");
    println!("  Pages: {}", result.pages.len());
    println!("  H2D time: {} ns", result.h2d_time_ns);
    println!("  Kernel time: {} ns", result.kernel_time_ns);
    println!("  D2H time: {} ns", result.d2h_time_ns);
    println!("  Total time: {} ns", result.total_time_ns);
    println!("  Compression ratio: {:.2}x", result.compression_ratio());
    println!("  PCIe 5x rule satisfied: {}", result.pcie_rule_satisfied());

    // Get compressor stats
    let stats = compressor.stats();
    println!("\nCompressor Stats:");
    println!("  Pages compressed: {}", stats.pages_compressed);
    println!("  Throughput: {:.2} GB/s", stats.throughput_gbps());

    Ok(())
}
```
