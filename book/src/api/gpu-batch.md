# GPU Batch API

The GPU batch API provides high-throughput compression for large page batches.

## GpuBatchConfig

```rust
use trueno_zram_core::gpu::GpuBatchConfig;
use trueno_zram_core::Algorithm;

let config = GpuBatchConfig {
    device_index: 0,        // CUDA device (0 = first GPU)
    algorithm: Algorithm::Lz4,
    batch_size: 1000,       // Pages per batch
    async_dma: true,        // Enable async transfers
    ring_buffer_slots: 4,   // Pipeline depth
};
```

## GpuBatchCompressor

### Creation

```rust
use trueno_zram_core::gpu::{GpuBatchCompressor, GpuBatchConfig};

let config = GpuBatchConfig::default();
let mut compressor = GpuBatchCompressor::new(config)?;
```

### Batch Compression

```rust
let pages: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; 1000];
let result = compressor.compress_batch(&pages)?;
```

### Statistics

```rust
let stats = compressor.stats();

println!("Pages compressed: {}", stats.pages_compressed);
println!("Input bytes: {}", stats.total_bytes_in);
println!("Output bytes: {}", stats.total_bytes_out);
println!("Time: {} ns", stats.total_time_ns);
println!("Ratio: {:.2}x", stats.compression_ratio());
println!("Throughput: {:.2} GB/s", stats.throughput_gbps());
```

### Configuration Access

```rust
let config = compressor.config();
println!("Batch size: {}", config.batch_size);
println!("Async DMA: {}", config.async_dma);
```

## BatchResult

```rust
pub struct BatchResult {
    /// Compressed pages
    pub pages: Vec<CompressedPage>,

    /// Host-to-device transfer time (ns)
    pub h2d_time_ns: u64,

    /// Kernel execution time (ns)
    pub kernel_time_ns: u64,

    /// Device-to-host transfer time (ns)
    pub d2h_time_ns: u64,

    /// Total wall clock time (ns)
    pub total_time_ns: u64,
}
```

### Methods

```rust
// Throughput in bytes/second
let throughput = result.throughput_bytes_per_sec(input_bytes);

// Compression ratio
let ratio = result.compression_ratio();

// Check PCIe 5x rule
let beneficial = result.pcie_rule_satisfied();
```

## Helper Functions

### `gpu_available()`

```rust
use trueno_zram_core::gpu::gpu_available;

if gpu_available() {
    println!("CUDA GPU detected");
}
```

### `select_backend()`

```rust
use trueno_zram_core::gpu::{select_backend, BackendSelection};

let backend = select_backend(batch_size, gpu_available());

match backend {
    BackendSelection::Gpu => { /* use GPU */ }
    BackendSelection::Simd => { /* use CPU SIMD */ }
    BackendSelection::Scalar => { /* use scalar */ }
}
```

### `meets_pcie_rule()`

```rust
use trueno_zram_core::gpu::meets_pcie_rule;

let pages = 10000;
let pcie_bandwidth = 64.0;  // GB/s (PCIe 5.0)
let gpu_throughput = 500.0; // GB/s

if meets_pcie_rule(pages, pcie_bandwidth, gpu_throughput) {
    println!("GPU offload beneficial");
}
```

## GpuDeviceInfo

```rust
use trueno_zram_core::gpu::GpuDeviceInfo;

let info = GpuDeviceInfo {
    index: 0,
    name: "RTX 4090".to_string(),
    total_memory: 24 * 1024 * 1024 * 1024,
    l2_cache_size: 72 * 1024 * 1024,
    compute_capability: (8, 9),
    backend: GpuBackend::Cuda,
};

println!("Optimal batch: {} pages", info.optimal_batch_size());
println!("Supported: {}", info.is_supported());
```
