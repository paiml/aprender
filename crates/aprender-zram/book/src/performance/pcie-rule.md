# PCIe 5x Rule

The PCIe 5x rule determines when GPU offload is beneficial for compression.

## The Rule

```
GPU beneficial when: T_compute > 5 × T_transfer
```

Where:
- `T_compute` = CPU computation time
- `T_transfer` = PCIe transfer time (H2D + D2H)

## Why 5x?

GPU offload has overhead:
1. **H2D transfer**: Copy data to GPU
2. **Kernel launch**: ~5-10us overhead
3. **D2H transfer**: Copy results back
4. **Synchronization**: Wait for completion

The 5x factor accounts for these overheads and ensures GPU provides net benefit.

## Calculation

```rust
use trueno_zram_core::gpu::meets_pcie_rule;
use trueno_zram_core::PAGE_SIZE;

fn check_gpu_benefit(
    pages: usize,
    pcie_bandwidth_gbps: f64,
    gpu_throughput_gbps: f64,
) -> bool {
    let data_bytes = pages * PAGE_SIZE;

    // Transfer time (round trip)
    let transfer_time = 2.0 * data_bytes as f64 / (pcie_bandwidth_gbps * 1e9);

    // GPU compute time
    let gpu_time = data_bytes as f64 / (gpu_throughput_gbps * 1e9);

    // CPU compute time (assume 4 GB/s baseline)
    let cpu_time = data_bytes as f64 / (4e9);

    // GPU beneficial if saves time
    cpu_time > (transfer_time + gpu_time) * 1.2  // 20% margin
}
```

## Examples

### PCIe 4.0 x16 (25 GB/s)

| Batch | Data Size | Transfer | GPU Time | Beneficial? |
|-------|-----------|----------|----------|-------------|
| 100 | 400 KB | 32 us | 4 us | No |
| 1,000 | 4 MB | 320 us | 40 us | Marginal |
| 10,000 | 40 MB | 3.2 ms | 400 us | Yes |
| 100,000 | 400 MB | 32 ms | 4 ms | Yes |

### PCIe 5.0 x16 (64 GB/s)

| Batch | Data Size | Transfer | GPU Time | Beneficial? |
|-------|-----------|----------|----------|-------------|
| 1,000 | 4 MB | 125 us | 40 us | No |
| 10,000 | 40 MB | 1.25 ms | 400 us | Yes |
| 100,000 | 400 MB | 12.5 ms | 4 ms | Yes |

## Checking in Code

```rust
use trueno_zram_core::gpu::{GpuBatchCompressor, GpuBatchConfig};

let mut compressor = GpuBatchCompressor::new(config)?;
let result = compressor.compress_batch(&pages)?;

if result.pcie_rule_satisfied() {
    println!("GPU offload was beneficial");
    println!("Kernel time: {} ns", result.kernel_time_ns);
    println!("Transfer time: {} ns",
        result.h2d_time_ns + result.d2h_time_ns);
} else {
    println!("Consider using CPU for this batch size");
}
```

## Backend Selection

```rust
use trueno_zram_core::gpu::{select_backend, BackendSelection, gpu_available};

let batch_size = 5000;
let has_gpu = gpu_available();

match select_backend(batch_size, has_gpu) {
    BackendSelection::Gpu => {
        // Use GPU batch compression
    }
    BackendSelection::Simd => {
        // Use CPU SIMD compression
    }
    BackendSelection::Scalar => {
        // Use scalar compression
    }
}
```

## Optimization Tips

1. **Batch larger**: Combine small batches into larger ones
2. **Use async DMA**: Overlap transfers with computation
3. **Profile first**: Measure actual transfer times
4. **Consider hybrid**: Use CPU for small batches, GPU for large

## Hardware-Specific Thresholds

| GPU | PCIe | Min Beneficial Batch |
|-----|------|---------------------|
| H100 | 5.0 x16 | 5,000 pages |
| A100 | 4.0 x16 | 8,000 pages |
| RTX 4090 | 4.0 x16 | 10,000 pages |
| RTX 3090 | 4.0 x16 | 12,000 pages |
