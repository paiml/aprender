# Tuning Guide

## Algorithm Selection

### LZ4 vs ZSTD

| Workload | Recommended | Reason |
|----------|-------------|--------|
| Real-time | LZ4 | Lowest latency |
| Memory-constrained | ZSTD | Better ratio |
| Mixed content | Adaptive | Auto-selects |
| Highly compressible | ZSTD-3 | Best ratio |

### Code Example

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

// For latency-sensitive workloads
let fast = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// For memory-constrained systems
let compact = CompressorBuilder::new()
    .algorithm(Algorithm::Zstd { level: 3 })
    .build()?;

// For unknown workloads
let adaptive = CompressorBuilder::new()
    .algorithm(Algorithm::Adaptive)
    .build()?;
```

## SIMD Backend Selection

The library auto-detects the best backend, but you can force one:

```rust
use trueno_zram_core::{CompressorBuilder, SimdBackend};

// Force AVX2 (e.g., for testing)
let compressor = CompressorBuilder::new()
    .prefer_backend(SimdBackend::Avx2)
    .build()?;
```

## GPU Batch Sizing

### Optimal Batch Size

| GPU | L2 Cache | Optimal Batch |
|-----|----------|---------------|
| H100 | 50 MB | 10,240 pages |
| A100 | 40 MB | 8,192 pages |
| RTX 4090 | 72 MB | 14,745 pages |
| RTX 3090 | 6 MB | 1,200 pages |

### Calculation

```rust
fn optimal_batch_size(l2_cache_bytes: usize) -> usize {
    // Each page needs ~3KB working memory
    // Target 80% L2 cache utilization
    (l2_cache_bytes * 80 / 100) / (3 * 1024)
}
```

## Async DMA

Enable async DMA for overlapping transfers:

```rust
let config = GpuBatchConfig {
    async_dma: true,
    ring_buffer_slots: 4,  // Pipeline depth
    ..Default::default()
};
```

Benefits:
- Overlaps H2D, compute, D2H
- 20-30% throughput improvement
- Higher GPU utilization

## Memory Configuration

### Working Memory

```rust
// Per-thread memory usage
const LZ4_HASH_TABLE: usize = 64 * 1024;   // 64 KB
const ZSTD_CONTEXT: usize = 256 * 1024;     // 256 KB
const WORKING_BUFFER: usize = 16 * 1024;    // 16 KB
```

### Reducing Memory

For memory-constrained systems:

```rust
// Use LZ4 (smaller context)
.algorithm(Algorithm::Lz4)

// Smaller batch sizes
let config = GpuBatchConfig {
    batch_size: 100,
    ..Default::default()
};
```

## Monitoring Performance

```rust
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// Process pages...

let stats = compressor.stats();

// Check throughput
if stats.throughput_gbps() < 3.0 {
    println!("Warning: Low throughput, check CPU affinity");
}

// Check ratio
if stats.ratio() < 1.5 {
    println!("Warning: Low compression, data may be incompressible");
}
```

## CPU Affinity

For best performance, pin compression threads to physical cores:

```bash
# Linux: Pin to cores 0-3
taskset -c 0-3 ./my_app

# Check NUMA topology
numactl --hardware
```

## Kernel Parameters

For zram integration:

```bash
# Increase zram size
echo $((8 * 1024 * 1024 * 1024)) > /sys/block/zram0/disksize

# Set compression streams
echo 4 > /sys/block/zram0/max_comp_streams

# Enable writeback
echo 1 > /sys/block/zram0/writeback
```
