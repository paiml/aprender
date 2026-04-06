# Benchmarks

## Running Benchmarks

```bash
# Criterion benchmarks
cargo bench --all-features

# With baseline comparison
cargo bench --all-features -- --save-baseline main

# Example benchmarks
cargo run -p trueno-zram-core --example compress_benchmark --release
```

## Results Summary

### LZ4 Performance

| Backend | Compress | Decompress | Ratio |
|---------|----------|------------|-------|
| AVX-512 | 4.4 GB/s | 5.4 GB/s | 2-4x |
| AVX2 | 3.2 GB/s | 4.1 GB/s | 2-4x |
| Scalar | 3.0 GB/s | 3.8 GB/s | 2-4x |

### ZSTD Performance

| Backend | Level | Compress | Decompress | Ratio |
|---------|-------|----------|------------|-------|
| AVX-512 | 1 | 11.2 GB/s | 46 GB/s | 3-5x |
| AVX-512 | 3 | 8.5 GB/s | 45 GB/s | 4-6x |
| AVX2 | 1 | 8.5 GB/s | 35 GB/s | 3-5x |

### Same-Fill Performance

| Backend | Detection | Ratio |
|---------|-----------|-------|
| AVX-512 | 22 GB/s | 2048:1 |
| AVX2 | 18 GB/s | 2048:1 |
| Scalar | 12 GB/s | 2048:1 |

## Data Patterns

### Zeros (Best Case)

```
Pattern: Zeros (100% same-fill)
Pages: 1000
Compression: 22 GB/s
Decompression: 46 GB/s
Ratio: 2048:1
```

### Text (Compressible)

```
Pattern: Text/Code
Pages: 1000
LZ4 Compression: 4.4 GB/s
LZ4 Decompression: 5.4 GB/s
Ratio: 3.2:1
```

### Random (Incompressible)

```
Pattern: Random bytes
Pages: 1000
LZ4 Compression: 1.6 GB/s
LZ4 Decompression: 32 GB/s
Ratio: 1.0:1 (pass-through)
```

## GPU Benchmarks

> **Note (2026-01-06):** GPU decompression is limited by PCIe transfer overhead.
> CPU parallel path (50+ GB/s) is faster for most workloads.

### RTX 4090 (Validated)

| Path | Throughput | Notes |
|------|------------|-------|
| CPU Parallel | 50+ GB/s | Primary recommended path |
| GPU End-to-End | ~6 GB/s | PCIe 4.0 transfer bottleneck |
| GPU Kernel-only | ~9 GB/s | Without H2D/D2H transfers |

### Recommendation

Use CPU parallel decompression for best performance. GPU useful for:
- Future PCIe 5.0+ systems with higher bandwidth
- Workloads where CPU is saturated

## Latency

| Operation | P50 | P99 | P99.9 |
|-----------|-----|-----|-------|
| LZ4 compress (4KB) | 45us | 85us | 120us |
| LZ4 decompress (4KB) | 38us | 72us | 95us |
| Same-fill detect | 8us | 15us | 25us |

## Memory Usage

| Component | Memory |
|-----------|--------|
| Hash table (LZ4) | 64 KB |
| Working buffer | 16 KB |
| ZSTD context | 256 KB |

## Comparison with Linux Kernel zram

**Validated 2026-01-06 on RTX 4090 + AMD Threadripper 7960X (AVX-512)**

### Block Device I/O (fio, Direct I/O)

| Metric | Kernel ZRAM | trueno-ublk | Speedup |
|--------|-------------|-------------|---------|
| Sequential Read | 9.2 GB/s | 16.5 GB/s | **1.8x** |
| Random 4K IOPS | 55K | 249K | **4.5x** |
| Compression Ratio | 2.5x | 3.87x | **+55%** |

### Compression Engine (cargo examples)

| Metric | Linux Kernel zram | trueno-zram | Speedup |
|--------|-------------------|-------------|---------|
| **Compress (parallel)** | 3-5 GB/s | 20-30 GB/s | **5-6x** |
| **Decompress (CPU)** | ~10 GB/s | 50+ GB/s | **5x** |
| **Same-fill detection** | ~8 GB/s | 22 GB/s | **2.75x** |
| **Compression ratio** | 2.5x | 3.87x | **+55%** |

### Architecture Difference

| Aspect | Linux Kernel | trueno-zram |
|--------|--------------|-------------|
| Threading | Single-threaded per page | Parallel (rayon) |
| SIMD | Limited | AVX-512/AVX2/NEON |
| Batch processing | No | Yes (5000+ pages) |
| GPU offload | No | Optional CUDA |

### Falsification Testing (2026-01-06)

```
Test Configuration:
├── Hardware: AMD Threadripper 7960X, 125GB RAM, RTX 4090
├── Device: trueno-ublk 8GB device
├── Tool: fio, cargo examples

Results:
├── Sequential I/O: 16.5 GB/s ✓ (claim: 12.5 GB/s)
├── Random IOPS: 249K ✓ (claim: 228K)
├── Compression: 30.66 GB/s ✓ (claim: 20-24 GB/s)
├── Ratio: 3.87x ✓ (claim: 3.7x)
├── mlock: 272 MB ✓ (claim: >100 MB)
├── Stress test: No deadlock ✓
└── Status: 6/8 PASS (GPU claims deprecated)
```

### Why trueno-zram is Faster

1. **SIMD vectorization**: AVX-512 processes 64 bytes per instruction vs byte-by-byte
2. **Parallel compression**: All CPU cores compress simultaneously via rayon
3. **Batch amortization**: Setup costs spread across thousands of pages
4. **Cache efficiency**: Sequential memory access patterns
5. **Zero-copy paths**: Same-fill pages detected without compression attempt
