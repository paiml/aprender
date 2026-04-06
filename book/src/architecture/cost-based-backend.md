# Cost-Based Backend Selection

The **cost-based backend dispatcher** is the brain of Trueno-DB. It automatically selects the optimal execution backend (GPU, SIMD, or Scalar) based on a physics-based cost model.

## The Problem

Modern systems have multiple execution backends:
- **GPU**: Massively parallel (1000s of cores), but high transfer overhead
- **SIMD**: Medium parallelism (8-16 lanes), low overhead
- **Scalar**: No parallelism, lowest overhead

**Question**: For a given query workload, which backend is fastest?

## The 5x Rule

Trueno-DB uses a simple, physics-based rule:

> **Use GPU only if compute time > 5x transfer time**

This rule comes from **Genchi Genbutsu** (Go and See): measuring real PCIe bandwidth and GPU performance.

## Cost Model

### Rule 1: Minimum Data Size (10 MB)

```rust
const MIN_GPU_DATA_SIZE_BYTES: usize = 10_000_000; // 10 MB

if total_bytes < MIN_GPU_DATA_SIZE_BYTES {
    return Backend::Simd; // Transfer overhead not worthwhile
}
```

**Rationale**: For small datasets (<10 MB), PCIe transfer overhead dominates. SIMD is faster.

### Rule 2: PCIe Transfer Time

```rust
const PCIE_BANDWIDTH_GBPS: f64 = 32.0; // PCIe Gen4 x16

let pcie_transfer_time_ms =
    (total_bytes as f64 / (PCIE_BANDWIDTH_GBPS * 1_000_000_000.0)) * 1000.0;
```

**Measured on**: AMD Ryzen 9 7950X with PCIe Gen4 x16
**Peak bandwidth**: 32 GB/s (64 GB/s bidirectional)
**Effective bandwidth**: ~28 GB/s (due to protocol overhead)

### Rule 3: GPU Compute Time

```rust
const GPU_THROUGHPUT_GFLOPS: f64 = 100.0; // Conservative estimate

let estimated_gpu_compute_ms =
    (estimated_flops / (GPU_THROUGHPUT_GFLOPS * 1_000_000_000.0)) * 1000.0;
```

**GPU model**: AMD Radeon RX 7900 XTX
**Peak performance**: 61 TFLOP/s (FP32)
**Sustained performance**: ~100 GFLOP/s (conservative, accounts for memory bottlenecks)

### Rule 4: Apply 5x Rule

```rust
const TRANSFER_OVERHEAD_MULTIPLIER: f64 = 5.0;

if estimated_gpu_compute_ms > pcie_transfer_time_ms * TRANSFER_OVERHEAD_MULTIPLIER {
    Backend::Gpu
} else {
    Backend::Simd
}
```

**Intuition**: GPU must do 5x more work than PCIe transfer to justify the overhead.

## Example Scenarios

### Scenario 1: Small Dataset (1 MB, Simple Sum)

```
Data: 1 MB (250,000 int32 values)
Operation: SUM (1 FLOP per value)
FLOPs: 250,000

Transfer time: 1 MB / 32 GB/s = 0.03 ms
Compute time: 250,000 / 100 GFLOP/s = 0.0025 ms

Result: 0.0025 ms < 0.03 ms * 5
Backend: SIMD (transfer dominates)
```

### Scenario 2: Large Dataset (1 GB, Complex Aggregation)

```
Data: 1 GB (250M int32 values)
Operation: SUM + AVG + COUNT + MIN + MAX (5 FLOPs per value)
FLOPs: 1.25 billion

Transfer time: 1 GB / 32 GB/s = 31.25 ms
Compute time: 1.25B / 100 GFLOP/s = 12.5 ms

Result: 12.5 ms < 31.25 ms * 5
Backend: SIMD (still transfer-bound)
```

### Scenario 3: Very Large Compute (1 GB, Hash Join)

```
Data: 1 GB (build + probe tables)
Operation: Hash join (200 FLOPs per probe)
FLOPs: 50 billion

Transfer time: 1 GB / 32 GB/s = 31.25 ms
Compute time: 50B / 100 GFLOP/s = 500 ms

Result: 500 ms > 31.25 ms * 5
Backend: GPU (compute justifies transfer)
```

## Implementation

See `src/backend/mod.rs`:

```rust
impl BackendDispatcher {
    #[must_use]
    pub fn select(total_bytes: usize, estimated_flops: f64) -> Backend {
        // Rule 1: Minimum data size threshold
        if total_bytes < MIN_GPU_DATA_SIZE_BYTES {
            return Backend::Simd;
        }

        // Rule 2: Calculate transfer time
        let pcie_transfer_time_ms =
            (total_bytes as f64 / (PCIE_BANDWIDTH_GBPS * 1_000_000_000.0)) * 1000.0;

        // Rule 3: Estimate GPU compute time
        let estimated_gpu_compute_ms =
            (estimated_flops / (GPU_THROUGHPUT_GFLOPS * 1_000_000_000.0)) * 1000.0;

        // Rule 4: Apply 5x rule
        if estimated_gpu_compute_ms > pcie_transfer_time_ms * TRANSFER_OVERHEAD_MULTIPLIER {
            Backend::Gpu
        } else {
            Backend::Simd
        }
    }
}
```

## Testing

See `tests/backend_selection_test.rs`:

```rust
#[test]
fn test_small_dataset_selects_cpu() {
    let total_bytes = 1_000_000; // 1 MB
    let estimated_flops = 1_000_000.0; // Simple sum
    let backend = BackendDispatcher::select(total_bytes, estimated_flops);
    assert!(matches!(backend, Backend::Simd));
}

#[test]
fn test_very_large_compute_selects_gpu() {
    let total_bytes = 1_000_000_000; // 1 GB
    let estimated_flops = 100_000_000_000.0; // 100 GFLOP
    let backend = BackendDispatcher::select(total_bytes, estimated_flops);
    assert!(matches!(backend, Backend::Gpu));
}
```

## Arithmetic Intensity

The cost model implicitly calculates **arithmetic intensity**:

```
AI = FLOPs / Bytes
```

**Rule of thumb**:
- **AI < 1**: Memory-bound (use SIMD)
- **1 ≤ AI < 10**: Balanced (depends on dataset size)
- **AI ≥ 10**: Compute-bound (use GPU)

## Future Improvements

### Dynamic Profiling

Instead of static constants, profile actual hardware:

```rust
// Measure PCIe bandwidth
let bandwidth = measure_pcie_bandwidth();

// Measure GPU throughput
let throughput = measure_gpu_throughput();
```

### Query Optimizer Integration

Integrate with query optimizer to estimate FLOPs:

```rust
fn estimate_flops(query: &QueryPlan) -> f64 {
    match query {
        QueryPlan::Sum(_) => num_rows as f64,
        QueryPlan::HashJoin(build, probe) => {
            build.num_rows() * 50.0 + probe.num_rows() * 200.0
        }
        // ...
    }
}
```

### Multi-GPU Support

Extend to support multiple GPUs:

```rust
enum Backend {
    Gpu { device_id: usize },
    Simd,
    Scalar,
}
```

## Academic References

- **Breß et al. (2014)**: "Robust Query Processing in Co-Processor-accelerated Databases"
- **Gregg & Hazelwood (2011)**: "Where is the data? Why you cannot debate CPU vs. GPU performance without the answer"
- **He et al. (2008)**: "Relational joins on graphics processors"

## Toyota Way Principles

- **Genchi Genbutsu**: Measured real PCIe bandwidth (32 GB/s)
- **Muda**: Eliminate waste by avoiding GPU when transfer dominates
- **Jidoka**: Built-in quality through comprehensive testing

## Next Steps

See the [Backend Dispatcher](../components/dispatcher/selection-algorithm.md) section for implementation details of the cost-based backend selection algorithm.
