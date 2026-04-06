# GPU Acceleration

GPU acceleration is the highest tier of the MoE backend selection in Phase 3. Batuta uses the wgpu crate (via Trueno) for portable GPU compute across Vulkan, Metal, DX12, and WebGPU.

## The 5x PCIe Dispatch Rule

GPU dispatch incurs overhead from data transfer across the PCIe bus. Based on Gregg and Hazelwood (2011), GPU compute is only beneficial when:

```
compute_time > 5 * transfer_time
```

The `BackendSelector` implements this as a cost model:

```rust
pub fn select_backend(&self, data_bytes: usize, flops: u64) -> Backend {
    let transfer_s = data_bytes as f64 / self.pcie_bandwidth;
    let compute_s = flops as f64 / self.gpu_gflops;

    if compute_s > self.min_dispatch_ratio * transfer_s {
        Backend::GPU
    } else {
        Backend::SIMD
    }
}
```

Default parameters assume PCIe 4.0 x16 (32 GB/s) and A100-class throughput (20 TFLOPS).

## When GPU Is Beneficial

| Operation | Data Size | Recommended Backend | Why |
|-----------|----------|-------------------|-----|
| Element-wise add | Any | Never GPU | Memory-bound, PCIe overhead dominates |
| Dot product | < 100K | SIMD | Transfer cost exceeds compute |
| Dot product | > 100K | GPU | Sufficient compute to amortize transfer |
| Matrix multiply | < 10K | SIMD | Small matrices fit in SIMD registers |
| Matrix multiply | > 10K | GPU | O(n^3) compute dominates O(n^2) transfer |

## Matrix Multiplication Example

```rust
let selector = BackendSelector::new();

// Small matrix: SIMD is faster
let backend = selector.select_for_matmul(64, 64, 64);
// --> Backend::SIMD

// Large matrix: GPU is faster
let backend = selector.select_for_matmul(1024, 1024, 1024);
// --> Backend::GPU
```

## Customizing Thresholds

The selector can be configured for different hardware:

```rust
let selector = BackendSelector::new()
    .with_pcie_bandwidth(64e9)       // PCIe 5.0
    .with_gpu_gflops(40e12)          // RTX 4090
    .with_min_dispatch_ratio(3.0);   // More aggressive dispatch
```

## GPU Backends via wgpu

Trueno abstracts GPU compute through wgpu, which maps to the native GPU API on each platform:

| Platform | API |
|----------|-----|
| Linux | Vulkan |
| macOS | Metal |
| Windows | DX12 / Vulkan |
| Browser | WebGPU |

## When to Avoid GPU

GPU dispatch should be avoided when:

- Data fits entirely in L1/L2 cache (SIMD will be faster)
- The operation is memory-bound (element-wise operations)
- The program will run in WASM without WebGPU support
- Latency matters more than throughput (kernel launch overhead is ~10us)

---

**Navigate:** [Table of Contents](../SUMMARY.md)
