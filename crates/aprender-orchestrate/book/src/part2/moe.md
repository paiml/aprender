# MoE Backend Selection

The Mixture-of-Experts (MoE) router is the core decision engine in Phase 3 optimization. It classifies each compute operation by complexity and data size, then selects the optimal backend: Scalar, SIMD, or GPU.

## How MoE Routing Works

The `BackendSelector::select_with_moe()` method takes two inputs:

1. **Operation complexity** -- Low, Medium, or High
2. **Data size** -- number of elements in the operation

```rust
pub fn select_with_moe(&self, complexity: OpComplexity, data_size: usize) -> Backend {
    match complexity {
        OpComplexity::Low => {
            if data_size > 1_000_000 { Backend::SIMD }
            else { Backend::Scalar }
        }
        OpComplexity::Medium => {
            if data_size > 100_000 { Backend::GPU }
            else if data_size > 10_000 { Backend::SIMD }
            else { Backend::Scalar }
        }
        OpComplexity::High => {
            if data_size > 10_000 { Backend::GPU }
            else if data_size > 1_000 { Backend::SIMD }
            else { Backend::Scalar }
        }
    }
}
```

## Complexity Classification

| Level | Operations | Algorithmic Complexity | Memory Pattern |
|-------|-----------|----------------------|----------------|
| Low | add, subtract, multiply, reshape | O(n) | Memory-bound |
| Medium | sum, mean, max, min, dot product | O(n) | Moderate compute |
| High | matmul, convolution, attention | O(n^2) or O(n^3) | Compute-bound |

## Threshold Table

| Complexity | Scalar | SIMD | GPU |
|-----------|--------|------|-----|
| Low | < 1M elements | >= 1M elements | Never |
| Medium | < 10K elements | 10K -- 100K elements | > 100K elements |
| High | < 1K elements | 1K -- 10K elements | > 10K elements |

These thresholds are derived from empirical benchmarks on Trueno SIMD kernels and the 5x PCIe dispatch rule from Gregg and Hazelwood (2011).

## Per-Converter Integration

Each framework converter embeds complexity metadata in its operation mappings:

```rust
// NumPy
NumPyOp::Add.complexity()                         // Low
NumPyOp::Sum.complexity()                         // Medium
NumPyOp::Dot.complexity()                         // High

// sklearn
SklearnAlgorithm::StandardScaler.complexity()     // Low
SklearnAlgorithm::LinearRegression.complexity()   // Medium
SklearnAlgorithm::KMeans.complexity()             // High

// PyTorch
PyTorchOperation::TensorCreation.complexity()     // Low
PyTorchOperation::Linear.complexity()             // Medium
PyTorchOperation::Forward.complexity()            // High
```

## End-to-End Example

```rust
let converter = NumPyConverter::new();

// Small array addition: Scalar
converter.recommend_backend(&NumPyOp::Add, 100);       // Scalar

// Large array addition: SIMD
converter.recommend_backend(&NumPyOp::Add, 2_000_000); // SIMD

// Large matrix multiply: GPU
converter.recommend_backend(&NumPyOp::Dot, 50_000);    // GPU
```

The cost model parameters are configurable for different hardware. See [GPU Acceleration](./gpu.md) for tuning details.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
