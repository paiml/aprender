# trueno-cuda-edge: GPU Edge-Case Testing

`trueno-cuda-edge` is a GPU edge-case test framework implementing Popperian falsificationism for CUDA/GPU code. It provides 5 falsification frameworks with a 50-point verification checklist.

## Overview

GPU code is notoriously difficult to test due to:
- Non-deterministic behavior
- Hardware-dependent edge cases
- Complex lifecycle management
- Numerical precision variations

trueno-cuda-edge addresses these challenges with systematic falsification testing.

## Integration with realizar

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
trueno-cuda-edge = "0.1"
```

### Quantization Parity Testing

The quantization oracle validates CPU/GPU numerical parity:

```rust
use trueno_cuda_edge::quant_oracle::{
    QuantFormat, BoundaryValueGenerator, check_values_parity, ParityConfig
};

#[test]
fn test_q4k_parity() {
    // Format-specific tolerances
    assert_eq!(QuantFormat::Q4K.tolerance(), 0.05);  // 5% for 4-bit
    assert_eq!(QuantFormat::Q6K.tolerance(), 0.01);  // 1% for 6-bit

    // Compare CPU and GPU results
    let config = ParityConfig::new(QuantFormat::Q4K);
    let report = check_values_parity(&cpu_values, &gpu_values, &config);
    assert!(report.passed());
}
```

### Boundary Value Generation

Test edge cases for quantized operations:

```rust
use trueno_cuda_edge::quant_oracle::{QuantFormat, BoundaryValueGenerator};

#[test]
fn test_boundary_values() {
    let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);

    // Universal boundaries: 0, NaN, Inf, format-specific levels
    let boundaries = gen.all_boundaries();

    for boundary in boundaries {
        // Test GPU kernel handles edge case
        let result = gpu_kernel.process(boundary);
        assert!(result.is_valid());
    }
}
```

### PTX Verification

Validate PTX kernels before deployment:

```rust
use trueno_cuda_edge::ptx_poison::{PtxVerifier, PtxMutator};

#[test]
fn test_ptx_kernel() {
    let verifier = PtxVerifier::new();

    // Structural verification (6 checks)
    let verified = verifier.verify(ptx_source)?;

    // Mutation testing for test quality
    let mutated = PtxMutator::FlipAddSub.apply(ptx_source);
    // Verify tests catch the mutation
}
```

### Shared Memory Boundary Testing

Validate shared memory allocations:

```rust
use trueno_cuda_edge::shmem_prober::{
    ComputeCapability, check_allocation, shared_memory_limit
};

#[test]
fn test_shmem_allocation() {
    let ampere = ComputeCapability::new(8, 0);
    assert_eq!(shared_memory_limit(ampere), 164 * 1024); // 164 KB

    // Validate allocation fits
    check_allocation(ampere, 128 * 1024)?;
}
```

## Falsification Protocol

The 50-point falsification checklist tracks verification status:

```rust
use trueno_cuda_edge::falsification::{FalsificationReport, all_claims};

#[test]
fn test_falsification_coverage() {
    let mut report = FalsificationReport::new();

    // Run tests and mark claims
    report.mark_verified("QO-001");  // Quantization parity
    report.mark_verified("PP-001");  // PTX verification

    // Track coverage
    assert!(report.coverage() >= 0.80);  // 80% minimum
}
```

## Framework Summary

| Framework | Purpose | Claims |
|-----------|---------|--------|
| F1: Null Fuzzer | Null pointer handling | 10 |
| F2: Shared Memory Prober | Memory allocation/access | 8 |
| F3: Lifecycle Chaos | Context lifecycle | 8 |
| F4: Quantization Oracle | CPU/GPU parity | 9 |
| F5: PTX Poison | Mutation testing | 9 |
| Supervision | Worker health | 6 |

## Example: Integration Test Suite

See `/tests/cuda_edge_cases.rs` for a complete integration test suite demonstrating all frameworks:

```bash
cargo test cuda_edge_cases --features cuda-testing
```

## See Also

- [Trueno Backend](./trueno-backend.md)
- [CUDA PTX Generation](./cuda-ptx.md)
- [CUDA Context Safety](./cuda-context-safety.md)
