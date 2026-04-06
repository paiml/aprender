# Code Coverage

Realizar maintains rigorous code coverage standards through the **T-COV-95** testing methodology, achieving **95.09% line coverage** across all platforms including CUDA.

## Coverage Targets

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Line Coverage | 95% | 95.09% | Achieved |
| Function Coverage | 95% | 95.91% | Exceeded |
| Region Coverage | 85% | 95.13% | Exceeded |

## CUDA-Last Coverage Architecture

Coverage collection uses a **CUDA-Last** strategy that runs all non-CUDA tests in parallel (8 threads), then runs CUDA tests single-threaded for GPU context safety:

```
Phase 1: All non-CUDA tests (8 threads, parallel)
Phase 2: CUDA tests (single-threaded, sequential)
Report: Combined coverage with COV_EXCLUDE quarantine
```

### Compute Quarantine

The `COV_EXCLUDE` regex filters hardware-specific files from the coverage report that would skew metrics due to conditional compilation:

```
src/cuda/                  # CUDA kernel wrappers (tested separately)
src/layers/                # Layer dispatch paths
src/simd                   # SIMD backend selection
```

These paths are verified through:
- **1,196 CUDA tests** running on RTX 4090
- **Parity tests**: Comparing CUDA output against CPU reference
- **Null pointer validation**: Pre-launch checks prevent GPU context poisoning
- **CPU fallback paths**: Automatic fallback when GPU constraints aren't met

## Running Coverage

```bash
# Full coverage report (recommended)
make coverage

# Quick coverage check
cargo llvm-cov --features cuda

# Coverage with HTML report
cargo llvm-cov --html --features cuda
```

## Coverage Philosophy: Popperian Falsification

Realizar follows **Popperian Falsification** -- tests attempt to *refute* the implementation, not verify it. This means:

1. **Property-based tests** generate millions of inputs looking for failures
2. **Mutation testing** verifies that changing code breaks tests
3. **Boundary testing** targets edge cases where bugs hide

See [Property-Based Testing](../tdd/property-based.md) for the proptest integration.

## T-COV-95 Campaign Results

The T-COV-95 campaign (January-February 2026) achieved:

- **14,614 total tests** (all passing)
- **1,196 CUDA/GPU tests** (RTX 4090)
- **33 proptest tests** generating millions of test cases
- **Security vulnerability discovered** via generative fuzzing
- **100% mutation score** on critical API paths
- **CUDA context safety**: Null pointer validation on all 20 kernel methods

### CUDA Context Safety (February 2026)

The coverage campaign revealed that kernel crashes from null device pointers permanently poison the CUDA context for the entire process. This led to:

1. **`validate_device_ptr()`** - Pre-launch null pointer check on all kernel methods
2. **Sync-on-drop** - `CudaExecutor::Drop` synchronizes before returning resources to pools
3. **CPU fallback** - `flash_attention_cached()` falls back to CPU when `seq_len < head_dim`
4. **Sentinel health checks** - Poisoned contexts are never returned to the pool

### Security Discovery

Proptest fuzzing revealed an **allocation attack vulnerability** in GGUF parsing. Corrupted headers with `tensor_count = u64::MAX` could cause multi-terabyte allocation attempts. This was fixed with bounds checks:

```rust
// src/gguf/loader.rs - Allocation attack prevention
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_METADATA_COUNT: u64 = 10_000;
const MAX_DIMS: u32 = 8;
const MAX_ARRAY_LEN: u64 = 10_000_000;

if tensor_count > MAX_TENSOR_COUNT {
    return Err(RealizarError::UnsupportedOperation {
        operation: "parse_gguf".to_string(),
        reason: format!(
            "tensor_count {} exceeds maximum allowed {}",
            tensor_count, MAX_TENSOR_COUNT
        ),
    });
}
```

## Continuous Integration

Coverage is checked on every PR via GitHub Actions:

```yaml
- name: Coverage Check
  run: |
    cargo llvm-cov --lcov --output-path lcov.info
    # Fail if below 95%
```

## Tools

- **cargo-llvm-cov**: LLVM-based coverage (fast, accurate)
- **proptest**: Property-based test generation
- **cargo-mutants**: Mutation testing

See [cargo llvm-cov](../tools/cargo-llvm-cov.md) for installation and usage.
