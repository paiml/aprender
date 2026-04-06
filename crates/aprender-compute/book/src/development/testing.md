# Testing

This chapter covers Trueno's comprehensive testing strategy and quality gates.

## Overview

Trueno follows **Extreme TDD** principles with multiple layers of testing:

- **Unit Tests**: Correctness for all operations
- **Property-Based Tests**: Mathematical invariants (proptest)
- **Backend Equivalence Tests**: All backends produce identical results
- **Mutation Testing**: >80% mutation kill rate
- **Coverage**: 90%+ line coverage required

## Running Tests

### Quick Tests (Development)

```bash
# Fast tests with nextest (parallel execution)
make test-fast

# Run all tests with output
make test

# Verbose output (single-threaded)
make test-verbose
```

### Coverage Commands

There is a single coverage command that handles everything:

| Command | Description | Time |
|---------|-------------|------|
| `make coverage` | Full coverage (trueno + trueno-gpu, CUDA if available) | ~45 seconds |

```bash
# Run coverage (the ONLY allowed command)
make coverage

# This generates:
# - LCOV report at lcov.info
# - HTML report at target/coverage/html/index.html
# - TOTAL coverage percentage (currently 95%+)
```

Never use `cargo llvm-cov` directly or install `cargo-tarpaulin`. See [Extreme TDD](./extreme-tdd.md) for details on the coverage mandate.

### Coverage Targets

| Component | Minimum | Target |
|-----------|---------|--------|
| Public API | 100% | 100% |
| SIMD backends | 90% | 95% |
| GPU backend | 85% | 90% |
| WASM backend | 90% | 95% |
| **Overall** | **90%** | **95%+** |

## Test Categories

### 1. Unit Tests

Basic correctness tests for all operations:

```rust
#[test]
fn test_add_correctness() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = add_f32(&a, &b).unwrap();
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_add_empty() {
    let result = add_f32(&[], &[]).unwrap();
    assert!(result.is_empty());
}
```

### 2. Property-Based Tests

Using `proptest` to verify mathematical properties:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_add_commutative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..1000),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..1000)
    ) {
        let len = a.len().min(b.len());
        let a = &a[..len];
        let b = &b[..len];

        let result1 = add_f32(a, b).unwrap();
        let result2 = add_f32(b, a).unwrap();

        assert_eq!(result1, result2);
    }
}
```

### 3. Backend Equivalence Tests

Verify all backends produce identical results:

```rust
#[test]
fn test_backend_equivalence_add() {
    let a = vec![1.0f32; 10000];
    let b = vec![2.0f32; 10000];

    let scalar = add_vectors_scalar(&a, &b);
    let sse2 = unsafe { add_vectors_sse2(&a, &b) };
    let avx2 = unsafe { add_vectors_avx2(&a, &b) };

    // Allow small floating-point tolerance
    for i in 0..scalar.len() {
        assert!((scalar[i] - sse2[i]).abs() < 1e-5);
        assert!((scalar[i] - avx2[i]).abs() < 1e-5);
    }
}
```

## Quality Gates

### Pre-Commit Checklist

Every commit must pass:

```bash
# Full quality gate check
make quality-gates

# Individual checks
make lint          # Zero clippy warnings
make fmt-check     # Proper formatting
make test-fast     # All tests pass
make coverage      # >90% coverage
```

### Tiered Testing (CI/CD)

```bash
# Tier 1: On-save (sub-second)
make tier1

# Tier 2: On-commit (1-5 minutes)
make tier2

# Tier 3: On-merge/nightly (hours)
make tier3
```

## GPU Testing

GPU tests require special handling due to hardware dependencies:

```bash
# Check if GPU is available
cargo test --all-features test_gpu_backend_available_check

# GPU tests are included in the standard coverage run
make coverage

# GPU tests use shared device pattern for faster execution
# See: src/backends/gpu/batch.rs
```

### GPU Test Patterns

GPU tests use a shared device to reduce initialization overhead:

```rust
use std::sync::OnceLock;

static SHARED_DEVICE: OnceLock<Option<GpuDevice>> = OnceLock::new();

fn get_shared_device() -> Option<GpuDevice> {
    SHARED_DEVICE
        .get_or_init(|| {
            if GpuDevice::is_available() {
                GpuDevice::new().ok()
            } else {
                None
            }
        })
        .clone()
}

#[test]
fn test_gpu_operation() {
    let Some(device) = get_shared_device() else {
        eprintln!("GPU not available, skipping");
        return;
    };
    // Test with device...
}
```

## Mutation Testing

Verify test quality with mutation testing:

```bash
# Run mutation testing (target: >80% kill rate)
make mutate

# Or directly with cargo-mutants
cargo mutants --timeout 60 --minimum-pass-rate 80
```

## Nextest Configuration

Trueno uses `cargo-nextest` for parallel test execution. Configuration is in `.config/nextest.toml`:

```toml
[profile.default]
slow-timeout = { period = "30s", terminate-after = 2 }
test-threads = "num-cpus"

[profile.coverage]
slow-timeout = { period = "20s", terminate-after = 2 }
# Exclude slow async GPU batch tests from fast coverage
default-filter = "not test(/test_matmul_parallel_1024/) and not test(/batch::tests::test_all_batch_operations/)"
```

## Troubleshooting

### Coverage Too Low

1. Check which files have low coverage:
   ```bash
   make coverage
   # Look at the detailed breakdown
   ```

2. For GPU code, coverage is included in the unified `make coverage` command (handles both crates automatically)

### Tests Timing Out

1. Increase timeout in `.config/nextest.toml`
2. Use `--test-threads=1` for GPU tests
3. Check for resource contention

### GPU Tests Failing

1. Verify GPU availability:
   ```bash
   cargo test --all-features test_gpu_backend_available_check
   ```

2. Check CUDA installation (for CUDA tests):
   ```bash
   nvidia-smi
   ```

3. Run GPU tests in isolation:
   ```bash
   cargo test --all-features -- --test-threads=1 gpu
   ```
