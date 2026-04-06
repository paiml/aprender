# Extreme TDD

Trueno follows an **Extreme TDD** methodology where test coverage is a non-negotiable quality gate, not a nice-to-have metric. Every commit must maintain 90%+ line coverage, and the project currently sits at **95%+** across both crates.

## Coverage Mandate

| Metric | Floor | Current |
|--------|-------|---------|
| Line coverage (overall) | 90% | 95%+ |
| trueno (core library) | 90% | ~96% |
| trueno-gpu (CUDA/PTX) | 90% | ~96% |
| New code | 100% | - |

The 90% floor is **automatically enforced** by a pre-commit hook. If coverage drops below 90%, the commit is blocked:

```
$ git commit -m "my change"
🔍 Checking test coverage before commit...
❌ COMMIT BLOCKED
   Coverage: 89.5%
   Required: 90%
   Gap: 0.5%
```

There is no `--no-verify` escape hatch in normal workflow. Fix the gap first.

## The Only Coverage Command

```bash
make coverage
```

This is the **only** allowed coverage command. It handles:

1. Both `trueno` and `trueno-gpu` crates in a single run
2. `--features cuda` when NVIDIA GPU is available
3. Exclusion patterns (`COV_EXCLUDE` for generated code, test helpers)
4. Combined LCOV report generation
5. HTML report at `target/coverage/html/index.html`
6. Final coverage percentage display

Never run `cargo llvm-cov` directly. Never install or use `cargo-tarpaulin`.

## Test Categories

Trueno uses five categories of tests, each catching different classes of defects:

### 1. Unit Tests

Standard correctness tests for every operation, including edge cases:

```rust
#[test]
fn test_add_empty() {
    let result = add_f32(&[], &[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_add_nan_propagation() {
    let a = vec![1.0, f32::NAN, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = add_f32(&a, &b).unwrap();
    assert!(result[1].is_nan());
}
```

### 2. Property-Based Tests (proptest)

Verify mathematical invariants hold across random inputs:

```rust
proptest! {
    #[test]
    fn test_add_commutative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..1000),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..1000)
    ) {
        let len = a.len().min(b.len());
        let result1 = add_f32(&a[..len], &b[..len]).unwrap();
        let result2 = add_f32(&b[..len], &a[..len]).unwrap();
        assert_eq!(result1, result2);
    }
}
```

Property tests run with `PROPTEST_CASES=256` in CI (Tier 2) and `PROPTEST_CASES=3` for fast local iteration.

### 3. Backend Equivalence Tests

All SIMD backends must produce identical results (within floating-point tolerance):

```rust
#[test]
fn test_backend_equivalence_add() {
    let a = vec![1.0f32; 10000];
    let b = vec![2.0f32; 10000];

    let scalar = add_vectors_scalar(&a, &b);
    let sse2 = unsafe { add_vectors_sse2(&a, &b) };
    let avx2 = unsafe { add_vectors_avx2(&a, &b) };

    for i in 0..scalar.len() {
        assert!((scalar[i] - sse2[i]).abs() < 1e-5);
        assert!((scalar[i] - avx2[i]).abs() < 1e-5);
    }
}
```

### 4. Mutation Testing

Verify that tests actually catch bugs, not just exercise code paths:

```bash
cargo mutants --timeout 120 --minimum-pass-rate 80
```

Target: **80%+ mutation kill rate**. If a mutant survives, the test suite has a blind spot.

### 5. Coverage Gap Tests

Targeted tests written specifically to close coverage gaps. These are identified using `pmat query --coverage-gaps` and target the highest-impact uncovered functions first.

## Writing a Coverage Gap Test

The workflow for closing coverage gaps:

1. **Find gaps** ranked by impact:
   ```bash
   pmat query --coverage-gaps --rank-by impact --limit 20 --exclude-tests
   ```

2. **Read the uncovered function**:
   ```bash
   pmat query "function_name" --include-source --limit 1
   ```

3. **Write a test** that exercises the uncovered paths. For GPU PTX generators, test via `kernel.emit_ptx()` (no hardware needed):
   ```rust
   #[test]
   fn test_softmax_kernel_ptx_generation() {
       let kernel = SoftmaxKernel::new(512);
       let ptx = kernel.emit_ptx();
       assert!(ptx.contains(".version"));
       assert!(ptx.contains(".entry"));
       assert!(ptx.contains(".target"));
   }
   ```

4. **Verify** the gap is closed:
   ```bash
   make coverage
   ```

## Pre-Commit Enforcement

The pre-commit hook at `.git/hooks/pre-commit` runs `make coverage` on every commit attempt. The hook:

1. Runs the full test suite with coverage instrumentation
2. Parses the final coverage percentage
3. Blocks the commit if coverage < 90%
4. Prints a clear error message with the gap

This is a hard gate. The only way to commit with lower coverage is `git commit --no-verify`, which should only be used in emergencies and documented in the commit message.

## See Also

- [Testing](./testing.md) — Test infrastructure and commands
- [Quality Gates](./quality-gates.md) — Full quality enforcement pipeline
