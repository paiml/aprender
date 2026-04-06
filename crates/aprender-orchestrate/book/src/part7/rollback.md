# Rollback Planning

Every migration step must be reversible. A rollback plan is a safety net that enables faster, bolder migration decisions.

## Feature Flags for Old/New Paths

Use compile-time feature flags to keep both implementations available:

```rust
#[cfg(feature = "legacy-python-ffi")]
pub fn compute(data: &[f32]) -> Vec<f32> {
    python_ffi::call_legacy_compute(data)
}

#[cfg(not(feature = "legacy-python-ffi"))]
pub fn compute(data: &[f32]) -> Vec<f32> {
    native_rust_compute(data)
}
```

```bash
cargo build --features legacy-python-ffi
```

## Runtime Feature Flags

For systems that cannot be recompiled:

```rust
pub fn compute(data: &[f32]) -> Vec<f32> {
    if std::env::var("USE_LEGACY_BACKEND").is_ok() {
        legacy_compute(data)
    } else {
        rust_compute(data)
    }
}
```

## Dual-Stack Testing

Run both implementations in parallel during migration:

```bash
batuta validate --trace --compare --dual-stack ./rust_out
```

| Aspect | Method | Tolerance |
|--------|--------|-----------|
| Numeric output | Absolute difference | 1e-6 (f32), 1e-12 (f64) |
| String output | Exact match | None |
| Syscall sequence | renacer trace diff | Order-insensitive for I/O |

## Git-Based Rollback

Tag each migration milestone:

```bash
git tag pre-migrate/data-loader

# If migration fails
git revert --no-commit HEAD~3..HEAD
git commit -m "Rollback data-loader migration"
```

## Rollback Checklist

Before declaring a module migration complete:

1. Feature flag allows instant revert to legacy code
2. All tests pass with both implementations
3. Performance benchmarks show no regression
4. renacer trace comparison shows equivalence
5. Rollback procedure documented and tested

---

**Navigate:** [Table of Contents](../SUMMARY.md)
