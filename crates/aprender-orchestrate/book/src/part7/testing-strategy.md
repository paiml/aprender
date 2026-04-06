# Testing Strategy

Testing during migration serves a dual purpose: verifying that the Rust code is correct on its own, and confirming that it preserves the behavior of the original. Batuta enforces a layered testing strategy aligned with the Certeza quality methodology.

## Testing Pyramid

```
              /\
             /  \        Tier 4: CI/CD
            / E2E\       Release tests, mutation, pmat analysis
           /──────\
          / Integ  \     Tier 3: Pre-push
         / ration   \    Full test suite, cross-module
        /────────────\
       /   Unit       \  Tier 2: Pre-commit
      /   Tests        \ cargo test --lib, clippy
     /──────────────────\
    /  Static Analysis   \ Tier 1: On-save
   / fmt, clippy, check   \ < 1 second
  /────────────────────────\
```

## Quality Tiers

| Tier | Trigger | Time Budget | What Runs |
|------|---------|-------------|-----------|
| Tier 1 | On save | < 1s | `cargo fmt`, `cargo clippy`, `cargo check` |
| Tier 2 | Pre-commit | < 5s | `cargo test --lib`, complexity gate |
| Tier 3 | Pre-push | 1-5 min | Full tests, integration tests |
| Tier 4 | CI/CD | 5-30 min | Release tests, mutation testing, pmat analysis |

Run tiers via Make:

```bash
make tier1   # On-save checks
make tier2   # Pre-commit gate
make tier3   # Pre-push validation
make tier4   # Full CI/CD pipeline
```

## Coverage Requirements

The Sovereign AI Stack enforces strict coverage targets:

- **90% minimum** (enforced, build fails below this)
- **95% preferred** (target for all new code)

```bash
make coverage   # Generates HTML + LCOV in target/coverage/
```

## Migration-Specific Testing

During migration, every transpiled module needs three test categories:

1. **Parity tests**: Output matches original implementation for the same input
2. **Property tests**: Invariants hold across random inputs (proptest)
3. **Regression tests**: Previously-fixed bugs stay fixed

```rust
#[test]
fn parity_with_python_output() {
    // Known input/output pairs captured from Python
    let input = vec![1.0, 2.0, 3.0];
    let expected = vec![2.0, 4.0, 6.0];
    assert_eq!(transform(&input), expected);
}
```

## Test Organization

```
src/
  module.rs           # Production code
  module/
    tests.rs          # Unit tests (use super::*)
tests/
  integration/
    module_test.rs    # Integration tests
  parity/
    module_parity.rs  # Python output comparison
```

See the following chapters for detailed guidance on [Test Migration](./test-migration.md), [Property-Based Testing](./property-testing.md), and [Regression Prevention](./regression.md).

---

**Navigate:** [Table of Contents](../SUMMARY.md)
