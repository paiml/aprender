# Coverage Requirements

Trueno-viz maintains strict test coverage requirements to ensure reliability.

## Coverage Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Line Coverage | 95% | 98% |
| Branch Coverage | 85% | 90% |
| Function Coverage | 98% | 100% |

## Measuring Coverage

```bash
# Install coverage tool
cargo install cargo-llvm-cov

# Generate coverage report
cargo llvm-cov --html

# View report
open target/llvm-cov/html/index.html
```

## Coverage in CI

The CI pipeline enforces coverage requirements:

```yaml
# .github/workflows/ci.yml
- name: Check coverage
  run: |
    cargo llvm-cov --fail-under-lines 95
```

## Excluding from Coverage

Some code is intentionally excluded:

```rust
// FFI bindings (tested via integration tests)
#[cfg(not(tarpaulin_include))]
mod ffi {
    // ...
}

// Platform-specific code
#[cfg(target_os = "windows")]
#[cfg(not(tarpaulin_include))]
fn windows_specific() {
    // ...
}
```

## Coverage by Module

Current coverage (as of latest release):

| Module | Coverage |
|--------|----------|
| `color` | 98.2% |
| `geometry` | 97.8% |
| `scale` | 96.5% |
| `plots` | 96.1% |
| `grammar` | 95.8% |
| `output` | 95.2% |
| **Overall** | **96.4%** |

## Writing for Coverage

### Cover all branches

```rust
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min              // Test: value < min
    } else if value > max {
        max              // Test: value > max
    } else {
        value            // Test: min <= value <= max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_below() {
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn test_clamp_above() {
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_clamp_within() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
    }
}
```

### Cover error paths

```rust
pub fn divide(a: f32, b: f32) -> Result<f32> {
    if b == 0.0 {
        Err(Error::DivisionByZero)  // Must test this path!
    } else {
        Ok(a / b)
    }
}

#[test]
fn test_divide_by_zero() {
    assert!(matches!(divide(5.0, 0.0), Err(Error::DivisionByZero)));
}
```

## Coverage Tools

### cargo-llvm-cov (Recommended)

Most accurate, uses LLVM instrumentation:

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --html
```

### tarpaulin

Alternative, works well with CI:

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

### grcov

Mozilla's coverage tool:

```bash
cargo install grcov
RUSTFLAGS="-C instrument-coverage" cargo test
grcov . -s . --binary-path ./target/debug/ -o ./coverage/
```

## Interpreting Reports

```text
File: src/scale.rs
┌────────┬───────────┬─────────┐
│ Line   │ Hits      │ Source  │
├────────┼───────────┼─────────┤
│ 15     │ ✓ 42      │ if x < 0│
│ 16     │ ✓ 12      │   -x    │
│ 17     │ ✓ 30      │ else    │
│ 18     │ ✗ 0       │   x * 2 │  ← NOT COVERED!
└────────┴───────────┴─────────┘
```

## Best Practices

1. **Write tests first** - TDD naturally achieves high coverage
2. **Test edge cases** - Empty inputs, boundaries, errors
3. **Don't game metrics** - Coverage doesn't guarantee correctness
4. **Use property tests** - Catch edge cases automatically
5. **Review uncovered lines** - Understand why they're not covered

## Complete Coverage Check

```bash
# Full coverage check with thresholds
cargo llvm-cov \
    --fail-under-lines 95 \
    --fail-under-branches 85 \
    --fail-under-functions 98 \
    --html
```

## Next Chapter

Continue to [Color Types](../api/color.md) for API reference.
