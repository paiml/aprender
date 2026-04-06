# Contributing to Trueno-Viz

Thank you for your interest in contributing to trueno-viz!

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/paiml/trueno-viz.git
cd trueno-viz
```

### Build and Test

```bash
# Build
cargo build

# Run tests
cargo test

# Run with all features
cargo test --all-features

# Check formatting
cargo fmt -- --check

# Run clippy
cargo clippy -- -D warnings
```

## Development Workflow

### 1. Create an Issue

Before starting work, create or find an issue describing the change.

### 2. Fork and Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Write Tests First (TDD)

```rust
#[test]
fn test_new_feature() {
    // Define expected behavior
    let result = new_feature(input);
    assert_eq!(result, expected);
}
```

### 4. Implement the Feature

Make tests pass with minimal implementation.

### 5. Run Quality Checks

```bash
# All tests pass
cargo test --all-features

# Coverage meets threshold
cargo llvm-cov --fail-under-lines 95

# No clippy warnings
cargo clippy -- -D warnings

# Properly formatted
cargo fmt -- --check
```

### 6. Submit Pull Request

Include:
- Description of changes
- Issue reference
- Test coverage report

## Code Style

### Formatting

Use `rustfmt` with default settings:

```bash
cargo fmt
```

### Documentation

All public items require documentation:

```rust
/// Computes the linear interpolation between two points.
///
/// # Arguments
///
/// * `self` - Start point
/// * `other` - End point
/// * `t` - Interpolation factor (0.0 to 1.0)
///
/// # Returns
///
/// The interpolated point.
///
/// # Example
///
/// ```
/// let p1 = Point::new(0.0, 0.0);
/// let p2 = Point::new(10.0, 10.0);
/// let mid = p1.lerp(p2, 0.5);
/// assert_eq!(mid, Point::new(5.0, 5.0));
/// ```
pub fn lerp(self, other: Self, t: f32) -> Self {
    // ...
}
```

### Error Handling

Use the crate's error types:

```rust
use crate::error::{Error, Result};

pub fn process(data: &[f32]) -> Result<Output> {
    if data.is_empty() {
        return Err(Error::EmptyData);
    }
    // ...
}
```

## Testing Guidelines

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Basic functionality
    #[test]
    fn test_basic_case() { ... }

    // Edge cases
    #[test]
    fn test_empty_input() { ... }

    #[test]
    fn test_single_element() { ... }

    // Error cases
    #[test]
    fn test_invalid_input() { ... }
}
```

### Property Tests

Use proptest for invariant testing:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_invariant(input in any::<Vec<f32>>()) {
        // Test invariants hold for all inputs
    }
}
```

## Pull Request Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt -- --check` passes
- [ ] Coverage meets 95% threshold
- [ ] CHANGELOG.md updated

## Questions?

- Open an issue for questions
- Join discussions on GitHub

Thank you for contributing!
