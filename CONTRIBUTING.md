# Contributing to Presentar

Thank you for your interest in contributing to Presentar!

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Rust 1.83.0+ (see `rust-toolchain.toml`)
- Git
- Optional: Nix (for reproducible dev environment)

### Setup

```bash
# Clone the repository
git clone https://github.com/paiml/presentar
cd presentar

# Option 1: Use Nix (recommended for reproducibility)
nix develop

# Option 2: Manual setup
rustup show  # Will install correct toolchain
cargo build
```

### Running Tests

```bash
# Fast tests (tier 1)
make tier1

# Full tests (tier 2)
make tier2

# With coverage
make coverage
```

## Development Process

### 1. Fork and Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
```

### 2. Write Tests First (Popperian TDD)

We follow Popperian falsificationism - write tests that TRY TO BREAK your code:

```rust
/// FALSIFICATION TEST: Feature X must handle edge case Y
#[test]
fn falsify_feature_x_edge_case_y() {
    let result = feature_x(edge_case_input);
    assert!(
        result.is_valid(),
        "FALSIFIED: Feature X fails on edge case Y"
    );
}
```

See: `docs/specifications/pixel-by-pixel-demo-ptop-ttop.md` Part 0

### 3. Implement

- Follow Rust idioms and clippy recommendations
- No `unwrap()` in production code
- Document public APIs

### 4. Quality Checks

```bash
# Must pass before PR
cargo clippy --all-features -- -D warnings
cargo fmt --check
cargo test
```

### 5. Submit PR

- Reference any related issues
- Include test results
- Describe changes clearly

## Commit Messages

Follow conventional commits:

```
feat(ptop): add temperature display for AMD k10temp
fix(ui): correct color gradient for high CPU usage
docs(spec): update falsification protocol
test(cpu): add k10temp edge case tests
```

## Questions?

- Open a [GitHub Issue](https://github.com/paiml/presentar/issues)
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
