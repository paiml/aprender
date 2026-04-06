# Contributing to simular

Thank you for your interest in contributing to simular! This document provides guidelines and workflows for contributors.

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Development Setup

### Prerequisites

- Rust 1.75+ (stable)
- Nix (optional, for reproducible environment)

### Using Nix (Recommended)

```bash
nix develop
```

### Manual Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-unknown-unknown

# Install development tools
cargo install cargo-tarpaulin cargo-mutants wasm-pack
```

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/simular.git
cd simular
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

Follow the coding standards below.

### 4. Test Your Changes

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run mutation tests
cargo mutants --jobs 4

# Check lints
cargo clippy -- -D warnings
```

### 5. Submit a Pull Request

- Ensure all tests pass
- Update documentation if needed
- Follow the commit message format

## Coding Standards

### Toyota Production System (TPS)

- **Jidoka**: Stop on error - use `Result<T, E>` everywhere
- **Poka-Yoke**: Type-safe units via `uom` crate
- **Zero SATD**: No TODO/FIXME comments - create issues instead

### JPL Power of 10 Rules

| Rule | Threshold |
|------|-----------|
| Max nesting depth | 4 |
| Max function lines | 60 |
| Max cyclomatic complexity | 15 |
| Min assertions per function | 2 |

### Commit Messages

Format: `type(scope): description (Refs ISSUE-ID)`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
```
feat(tsp): Add 2-opt local search (Refs OR-001)
fix(orbit): Correct energy conservation (Refs ORBIT-042)
docs: Update installation guide
```

## Testing Requirements

### Coverage Requirements

- Minimum 95% line coverage
- Minimum 80% mutation coverage

### Test Types

1. **Unit Tests**: Test individual functions
2. **Property Tests**: Use `proptest` for fuzzing
3. **Integration Tests**: Test module interactions
4. **EDD Tests**: Follow 5-phase methodology

### EDD 5-Phase Test Structure

```rust
#[cfg(test)]
mod tests {
    // Phase 1: Equation documentation
    // Phase 2: Failing test
    // Phase 3: Implementation test
    // Phase 4: Verification test
    // Phase 5: Falsification test
}
```

## Pull Request Checklist

- [ ] All tests pass (`cargo test`)
- [ ] No clippy warnings (`cargo clippy -- -D warnings`)
- [ ] Code formatted (`cargo fmt`)
- [ ] Coverage meets 95% threshold
- [ ] Documentation updated
- [ ] Commit messages follow format
- [ ] No new SATD comments

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
