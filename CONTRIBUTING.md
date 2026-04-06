# Contributing to Realizar

Thank you for your interest in contributing to Realizar! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guide](#style-guide)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Use inclusive language
- Accept constructive criticism gracefully
- Focus on what is best for the community

## Getting Started

### Prerequisites

- Rust 1.83.0 or later (see `rust-toolchain.toml`)
- Git
- Make

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/realizar.git
cd realizar
```

## Development Setup

### Install Dependencies

```bash
# Build the project
cargo build

# Run tests
cargo test

# Run quality gates
make quality-gates
```

### IDE Setup

We recommend VS Code with rust-analyzer extension:

```json
{
  "rust-analyzer.cargo.features": ["full"]
}
```

## Making Changes

### Branch Policy

We use a **trunk-based development** model:

- All commits go directly to `master`
- No feature branches
- Small, incremental changes

### EXTREME TDD

All changes must follow Test-Driven Development:

1. **RED**: Write a failing test first
2. **GREEN**: Write minimal code to pass
3. **REFACTOR**: Clean up while tests pass

```bash
# Run tests continuously
cargo watch -x test
```

### Commit Messages

Use conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `style`, `chore`

Example:
```
feat(parser): add GGUF v3 header support

Implement parser for GGUF version 3 header format.
Required for llama.cpp 0.3+ compatibility.

Refs #234
```

## Testing

### Running Tests

```bash
# All tests
cargo test

# Specific module
cargo test parser::

# With output
cargo test -- --nocapture

# Fast tests only
make test-fast
```

### Test Coverage

```bash
# Generate coverage report
make coverage

# View HTML report
open target/llvm-cov/html/index.html
```

Target: 85% minimum coverage

### Property-Based Tests

Use proptest for mathematical invariants:

```rust
proptest! {
    #[test]
    fn test_softmax_sums_to_one(v in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let result = softmax(&v);
        let sum: f32 = result.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5);
    }
}
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench --bench inference
```

## Submitting Changes

### Before Submitting

1. Run quality gates:
   ```bash
   make quality-gates
   ```

2. Ensure no warnings:
   ```bash
   cargo clippy -- -D warnings
   ```

3. Format code:
   ```bash
   cargo fmt
   ```

### Pull Request Process

1. Create PR with clear title and description
2. Link related issues
3. Wait for CI to pass
4. Address review feedback
5. Squash commits if requested

### PR Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] `make quality-gates` passes
```

## Style Guide

### Rust Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- No `unsafe` in public API
- Prefer `expect()` over `unwrap()` with descriptive messages

### Documentation

- Document all public items
- Include examples in doc comments
- Keep README up to date

### Error Handling

```rust
// Good: descriptive error message
let file = File::open(path)
    .expect("Failed to open model file");

// Bad: no context
let file = File::open(path).unwrap();
```

## Questions?

- Open a GitHub issue
- Check existing documentation
- Ask in discussions

Thank you for contributing!
