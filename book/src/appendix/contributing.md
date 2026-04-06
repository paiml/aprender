# Contributing Guide

Thank you for your interest in contributing to Batuta!

## Getting Started

### Prerequisites

- Rust 1.75+ (stable)
- Git
- Cargo

### Clone and Build

```bash
git clone https://github.com/paiml/batuta.git
cd batuta
cargo build
cargo test
```

## Development Workflow

### Branch Strategy

All work happens on `main` branch. No feature branches.

### Quality Gates

Before committing, ensure:

```bash
# Format code
cargo fmt

# Run lints
cargo clippy -- -D warnings

# Run tests
cargo test

# Check demo-score (must be A- or higher)
pmat demo-score
```

### Commit Messages

Follow conventional commits:

```
type(scope): description

- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Tests
- chore: Maintenance
```

Example:
```
feat(stack): Add diagnostics module

- Add anomaly detection
- Add graph metrics
- Add dashboard rendering

(Refs STACK-DIAG)
```

## Code Style

### Rust Guidelines

- Use `rustfmt` defaults
- No `unwrap()` in library code (use `?` or `expect()` with message)
- Document public APIs with doc comments
- Add tests for new functionality

### Documentation

- Update book chapters for new features
- Keep README current
- Add examples for complex features

## Testing

### Test Categories

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# Examples
cargo run --example <name>
```

### Quality Metrics

- Coverage: 85%+ target
- Mutation score: 80%+ target
- Demo score: A- (85) minimum

## Pull Requests

1. Ensure all quality gates pass
2. Update documentation
3. Add tests for new code
4. Reference issue/ticket in commit

## Questions?

- Open an issue on GitHub
- Check existing documentation

---

**Navigate:** [Table of Contents](../SUMMARY.md)
