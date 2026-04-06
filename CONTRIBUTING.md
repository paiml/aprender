# Contributing to Pacha

Thank you for your interest in contributing to Pacha.

## Development Setup

```bash
git clone https://github.com/paiml/pacha.git
cd pacha
cargo build
cargo test
```

## Quality Gates

All contributions must pass the tiered quality gates:

```bash
make tier1   # Fast feedback: fmt, clippy, check
make tier2   # Pre-commit: tests + clippy
make tier3   # Pre-push: full validation
```

## Code Standards

- Run `cargo fmt` before committing
- All clippy warnings must be resolved: `cargo clippy -- -D warnings`
- No `unwrap()` in production code (use `.expect()` or `?` operator)
- Minimum test coverage: 80%

## Pull Request Process

1. Ensure all quality gates pass
2. Update documentation if applicable
3. Add tests for new functionality
4. Submit a pull request against `main`
