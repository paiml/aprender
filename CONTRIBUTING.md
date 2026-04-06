# Contributing to alimentar

Thank you for considering contributing to alimentar! This document outlines how to contribute effectively.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/paiml/alimentar.git
cd alimentar

# Install development dependencies
make dev-deps

# Run quality checks
make check
```

## Quality Standards

- **Coverage**: Minimum 85% line coverage (`make coverage`)
- **Linting**: Zero clippy warnings (`make lint`)
- **Formatting**: `cargo fmt` compliance (`make fmt-check`)
- **Security**: `cargo audit` clean (`cargo audit`)

## Workflow

1. Fork the repository
2. Create your feature branch
3. Write tests for new functionality
4. Ensure all quality gates pass: `make quality-gate`
5. Submit a pull request

## Code Style

- Follow Rust idioms and the project's `.clippy.toml` configuration
- Use `expect()` with descriptive messages instead of `unwrap()`
- Add doc comments for all public items
- Keep functions under 100 lines and cyclomatic complexity under 15

## Testing

```bash
make test          # All tests
make test-fast     # Quick parallel tests
make coverage      # Coverage report
make mutants       # Mutation testing
```

## Commit Messages

Use conventional commit style:

```
feat: add parquet schema inference
fix: correct CSV delimiter detection
docs: update API examples
test: add property tests for streaming
```

## License

By contributing, you agree that your contributions will be licensed under the MIT license.
