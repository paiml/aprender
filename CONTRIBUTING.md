# Contributing to Aprender

Thank you for your interest in contributing to Aprender!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/paiml/aprender.git
cd aprender

# Build
cargo build --release

# Run tests
cargo test

# Run quality gates
make tier2
```

## Quality Standards

Aprender follows EXTREME TDD methodology with strict quality gates:

- **Test Coverage**: 95%+ required (current: 96.94%)
- **Clippy**: Zero warnings (`cargo clippy -- -D warnings`)
- **Formatting**: `cargo fmt --check`
- **Mutation Testing**: 85%+ mutation score

## Tiered Quality Gates

| Tier | When | Commands |
|------|------|----------|
| Tier 1 | On save | `cargo fmt --check && cargo clippy && cargo check` |
| Tier 2 | Pre-commit | `cargo test --lib && cargo clippy -- -D warnings` |
| Tier 3 | Pre-push | `cargo test --all && make coverage` |
| Tier 4 | CI/CD | Full mutation testing + PMAT analysis |

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests first (TDD)
4. Implement your changes
5. Ensure all quality gates pass
6. Submit PR with clear description

## Code Style

- Follow Rust idioms and conventions
- Use meaningful variable names
- Add documentation for public APIs
- Keep functions focused and small

## Reporting Issues

- Use GitHub Issues for bug reports
- Include minimal reproduction steps
- Specify Rust version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
