# Contributing to Renacer

Thank you for your interest in contributing to Renacer!

## How to Contribute

1. Fork the repository
2. Create your changes on master
3. Run `cargo fmt && cargo clippy -- -D warnings && cargo test`
4. Submit a pull request

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- All clippy warnings must be resolved
- No `unwrap()` in production code — use `expect()` or proper error handling
- Property-based tests required for new syscall handling logic

## Testing

```bash
cargo test                                    # All tests
cargo test --test property_based_comprehensive # Property-based tests
make tier1                                    # Fast tests (<5s)
make tier2                                    # Integration tests (<30s)
make tier3                                    # Full validation (<5m)
```

## Pull Request Process

1. Ensure all quality gates pass (`make tier2`)
2. Update documentation for any public API changes
3. Add tests for new functionality
4. Include property-based tests for new parsing or filtering logic

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
