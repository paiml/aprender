# Contributing to Entrenar

Thank you for your interest in contributing to Entrenar!

## How to Contribute

1. Fork the repository
2. Create your changes on master
3. Run `cargo fmt && cargo clippy -- -D warnings && cargo test`
4. Submit a pull request

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- All clippy warnings must be resolved
- No `unwrap()` in production code — use `expect()` or proper error handling

## Pull Request Process

1. Ensure all tests pass
2. Update documentation for any public API changes
3. Add tests for new functionality

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
