# Contributing to trueno-graph

Thank you for your interest in contributing to trueno-graph.

## How to Contribute

1. Fork the repository and create your branch from `main`.
2. Make your changes and ensure all tests pass.
3. Add tests for any new functionality.
4. Ensure your code passes all quality gates (`make lint`, `make test`).
5. Submit a pull request.

## Code Style

- Follow standard Rust conventions (`cargo fmt`).
- All code must pass `cargo clippy -- -D warnings` with zero warnings.
- Maintain test coverage at or above 95%.

## Pull Request Process

1. Update documentation if your change affects public APIs.
2. Ensure CI passes (formatting, linting, tests, coverage).
3. PRs require at least one approving review before merge.
4. Squash commits into a single meaningful commit when merging.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
