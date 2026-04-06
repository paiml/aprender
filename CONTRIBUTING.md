# Contributing to trueno-rag

Thank you for your interest in contributing to trueno-rag! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/trueno-rag.git`
3. Create a branch for your changes
4. Make your changes following the guidelines below
5. Submit a pull request

## Development Setup

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/paiml/trueno-rag.git
cd trueno-rag
cargo build

# Run tests
cargo test

# Run lints
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt --all
```

## Quality Standards

### Code Quality

- **No `unwrap()` in production code** -- use `.expect("descriptive message")` or proper error handling with `?`
- All public APIs must have documentation with examples
- Follow Rust API guidelines: <https://rust-lang.github.io/api-guidelines/>
- Maximum cognitive complexity of 25 per function

### Testing

- All new features must include tests
- Target 95% or higher line coverage
- Use property-based testing (`proptest`) for algorithmic code
- Run the full test suite before submitting: `cargo test --all-features`

### Linting

- Code must pass `cargo clippy -- -D warnings`
- Code must be formatted with `cargo fmt`
- The `.clippy.toml` enforces additional rules including the `unwrap()` ban

## Pull Request Process

1. Ensure all tests pass: `cargo test --all-features`
2. Ensure clippy is clean: `cargo clippy --all-targets --all-features -- -D warnings`
3. Ensure code is formatted: `cargo fmt --all --check`
4. Update documentation if adding new public APIs
5. Add a changelog entry under `[Unreleased]` in `CHANGELOG.md`
6. Request review from a maintainer

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add semantic chunking strategy
fix: correct BM25 scoring for empty documents
docs: update API reference for rerankers
test: add property tests for fusion strategies
perf: optimize vector similarity computation
```

## Architecture Overview

- `src/chunk/` -- Document chunking strategies
- `src/embed/` -- Embedding providers (Mock, TF-IDF, FastEmbed, Nemotron)
- `src/retrieve.rs` -- Dense, sparse, and hybrid retrieval
- `src/fusion.rs` -- Result fusion strategies (RRF, Linear, DBSF)
- `src/rerank.rs` -- Reranking implementations
- `src/pipeline/` -- High-level RAG pipeline builder
- `src/sqlite/` -- SQLite-backed persistent storage
- `src/compressed.rs` -- LZ4/ZSTD index compression

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Include Rust version (`rustc --version`) and OS information

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
