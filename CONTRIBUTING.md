# Contributing to trueno-zram

Thank you for your interest in contributing to trueno-zram! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/trueno-zram.git`
3. Create a branch for your changes
4. Make your changes following the guidelines below
5. Submit a pull request

## Development Setup

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/paiml/trueno-zram.git
cd trueno-zram
cargo build --release

# Run tests
cargo test -p trueno-zram-core

# Run full quality gate
make quality-gate

# Run benchmarks
cargo bench -p trueno-zram-core
```

## Requirements

- Linux Kernel >= 6.0 (for ublk support)
- x86_64 with AVX2/AVX-512 or AArch64 with NEON
- Rust >= 1.82.0

## Quality Standards

### Code Quality

- **No `unwrap()` in production code** -- use `.expect("descriptive message")` or proper error handling with `?`
- All public APIs must have documentation with examples
- Follow Rust API guidelines: <https://rust-lang.github.io/api-guidelines/>
- Maximum cognitive complexity of 25 per function

### Testing

- All new features must include tests
- Target 95% or higher line coverage
- Use property-based testing (`proptest`) for compression roundtrip invariants
- Run the full test suite before submitting: `cargo test --all-features`

### Linting

- Code must pass `cargo clippy -- -D warnings`
- Code must be formatted with `cargo fmt`
- The `.clippy.toml` enforces additional rules including the `unwrap()` ban

### Performance

- Compression/decompression changes must include benchmark results
- No regressions allowed on ZSTD or LZ4 throughput
- Run `cargo bench -p trueno-zram-core` to verify

## Pull Request Process

1. Ensure all tests pass: `cargo test --all-features`
2. Ensure clippy is clean: `cargo clippy --all-targets --all-features -- -D warnings`
3. Ensure code is formatted: `cargo fmt --all --check`
4. Include benchmark results for performance-related changes
5. Update documentation if adding new public APIs
6. Add a changelog entry under `[Unreleased]` in `CHANGELOG.md`
7. Request review from a maintainer

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ZSTD level 3 compression support
fix: correct page alignment for AVX-512
docs: update benchmark comparison table
test: add property tests for entropy routing
perf: optimize decompression hot path
```

## Architecture Overview

| Crate | Description |
|-------|-------------|
| `trueno-zram-core` | SIMD compression library (LZ4, ZSTD, AVX-512) |
| `trueno-zram-adaptive` | Entropy-based algorithm selection |
| `trueno-ublk` | Userspace block device daemon |

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Include Rust version (`rustc --version`), kernel version (`uname -r`), and CPU features

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 License.
