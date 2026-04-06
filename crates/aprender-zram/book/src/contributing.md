# Contributing

## Development Setup

```bash
# Clone the repository
git clone https://github.com/paiml/trueno-zram
cd trueno-zram

# Build
cargo build --all-features

# Run tests
cargo test --workspace --all-features

# Run with CUDA
cargo test --workspace --features cuda
```

## Code Style

- Format with `cargo fmt`
- Lint with `cargo clippy --all-features -- -D warnings`
- No panics in library code
- All public items must be documented

## Testing

### Unit Tests

```bash
cargo test --workspace --all-features
```

### Coverage

```bash
cargo llvm-cov --workspace --all-features
```

Target: 95% line coverage.

### Mutation Testing

```bash
cargo mutants --package trueno-zram-core
```

Target: 80% mutation score.

## Quality Gates

Before submitting a PR:

1. **Formatting**: `cargo fmt --check`
2. **Linting**: `cargo clippy --all-features -- -D warnings`
3. **Tests**: `cargo test --workspace --all-features`
4. **Documentation**: `cargo doc --no-deps`
5. **Coverage**: >= 95%

## Commit Messages

Follow conventional commits:

```
feat: Add new compression algorithm
fix: Handle edge case in decompression
perf: Optimize hash table lookup
docs: Update API documentation
test: Add property-based tests
refactor: Simplify SIMD dispatch
```

Always reference the work item:

```
feat: Add GPU batch compression (Refs ZRAM-001)
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality gates
5. Submit PR with description
6. Address review feedback

## Architecture

See [Design Overview](./architecture/overview.md) for architecture decisions.

## Adding a New Algorithm

1. Create module in `src/algorithms/`
2. Implement `PageCompressor` trait
3. Add SIMD implementations
4. Add to `Algorithm` enum
5. Write tests and benchmarks
6. Update documentation

## Adding SIMD Backend

1. Add detection in `src/simd/detect.rs`
2. Create implementation file (e.g., `avx512.rs`)
3. Add dispatch in `src/simd/dispatch.rs`
4. Write correctness tests
5. Run benchmarks

## Reporting Issues

Use GitHub Issues with:

- Clear description
- Reproduction steps
- Expected vs actual behavior
- System info (CPU, OS, Rust version)

## License

Contributions are licensed under MIT OR Apache-2.0.
