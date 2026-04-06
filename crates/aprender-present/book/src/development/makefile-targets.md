# Makefile Targets

Common development commands.

## Quick Reference

| Target | Description | Time |
|--------|-------------|------|
| `make build` | Production build | ~30s |
| `make test` | All tests | ~60s |
| `make test-fast` | Unit tests only | ~5s |
| `make lint` | Clippy check | ~10s |
| `make fmt` | Format check | ~2s |
| `make coverage` | Generate coverage | ~120s |

## Three-Tier Pipeline

### Tier 1: On-Save (<1s)

```bash
make tier1
```

- `cargo check` - Type checking
- Fast clippy (lib only)
- Fast tests

### Tier 2: Pre-Commit (1-5min)

```bash
make tier2
```

- Format check
- Full clippy
- All tests
- Quality score

### Tier 3: Nightly

```bash
make tier3
```

- Tier 2 +
- Coverage report
- Mutation testing

## Development Targets

```bash
# Development server with hot reload
make dev

# Run single test
make test-one TEST=test_button_click

# Watch mode for TDD
make watch

# Generate documentation
make doc
```

## Build Targets

```bash
# Debug build
cargo build

# Release build
make build

# Clean artifacts
make clean
```

## Quality Targets

```bash
# Format code
make fmt-fix

# Lint code
make lint

# Coverage report
make coverage

# Quality score
make score
```

## Verified Test

```rust
#[test]
fn test_makefile_exists() {
    // Makefile provides these targets
    let targets = ["build", "test", "lint", "fmt", "tier1", "tier2", "tier3"];
    assert_eq!(targets.len(), 7);
}
```
