# ADR-0005: Modular Feature Flags

## Status

Accepted

## Date

2024-11-25

## Context

Different deployment targets have different needs:

- **AWS Lambda**: Minimal binary size, fast cold start
- **Server**: Full HTTP stack, metrics, CLI
- **Embedded**: No std, minimal dependencies
- **WASM**: Browser deployment

## Decision

Use Cargo feature flags for modular compilation.

## Features

```toml
[features]
default = ["server", "cli", "gpu"]
minimal = []           # Core inference only
server = ["axum", "tokio", "tower"]
cli = ["clap"]
gpu = ["trueno/gpu"]
full = ["server", "cli", "gpu", "metrics"]
metrics = ["prometheus"]
```

## Rationale

1. **Binary size** - `minimal` feature produces <5MB binaries
2. **Cold start** - Less code = faster startup
3. **Dependencies** - Only pull what you need
4. **CI efficiency** - Faster builds for targeted tests

## Build Configurations

| Feature | Binary Size | Dependencies | Use Case |
|---------|-------------|--------------|----------|
| minimal | ~3 MB | 8 | Lambda, WASM |
| server | ~8 MB | 25 | Production API |
| full | ~12 MB | 35 | Development |

## Usage

```bash
# Lambda deployment (minimal)
cargo build --release --no-default-features --features minimal

# Server only (no CLI)
cargo build --release --no-default-features --features server,gpu

# Everything
cargo build --release --features full
```

## Consequences

### Positive
- Tailored builds for each deployment target
- Reduced attack surface with fewer dependencies
- Faster CI builds for targeted testing

### Negative
- Feature flag complexity
- Testing matrix increases
- Documentation must cover all combinations

## Validation

**Falsifiable claim**: `--features minimal` produces binaries <5MB.

**Test**:
```bash
cargo build --release --no-default-features --features minimal
ls -la target/release/realizar
```

## References

- [Cargo Features](https://doc.rust-lang.org/cargo/reference/features.html)
- [Feature Flag Best Practices](https://doc.rust-lang.org/cargo/reference/features.html#feature-resolver-version-2)
