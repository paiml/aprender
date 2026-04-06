# Build Options

The `[build]` section controls Phase 5: compiling the transpiled Rust code into a release binary, WASM module, or cross-compiled target.

## Settings

```toml
[build]
release = true
wasm = false
cargo_flags = []
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `release` | bool | `true` | Build with `--release` optimizations |
| `target` | string | (none) | Rust target triple for cross-compilation |
| `wasm` | bool | `false` | Build a WebAssembly module instead of a native binary |
| `cargo_flags` | array | `[]` | Additional flags passed to `cargo build` |

## Release Profile

When `release` is `true` (the default), the build uses Cargo's release profile. Set it to `false` during development for faster compile times and debug symbols.

### LTO and Strip

Pass Cargo profile flags through `cargo_flags` to enable link-time optimization or strip symbols:

```toml
[build]
release = true
cargo_flags = ["--config", "profile.release.lto=true", "--config", "profile.release.strip=true"]
```

## WASM Target Configuration

Set `wasm = true` to target `wasm32-unknown-unknown`. Batuta uses `wasm-pack` if available, falling back to raw `cargo build --target wasm32-unknown-unknown`. The `wasm` feature flag is enabled automatically, gating out native-only code paths.

```toml
[build]
wasm = true
release = true
```

## Cross-Compilation Targets

Set the `target` field to any Rust target triple.

```toml
[build]
target = "aarch64-unknown-linux-gnu"
```

Common targets:

| Triple | Platform |
|--------|----------|
| `x86_64-unknown-linux-gnu` | Linux x86-64 (glibc) |
| `x86_64-unknown-linux-musl` | Linux x86-64 (static musl) |
| `aarch64-unknown-linux-gnu` | Linux ARM64 |
| `aarch64-apple-darwin` | macOS Apple Silicon |
| `wasm32-unknown-unknown` | WebAssembly (prefer `wasm = true`) |

Ensure the corresponding toolchain is installed before cross-compiling:

```bash
rustup target add aarch64-unknown-linux-gnu
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
