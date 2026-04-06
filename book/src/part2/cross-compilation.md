# Cross-Compilation

Cross-compilation builds the transpiled Rust project for a target platform different from the host. Batuta supports cross-compilation through Cargo's target triple system and the `cross` tool.

## Target Triples

A target triple specifies the architecture, vendor, OS, and ABI:

```
<arch>-<vendor>-<os>-<abi>
```

## Common Targets

| Target Triple | Platform | Use Case |
|--------------|----------|----------|
| `x86_64-unknown-linux-gnu` | Linux x86-64 (glibc) | Standard Linux servers |
| `x86_64-unknown-linux-musl` | Linux x86-64 (musl) | Static binaries, Alpine |
| `aarch64-unknown-linux-gnu` | Linux ARM64 | AWS Graviton, Raspberry Pi 4 |
| `x86_64-apple-darwin` | macOS Intel | Mac development |
| `aarch64-apple-darwin` | macOS Apple Silicon | M1/M2/M3 Macs |
| `x86_64-pc-windows-msvc` | Windows x86-64 | Windows deployment |
| `wasm32-unknown-unknown` | WebAssembly | Browser deployment |

## Using Cargo Directly

```bash
# Install target toolchain
rustup target add aarch64-unknown-linux-gnu

# Cross-compile
batuta build --release --target aarch64-unknown-linux-gnu
```

## Using the `cross` Tool

The `cross` tool uses Docker containers with pre-configured cross-compilation toolchains:

```bash
# Install cross
cargo install cross

# Cross-compile without manual toolchain setup
cross build --release --target aarch64-unknown-linux-gnu
```

This is the recommended approach because it handles linker configuration, system libraries, and C dependencies automatically.

## musl Static Linking

The `musl` target produces fully static binaries with no dynamic library dependencies, ideal for Docker scratch containers, Lambda functions, and air-gapped environments:

```bash
rustup target add x86_64-unknown-linux-musl
batuta build --release --target x86_64-unknown-linux-musl
```

## WebAssembly Target

WASM builds require special handling. See the Batuta `wasm` feature flag:

```bash
# WASM debug build
batuta build --wasm

# WASM release build
batuta build --wasm --release
```

The WASM build disables filesystem access and uses in-memory analysis, controlled by the `wasm` feature flag in Cargo.toml.

## Configuration

Cross-compilation settings in `batuta.toml`:

```toml
[build]
target = "x86_64-unknown-linux-musl"
cargo_flags = ["--locked"]
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
