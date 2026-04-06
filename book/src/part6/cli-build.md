# `batuta build`

Build the transpiled Rust project into a final binary (Phase 5: Deployment).

## Synopsis

```bash
batuta build [OPTIONS]
```

## Description

The build command compiles the transpiled Rust project using `cargo build`. It loads project configuration from `batuta.toml` to locate the transpiled output directory and any extra cargo flags.

This is Phase 5 of the 5-phase transpilation pipeline. It requires Phase 4 (Validation) to be completed first.

## Options

| Option | Description |
|--------|-------------|
| `--release` | Build in release mode (optimized) |
| `--target <TARGET>` | Cross-compile for a specific target platform |
| `--wasm` | Build for WebAssembly (`wasm32-unknown-unknown`) |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Configuration

The build command reads settings from `batuta.toml`:

```toml
[transpilation]
output_dir = "./rust-output"  # Where to find the transpiled project

[build]
cargo_flags = ["--locked"]    # Extra flags passed to cargo build
```

## What It Does

1. Loads `batuta.toml` to find `transpilation.output_dir`
2. Verifies `Cargo.toml` exists in the output directory
3. Builds cargo arguments: `cargo build [--release] [--target <T>] [extra_flags...]`
4. Executes `cargo build` with inherited stdio (output streams through)
5. Updates workflow state on success/failure

## Examples

### Debug Build

```bash
$ batuta build

🔨 Building Rust project...

Build Settings:
  • Build mode: debug
  • WebAssembly: disabled
  • Project: ./rust-output

Running: cargo build
   Compiling my-project v0.1.0 (/path/to/rust-output)
    Finished `dev` profile

✅ Build completed successfully!
```

### Release Build

```bash
$ batuta build --release
```

### WebAssembly Build

```bash
$ batuta build --wasm --release
```

### Cross-Compilation

```bash
$ batuta build --release --target aarch64-unknown-linux-gnu
```

## See Also

- [Phase 5: Deployment](../part2/phase5-deployment.md)
- [Cross-compilation](../part2/cross-compilation.md)
- [WebAssembly](../part2/wasm.md)

---

**Previous:** [`batuta validate`](./cli-validate.md)
**Next:** [`batuta report`](./cli-report.md)
