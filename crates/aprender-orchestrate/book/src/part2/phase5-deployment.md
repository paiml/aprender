# Phase 5: Deployment

Phase 5 builds the transpiled Rust project into a final binary, with support for release optimization, cross-compilation, and WebAssembly targets.

## Overview

Deployment is the final phase of the transpilation pipeline. It compiles the validated Rust code into a distributable binary:

```
Validated .rs project
       │
       ▼
┌──────────────────────────┐
│  cargo build             │
│  --release               │ ← Optional: release mode
│  --target <triple>       │ ← Optional: cross-compile
│  --target wasm32-unknown │ ← Optional: WebAssembly
│  [extra cargo_flags]     │ ← From batuta.toml
└────────────┬─────────────┘
             │
             ▼
    Final Binary / .wasm
```

## Build Modes

### Debug Build

Default mode for quick iteration:

```bash
batuta build
```

### Release Build

Optimized binary with the profile settings from Phase 3:

```bash
batuta build --release
```

### WebAssembly

Builds for `wasm32-unknown-unknown` target:

```bash
batuta build --wasm --release
```

### Cross-Compilation

Target a specific platform:

```bash
batuta build --release --target aarch64-unknown-linux-gnu
batuta build --release --target x86_64-apple-darwin
```

## Configuration

Build settings are read from `batuta.toml`:

```toml
[transpilation]
output_dir = "./rust-output"    # Compiled project location

[build]
cargo_flags = ["--locked"]      # Extra flags for cargo build
```

The build command:
1. Reads `transpilation.output_dir` to locate the project
2. Verifies `Cargo.toml` exists
3. Appends `build.cargo_flags` to the cargo command
4. Runs `cargo build` with inherited stdio

## Jidoka Integration

Build failures (non-zero cargo exit code) mark the Deployment phase as failed in the workflow state. The exit code is captured and reported. Success marks the full 5-phase migration as complete.

## Beyond `batuta build`

For production deployment of ML models (not transpiled code), Batuta also provides:

- **`batuta serve`** — Serve models via Realizar with OpenAI-compatible API
- **`batuta deploy`** — Generate Docker, Lambda, K8s, Fly.io, or Cloudflare deployments
- **`batuta pacha`** — Model registry with versioning and Ed25519 signatures

## CLI Reference

See [`batuta build`](../part6/cli-build.md) for full command documentation.

---

**Previous:** [Phase 4: Validation](./phase4-validation.md)
**Next:** [Part III: The Tool Ecosystem](../part3/tool-overview.md)
