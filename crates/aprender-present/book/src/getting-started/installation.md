# Installation

## Prerequisites

- **Rust 1.75+** with the WASM target
- **wasm-bindgen-cli** for WASM bindings
- **wasm-opt** for production optimization (optional)

## Install Rust WASM Target

```bash
rustup target add wasm32-unknown-unknown
```

## Install Development Tools

```bash
# WASM bindings generator
cargo install wasm-bindgen-cli

# Production optimizer (optional)
cargo install wasm-opt

# File watcher for hot reload (optional)
cargo install cargo-watch
```

## Add Presentar to Your Project

```toml
# Cargo.toml
[dependencies]
presentar = "0.1"
presentar-core = "0.1"
presentar-widgets = "0.1"
presentar-yaml = "0.1"

[dev-dependencies]
presentar-test = "0.1"
```

## Verify Installation

```bash
# Create a new project
cargo new my-presentar-app
cd my-presentar-app

# Add dependencies and build
cargo build --target wasm32-unknown-unknown
```

## IDE Setup

### VS Code

Install the following extensions:
- **rust-analyzer** - Rust language support
- **YAML** - YAML syntax highlighting
- **WebGL GLSL Editor** - WGSL shader support

### IntelliJ/CLion

- Install the Rust plugin
- Enable WASM target in build configuration

## Next Steps

Continue to [Quick Start](./quick-start.md) to build your first Presentar app.
