# Installation

## Requirements

- Rust 1.75 or later
- Cargo (comes with Rust)

## Adding to Your Project

Add alimentar to your `Cargo.toml`:

```toml
[dependencies]
alimentar = "0.1"
```

### Feature Flags

Alimentar uses feature flags to control optional functionality:

```toml
[dependencies]
# Default features (local filesystem, tokio runtime)
alimentar = "0.1"

# With S3 support
alimentar = { version = "0.1", features = ["s3"] }

# With HuggingFace Hub integration
alimentar = { version = "0.1", features = ["hf-hub"] }

# For WASM targets
alimentar = { version = "0.1", default-features = false, features = ["wasm"] }

# All features
alimentar = { version = "0.1", features = ["s3", "hf-hub", "http"] }
```

### Available Features

| Feature | Description | Default |
|---------|-------------|---------|
| `local` | Local filesystem backend | Yes |
| `tokio-runtime` | Async runtime for non-WASM targets | Yes |
| `s3` | S3-compatible storage backend | No |
| `http` | HTTP data sources | No |
| `hf-hub` | HuggingFace Hub integration | No |
| `wasm` | WebAssembly target support | No |

## Verifying Installation

Create a simple test program:

```rust
use alimentar::{ArrowDataset, Dataset};

fn main() -> alimentar::Result<()> {
    // Create a simple dataset from a CSV string
    let csv_data = "name,age,score\nAlice,30,95.5\nBob,25,87.3\n";

    // For testing, we'll use the library's types
    println!("Alimentar {} installed successfully!", env!("CARGO_PKG_VERSION"));

    Ok(())
}
```

Run with:

```bash
cargo run
```

## CLI Installation

Alimentar includes a command-line tool. Install it globally:

```bash
cargo install alimentar
```

Verify the CLI:

```bash
alimentar --version
alimentar --help
```

## Building from Source

```bash
git clone https://github.com/paiml/alimentar.git
cd alimentar

# Run tests
cargo test --all-features

# Build release
cargo build --release

# Install CLI
cargo install --path .
```

## WASM Setup

For WebAssembly targets:

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for WASM
wasm-pack build --target web
```

See [Browser Integration](../serve/browser-integration.md) for detailed WASM usage.
