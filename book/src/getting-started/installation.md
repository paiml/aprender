# Installation

## Install the CLI

```bash
cargo install aprender
```

This installs the `apr` binary — a single command for inference, training, serving,
model operations, and profiling.

```bash
apr --version
# apr 0.4.17
```

## Verify

```bash
# List available commands
apr --help

# Download a model
apr pull qwen2.5-coder-1.5b

# Run inference
apr run qwen2.5-coder-1.5b "What is 2+2?"
```

## Library Usage

For the ML library (algorithms, data structures, format I/O):

```toml
[dependencies]
aprender-core = "0.29"
```

```rust
use aprender::linear_regression::LinearRegression;
use aprender::traits::Estimator;
```

## From Source

```bash
git clone https://github.com/paiml/aprender
cd aprender
cargo install --path .
```

## Requirements

- Rust 1.89+ (stable)
- Linux, macOS, or Windows
- GPU optional (CUDA for NVIDIA, wgpu for AMD/Intel/Apple)
