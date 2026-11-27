<!-- PCU: getting-started-installation | contract: contracts/apr-page-getting-started-installation-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# Installation

## Install the CLI

Pre-built and ready to use (Linux x86_64/aarch64 only today — other platforms fall
through to `cargo install` below):

```bash
curl -LsSf https://paiml.com/apr/install.sh | sh
```

To read the script before it runs, download it with `curl -LsSf https://paiml.com/apr/install.sh -o install.sh`, review it, then run `sh install.sh`. The installer is POSIX `sh` and checks the downloaded release archive against its published `.sha256` before installing.

Compile and build:

```bash
cargo install aprender
```

This installs the `apr` binary — a single command for inference, training, serving,
model operations, and profiling.

```bash
apr --version
# apr 0.75.4
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
aprender = "0.75"
```

```rust
use aprender::linear_model::LinearRegression;
use aprender::prelude::{Matrix, Vector};
use aprender::traits::Estimator;

let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
let mut model = LinearRegression::new();
model.fit(&x, &y).expect("linear regression fit should succeed");
```

## From Source

```bash
git clone https://github.com/paiml/aprender
cd aprender
cargo install --path .
```

## Requirements

- Rust 1.91+ (stable)
- Linux, macOS, or Windows
- GPU optional (CUDA for NVIDIA, wgpu for AMD/Intel/Apple)
