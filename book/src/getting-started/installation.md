# Installation

## From crates.io

Add trueno-zram-core to your `Cargo.toml`:

```toml
[dependencies]
trueno-zram-core = "0.1"
```

Or use cargo add:

```bash
cargo add trueno-zram-core
```

## With CUDA Support

For GPU acceleration, enable the `cuda` feature:

```toml
[dependencies]
trueno-zram-core = { version = "0.1", features = ["cuda"] }
```

Or:

```bash
cargo add trueno-zram-core --features cuda
```

### CUDA Requirements

- CUDA Toolkit 12.8 or later
- NVIDIA driver supporting CUDA 12.8
- GPU with compute capability >= 7.0 (Volta or newer)

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library support | Yes |
| `nightly` | Nightly-only SIMD features | No |
| `cuda` | CUDA GPU acceleration | No |

## System Requirements

- **OS**: Linux (kernel >= 5.10 LTS)
- **CPU**: x86_64 (AVX2/AVX-512) or AArch64 (NEON)
- **Rust**: 1.82.0 or later (MSRV)

## Verifying Installation

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()?;

    println!("trueno-zram installed successfully!");
    println!("SIMD backend: {:?}", compressor.backend());

    Ok(())
}
```

## Building from Source

```bash
git clone https://github.com/paiml/trueno-zram
cd trueno-zram

# Build all crates
cargo build --release --all-features

# Run tests
cargo test --workspace --all-features

# Build with CUDA
cargo build --release --features cuda
```
