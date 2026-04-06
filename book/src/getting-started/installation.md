# Installation

This guide will help you install Entrenar and set up your development environment for neural network training with
autograd, optimizers, and LoRA/QLoRA fine-tuning.

## Prerequisites

Before installing Entrenar, ensure you have:

- **Rust 1.70+**: Install from [rustup.rs](https://rustup.rs)
- **Cargo**: Comes bundled with Rust
- **Git**: For cloning the repository (optional)

```bash
# Verify Rust installation
rustc --version  # Should show 1.70 or higher
cargo --version
```

## Installation Methods

### Method 1: Add as Cargo Dependency (Recommended)

Add Entrenar to your `Cargo.toml`:

```toml
[dependencies]
entrenar = "0.2"
ndarray = "0.16"  # Required for tensor operations
```

Then run:

```bash
cargo build
```

### Method 2: Clone and Build from Source

For development or to run examples:

```bash
# Clone the repository
git clone https://github.com/paiml/entrenar.git
cd entrenar

# Run tests to verify installation
cargo test

# Run quality gates
cargo clippy -- -D warnings
cargo fmt --check

# Build in release mode for performance
cargo build --release
```

## Verifying Installation

Create a simple test file `test_install.rs`:

```rust
use entrenar::Tensor;

fn main() {
    // Create a simple tensor
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    println!("Tensor created: {:?}", x.data());

    // Test autograd
    let y = &x * &x;  // y = x²
    println!("Forward pass successful!");

    println!("✅ Entrenar is installed correctly!");
}
```

Run it:

```bash
cargo run --example test_install
```

Expected output:
```
Tensor created: [1.0, 2.0, 3.0]
Forward pass successful!
✅ Entrenar is installed correctly!
```

## Feature Flags

Entrenar supports optional features via Cargo feature flags:

```toml
[dependencies]
entrenar = { version = "0.2", features = ["simd", "quantization"] }
```

Available features:

| Feature | Description | Default |
|---------|-------------|---------|
| `simd` | SIMD-accelerated optimizer updates via Trueno | ✅ Enabled |
| `quantization` | 4-bit quantization for QLoRA | ✅ Enabled |
| `serde` | Serialization support for adapters | ✅ Enabled |

## Development Dependencies

For contributing or running the full test suite:

```toml
[dev-dependencies]
proptest = "1.0"         # Property-based testing
approx = "0.5"           # Floating-point comparisons
serde_json = "1.0"       # JSON serialization
criterion = "0.5"        # Benchmarking
cargo-mutants = "24.0"   # Mutation testing
```

Install development tools:

```bash
# Code coverage
cargo install cargo-llvm-cov

# Mutation testing
cargo install cargo-mutants

# Benchmarking
cargo install cargo-criterion
```

## Platform-Specific Notes

### Linux

No special configuration required. SIMD acceleration works out of the box on x86_64 and ARM64.

### macOS

Apple Silicon (M1/M2) users get native ARM64 SIMD support:

```bash
# Verify ARM64 build
cargo build --release
file target/release/entrenar
# Should show: Mach-O 64-bit executable arm64
```

### Windows

Windows users should use the MSVC toolchain:

```bash
rustup default stable-msvc
cargo build
```

## IDE Setup

### Visual Studio Code

Recommended extensions:

- **rust-analyzer**: IntelliSense and code completion
- **CodeLLDB**: Debugging support
- **Even Better TOML**: Cargo.toml syntax highlighting

### RustRover / IntelliJ IDEA

The Rust plugin provides excellent support for Entrenar development.

## Troubleshooting

### Error: "cannot find crate `ndarray`"

**Solution**: Add `ndarray = "0.15"` to your `Cargo.toml` dependencies.

### Error: "SIMD operations not available"

**Solution**: Ensure you're compiling in release mode for SIMD optimizations:

```bash
cargo build --release
```

### Tests Failing on Fresh Install

**Solution**: Run with increased stack size for gradient checking tests:

```bash
RUST_MIN_STACK=8388608 cargo test
```

### Slow Compile Times

**Solution**: Enable parallel compilation:

```bash
# Add to ~/.cargo/config.toml
[build]
jobs = 4  # Or number of CPU cores
```

## Next Steps

Now that Entrenar is installed:

1. **[Quick Start](./quick-start.md)** - Train your first neural network
2. **[First Training Loop](./first-training-loop.md)** - Build a complete training pipeline
3. **[Core Concepts](./core-concepts.md)** - Understand Entrenar's architecture

## Getting Help

- **Documentation**: [https://docs.rs/entrenar](https://docs.rs/entrenar)
- **Issues**: [GitHub Issues](https://github.com/paiml/entrenar/issues)
- **Examples**: See `examples/` directory in the repository
- **Tests**: See `src/*/tests.rs` for usage patterns

---

**Ready to train?** Continue to [Quick Start](./quick-start.md) →
