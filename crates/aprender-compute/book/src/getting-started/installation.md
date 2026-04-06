# Installation

This guide covers installing Trueno and its dependencies.

## Prerequisites

### Rust Toolchain

Trueno requires Rust 1.70 or later. Install via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

Verify installation:

```bash
rustc --version  # Should be >= 1.70.0
cargo --version
```

### Platform-Specific Requirements

#### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential pkg-config

# Fedora/RHEL
sudo dnf install gcc pkg-config
```

#### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install
```

#### Windows

Install [Visual Studio 2022](https://visualstudio.microsoft.com/) with:
- Desktop development with C++
- Windows 10/11 SDK

### Optional: GPU Support

For GPU acceleration, install graphics drivers:

**NVIDIA (CUDA/Vulkan)**:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-driver-535 vulkan-tools

# Verify
vulkaninfo
```

**AMD (Vulkan)**:
```bash
# Ubuntu/Debian
sudo apt-get install mesa-vulkan-drivers vulkan-tools

# Verify
vulkaninfo
```

**Intel (Vulkan)**:
```bash
# Ubuntu/Debian
sudo apt-get install intel-media-va-driver vulkan-tools
```

**macOS (Metal)**:
Metal support is built-in on macOS 10.13+. No additional installation required.

## Installing Trueno

### From crates.io (Recommended)

Add Trueno to your `Cargo.toml`:

```toml
[dependencies]
trueno = "0.14"
```

Or use `cargo add`:

```bash
cargo add trueno
```

### From GitHub (Development)

For the latest development version:

```toml
[dependencies]
trueno = { git = "https://github.com/paiml/trueno", branch = "main" }
```

### With Specific Features

Trueno supports feature flags for selective compilation:

```toml
[dependencies]
# Default: SIMD backends only (no GPU)
trueno = "0.14"

# Enable GPU support
trueno = { version = "0.14", features = ["gpu"] }

# Enable all features
trueno = { version = "0.14", features = ["gpu", "parallel"] }

# Minimal (scalar only, for testing)
trueno = { version = "0.14", default-features = false }
```

**Available features:**
- `gpu` - Enable GPU backend via wgpu (adds ~5MB to binary)
- `wasm` - Enable WebAssembly SIMD128 support
- `f16` - Enable half-precision (f16) support (requires nightly)

## Verifying Installation

Create a test project:

```bash
cargo new trueno-test
cd trueno-test
```

Add Trueno to `Cargo.toml`:

```toml
[dependencies]
trueno = "0.14"
```

Replace `src/main.rs` with:

```rust
use trueno::Vector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create two vectors
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

    // Add them (uses best available SIMD backend)
    let result = a.add(&b)?;

    println!("Result: {:?}", result.as_slice());
    // Output: [6.0, 8.0, 10.0, 12.0]

    // Check which backend was used
    println!("Backend: {:?}", a.backend());

    Ok(())
}
```

Run the test:

```bash
cargo run --release
```

Expected output:

```
Result: [6.0, 8.0, 10.0, 12.0]
Backend: Avx2  # (or Sse2, Neon, etc. depending on your CPU)
```

## Development Installation

For contributing to Trueno or running tests:

```bash
# Clone repository
git clone https://github.com/paiml/trueno.git
cd trueno

# Build with all features
cargo build --all-features --release

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench

# Generate coverage report
cargo llvm-cov --all-features --workspace
```

### Development Dependencies

Install additional tools for development:

```bash
# Code coverage
cargo install cargo-llvm-cov

# Mutation testing
cargo install cargo-mutants

# Benchmarking (included in Cargo.toml dev-dependencies)
# criterion is automatically available

# Formatting and linting (included with rustup)
rustup component add rustfmt clippy
```

## Platform-Specific Notes

### x86_64 (Intel/AMD)

Trueno automatically detects and uses the best available SIMD instruction set:

- **SSE2**: Baseline (guaranteed on all x86_64)
- **AVX**: Sandy Bridge+ (2011+)
- **AVX2**: Haswell+ (2013+)
- **AVX-512**: Zen4, Sapphire Rapids+ (2022+)

Check your CPU features:

```bash
# Linux
cat /proc/cpuinfo | grep flags

# macOS
sysctl -a | grep cpu.features

# Windows (PowerShell)
Get-WmiObject -Class Win32_Processor | Select-Object -Property Name, Features
```

### ARM64 (Apple Silicon, AWS Graviton)

Trueno uses NEON SIMD on ARM64:

- **Apple M1/M2/M3**: Full NEON support (128-bit)
- **AWS Graviton2/3**: Full NEON support
- **Raspberry Pi 4**: Limited NEON support

### WebAssembly

For WASM targets:

```bash
# Install wasm32 target
rustup target add wasm32-unknown-unknown

# Build for WASM
cargo build --target wasm32-unknown-unknown --release

# Enable SIMD128 (requires nightly for now)
rustup toolchain install nightly
cargo +nightly build --target wasm32-unknown-unknown \
    -Z build-std=std,panic_abort \
    --release
```

## Troubleshooting

### "No suitable backend found" error

If you see this error, Trueno couldn't detect any SIMD support. Possible causes:

1. **Running on ancient CPU** (pre-2011 x86_64):
   - Solution: Use `Backend::Scalar` explicitly

2. **Cross-compiling** without proper target configuration:
   - Solution: Set `RUSTFLAGS` for target CPU:
     ```bash
     RUSTFLAGS="-C target-cpu=native" cargo build --release
     ```

3. **WASM without SIMD128**:
   - Solution: Enable SIMD in browser flags or use scalar fallback

### GPU not detected

If GPU is available but not being used:

1. **Check Vulkan/Metal installation**:
   ```bash
   # Linux/Windows
   vulkaninfo

   # macOS - Metal is built-in, check system version
   sw_vers  # Should be >= 10.13
   ```

2. **Verify GPU feature flag**:
   ```toml
   trueno = { version = "0.14", features = ["gpu"] }
   ```

3. **Check workload size** (GPU only used for 100K+ elements):
   ```rust
   let large = Vector::from_slice(&vec![1.0; 200_000]);
   println!("Backend: {:?}", large.backend());
   // Should show: Gpu
   ```

### Compilation errors

**Error: `feature 'avx512' requires nightly`**
- Trueno uses stable Rust. This error indicates you're on an old rustc version.
- Solution: `rustup update stable`

**Error: `wgpu` fails to compile**
- This is usually a missing system dependency.
- Solution (Ubuntu): `sudo apt-get install libvulkan-dev`

**Error: Link errors on Windows**
- Solution: Install Visual Studio 2022 with C++ build tools

## Next Steps

Now that Trueno is installed:

- [Quick Start](./quick-start.md) - Run your first program
- [Core Concepts](./core-concepts.md) - Understand key abstractions
- [API Reference](../api-reference/vector-operations.md) - Explore available operations
