# Installation

## Basic Installation

Add to `Cargo.toml`:

```toml
[dependencies]
trueno-viz = "0.1"
```

## Feature Flags

Trueno-viz provides granular feature flags:

```toml
[dependencies]
# Minimal installation (PNG output only)
trueno-viz = { version = "0.1", default-features = false }

# With SVG support
trueno-viz = { version = "0.1", features = ["svg"] }

# With terminal output
trueno-viz = { version = "0.1", features = ["terminal"] }

# With GPU acceleration
trueno-viz = { version = "0.1", features = ["gpu"] }

# With ML integration (aprender)
trueno-viz = { version = "0.1", features = ["ml"] }

# All features
trueno-viz = { version = "0.1", features = ["full"] }
```

### Feature Matrix

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | PNG + basic plots | trueno |
| `svg` | SVG output | None |
| `terminal` | ASCII/Unicode output | None |
| `gpu` | GPU compute | wgpu |
| `ml` | ML visualization | aprender |
| `graph` | Graph visualization | trueno-graph |
| `parallel` | Parallel rendering | rayon |
| `wasm` | WebAssembly target | wasm-bindgen |
| `full` | All features | All above |

## Verifying Installation

Create a test file to verify installation:

```rust
// tests/installation_test.rs
use trueno_viz::prelude::*;

#[test]
fn test_installation() {
    // Verify core types are available
    let color = Rgba::new(255, 0, 0, 255);
    assert_eq!(color.r, 255);

    // Verify framebuffer works
    let fb = Framebuffer::new(100, 100);
    assert_eq!(fb.width(), 100);
    assert_eq!(fb.height(), 100);

    // Verify point geometry
    let p = Point::new(1.0, 2.0);
    assert!((p.x - 1.0).abs() < f32::EPSILON);
}
```

Run with:

```bash
cargo test test_installation
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Full | AVX2/AVX512 SIMD |
| macOS x86_64 | Full | AVX2 SIMD |
| macOS ARM64 | Full | NEON SIMD |
| Windows x86_64 | Full | AVX2 SIMD |
| WebAssembly | Full | SIMD128 |
| Linux ARM64 | Full | NEON SIMD |

## Troubleshooting

### Missing SIMD Support

If you encounter SIMD-related errors:

```bash
# Check CPU features
cat /proc/cpuinfo | grep -E 'avx|sse|neon'

# Force scalar fallback
RUSTFLAGS="-C target-feature=-avx2" cargo build
```

### GPU Not Detected

For GPU features:

```bash
# List available GPU devices
cargo run --features gpu --example list_devices
```

## Next Steps

Continue to [Your First Plot](./first-plot.md) to create a visualization.
