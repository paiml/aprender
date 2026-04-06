# Reproducibility Guide

This document describes simular's reproducibility guarantees and how to achieve bit-identical results.

## Random Number Generation

### Seed Configuration

All random operations use deterministic seeding:

```rust
use simular::prelude::SimRng;

// Create deterministic RNG
let rng = SimRng::new(42);  // seed = 42

// Same seed = same sequence
assert_eq!(SimRng::new(42).next_u64(), SimRng::new(42).next_u64());
```

### Default Seeds

| Component | Default Seed | Description |
|-----------|--------------|-------------|
| TSP GRASP | 42 | Tour construction randomness |
| Monte Carlo | 42 | Point sampling |
| Orbit perturbation | 12345 | Initial condition noise |

### Per-Thread Partitioning

For parallel execution, seeds are partitioned:
- Thread 0: seed + 0
- Thread 1: seed + 1
- Thread N: seed + N

This ensures deterministic parallel execution.

## Build Reproducibility

### Rust Toolchain

Pin exact Rust version in `rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.83.0"
```

### Cargo Lock

Always commit `Cargo.lock` for exact dependency versions.

### Docker

Use versioned base images:

```dockerfile
ARG RUST_VERSION=1.83.0
FROM rust:${RUST_VERSION}-slim-bookworm
```

### Nix

For hermetic builds:

```bash
nix develop
cargo build --release
```

## Platform Consistency

### IEEE 754 Strict Mode

Enable strict floating-point:

```rust
// In simulation code
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
// Avoid fused multiply-add for reproducibility
```

### WASM Considerations

WASM uses IEEE 754 strictly, ensuring cross-platform consistency.

## Verification

### Hash Verification

```bash
# Generate state hash
cargo run --example verify_hash -- --seed 42

# Compare across platforms
echo "Expected: 0x1234...abcd"
```

### CI Verification

GitHub Actions runs on Linux, macOS, and Windows to verify cross-platform reproducibility.

## Troubleshooting

### Non-Deterministic Results

1. Check seed is set explicitly
2. Verify Cargo.lock is committed
3. Check for floating-point optimizations
4. Verify same Rust version

### Performance vs Reproducibility

If performance is prioritized over exact reproducibility:

```rust
// Allow FMA (faster but less portable)
#[cfg(feature = "fast-math")]
let result = a.mul_add(b, c);
```
