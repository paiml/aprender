# ARM64 Graviton Optimization

Realizar is optimized for AWS Graviton processors, providing better price-performance for Lambda deployments.

## Why ARM64?

| Metric | x86_64 Lambda | ARM64 Lambda |
|--------|---------------|--------------|
| Price | $0.0000166667/GB-second | $0.0000133334/GB-second |
| Savings | Baseline | **20% cheaper** |
| Performance | Good | Better for inference |
| SIMD | AVX2 (256-bit) | NEON (128-bit, more units) |

## Graviton Generations

### Graviton2 (2020)
- 64 Neoverse N1 cores
- 64 NEON SIMD units
- Ideal for Lambda

### Graviton3 (2022)
- 64 Neoverse V1 cores
- 2x vector processing width
- BFloat16 support

## Detection at Runtime

```rust
pub fn is_arm64_optimized() -> bool {
    cfg!(target_arch = "aarch64")
}

pub fn has_neon_simd() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
```

## Building for ARM64

### Cross-Compilation

```bash
# Install cross-compilation toolchain
rustup target add aarch64-unknown-linux-gnu

# Install linker
sudo apt-get install gcc-aarch64-linux-gnu

# Build
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    cargo build --release --target aarch64-unknown-linux-gnu --features lambda
```

### Using cargo-lambda

```bash
# Install cargo-lambda
cargo install cargo-lambda

# Build for ARM64 Lambda
cargo lambda build --release --arm64 --features lambda

# Package
zip -j function.zip target/lambda/bootstrap
```

## Lambda Configuration

```yaml
# SAM template
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  InferenceFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: bootstrap
      Runtime: provided.al2023
      Architectures:
        - arm64  # Use Graviton
      MemorySize: 1024
      Timeout: 30
      CodeUri: target/lambda/
```

## NEON SIMD Operations

Trueno automatically uses NEON on ARM64:

```rust
// Vector addition - uses NEON on ARM64
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let c = a.add(&b);  // NEON-accelerated
```

## Performance Comparison

From Lambda benchmarks:

| Operation | x86_64 | ARM64 | Speedup |
|-----------|--------|-------|---------|
| Handler creation | ~110µs | ~100µs | 1.1x |
| Warm inference | ~45ns | ~35ns | 1.3x |
| Batch (100) | ~4.5µs | ~3.8µs | 1.2x |

Combined with 20% cost savings, ARM64 offers **~35% better price-performance**.

## Best Practices

1. **Always use ARM64** for new Lambda deployments
2. **Test on x86_64 first** for easier debugging
3. **Profile both architectures** for your workload
4. **Use cargo-lambda** for simplified builds
5. **Set memory appropriately** - more memory = more CPU
