# Bundle Size

Keep WASM bundles small for fast loading.

## Target

**<500KB** gzipped for production

## Size Breakdown

| Component | Size |
|-----------|------|
| Core framework | ~80KB |
| Basic widgets | ~50KB |
| Layout engine | ~30KB |
| YAML parser | ~40KB |
| **Total** | ~200KB |

## Optimization Steps

### 1. Release Build

```bash
cargo build --target wasm32-unknown-unknown --release
```

### 2. LTO (Link-Time Optimization)

```toml
[profile.release]
lto = true
codegen-units = 1
```

### 3. wasm-opt

```bash
wasm-opt -O3 -o optimized.wasm output.wasm
```

### 4. Compression

```bash
gzip -9 optimized.wasm
# Or brotli for better compression
brotli -9 optimized.wasm
```

## Measuring

```bash
# Raw size
ls -lh output.wasm

# Compressed size
gzip -c output.wasm | wc -c
```

## Reducing Size

| Technique | Savings |
|-----------|---------|
| Remove unused features | 10-30% |
| Strip debug info | 20-40% |
| wasm-opt | 20-30% |
| gzip | 60-70% |

## Feature Flags

```toml
[features]
default = ["basic"]
basic = []
charts = ["dep:chart-lib"]
full = ["basic", "charts"]
```

## Verified Test

```rust
#[test]
fn test_bundle_size_concerns() {
    // Verify we're thinking about size
    use std::mem::size_of;

    // Small types = small bundle
    assert!(size_of::<presentar_core::Size>() <= 8);
    assert!(size_of::<presentar_core::Point>() <= 8);
}
```
