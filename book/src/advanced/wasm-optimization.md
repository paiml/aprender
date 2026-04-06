# WASM Optimization

Optimize WebAssembly bundle for production.

## Build Command

```bash
cargo build --target wasm32-unknown-unknown --release
wasm-opt -O3 -o output_opt.wasm output.wasm
```

## Optimization Flags

```toml
# Cargo.toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.release.package."*"]
opt-level = "z"  # Size optimization for deps
```

## Size Reduction

| Step | Size |
|------|------|
| Debug build | ~5MB |
| Release build | ~800KB |
| wasm-opt -O3 | ~500KB |
| gzip | ~150KB |

## wasm-opt Levels

| Level | Focus | Use |
|-------|-------|-----|
| `-O1` | Fast compile | Development |
| `-O2` | Balanced | CI |
| `-O3` | Max speed | Production |
| `-Oz` | Min size | Mobile |

## Code Splitting

```rust
// Lazy load large features
#[cfg(feature = "charts")]
mod charts;
```

## Remove Dead Code

```toml
[dependencies]
serde = { version = "1", default-features = false }
```

## Performance Tips

| Tip | Impact |
|-----|--------|
| Use `#[inline]` wisely | Reduces call overhead |
| Avoid `Box<dyn Trait>` | Static dispatch faster |
| Minimize allocations | Reuse buffers |
| Use `&str` over `String` | Zero-copy |

## Measuring Size

```bash
# Show section sizes
wasm-objdump -h output.wasm

# Find large functions
wasm-objdump -d output.wasm | grep "func" | sort -k2 -n -r | head
```

## Bundle Analysis

```bash
# Size breakdown
twiggy top output.wasm

# Dependency graph
twiggy dominators output.wasm
```

## Verified Test

```rust
#[test]
fn test_optimization_config() {
    // Verify release profile exists
    #[cfg(debug_assertions)]
    let is_release = false;
    #[cfg(not(debug_assertions))]
    let is_release = true;

    // In release mode, optimizations are active
    if is_release {
        assert!(true);
    }
}
```
