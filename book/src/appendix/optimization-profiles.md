# Appendix D: Optimization Profiles

Cargo profiles control compilation settings that affect binary size, speed, and debug experience.

## Profile Summary

| Profile | Use Case | Binary Size | Speed | Debug Info |
|---------|----------|-------------|-------|------------|
| `dev` | Development, testing | Large | Moderate | Full |
| `release` | Production deployment | Small | Maximum | Minimal |
| `release-wasm` | Browser deployment | Smallest | Maximum | None |
| `bench` | Benchmarking | Small | Maximum | Line tables |

## Profile Configuration

### dev (Default)

```toml
[profile.dev]
opt-level = 0
debug = true
overflow-checks = true
incremental = true
```

### release

```toml
[profile.release]
opt-level = 3
debug = true          # Debug info for profiling, stripped at deploy
lto = "thin"          # Link-Time Optimization (cross-crate inlining)
codegen-units = 1     # Single codegen unit for maximum optimization
strip = "none"        # Keep symbols for flamegraph; strip at deploy
panic = "abort"       # Smaller binary, no unwinding overhead
```

### release-wasm

```toml
[profile.release-wasm]
inherits = "release"
opt-level = "z"       # Optimize for size (critical for WASM download)
lto = "fat"           # Maximum cross-crate optimization
strip = "symbols"     # Remove all symbols
codegen-units = 1
```

## LTO Options

| LTO Setting | Compile Time | Runtime Speed | Binary Size |
|-------------|-------------|---------------|-------------|
| `false` | Fastest | Baseline | Largest |
| `"thin"` | +20-40% | +5-15% | -10-20% |
| `"fat"` | +100-200% | +10-20% | -15-25% |

Thin LTO is the best tradeoff for most use cases. Fat LTO is worth it only for WASM where binary size is critical.

## Size vs Speed Tradeoffs

| Goal | `opt-level` | `lto` | `strip` | `codegen-units` |
|------|-------------|-------|---------|-----------------|
| Maximum speed | `3` | `"thin"` | `"none"` | `1` |
| Minimum size | `"z"` | `"fat"` | `"symbols"` | `1` |
| Fast compile | `0` | `false` | `"none"` | `16` |

## Target-Specific Flags

Enable CPU-specific instructions via `.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=x86-64-v3"]  # AVX2 baseline

[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]  # WASM SIMD
```

| Target | ISA Extensions | Performance Impact |
|--------|---------------|-------------------|
| `x86-64` (default) | SSE2 | Baseline |
| `x86-64-v3` | AVX2, FMA | 2-4x for vectorizable code |
| `native` | All available (e.g., AVX-512) | 4-16x for SIMD-heavy code |
| `wasm32+simd128` | WASM SIMD | 2-4x in browser |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
