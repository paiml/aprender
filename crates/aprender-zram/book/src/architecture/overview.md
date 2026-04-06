# Design Overview

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Public API                           │
│  CompressorBuilder, Algorithm, CompressedPage           │
├─────────────────────────────────────────────────────────┤
│                 Algorithm Selection                      │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐   │
│  │   LZ4   │  │  ZSTD   │  │ Adaptive │  │Samefill │   │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬────┘   │
├───────┼────────────┼────────────┼─────────────┼────────┤
│       │     SIMD Dispatch       │             │         │
│  ┌────▼────┐  ┌────▼────┐  ┌───▼───┐  ┌─────▼─────┐   │
│  │ AVX-512 │  │  AVX2   │  │ NEON  │  │  Scalar   │   │
│  └─────────┘  └─────────┘  └───────┘  └───────────┘   │
├─────────────────────────────────────────────────────────┤
│                   GPU Backend (Optional)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  CUDA Batch Compressor (trueno-gpu PTX)         │   │
│  │  ├── H2D Transfer                               │   │
│  │  ├── Warp-Cooperative LZ4 Kernel                │   │
│  │  └── D2H Transfer                               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Crate Structure

```
trueno-zram/
├── crates/
│   ├── trueno-zram-core/     # Core compression library
│   │   ├── src/
│   │   │   ├── lib.rs        # Public API
│   │   │   ├── error.rs      # Error types
│   │   │   ├── page.rs       # CompressedPage
│   │   │   ├── lz4/          # LZ4 implementation
│   │   │   ├── zstd/         # ZSTD implementation
│   │   │   ├── gpu/          # GPU batch compression
│   │   │   ├── simd/         # SIMD detection/dispatch
│   │   │   ├── samefill.rs   # Same-fill detection
│   │   │   ├── compat.rs     # Kernel compatibility
│   │   │   └── benchmark.rs  # Benchmarking utilities
│   │   └── examples/
│   ├── trueno-zram-adaptive/ # ML-driven selection
│   ├── trueno-zram-generator/# systemd integration
│   └── trueno-zram-cli/      # CLI tool
└── bins/
    └── trueno-ublk/          # ublk daemon
```

## Key Design Decisions

### 1. Runtime SIMD Dispatch

CPU features are detected at runtime, not compile time:

```rust
// Detection happens once at startup
let features = simd::detect();

// Dispatch based on available features
if features.has_avx512() {
    lz4::avx512::compress(input, output)
} else if features.has_avx2() {
    lz4::avx2::compress(input, output)
} else {
    lz4::scalar::compress(input, output)
}
```

### 2. Page-Based Compression

All compression operates on fixed 4KB pages:

```rust
pub const PAGE_SIZE: usize = 4096;

// This is enforced at the type level
pub fn compress(page: &[u8; PAGE_SIZE]) -> Result<CompressedPage>;
```

### 3. Builder Pattern

Configuration via builder pattern:

```rust
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .prefer_backend(SimdBackend::Avx512)
    .build()?;
```

### 4. Trait-Based Abstraction

The `PageCompressor` trait enables polymorphism:

```rust
pub trait PageCompressor {
    fn compress(&self, page: &[u8; PAGE_SIZE]) -> Result<CompressedPage>;
    fn decompress(&self, page: &CompressedPage) -> Result<[u8; PAGE_SIZE]>;
}
```

### 5. Zero-Copy Where Possible

Minimize allocations in hot paths:

```rust
// Output buffer passed in, not allocated
fn compress_into(input: &[u8], output: &mut [u8]) -> Result<usize>;
```

### 6. No Panics in Library Code

All errors are returned as `Result`:

```rust
#![deny(clippy::panic)]
#![deny(clippy::unwrap_used)]
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `thiserror` | Error derive macros |
| `cudarc` | CUDA driver bindings |
| `rayon` | Parallel iteration |
| `trueno-gpu` | Pure Rust PTX generation |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | Yes | Standard library |
| `nightly` | No | Nightly SIMD features |
| `cuda` | No | CUDA GPU support |
