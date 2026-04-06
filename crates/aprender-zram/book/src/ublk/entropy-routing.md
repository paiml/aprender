# Entropy Routing

trueno-ublk uses Shannon entropy analysis to route data to the optimal compression backend.

## Shannon Entropy

Shannon entropy measures the information density of data in bits per byte. The formula is:

```
H(X) = -Σ p(x) * log2(p(x))
```

Where p(x) is the probability of each byte value occurring.

| Entropy (bits/byte) | Data Type | Compressibility |
|---------------------|-----------|-----------------|
| 0.0 | All same value | Extremely high |
| 1.0-3.0 | Simple patterns | Very high |
| 4.0-6.0 | Text, code | Good |
| 7.0-7.5 | Compressed data | Poor |
| 8.0 | Random/encrypted | None |

## Routing Strategy

trueno-ublk routes pages based on their entropy:

```
┌──────────────────────────────────────────────────────────┐
│                    Incoming Page                          │
│                         │                                 │
│                    Calculate Entropy                      │
│                         │                                 │
│           ┌─────────────┼─────────────┐                  │
│           ▼             ▼             ▼                  │
│      entropy < 4.0  4.0 ≤ e ≤ 7.0  entropy > 7.0        │
│           │             │             │                  │
│           ▼             ▼             ▼                  │
│      GPU Batch      SIMD Path    Scalar/Skip            │
│    (highly comp.)  (normal data)  (incompress.)         │
└──────────────────────────────────────────────────────────┘
```

### GPU Batch Path (entropy < 4.0)
- Used for highly compressible data
- Benefits from GPU parallelism
- Batches multiple pages for efficiency
- Best for: zeros, repeating patterns, sparse data

### SIMD Path (4.0 ≤ entropy ≤ 7.0)
- Used for typical data
- AVX2/AVX-512/NEON acceleration
- Single-page processing
- Best for: text, code, structured data

### Scalar/Skip Path (entropy > 7.0)
- Used for incompressible data
- Avoids wasting CPU cycles
- May store uncompressed
- Best for: encrypted data, already-compressed media

## Configuration

Set the entropy threshold when creating a device:

```rust
use trueno_ublk::BlockDevice;
use trueno_zram_core::{Algorithm, CompressorBuilder};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// Lower threshold = more aggressive SIMD usage
let device = BlockDevice::with_entropy_threshold(
    1 << 30,    // Size
    compressor,
    6.5,        // Custom threshold (default is 7.0)
);
```

## Monitoring Entropy

Use the CLI to analyze device entropy:

```bash
# Show entropy distribution
trueno-ublk entropy /dev/ublkb0

# Output:
# Entropy Distribution:
#   0.0-2.0 bits: ████████████████████ 45% (GPU)
#   2.0-4.0 bits: ████████ 18% (GPU)
#   4.0-6.0 bits: ██████████ 22% (SIMD)
#   6.0-7.0 bits: ████ 10% (SIMD)
#   7.0-8.0 bits: ██ 5% (Scalar)
#
# Average entropy: 3.2 bits/byte
# Recommended threshold: 7.0
```

## Statistics

Track routing decisions via stats:

```rust
let stats = device.stats();

println!("Routing Statistics:");
println!("  GPU pages:    {} ({:.1}%)",
    stats.gpu_pages,
    100.0 * stats.gpu_pages as f64 / stats.pages_stored as f64);
println!("  SIMD pages:   {} ({:.1}%)",
    stats.simd_pages,
    100.0 * stats.simd_pages as f64 / stats.pages_stored as f64);
println!("  Scalar pages: {} ({:.1}%)",
    stats.scalar_pages,
    100.0 * stats.scalar_pages as f64 / stats.pages_stored as f64);
println!("  Zero pages:   {} ({:.1}%)",
    stats.zero_pages,
    100.0 * stats.zero_pages as f64 / stats.pages_stored as f64);
```

## Zero-Page Optimization

All-zero pages receive special handling regardless of entropy:

```rust
// Zero pages are detected and deduplicated
let zeros = vec![0u8; PAGE_SIZE];
device.write(0, &zeros)?;
device.write(PAGE_SIZE as u64, &zeros)?;
device.write(2 * PAGE_SIZE as u64, &zeros)?;

let stats = device.stats();
// All three pages share the same zero-page representation
assert_eq!(stats.zero_pages, 3);
```

This achieves effective 2048:1 compression for sparse data like freshly-allocated memory.
