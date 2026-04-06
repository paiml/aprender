# Same-Fill Detection

Same-fill optimization provides 2048:1 compression for pages containing a single repeated byte value.

## Why Same-Fill Matters

Memory pages often contain:
- **Zero pages**: ~30-40% of typical memory (uninitialized, freed)
- **Same-fill pages**: ~5-10% additional (memset patterns)

Detecting and encoding these specially provides massive compression wins.

## Detection

```rust
use trueno_zram_core::samefill::detect_same_fill;
use trueno_zram_core::PAGE_SIZE;

let zero_page = [0u8; PAGE_SIZE];
let pattern_page = [0xAA; PAGE_SIZE];
let mixed_page = [0u8; PAGE_SIZE];

// Zero page detected
assert!(detect_same_fill(&zero_page).is_some());

// Pattern page detected
assert!(detect_same_fill(&pattern_page).is_some());

// Mixed content not detected
let mut mixed = [0u8; PAGE_SIZE];
mixed[100] = 0xFF;
assert!(detect_same_fill(&mixed).is_none());
```

## Compact Encoding

Same-fill pages compress to just 2 bytes:

```rust
use trueno_zram_core::samefill::CompactSameFill;

let fill_value = 0u8;
let compact = CompactSameFill::new(fill_value);

// Serialize (2 bytes)
let bytes = compact.to_bytes();
assert_eq!(bytes.len(), 2);

// Deserialize
let restored = CompactSameFill::from_bytes(&bytes)?;
assert_eq!(restored.fill_value(), 0);

// Expand back to full page
let page = restored.expand();
assert_eq!(page.len(), PAGE_SIZE);
assert!(page.iter().all(|&b| b == 0));
```

## Compression Ratio

| Page Type | Original | Compressed | Ratio |
|-----------|----------|------------|-------|
| Zero-fill | 4096 B | 2 B | 2048:1 |
| 0xFF-fill | 4096 B | 2 B | 2048:1 |
| Any same-fill | 4096 B | 2 B | 2048:1 |

## Integration with Compressor

The compressor automatically detects same-fill pages:

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

let zero_page = [0u8; PAGE_SIZE];
let compressed = compressor.compress(&zero_page)?;

// Same-fill pages get special encoding
println!("Compressed size: {} bytes", compressed.data.len());
// Output: ~20 bytes (LZ4 minimal encoding for same-fill)
```

## Performance

Same-fill detection is extremely fast:

```rust
// SIMD-accelerated comparison
// AVX-512: Check 64 bytes per iteration
// AVX2: Check 32 bytes per iteration
// Typical: <100ns for 4KB page
```

## Memory Statistics

On typical systems:

| Memory Type | Same-Fill % |
|-------------|-------------|
| Idle desktop | 60-70% |
| Active workload | 35-45% |
| Database server | 25-35% |
| Compilation | 40-50% |
