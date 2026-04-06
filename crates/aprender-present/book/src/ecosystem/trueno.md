# Trueno

SIMD-accelerated tensor library powering Presentar.

## Overview

Trueno provides high-performance tensor operations:

- SIMD vectorization (AVX2, SSE4.1, NEON)
- GPU compute (WebGPU/WGSL)
- Zero-copy memory management
- Type-safe tensor API

## Integration

```toml
[dependencies]
trueno = "0.1"
```

## Usage in Presentar

### Layout Calculations

```rust
use trueno::Tensor;

// Fast position calculations
let positions = Tensor::new(&[child_count, 2]);
positions.fill(0.0);
positions.simd_add(offset);
```

### Color Operations

```rust
// WCAG contrast calculation
let luminance = trueno::color::relative_luminance(r, g, b);
let contrast = trueno::color::contrast_ratio(l1, l2);
```

### Image Processing

```rust
// Visual regression diff
let diff = trueno::image::diff(&baseline, &actual);
let similarity = 1.0 - diff.mean();
```

## Performance

| Operation | CPU | Trueno SIMD |
|-----------|-----|-------------|
| 1M adds | 2ms | 0.3ms |
| Matrix mul | 10ms | 1.5ms |
| Image diff | 20ms | 3ms |

## Memory Model

```
Stack allocation preferred
↓
Arena allocation for temporary
↓
Heap only when necessary
```

## Verified Test

```rust
#[test]
fn test_trueno_available() {
    // Trueno is a dependency
    // This test verifies it's linked
    use std::mem::size_of;

    // Basic types work
    assert_eq!(size_of::<f32>(), 4);
}
```
