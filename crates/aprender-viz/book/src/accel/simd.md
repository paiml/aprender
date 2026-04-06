# SIMD Acceleration

Trueno-viz uses SIMD (Single Instruction, Multiple Data) to accelerate
rendering and data processing through the trueno core library and the
`monitor::simd` module.

## Automatic Dispatch

SIMD acceleration is automatic based on CPU capabilities:

```rust
use trueno_viz::accel;

// Query available SIMD support
let info = accel::cpu_info();
println!("SSE2: {}", info.has_sse2());
println!("AVX2: {}", info.has_avx2());
println!("AVX512: {}", info.has_avx512());
println!("NEON: {}", info.has_neon());  // ARM
```

## Supported Operations

| Operation | SIMD Benefit |
|-----------|--------------|
| Color blending | 4-8x faster |
| Pixel filling | 4-16x faster |
| Statistics (min/max/sum/mean) | 4-8x faster |
| Scale transforms | 4-8x faster |
| Batch normalization | 4-5x faster |
| Line clipping | 2-4x faster |

## Monitor SIMD Kernels

The `monitor` feature provides low-level SIMD kernels for TUI applications:

```rust
use trueno_viz::monitor::simd::kernels::{
    simd_sum, simd_mean, simd_min, simd_max,
    simd_statistics, simd_normalize,
};

let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

// Individual operations
let sum = simd_sum(&data);      // AVX2 horizontal reduction
let mean = simd_mean(&data);    // Vectorized mean
let min = simd_min(&data);      // Parallel comparison
let max = simd_max(&data);      // Parallel comparison

// Combined statistics (single pass)
let stats = simd_statistics(&data);
println!("Min: {}, Max: {}, Mean: {}", stats.min, stats.max, stats.mean());

// Batch normalization
let normalized = simd_normalize(&data, 999.0);
```

### SimdRingBuffer

SIMD-optimized circular buffer for real-time metrics:

```rust
use trueno_viz::monitor::simd::SimdRingBuffer;

let mut buffer = SimdRingBuffer::new(1000);

// O(1) push
for value in metrics {
    buffer.push(value);
}

// SIMD-accelerated statistics
let stats = buffer.statistics();
println!("Mean: {}, Stddev: {}", stats.mean(), stats.std_dev());
```

## Explicit SIMD Control

Force specific SIMD level:

```rust
use trueno_viz::accel::{SimdLevel, set_simd_level};

// Force AVX2 (disable AVX512)
set_simd_level(SimdLevel::Avx2);

// Force scalar (disable all SIMD)
set_simd_level(SimdLevel::Scalar);

// Auto-detect (default)
set_simd_level(SimdLevel::Auto);
```

## Run the Example

```bash
cargo run --example simd_kernels --release --features monitor
```

**Example output:**
```text
SIMD Kernels Demo (trueno-viz monitor module)
=============================================

Processing 10,000 f64 values...

Individual SIMD Operations:
---------------------------
  simd_sum:  499950.00 (1.234µs)
  simd_mean: 49.99 (1.456µs)
  simd_min:  0.00 (890ns)
  simd_max:  99.99 (912ns)

Combined Statistics (single SIMD pass):
---------------------------------------
  Min:      0.00
  Max:      99.99
  Mean:     49.99
  Sum:      499950.00
  Variance: 833.25
  Stddev:   28.87
  Time:     2.345µs

Performance Scaling (1000 iterations each):
-------------------------------------------
  Size   100: SIMD     0.12us, Scalar     0.48us, Speedup: 4.0x
  Size  1000: SIMD     0.45us, Scalar     2.10us, Speedup: 4.7x
  Size 10000: SIMD     3.21us, Scalar    14.50us, Speedup: 4.5x

SIMD kernels provide consistent >4x speedup for data aggregation.
```

## Benchmark Results

| Size | SIMD | Scalar | Speedup |
|------|------|--------|---------|
| 100 | 8.4ns | 34.7ns | **4.1x** |
| 300 | 29.6ns | 142ns | **4.8x** |
| 1000 | 122ns | 564ns | **4.6x** |
| 10000 | 1.47µs | 5.94µs | **4.0x** |

## Color Operations

```rust
use trueno_viz::color::Rgba;
use trueno_viz::accel::simd_ops;

// Batch alpha blending (8 colors at once on AVX2)
let src_colors = [Rgba::RED; 8];
let dst_colors = [Rgba::BLUE; 8];
let result = simd_ops::blend_colors_batch(&src_colors, &dst_colors);
```

## Scale Transforms

```rust
use trueno_viz::scale::LinearScale;
use trueno_viz::accel::simd_ops;

let scale = LinearScale::new().domain(0.0, 100.0).range(0.0, 800.0);
let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();

// SIMD batch transform
let pixels = simd_ops::transform_batch(&scale, &values);
```

## Platform-Specific Notes

### x86_64

- SSE2: Always available (baseline)
- AVX2: Haswell and newer (2013+)
- AVX512: Skylake-X and newer

### ARM64 (Apple Silicon, Raspberry Pi 4+)

- NEON: Always available
- 128-bit vectors

### WebAssembly

- SIMD128: Supported in modern browsers
- Auto-detected at runtime

## Feature Flags

Enable monitor SIMD kernels:

```toml
[dependencies]
trueno-viz = { version = "0.1.15", features = ["monitor"] }
```

## Non-AVX2 Fallback

SIMD functions gracefully fall back on older hardware:

```bash
# Test fallback mode
RUSTFLAGS="-C target-feature=-avx2" cargo run --example simd_kernels --release --features monitor
```

No SIGILL crash - operations work correctly at scalar speed.

## Next Chapter

Continue to [GPU Compute](./gpu.md) for even more acceleration.
