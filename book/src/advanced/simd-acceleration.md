# SIMD Acceleration

SIMD (Single Instruction Multiple Data) acceleration provides hardware-optimized statistical calculations for real-time visualization and trace analysis.

> **TDD-Verified:** SIMD operations tested in [`tests/sprint19_enhanced_stats_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Statistical Analysis](./statistical-analysis.md) for overview

## Overview

Renacer uses **trueno-viz SIMD kernels** for vectorized data processing:

| Operation | SIMD Benefit | Use Case |
|-----------|--------------|----------|
| Sum/Mean | 4-5x faster | Ring buffer aggregation |
| Min/Max | 4-5x faster | Sparkline normalization |
| Statistics | 3-4x faster | Combined metrics |
| Normalize | 4x faster | Batch scaling |

**Benefits:**
- **4-5x faster** than scalar code (x86_64 AVX2)
- **Zero allocation** - operates directly on buffer storage
- **Automatic fallback** - works on non-AVX2 hardware

## Visualization Module SIMD

The `visualize` module uses SIMD for real-time TUI rendering:

### HistoryBuffer (Ring Buffer)

```rust,ignore
use renacer::visualize::ring_buffer::HistoryBuffer;

// Create a buffer for metric history
let mut buffer = HistoryBuffer::new(300);  // 300 samples

// Push values (O(1), no allocation)
for latency in syscall_latencies {
    buffer.push(latency);
}

// SIMD-accelerated statistics (zero allocation)
let sum = buffer.sum();       // simd_sum via trueno-viz
let avg = buffer.avg();       // simd_mean via trueno-viz
let min = buffer.min();       // simd_min via trueno-viz
let max = buffer.max();       // simd_max via trueno-viz

// Combined statistics in single SIMD pass
let (min, max, mean) = buffer.stats();  // simd_statistics via trueno-viz
```

### Sparkline Generation

```rust,ignore
use renacer::visualize::theme::{sparkline, normalize_batch};

// SIMD-accelerated sparkline (uses simd_min/simd_max)
let values = buffer.latest(50);
let spark = sparkline(&values, 50);  // "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"

// SIMD batch normalization
let normalized = normalize_batch(&values);  // [0.0, ..., 1.0]
```

## How SIMD Works

**Vector processing:**
```text
Scalar (1 value at a time):
  [1234] → process → result

AVX2 SIMD (4 f64 values at once):
  [1234, 5678, 9012, 3456] → process → [4 results]
```

**Implementation:**
- Uses `trueno-viz::monitor::simd::kernels` for low-level operations
- Automatic feature detection (AVX2, SSE2, NEON)
- Graceful fallback to scalar on unsupported hardware

## Run the Example

```bash
cargo run --example simd_visualization --release
```

**Example output:**
```text
SIMD-Accelerated Visualization Demo
====================================

Buffer filled with 1000 simulated latency values

SIMD-Accelerated Statistics:
-----------------------------
  Sum:    99875.32 (1.234µs)
  Avg:    99.88 (1.456µs)
  Min:    20.12
  Max:    179.88
  Mean:   99.88
  Stddev: 35.67 (2.345µs)
  Stats:  (890ns for min/max/mean combined)

SIMD-Accelerated Sparkline:
---------------------------
  ▃▅▇█▇▆▄▂▁▂▄▆▇█▇▅▃▁▂▃▅▇█▇▆▄▂▁▂▄▆▇█▇▅▃▁▂▃▅▇█▇▆▄▂▁▂▄
  (Generated in 234ns)

Performance Scaling:
--------------------
  Size   100: 0.12 us/op (1000 iterations)
  Size  1000: 0.45 us/op (1000 iterations)
  Size 10000: 3.21 us/op (1000 iterations)

SIMD acceleration powered by trueno-viz monitor::simd::kernels
```

## Benchmark Results

From `cargo bench --bench visualization_simd`:

| Size | SIMD Sum | Scalar Sum | Speedup |
|------|----------|------------|---------|
| 100 | 8.4ns | 34.7ns | **4.1x** |
| 300 | 29.6ns | 142ns | **4.8x** |
| 1000 | 122ns | 564ns | **4.6x** |
| 10000 | 1.47µs | 5.94µs | **4.0x** |

## Platform Support

| Platform | SIMD Level | Performance |
|----------|------------|-------------|
| x86_64 AVX2 | Full | 4-5x speedup |
| x86_64 SSE2 | Partial | 2-3x speedup |
| ARM64 NEON | Full | 4x speedup |
| WASM SIMD128 | Full | 3-4x speedup |
| Fallback | Scalar | Baseline |

## Non-AVX2 Fallback

Renacer gracefully falls back on older hardware:

```bash
# Test fallback mode
RUSTFLAGS="-C target-feature=-avx2" cargo run --release -- visualize -- ls
```

No SIGILL crash - operations work correctly at scalar speed.

## Dependencies

SIMD acceleration requires trueno-viz with the `monitor` feature:

```toml
[dependencies]
trueno-viz = { version = "0.1.15", features = ["monitor"] }
```

## Summary

SIMD acceleration provides:
- **4-5x speedup** for visualization statistics
- **Zero allocation** in hot paths
- **Automatic fallback** on older hardware
- **Seamless integration** via trueno-viz kernels

**All SIMD operations tested in:**
- [`tests/sprint19_enhanced_stats_tests.rs`](../../../tests/)
- [`benches/visualization_simd.rs`](../../../benches/)

## Related

- [Statistical Analysis](./statistical-analysis.md) - Parent chapter
- [Percentile Analysis](./percentiles.md) - Percentile calculations
- [Real-time Visualization](./realtime-anomaly.md) - TUI visualization
