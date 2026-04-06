# BLIS-Style Matrix Multiplication

High-performance GEMM (General Matrix Multiply) implementation based on the BLIS framework.

## Overview

The BLIS implementation achieves **72 GFLOP/s** on modern x86_64 CPUs, a significant improvement over naive implementations (~10 GFLOP/s).

## Running the Benchmark

```bash
cargo run --release --example blis_benchmark
```

Expected output:
```
=== BLIS GEMM Benchmark ===

--- Reference vs BLIS Comparison ---
64x64: Reference      0.1ms, BLIS      0.0ms, Speedup:   3.5x
128x128: Reference    1.2ms, BLIS      0.1ms, Speedup:   8.7x
256x256: Reference   12.4ms, BLIS      0.8ms, Speedup:  14.6x
512x512: Reference  159.0ms, BLIS      4.5ms, Speedup:  35.1x

--- BLIS Performance ---
BLIS                   64x  64:     20.3 us,   25.9 GFLOP/s
BLIS                  128x 128:     99.5 us,   42.2 GFLOP/s
BLIS                  256x 256:    544.1 us,   61.7 GFLOP/s
BLIS                  512x 512:   3752.0 us,   71.5 GFLOP/s
BLIS                 1024x1024:  30965.0 us,   69.4 GFLOP/s
```

## Algorithm

The implementation uses the 5-loop BLIS algorithm:

```
Loop 5: jc (N dimension, L3 blocking)
  Loop 4: pc (K dimension, L2 blocking)
    Pack B → B̃ (nc × kc panel)
    Loop 3: ic (M dimension, L1 blocking)
      Pack A → Ã (mc × kc panel)
      Loop 2: jr (NR micro-tiles)
        Loop 1: ir (MR micro-tiles)
          MICROKERNEL: C[ir,jr] += Ã[ir,:] × B̃[:,jr]
```

### Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| MR | 8 | Microkernel rows (AVX2 = 8 f32) |
| NR | 6 | Microkernel columns |
| KC | 256 | K-blocking for L1 cache |
| MC | 72 | M-blocking for L2 cache |
| NC | 4096 | N-blocking for L3 cache |

## API Usage

### Basic GEMM

```rust
use trueno::blis::gemm;

let m = 256;
let n = 256;
let k = 256;

let a: Vec<f32> = vec![1.0; m * k];
let b: Vec<f32> = vec![1.0; k * n];
let mut c: Vec<f32> = vec![0.0; m * n];

// C += A × B
gemm(m, n, k, &a, &b, &mut c).unwrap();
```

### With Profiling

```rust
use trueno::blis::{gemm_profiled, BlisProfiler};

let mut profiler = BlisProfiler::enabled();

gemm_profiled(m, n, k, &a, &b, &mut c, &mut profiler).unwrap();

println!("{}", profiler.summary());
```

Output:
```
BLIS Profiler Summary
=====================
Macro: 578.8us avg, 58.0 GFLOP/s, 1 calls
Midi:  130.0us avg, 64.6 GFLOP/s, 4 calls
Micro: 0.4us avg, 69.2 GFLOP/s, 1376 calls
Pack:  9.7us avg, 5 calls
Total: 58.0 GFLOP/s
```

## Toyota Production System Integration

### Jidoka (Stop on Defect)

Runtime guards catch numerical errors:

```rust
use trueno::blis::{JidokaGuard, gemm_reference_with_jidoka};

let guard = JidokaGuard::strict();

// Will return Err if NaN/Inf detected
gemm_reference_with_jidoka(m, n, k, &a, &b, &mut c, &guard)?;
```

### Heijunka (Load Leveling)

Parallel execution uses balanced partitioning:

```rust
use trueno::blis::HeijunkaScheduler;

let scheduler = HeijunkaScheduler::default();
let partitions = scheduler.partition_m(1024, 72);
// Returns balanced work ranges for each thread
```

## Performance Analysis

### Roofline Model

For GEMM with optimal blocking:
- Arithmetic Intensity: AI ≈ K/2 FLOP/byte
- At K=512: AI ≈ 256 → **compute-bound** (good)

### Achieved vs Theoretical

| Size | Achieved | Theoretical | Efficiency |
|------|----------|-------------|------------|
| 256×256 | 61 GFLOP/s | ~400 GFLOP/s | 15% |
| 512×512 | 71 GFLOP/s | ~400 GFLOP/s | 18% |
| 1024×1024 | 72 GFLOP/s | ~400 GFLOP/s | 18% |

## References

1. Goto, K., & Van de Geijn, R. A. (2008). Anatomy of High-Performance Matrix Multiplication. *ACM TOMS*, 34(3).

2. Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for Rapidly Instantiating BLAS Functionality. *ACM TOMS*, 41(3).

3. Low, T. M., et al. (2016). Analytical Modeling Is Enough for High-Performance BLIS. *ACM TOMS*, 43(2).
