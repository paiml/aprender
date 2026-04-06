# Profiling and Performance Tuning

This chapter documents performance profiling techniques and optimization discoveries from the Sovereign AI Stack.

## Thread Pool Optimization

### The 2.05x Discovery

A major performance breakthrough was discovered through systematic profiling: **reducing thread count from 48 to 16 yielded a 2.05x speedup** in CPU inference.

| Metric | 48 Threads | 16 Threads | Improvement |
|--------|------------|------------|-------------|
| Throughput | 12.4 tok/s | 25.4 tok/s | **2.05x** |
| Overhead | 3.5x | 1.7x | 2.06x |
| Per-token latency | 80.6 ms | 39.4 ms | 2.05x |

### Root Cause Analysis

The default rayon thread pool uses all available logical cores (hyperthreads). For small work units like single-token inference, this causes:

1. **Cache line bouncing** - 48 threads invalidating L1/L2 constantly
2. **False sharing** - Adjacent output writes causing coherency traffic
3. **Hyperthread contention** - HT pairs fighting for same FPU
4. **Rayon sync overhead** - Work units too small for 48-way split

### Optimal Thread Count Formula

```
Optimal threads = min(physical_cores, work_size / cache_line_size)
```

For Qwen 1.5B with 1536 hidden dimension:
- 1536 elements / 16 elements per cache line = 96 cache lines
- 12-16 threads = 6-8 cache lines per thread (optimal)
- 48 threads = 2 cache lines per thread (too fine-grained)

### Implementation

The `configure_optimal_thread_pool()` function in realizar sets the optimal thread count:

```rust
use realizar::inference::configure_optimal_thread_pool;

// Set to 16 threads (or physical core count)
configure_optimal_thread_pool();

// Or set explicitly via environment
std::env::set_var("RAYON_NUM_THREADS", "16");
```

### Profiling Tools

#### Micro-Level Profiling

```bash
cargo run --release --example micro_profile
```

Profiles individual operations (matmul, attention, FFN) to identify bottlenecks.

#### Layer-Level Profiling

```bash
cargo run --release --example layer_profile
```

Profiles generation timing to measure per-token latency and throughput.

#### Thread Sweep

```bash
for t in 8 10 12 14 16 18 20 24 32 48; do
    echo "=== $t threads ==="
    RAYON_NUM_THREADS=$t cargo run --release --example instrumented_forward 2>&1 | grep -E "Throughput|Per token"
done
```

### Results Interpretation

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low throughput, high thread count | Thread overhead | Reduce threads |
| Low bandwidth utilization (<20%) | Compute-bound | SIMD optimization |
| High bandwidth, low throughput | Memory-bound | Better tiling |
| Variable latency | Cache thrashing | Thread affinity |

## Tile-Level Profiling (TILING-SPEC-001)

Trueno's `BrickProfiler` supports hierarchical tile profiling:

```rust
use trueno::{BrickProfiler, TileLevel};

let mut profiler = BrickProfiler::new();
profiler.enable_tile_profiling();

// Profile a macro tile (L3/Global memory level)
let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
// ... execute computation ...
profiler.stop_tile(timer, elements, flops);

// Get results
println!("{}", profiler.tile_summary());
```

### Tile Hierarchy

| Level | Memory | Typical Size | Use Case |
|-------|--------|--------------|----------|
| Macro | L3/Global | 32MB | Layer-level |
| Midi | L2/Shared | 256KB | Head-level |
| Micro | L1/Registers | 32KB | SIMD-level |

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| GFLOP/s | flops / seconds / 1e9 | Compute throughput |
| Arithmetic Intensity | flops / bytes | >10 = compute-bound |
| Cache Efficiency | actual / peak | Target >50% |

## Remaining Optimization Opportunities

After thread optimization (25.4 tok/s), the remaining gap to 42 tok/s target is 1.66x:

| Optimization | Expected Gain | Status |
|--------------|---------------|--------|
| Thread count optimization | 2.05x | Done |
| Fuse parallel regions | 1.2-1.3x | Pending |
| SIMD attention (AVX-512) | 1.2-1.4x | Pending |
| Reduce Vec allocations | 1.1x | Pending |

---

**Previous:** [Optimization Iteration](./optimization-iteration.md)
**Next:** [Code Review](./code-review.md)
