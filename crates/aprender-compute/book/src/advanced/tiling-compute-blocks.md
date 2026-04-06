# Tiling Compute Blocks (TCB)

Tiling Compute Blocks represent the fundamental unit of work partitioning within high-performance `ComputeBrick` kernels. While a `ComputeBrick` defines a logical operation (e.g., Q4_K MatMul), a TCB defines the **physical execution strategy** - how data is partitioned across the hardware's memory hierarchy.

## Core Insight

The 37x performance gap identified in the B4 Investigation was largely due to:
1. **Lack of tiling** in the CPU path
2. **Suboptimal tiling** (low occupancy) in early GPU kernels

This module formalizes the tiling strategies that enabled the >200 tok/s target for Qwen2.5-Coder inference.

## Design Philosophy

> "Cache-First, Computation-Second"

Execution is planned around memory movement, ensuring that every byte loaded from Global Memory is reused maximum times within the TCB.

## Hierarchical Tiling

Tiling occurs at three distinct levels, each corresponding to a hardware memory tier:

| Level | Memory | CPU | GPU |
|-------|--------|-----|-----|
| **Macro-Tile** | L3/Global | Socket-level parallelism | SM-level partitioning |
| **Midi-Tile** | L2/Shared | Rayon task | Thread block |
| **Micro-Tile** | Registers | SIMD lanes | Warp-level |

### Cache Hierarchy Sizes

```rust
use trueno::tiling::TcbLevel;

// Typical x86_64 cache sizes
assert_eq!(TcbLevel::Macro.typical_cache_bytes(), 32 * 1024 * 1024); // 32 MB L3
assert_eq!(TcbLevel::Midi.typical_cache_bytes(), 256 * 1024);        // 256 KB L2
assert_eq!(TcbLevel::Micro.typical_cache_bytes(), 32 * 1024);        // 32 KB L1
```

## TcbGeometry

The `TcbGeometry` struct captures tile dimensions for GEMM-style operations:

```rust
use trueno::tiling::TcbGeometry;

// Create a 4x8x256 micro-tile (M=4 rows, N=8 cols, K=256 reduction)
let tile = TcbGeometry::new(4, 8, 256);

// Check Q4_K alignment (K must be multiple of 256)
assert!(tile.is_q4k_aligned());

// Calculate arithmetic intensity (FLOP/byte)
let ai = tile.arithmetic_intensity();
println!("AI: {:.2} FLOP/byte", ai); // ~1.33

// Check if tile fits in cache
assert!(tile.fits_in_cache(32 * 1024)); // 32KB L1
```

### Arithmetic Intensity Formula

For GEMM operations:

```
AI = (2 * M * N * K) / ((M * K + K * N) * sizeof(f32))
```

Higher AI means the operation is **compute-bound**; lower means **memory-bound**.

## TilingConfig

A complete tiling configuration specifies all three hierarchy levels:

```rust
use trueno::tiling::{TilingConfig, TilingBackend};

// GPU Q4_K MatVec (single-token generation)
let gpu_config = TilingConfig::gpu_q4k_matvec();
assert_eq!(gpu_config.macro_tile.m, 1);      // Single row
assert_eq!(gpu_config.macro_tile.k, 256);    // Q4_K superblock
assert_eq!(gpu_config.backend, TilingBackend::Gpu);

// CPU AVX-512 MatMul
let avx512_config = TilingConfig::cpu_avx512_matmul();
assert_eq!(avx512_config.micro_tile.n, 16);  // 16 floats = 512 bits
assert_eq!(avx512_config.micro_tile.alignment, 64);

// Validate hierarchy consistency
assert!(gpu_config.validate().is_ok());
```

### Pre-defined Configurations

| Configuration | Backend | Use Case |
|---------------|---------|----------|
| `gpu_q4k_matvec()` | GPU | Single-token Q4_K inference |
| `gpu_q4k_matmul()` | GPU | Batched Q4_K prefill |
| `gpu_softmax()` | GPU | Vocabulary-sized softmax |
| `cpu_avx512_matmul()` | AVX-512 | Dense GEMM |
| `cpu_avx512_q4k_matvec()` | AVX-512 | Q4_K inference |
| `cpu_avx512_vnni_q4k_q8k()` | AVX-512 VNNI | Pure integer Q4K x Q8K |
| `cpu_avx2_matmul()` | AVX2 | Dense GEMM (256-bit) |
| `cpu_avx2_q4k_matvec()` | AVX2 | Q4_K inference (256-bit) |
| `cpu_rmsnorm()` | CPU | RMS normalization |

## TcbIndexCalculator

The index calculator maps hierarchical tile indices to memory offsets:

```rust
use trueno::tiling::{TilingConfig, TcbIndexCalculator};

let config = TilingConfig::cpu_avx2_matmul();
let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

// Get tile offsets
let (row, col) = calc.macro_tile_offset(0);
assert_eq!((row, col), (0, 0));

// Check boundary conditions
let is_boundary = calc.is_boundary_tile(0);

// Get actual tile dimensions at boundaries
let (actual_m, actual_n) = calc.actual_tile_dims(0);
```

### Boundary Handling

When problem sizes don't divide evenly by tile sizes, boundary tiles require special handling:

```rust
// 100x100 problem with 256x256 tiles
let calc = TcbIndexCalculator::new(config, 100, 100, 256);
assert!(calc.is_boundary_tile(0));  // First tile is boundary

let (actual_m, actual_n) = calc.actual_tile_dims(0);
assert_eq!((actual_m, actual_n), (100, 100));  // Clamped to problem size
```

## Q4_K Quantized Tiling

### Q4_K Format Constants

```rust
use trueno::tiling::{Q4K_SUPERBLOCK_SIZE, Q4K_SUPERBLOCK_BYTES};

assert_eq!(Q4K_SUPERBLOCK_SIZE, 256);  // Elements per superblock
assert_eq!(Q4K_SUPERBLOCK_BYTES, 144); // Bytes per superblock

// Compression ratio vs f32
let ratio = (256.0 * 4.0) / 144.0;
println!("Q4_K compression: {:.2}x", ratio); // 7.11x
```

### TiledQ4KMatvec

Executor for cache-blocked Q4_K matrix-vector multiplication:

```rust
use trueno::tiling::TiledQ4KMatvec;

let matvec = TiledQ4KMatvec::new(4096, 4096);

// Get superblock layout
println!("Superblocks per row: {}", matvec.superblocks_per_row()); // 16
println!("Total superblocks: {}", matvec.total_superblocks());     // 65536

// Calculate optimal parallelism for L2 cache
let parallel_rows = matvec.optimal_parallel_rows(256 * 1024);
println!("Optimal parallel rows: {}", parallel_rows); // ~100

// Get statistics
let stats = matvec.stats();
println!("Weight bytes: {:.2} MB", stats.total_weight_bytes as f64 / 1e6);
println!("Arithmetic intensity: {:.2} FLOP/byte", stats.arithmetic_intensity);
```

## Memory Layout Helpers

### Goto Algorithm Panel Packing

The Goto algorithm packs matrices into panel-major format for optimal cache reuse:

```rust
use trueno::tiling::{pack_a_index, pack_b_index};

// Panel A: mr=4, kc=256
let idx = pack_a_index(0, 0, 4, 256, 64); // First element
let idx_next_panel = pack_a_index(4, 0, 4, 256, 64); // Next panel

// Panel B: nr=8, kc=64
let idx_b = pack_b_index(0, 0, 8, 64, 64);
```

### Bank Conflict Avoidance

XOR swizzling prevents shared memory bank conflicts on GPUs:

```rust
use trueno::tiling::swizzle_index;

// Without swizzling: indices 0 and 32 both hit bank 0
// With swizzling: they map to different banks
let idx0 = swizzle_index(0);   // 0 -> bank 0
let idx32 = swizzle_index(32); // 33 -> bank 1
assert_ne!(idx0 % 32, idx32 % 32);
```

## Prefetch Optimization

Calculate optimal prefetch distances based on cache latency:

```rust
use trueno::tiling::{TcbGeometry, TcbLevel, optimal_prefetch_distance};

let micro_tile = TcbGeometry::new(4, 8, 64);

let l1_dist = optimal_prefetch_distance(&micro_tile, TcbLevel::Micro);
let l2_dist = optimal_prefetch_distance(&micro_tile, TcbLevel::Midi);
let l3_dist = optimal_prefetch_distance(&micro_tile, TcbLevel::Macro);

// L3 prefetch should be further ahead (higher latency)
assert!(l3_dist >= l2_dist);
```

## Tile Profiling

The `BrickProfiler` supports tile-level profiling to analyze performance at each level of the hierarchy:

```rust
use trueno::brick::{BrickProfiler, TileLevel};

let mut profiler = BrickProfiler::new();
profiler.enable_tile_profiling();

// Profile a macro tile
let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
// ... execute tile computation ...
let elements = 256 * 256;  // Elements processed
let flops = 2 * 256 * 256 * 256;  // FMA operations
profiler.stop_tile(timer, elements, flops);

// Get statistics
let stats = profiler.tile_stats(TileLevel::Macro);
println!("GFLOP/s: {:.2}", stats.gflops());
println!("Arithmetic Intensity: {:.2} FLOP/byte", stats.arithmetic_intensity());

// Summary report
println!("{}", profiler.tile_summary());

// JSON export for pmat integration
let json = profiler.tile_stats_to_json();
```

### TileStats Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `avg_us()` | total_ns / count / 1000 | Average tile time in µs |
| `throughput()` | elements / seconds | Elements per second |
| `gflops()` | flops / seconds / 1e9 | GFLOP/s throughput |
| `arithmetic_intensity()` | flops / (elements * 4) | FLOP/byte (assuming f32) |
| `cache_efficiency(peak)` | gflops / peak | Efficiency vs theoretical peak |

### Running the Tile Profiling Demo

```bash
cargo run --example tile_profiler_demo
```

Output shows tile-level statistics, GFLOP/s measurements, and JSON export.

## Running the Tiling Demo

```bash
cargo run --example tiling_demo
```

Output shows hierarchical configurations, index mapping, Q4_K statistics, and backend comparisons.

## Scientific References

1. **Lam et al. (1991)** - "The Cache Performance and Optimizations of Blocked Algorithms" - ASPLOS
2. **Goto & van de Geijn (2008)** - "Anatomy of High-Performance Matrix Multiplication" - ACM TOMS
3. **Volkov (2010)** - "Better Performance at Lower Occupancy" - GTC
4. **Dao et al. (2022)** - "FlashAttention" - NeurIPS (tiling for attention)

## See Also

- [Phase 2 Micro-Kernel](./phase2-microkernel.md) - Micro-kernel implementation
- [Phase 15 Fused Q4K Kernels](./phase15-fused-q4k.md) - Q4_K kernel fusion
- [Memory Alignment](./memory-alignment.md) - SIMD alignment requirements
- [AVX-512 Backend](../architecture/avx512-backend.md) - 512-bit SIMD implementation
