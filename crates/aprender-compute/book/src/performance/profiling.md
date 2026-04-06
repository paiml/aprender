# Profiling

## The Real Profiling Mandate

Trueno enforces a strict **"Real Profiling"** mandate. All performance metrics reported by the ecosystem MUST be measured, not derived.

> **Forbidden**: Calculating per-brick time by taking total throughput and multiplying by a budget fraction.
> **Required**: Measuring start/end times for every operation, with full synchronization.

### Why?

Simulated or derived metrics mask bottlenecks. If you assume an operation takes 10% of the time, you will never discover when it actually takes 50% due to a regression.

## BrickProfiler v2 (PAR-200)

The `BrickProfiler` is the core profiling tool built into `trueno`. Version 2 (PAR-200) introduces O(1) hot-path profiling with deferred sync support.

### Key Features

| Feature | v1 | v2 (PAR-200) |
|---------|----|----|
| Brick lookup | HashMap<String> O(n) | BrickId enum O(1) |
| GPU sync | Immediate (~200% overhead) | Deferred (~5% overhead) |
| Category aggregation | Manual | Automatic (Norm, Attention, FFN) |

### BrickId Enum

```rust
use trueno::{BrickProfiler, BrickId, BrickCategory, SyncMode};

let mut profiler = BrickProfiler::new();
profiler.enable();

// O(1) brick timing with enum-based lookup
let timer = profiler.start_brick(BrickId::QkvProjection);
// ... perform QKV projection ...
profiler.stop_brick(timer, num_elements);

// Category breakdown
let cats = profiler.category_stats();
println!("Attention: {:.1}%", cats[BrickCategory::Attention as usize].percentage(profiler.total_ns()));
```

### Deferred Sync Mode

For GPU workloads, immediate synchronization after every operation adds ~200% overhead. Deferred sync batches measurements:

```rust
profiler.set_sync_mode(SyncMode::Deferred);
profiler.reset_epoch();

// Record without sync (timestamps only)
let start = profiler.elapsed_ns();
// ... GPU kernel launch ...
profiler.record_deferred(BrickId::AttentionScore, start, elements);

// Single sync point at end of layer/batch
let end = profiler.elapsed_ns();
profiler.finalize(end);  // Apply all pending measurements
```

**Sync Modes:**
- `Immediate`: Sync after every brick (~200% overhead, accurate per-brick)
- `PerLayer`: Sync once per transformer layer (~20% overhead)
- `Deferred`: Single sync at batch end (~5% overhead)
- `None`: No timing (0% overhead, for production)

### Running the Example

```bash
cargo run --example brick_profiler_v2
```

Output:
```
=== PAR-200: BrickProfiler v2 Demo ===

Per-Brick Timing:
Brick                  Avg (µs) Total (µs)    Count
----------------------------------------------------
RmsNorm                   104.6      313.9        3
QkvProjection             253.8      761.3        3
...

Category Breakdown:
Category       Avg (µs)      Pct    Samples
--------------------------------------------
Norm              104.6     8.3%          3
Attention         228.7    36.4%          6
FFN               348.0    55.3%          6
```

### Integration with Realizar

The realizar inference engine integrates BrickProfiler v2:

```rust
// In CudaExecutor
executor.set_profiler_sync_mode(SyncMode::Deferred);

// During forward pass
let timer = executor.start_brick_id(BrickId::QkvProjection);
// ... kernel launch ...
executor.stop_brick_id(timer, hidden_dim as u64);
```

## Falsification Protocols (F101-F110)

To prove profiling is real, we apply Popperian Falsification:

1. **F101**: `BrickId::COUNT == 15` (all brick types defined)
2. **F102**: Category mapping correct for all BrickIds
3. **F103**: Deferred mode accumulates pending measurements
4. **F104**: `finalize()` clears pending queue
5. **F105**: Zero-overhead when disabled
6. **F106**: Array indexing is O(1)
7. **F107**: Thread-safe (Send + Sync)
8. **F108**: BrickIdTimer fits in 32 bytes
9. **F109**: `elapsed_ns()` monotonic
10. **F110**: Category stats sum correctly

## Execution Path Graph (PAR-201)

BrickProfiler v2 also supports execution path graphs for tracking the full hierarchy:

```
Layer(0)
  └─► Brick(QkvProjection) ─────► Kernel(batched_q4k_gemv, ptx_hash=0x7a3b...)
  │                                   └─► PTX source lookup
  └─► Brick(AttentionScore) ────► Kernel(incremental_attention, ptx_hash=0x9f1c...)
```

### Enabling Graph Recording

```rust
use trueno::{BrickProfiler, BrickId, ExecutionNode, PtxRegistry};

let mut profiler = BrickProfiler::new();
profiler.enable();
profiler.enable_graph();  // Enable execution graph tracking

// Record layer scope
profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });

  // Record brick
  let timer = profiler.start_brick(BrickId::QkvProjection);
  // ... work ...
  profiler.stop_brick(timer, elements);
  profiler.graph_record_brick(BrickId::QkvProjection, timing_ns, elements);

  // Record kernel launch
  profiler.graph_record_kernel(
      "batched_q4k_gemv",
      ptx_hash,
      (32, 1, 1),   // grid
      (256, 1, 1),  // block
      4096,         // shared_mem
  );

profiler.graph_pop_scope();
```

### PTX Registry

Track PTX source code for debugging:

```rust
let mut registry = PtxRegistry::new();
registry.register("kernel_name", ptx_source, Some(Path::new("src/kernel.ptx")));

// Lookup by hash
let hash = PtxRegistry::hash_ptx(ptx_source);
let source = registry.lookup(hash);
```

### Visualization

**Option 1: Headless ASCII Tree (CI/CD, Automation)**

Zero-dependency tree visualization for testing and automation:

```rust
let graph = profiler.execution_graph();
let tree = graph.to_ascii_tree();
println!("{}", tree);

// Output can be used for:
// - Snapshot tests (deterministic output)
// - CI/CD logs
// - File export
std::fs::write("execution_tree.txt", &tree)?;
```

Output:
```
Layer 0
├── RmsNorm  50.0µs (4096 elem)
│   └── rmsnorm_kernel  <<<16,256,1>>> smem=1024B
├── QkvProjection  200.0µs (4096 elem)
│   └── batched_q4k_gemv  <<<32,256,1>>> smem=4096B
```

**Option 2: Interactive TUI (presentar-terminal)**

Native TUI widget for interactive exploration (requires `presentar-tui` feature):

```rust
use trueno::ExecutionGraph;
use presentar_terminal::{Tree, TuiApp};

// Convert execution graph to tree widget
let tree_node = profiler.execution_graph().to_tree_node();
let tree = Tree::new().with_root(tree_node).expand_all();

// Use in TUI app or render headless via HeadlessCanvas
```

**Option 3: Graphviz DOT Export**

Export to Graphviz DOT format for SVG rendering:

```bash
# In code
let dot = profiler.graph_to_dot();
std::fs::write("graph.dot", dot)?;

# Visualize
dot -Tsvg graph.dot -o graph.svg
```

### Running the Example

```bash
# Headless ASCII tree (default, no dependencies)
cargo run --example execution_graph

# With presentar-terminal TreeNode
cargo run --example execution_graph --features presentar-tui
```

## Backend-Specific Profiling (CPU/SIMD/GPU)

Different compute backends require different profiling approaches. See the full specification in `docs/specifications/ml-tuner-bricks.md` (Appendix E.8).

### Instrumentation Status

| Backend | Path | BrickProfiler | Overhead |
|---------|------|---------------|----------|
| CUDA | `CudaExecutor::forward()` | Full | ~5% (deferred) |
| CPU | `forward()` | **None** | N/A |
| CPU | `forward_profiled()` | Full | ~10% |
| SIMD | trueno ops | Per-op | ~2% |

**Key Insight**: The legacy CPU `forward()` function lacks BrickProfiler instrumentation. For CPU profiling, use `forward_profiled()` or add instrumentation manually.

### SIMD Backend Profiling

Profile SIMD operations at the brick level:

```rust
use trueno::{BrickProfiler, BrickId};

let mut profiler = BrickProfiler::new();
profiler.enable();

// Profile SIMD operation
let timer = profiler.start_brick(BrickId::RmsNorm);
trueno::simd::rms_norm_avx2(&input, &mut output);  // AVX2 backend
profiler.stop_brick(timer, input.len() as u64);

// Get throughput
let stats = profiler.stats_for(BrickId::RmsNorm);
let throughput = stats.total_elements as f64 / stats.total_ns as f64 * 1000.0;
println!("RmsNorm: {:.2} Melem/s", throughput);
```

### Backend Comparison

Compare performance across backends:

```rust
use trueno::{BrickProfiler, BrickId, detect_backend, Backend};

let backend = detect_backend();
let mut profiler = BrickProfiler::new();
profiler.enable();

// Same brick, different backends
match backend {
    Backend::Avx512 => { /* AVX-512 path */ }
    Backend::Avx2 => { /* AVX2 path */ }
    Backend::Neon => { /* ARM NEON path */ }
    _ => { /* Scalar fallback */ }
}

// Report includes backend name
println!("Backend: {:?}", backend);
println!("{}", profiler.report());
```

### Backend-Specific Roofline

Different backends have different theoretical peaks:

| Backend | Peak TFLOPS (FP32) | Memory BW (GB/s) |
|---------|-------------------|------------------|
| RTX 4090 | 83.0 | 1008 |
| AVX-512 | ~2.0 | ~100 |
| AVX2 | ~0.5 | ~50 |
| ARM NEON | ~0.2 | ~40 |
| Scalar | ~0.1 | ~25 |

```rust
// Backend-aware roofline distance
let distance = match backend {
    Backend::Cuda => graph.roofline_distance(83.0, 1008.0),
    Backend::Avx512 => graph.roofline_distance(2.0, 100.0),
    Backend::Avx2 => graph.roofline_distance(0.5, 50.0),
    _ => graph.roofline_distance(0.1, 25.0),
};
```

### Critical Path Analysis (Phase 9)

Identify true bottlenecks vs parallelizable work:

```rust
let graph = profiler.execution_graph();

// Get critical path
let (critical_path, total_ns) = graph.critical_path();
println!("Critical path: {} nodes, {:.2}ms", critical_path.len(), total_ns as f64 / 1_000_000.0);

// Find parallelization opportunities
let slack = graph.compute_slack();
for (node_id, slack_ns) in &slack {
    if *slack_ns > 0 {
        println!("Node {} can be parallelized (slack: {}µs)", node_id.0, slack_ns / 1000);
    }
}

// Formatted summary
println!("{}", graph.critical_path_summary());
```

## Tools

- **presentar-terminal Tree**: Native TUI tree widget for hierarchical execution graphs.
- **cbtop**: The primary visualization tool for ComputeBrick pipelines. Supports backend-specific profiling display.
- **perf / flamegraph**: For CPU-side overhead analysis.
- **nsight**: For deep GPU kernel inspection (external to the pure Rust stack).
