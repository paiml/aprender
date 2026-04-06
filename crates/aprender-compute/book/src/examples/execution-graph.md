# Execution Path Graph

The Execution Path Graph (PAR-201) tracks the full hierarchy of operations during inference: Layer → Brick → Kernel → PTX. This enables precise profiling and bottleneck detection.

## Running the Example

```bash
# Basic (headless ASCII tree)
cargo run --example execution_graph

# With presentar-terminal TreeNode
cargo run --example execution_graph --features presentar-tui
```

## Headless ASCII Tree

Zero-dependency visualization for CI/CD and automation:

```rust
use trueno::{BrickProfiler, BrickId, ExecutionNode};

let mut profiler = BrickProfiler::new();
profiler.enable();
profiler.enable_graph();

// Record a transformer layer
profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });

// Record a brick with its kernel
profiler.graph_push_scope(ExecutionNode::Brick {
    id: BrickId::QkvProjection,
    timing_ns: 200_000,
    elements: 4096,
});
profiler.graph_record_kernel(
    "batched_q4k_gemv",
    0xDEADBEEF,
    (32, 1, 1),   // grid
    (256, 1, 1),  // block
    4096,         // shared_mem
);
profiler.graph_pop_scope(); // pop brick
profiler.graph_pop_scope(); // pop layer

// Render to ASCII (no dependencies)
let tree = profiler.execution_graph().to_ascii_tree();
println!("{}", tree);
```

Output:
```
Layer 0
└── QkvProjection  200.0µs (4096 elem)
    └── batched_q4k_gemv  <<<32,256,1>>> smem=4096B
```

## Full Example Output

```
Execution Graph
├── Layer 0
│   ├── RmsNorm  50.0µs (4096 elem)
│   │   └── rmsnorm_kernel  <<<16,256,1>>> smem=1024B
│   ├── QkvProjection  200.0µs (4096 elem)
│   │   └── batched_q4k_gemv  <<<32,256,1>>> smem=4096B
│   ├── AttentionScore  150.0µs (4096 elem)
│   │   └── incremental_attention  <<<8,256,1>>> smem=2048B
│   └── GateProjection  300.0µs (4096 elem)
│       └── batched_q6k_gemv  <<<64,256,1>>> smem=8192B
└── Layer 1
    ├── RmsNorm  50.0µs (4096 elem)
    │   └── rmsnorm_kernel  <<<16,256,1>>> smem=1024B
    ...
```

## Use Cases

| Use Case | Method | Dependencies |
|----------|--------|--------------|
| CI/CD logs | `to_ascii_tree()` | None |
| Snapshot tests | `to_ascii_tree()` | None |
| File export | `to_ascii_tree()` | None |
| Interactive TUI | `to_tree_node()` | `presentar-tui` feature |
| SVG visualization | `to_dot()` | External graphviz |

## PTX Registry

Track PTX source code for kernel debugging:

```rust
use trueno::PtxRegistry;

let mut registry = PtxRegistry::new();
registry.register("kernel_name", ptx_source, Some(Path::new("src/kernel.ptx")));

// Lookup by hash
let hash = PtxRegistry::hash_ptx(ptx_source);
let source = registry.lookup(hash);
```

## Graphviz Export

```bash
# Generate DOT file
cargo run --example execution_graph 2>/dev/null | grep -A1000 "digraph" > /tmp/graph.dot

# Or in code:
let dot = profiler.graph_to_dot();
std::fs::write("graph.dot", dot)?;

# Render to SVG
dot -Tsvg graph.dot -o graph.svg
```

## Query Helpers

```rust
let graph = profiler.execution_graph();

// Find all kernel nodes
for (id, node) in graph.kernel_nodes() {
    println!("{}: {:?}", id.0, node);
}

// Find slowest brick with kernel
if let Some((id, node, timing_ns)) = graph.slowest_kernel() {
    println!("Bottleneck: {:?} at {}µs", node, timing_ns / 1000);
}

// Check scope balance
assert!(graph.is_scope_balanced());
```

## Critical Path Analysis (Phase 9)

Identify true bottlenecks vs parallelizable work using longest-path analysis:

```rust
use trueno::ExecutionGraph;

// After recording execution...
let (critical_path, total_ns) = graph.critical_path();

println!("Critical path: {} nodes, {:.2}ms total",
    critical_path.len(),
    total_ns as f64 / 1_000_000.0);

// Get formatted summary with parallelization opportunities
println!("{}", graph.critical_path_summary());
```

Output:
```
Critical Path: 0.70ms (3 nodes)
──────────────────────────────────────────────────
┌ RmsNorm (100.0µs)
│ QkvProjection (200.0µs)
└ GateProjection (400.0µs)

Parallelization Opportunities (high slack):
  AttentionScore slack=100.0µs
```

## Slack Calculation

Nodes with positive slack can be parallelized without affecting total time:

```rust
let slack = graph.compute_slack();

for (node_id, slack_ns) in &slack {
    if *slack_ns > 0 {
        println!("Node {} can be delayed by {}µs", node_id.0, slack_ns / 1000);
    }
}
```

## Roofline Integration

Measure distance from theoretical peak performance:

```rust
// Device: RTX 4090 (83 TFLOPS, 1008 GB/s)
let distances = graph.roofline_distance(83.0, 1008.0);

for (node_id, distance) in &distances {
    let efficiency = (1.0 - distance) * 100.0;
    println!("Kernel {} at {:.1}% of roofline", node_id.0, efficiency);
}
```

Record kernels with roofline metrics:

```rust
graph.record_kernel_launch_with_metrics(
    "matmul_kernel",
    ptx_hash,
    (128, 1, 1),      // grid
    (256, 1, 1),      // block
    16384,            // shared_mem
    150_000,          // timing_ns
    50.0,             // arithmetic_intensity (FLOPs/byte)
    42.0,             // achieved_tflops
);
```

## Data Movement Tracking

Track H2D/D2H/D2D transfers and detect wasteful ping-pong patterns:

```rust
use trueno::TransferDirection;

// Record transfers
graph.record_transfer("host_weights", "device_weights",
    4 * 1024 * 1024, // 4MB
    TransferDirection::H2D,
    Some(50_000)); // 50µs

// Detect ping-pong anti-pattern
let ping_pongs = graph.detect_ping_pong();
if !ping_pongs.is_empty() {
    println!("Warning: {} wasteful transfer patterns detected", ping_pongs.len());
}
```

## Edge Types

| Edge Type | Purpose |
|-----------|---------|
| `Contains` | Layer contains bricks |
| `Launches` | Brick launches kernel |
| `Calls` | Function calls function |
| `Sequence` | Sequential execution |
| `DependsOn` | CUDA event dependency |
| `Transfer` | Memory transfer with bytes and direction |

## Integration with realizar

The execution graph integrates with the realizar inference engine:

```rust
// In CudaExecutor
executor.set_profiler_sync_mode(SyncMode::Deferred);

// During forward pass - graph records automatically
let timer = executor.start_brick_id(BrickId::QkvProjection);
// ... kernel launch ...
executor.stop_brick_id(timer, hidden_dim as u64);

// Export graph after inference
let graph = executor.profiler().execution_graph();
println!("{}", graph.to_ascii_tree());

// Phase 9: Analyze critical path
println!("{}", graph.critical_path_summary());
```
