# Examples

This directory contains runnable examples demonstrating trueno-graph capabilities.

## Available Examples

| Example | Description | Command |
|---------|-------------|---------|
| `simple_graph` | Basic graph operations | `cargo run --example simple_graph` |
| `graph_algorithms` | BFS, PageRank, find_callers | `cargo run --example graph_algorithms` |
| `comprehensive_demo` | Full feature demonstration | `cargo run --example comprehensive_demo` |
| `paging_demo` | GPU memory paging (requires gpu feature) | `cargo run --example paging_demo --features gpu` |

## Running Examples

```bash
# Run without GPU
cargo run --example simple_graph
cargo run --example graph_algorithms

# Run with GPU acceleration
cargo run --example paging_demo --features gpu
```

## Expected Output

### simple_graph
```
Graph created with 3 nodes
Node 0 neighbors: [1, 2]
```

### graph_algorithms
```
BFS from node 0: visited 5 nodes
PageRank converged in 15 iterations
Top ranked node: 2 (score: 0.234)
```
