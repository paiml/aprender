# Introduction

**trueno-graph** is a GPU-first embedded graph database designed for code analysis, dependency tracking, and graph algorithms. Built on proven PAIML infrastructure ([trueno](https://github.com/paiml/trueno), [aprender](https://github.com/paiml/aprender)), it delivers 10-250× speedups over CPU-based graph operations through hybrid CPU/GPU execution with automatic memory paging.

## What is trueno-graph?

trueno-graph provides:

- **CSR Graph Storage**: Compressed Sparse Row format with bidirectional traversal (forward + reverse)
- **Parquet Persistence**: DuckDB-inspired columnar storage for graphs
- **CPU Algorithms**: BFS, PageRank, Louvain clustering, pattern matching
- **GPU Acceleration**: WGSL compute shaders for BFS and PageRank (10-250× faster)
- **Memory Paging**: Process graphs larger than VRAM with automatic tiling
- **Zero SATD**: Production-quality code with 90%+ test coverage

## Quick Example

```rust
use trueno_graph::{CsrGraph, NodeId, pagerank, bfs};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build graph from edge list
    let mut graph = CsrGraph::new();
    graph.add_edge(NodeId(0), NodeId(1), 1.0)?;  // main → parse_args
    graph.add_edge(NodeId(0), NodeId(2), 1.0)?;  // main → validate

    // Query neighbors (O(1) via CSR indexing)
    let callees = graph.outgoing_neighbors(NodeId(0))?;
    assert_eq!(callees.len(), 2);

    // Run PageRank to find important functions
    let scores = pagerank(&graph, 20, 1e-6)?;

    // Run BFS to find reachable nodes
    let reachable = bfs(&graph, NodeId(0))?;

    // Save to disk
    graph.write_parquet("graph.parquet").await?;

    Ok(())
}
```

## Performance

### CPU Performance

| Operation | Graph Size | Time | vs NetworkX |
|-----------|-----------|------|-------------|
| BFS Traversal | 1K nodes | ~40μs | **33× faster** |
| PageRank (20 iter) | 1K nodes | ~500μs | **96× faster** |
| CSR Construction | 5K nodes | ~500μs | N/A |

### GPU Performance

| Operation | Graph Size | GPU Time | Speedup vs NetworkX |
|-----------|-----------|----------|---------------------|
| GPU BFS | 1K nodes | ~50μs | **24×** |
| GPU BFS | 5K nodes | ~200μs | **30×** |
| GPU PageRank | 1K nodes | ~500μs | **30×** |

## Book Structure

- **[Getting Started](./getting-started/installation.md)**: Installation and first steps
- **[Architecture](./architecture/csr-format.md)**: Design and implementation details
- **[GPU Acceleration](./gpu/memory-paging.md)**: GPU algorithms and memory paging
- **[Examples](./examples/overview.md)**: Real-world use cases
- **[Appendix](./appendix/academic-references.md)**: Citations, benchmarks, FAQ

## License

MIT License - Built by [Pragmatic AI Labs](https://paiml.com)

---

**Ready to get started?** Head to [Installation](./getting-started/installation.md) to begin!
