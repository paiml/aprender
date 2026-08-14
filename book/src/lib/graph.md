<!-- PCU: lib-graph | contract: contracts/apr-page-lib-graph-v1.yaml -->

# Module: `aprender::graph`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/graph/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/graph/mod.rs) or directory.

## Example

```rust
use aprender::graph::{Graph, GraphCentrality};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::graph` is the classical graph-algorithms layer, built on a
cache-friendly Compressed Sparse Row representation. It exposes a `Graph`
type for directed and undirected graphs (weighted or unweighted), shortest
paths (Dijkstra-based `shortest_path`), centrality measures (degree,
PageRank, betweenness, closeness, eigenvector, Katz, harmonic) via the
`GraphCentrality` extension trait, plus community detection (Louvain) and
graph metrics (`density`, `diameter`, `clustering_coefficient`,
`assortativity`).

## Key types

| Type | Description |
|------|-------------|
| `Graph` | CSR-backed graph. Constructors: `new(directed)`, `from_edges(...)`, `from_weighted_edges(...)`. |
| `Edge` | Source / target / optional weight tuple used at construction time. |
| `NodeId` | Type alias `usize` for contiguous node ids. |
| `GraphCentrality` | Extension trait providing `degree_centrality`, `pagerank`, `betweenness_centrality`, etc. |

Top-level `Graph` methods include `num_nodes`, `num_edges`, `neighbors`,
`shortest_path`, `louvain`, `modularity`, `density`, `diameter`,
`clustering_coefficient`, `assortativity`.

## Usage patterns

### Pattern 1: Build a triangle and measure centrality

```rust
use aprender::graph::{Graph, GraphCentrality};

// Undirected triangle 0 — 1 — 2 — 0
let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
assert_eq!(g.num_nodes(), 3);

let dc = g.degree_centrality();
assert_eq!(dc.len(), 3);

let pr = g.pagerank(0.85, 100, 1e-6).expect("pagerank converges");
println!("pagerank scores: {:?}", pr);
```

### Pattern 2: Shortest path and Louvain communities

```rust
use aprender::graph::Graph;

let g = Graph::from_weighted_edges(&[
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 1.0),
    (0, 3, 5.0),
], false);

let path = g.shortest_path(0, 3);
assert!(path.is_some(), "0 and 3 are connected");

let communities = g.louvain();
println!("found {} communities", communities.len());
```

## See also

- [`gnn`](gnn.md) — graph neural network layers (GCN, GAT, GIN, SAGE) that consume graph topology
- [`primitives`](primitives.md) — `Matrix` / `Vector` for adjacency-matrix views
- [`mining`](mining.md) — frequent-pattern mining on graph-like transactional data

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
