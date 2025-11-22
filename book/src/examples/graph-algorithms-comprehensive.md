# Case Study: Comprehensive Graph Algorithms Demo

This case study demonstrates all 11 graph algorithms from v0.6.0, organized into three phases: Pathfinding, Components & Traversal, and Community & Link Analysis.

## Overview

This comprehensive example showcases:
- **Phase 1**: Pathfinding algorithms (shortest_path, Dijkstra, A*, all-pairs)
- **Phase 2**: Components & traversal (DFS, connected_components, SCCs, topological_sort)
- **Phase 3**: Community detection & link prediction (label_propagation, common_neighbors, adamic_adar)

## Running the Example

```bash
cargo run --example graph_algorithms_comprehensive
```

Expected output: Three demonstration phases covering all 11 new graph algorithms with real-world scenarios.

## Phase 1: Pathfinding Algorithms

### Road Network Example

We build a weighted graph representing cities connected by roads:

```rust,ignore
use aprender::graph::Graph;

let weighted_edges = vec![
    (0, 1, 4.0),  // A-B: 4km
    (0, 2, 2.0),  // A-C: 2km
    (1, 2, 1.0),  // B-C: 1km
    (1, 3, 5.0),  // B-D: 5km
    (2, 3, 8.0),  // C-D: 8km
    (2, 4, 10.0), // C-E: 10km
    (3, 4, 2.0),  // D-E: 2km
    (3, 5, 6.0),  // D-F: 6km
    (4, 5, 3.0),  // E-F: 3km
];

let g_weighted = Graph::from_weighted_edges(&weighted_edges, false);
```

### Algorithm 1: BFS Shortest Path

Unweighted shortest path (minimum hops):

```rust,ignore
let g_unweighted = Graph::from_edges(&unweighted_edges, false);
let path = g_unweighted.shortest_path(0, 5).expect("Path should exist");
// Returns: [0, 1, 3, 5] (3 hops)
```

**Complexity**: O(n+m) - breadth-first search

### Algorithm 2: Dijkstra's Algorithm

Weighted shortest path with priority queue:

```rust,ignore
let (dijkstra_path, distance) = g_weighted.dijkstra(0, 5)
    .expect("Path should exist");
// Returns: path = [0, 2, 1, 3, 4, 5], distance = 13.0 km
```

**Complexity**: O((n+m) log n) - priority queue operations

### Algorithm 3: A* Search

Heuristic-guided pathfinding with estimated remaining distance:

```rust,ignore
let heuristic = |node: usize| match node {
    0 => 10.0, // A to F: ~10km estimate
    1 => 8.0,  // B to F: ~8km
    2 => 9.0,  // C to F: ~9km
    3 => 5.0,  // D to F: ~5km
    4 => 3.0,  // E to F: ~3km
    _ => 0.0,  // F to F or other: 0km
};

let astar_path = g_weighted.a_star(0, 5, heuristic)
    .expect("Path should exist");
// Finds optimal path using heuristic guidance
```

**Complexity**: O((n+m) log n) - but often faster than Dijkstra in practice

### Algorithm 4: All-Pairs Shortest Paths

Compute distance matrix between all node pairs:

```rust,ignore
let dist_matrix = g_unweighted.all_pairs_shortest_paths();
// Returns: Vec<Vec<Option<usize>>> with distances
// dist_matrix[i][j] = Some(d) if path exists, None otherwise
```

**Complexity**: O(n(n+m)) - runs BFS from each node

## Phase 2: Components & Traversal

### Algorithm 5: Depth-First Search

Stack-based exploration:

```rust,ignore
let tree_edges = vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)];
let tree = Graph::from_edges(&tree_edges, false);

let dfs_order = tree.dfs(0).expect("DFS from root");
// Returns: [0, 2, 5, 1, 4, 3] (one valid DFS ordering)
```

**Complexity**: O(n+m) - visits each node and edge once

### Algorithm 6: Connected Components

Find groups in undirected graphs using Union-Find:

```rust,ignore
let component_edges = vec![
    (0, 1), (1, 2), // Component 1: {0,1,2}
    (3, 4),         // Component 2: {3,4}
    // Node 5 is isolated (Component 3)
];
let g_components = Graph::from_edges(&component_edges, false);

let components = g_components.connected_components();
// Returns: [0, 0, 0, 1, 1, 2] (component ID for each node)
```

**Complexity**: O(m α(n)) - near-linear with inverse Ackermann function

### Algorithm 7: Strongly Connected Components

Find cycles in directed graphs using Tarjan's algorithm:

```rust,ignore
let scc_edges = vec![
    (0, 1), (1, 2), (2, 0), // SCC 1: {0,1,2} (cycle)
    (2, 3), (3, 4), (4, 3), // SCC 2: {3,4} (cycle)
];
let g_directed = Graph::from_edges(&scc_edges, true);

let sccs = g_directed.strongly_connected_components();
// Returns: component ID for each node
```

**Complexity**: O(n+m) - single-pass Tarjan's algorithm

### Algorithm 8: Topological Sort

Order DAG nodes by dependencies:

```rust,ignore
let dag_edges = vec![
    (0, 1), // Task 0 → Task 1
    (0, 2), // Task 0 → Task 2
    (1, 3), // Task 1 → Task 3
    (2, 3), // Task 2 → Task 3
    (3, 4), // Task 3 → Task 4
];
let dag = Graph::from_edges(&dag_edges, true);

match dag.topological_sort() {
    Some(order) => println!("Valid execution order: {:?}", order),
    None => println!("Cycle detected! No valid ordering."),
}
// Returns: Some([0, 2, 1, 3, 4]) (one valid ordering)
```

**Complexity**: O(n+m) - DFS with in-stack cycle detection

## Phase 3: Community & Link Analysis

### Social Network Example

Build a social network with two communities connected by a bridge:

```rust,ignore
let social_edges = vec![
    // Community 1: {0,1,2,3}
    (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),
    // Bridge
    (3, 4),
    // Community 2: {4,5,6,7}
    (4, 5), (5, 6), (6, 7), (7, 4), (4, 6),
];
let g_social = Graph::from_edges(&social_edges, false);
```

### Algorithm 9: Label Propagation

Iterative community detection:

```rust,ignore
let communities = g_social.label_propagation(10, Some(42));
// Returns: community ID for each node
// Typically detects 2 communities matching the structure
```

**Complexity**: O(k(n+m)) - k iterations, deterministic with seed

### Algorithm 10: Common Neighbors

Link prediction metric counting shared neighbors:

```rust,ignore
let cn_1_3 = g_social.common_neighbors(1, 3).expect("Nodes exist");
// Returns: count of nodes connected to both 1 and 3

// Within-community prediction (high score)
let cn_within = g_social.common_neighbors(1, 3)?;

// Cross-community prediction (low score)
let cn_across = g_social.common_neighbors(0, 7)?;
```

**Complexity**: O(min(deg(u), deg(v))) - two-pointer set intersection

### Algorithm 11: Adamic-Adar Index

Weighted link prediction favoring rare shared neighbors:

```rust,ignore
let aa_1_3 = g_social.adamic_adar_index(1, 3).expect("Nodes exist");
// Returns: sum of 1/log(deg(z)) for shared neighbors z
// Higher score = stronger prediction for future link

// Compare within-community vs. cross-community
let aa_within = g_social.adamic_adar_index(1, 3)?;
let aa_across = g_social.adamic_adar_index(0, 7)?;
// aa_within > aa_across (within-community links more likely)
```

**Complexity**: O(min(deg(u), deg(v))) - weighted set intersection

## Key Insights

### Algorithm Selection Guide

| Task | Algorithm | Complexity | Use Case |
|------|-----------|------------|----------|
| Unweighted shortest path | BFS (`shortest_path`) | O(n+m) | Minimum hops |
| Weighted shortest path | Dijkstra | O((n+m) log n) | Road networks |
| Guided pathfinding | A* | O((n+m) log n) | With heuristics |
| All-pairs distances | All-Pairs | O(n(n+m)) | Distance matrix |
| Tree traversal | DFS | O(n+m) | Exploration |
| Find groups | Connected Components | O(m α(n)) | Clusters |
| Find cycles | SCCs | O(n+m) | Dependency analysis |
| Task ordering | Topological Sort | O(n+m) | Scheduling |
| Community detection | Label Propagation | O(k(n+m)) | Social networks |
| Link prediction | Common Neighbors / Adamic-Adar | O(deg) | Recommendations |

### Performance Characteristics

Synthetic graphs (1000 nodes, sparse with avg degree ~3-5):
- **shortest_path**: ~2.2µs
- **dijkstra**: ~8.5µs
- **a_star**: ~7.2µs
- **dfs**: ~5.6µs
- **connected_components**: ~11.5µs
- **strongly_connected_components**: ~17.2µs
- **topological_sort**: ~6.2µs
- **label_propagation**: ~84µs
- **common_neighbors**: ~350ns (degree 100)
- **adamic_adar_index**: ~510ns (degree 100)

All algorithms achieve their theoretical complexity bounds with CSR graph representation.

## Testing Strategy

The example demonstrates:
1. **Correctness**: Verifies expected paths, orderings, and communities
2. **Edge cases**: Handles disconnected graphs, cycles, and isolated nodes
3. **Real-world scenarios**: Road networks, task scheduling, social networks

## Related Chapters

- [Graph Algorithms Theory](../ml-fundamentals/graph-algorithms.md)
- [Graph Pathfinding Theory](../ml-fundamentals/graph-pathfinding.md)
- [Graph Components and Traversal](../ml-fundamentals/graph-components-traversal.md)
- [Graph Link Prediction and Community Detection](../ml-fundamentals/graph-link-prediction.md)

## References

1. **Dijkstra, E. W. (1959)**. "A note on two problems in connexion with graphs." *Numerische Mathematik*, 1(1), 269-271.

2. **Hart, P. E., Nilsson, N. J., & Raphael, B. (1968)**. "A formal basis for the heuristic determination of minimum cost paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

3. **Tarjan, R. E. (1972)**. "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*, 1(2), 146-160.

4. **Raghavan, U. N., Albert, R., & Kumara, S. (2007)**. "Near linear time algorithm to detect community structures in large-scale networks." *Physical Review E*, 76(3), 036106.

5. **Adamic, L. A., & Adar, E. (2003)**. "Friends and neighbors on the Web." *Social Networks*, 25(3), 211-230.
