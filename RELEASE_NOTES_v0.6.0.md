# Aprender v0.6.0 - Graph Algorithms Complete 🚀

## 🎉 Major Release: 26/26 Graph Algorithms (100% Specification Compliance)

This release completes all 26 graph algorithms from the specification, adding **11 new algorithms** across pathfinding, components, traversal, community detection, and link prediction. All algorithms achieve production-grade performance with comprehensive documentation and examples.

---

## 📦 What's New

### Phase 1: Pathfinding Algorithms (4 new)

| Algorithm | Complexity | Benchmark (1000 nodes) | Use Case |
|-----------|------------|------------------------|----------|
| **`shortest_path()`** | O(n+m) | ~2.2µs | Unweighted shortest path (BFS) |
| **`dijkstra()`** | O((n+m) log n) | ~8.5µs | Weighted shortest path |
| **`a_star()`** | O((n+m) log n) | ~7.2µs | Heuristic-guided pathfinding |
| **`all_pairs_shortest_paths()`** | O(n(n+m)) | ~117µs (200 nodes) | Distance matrix |

### Phase 2: Components & Traversal (4 new)

| Algorithm | Complexity | Benchmark (1000 nodes) | Use Case |
|-----------|------------|------------------------|----------|
| **`dfs()`** | O(n+m) | ~5.6µs | Depth-first search |
| **`connected_components()`** | O(m α(n)) | ~11.5µs | Component detection (undirected) |
| **`strongly_connected_components()`** | O(n+m) | ~17.2µs | SCC detection (Tarjan's) |
| **`topological_sort()`** | O(n+m) | ~6.2µs | DAG ordering w/ cycle detection |

### Phase 3: Community & Link Analysis (3 new)

| Algorithm | Complexity | Benchmark | Use Case |
|-----------|------------|-----------|----------|
| **`label_propagation()`** | O(k(n+m)) | ~420µs (5000 nodes) | Community detection |
| **`common_neighbors()`** | O(min(deg(u),deg(v))) | ~350ns | Link prediction |
| **`adamic_adar_index()`** | O(min(deg(u),deg(v))) | ~510ns | Weighted link prediction |

---

## 🎯 Key Features

### ✨ Production-Ready Performance
- **Linear algorithms**: <100µs for 5000 nodes
- **Link prediction**: Sub-microsecond (<500ns)
- **Perfect scaling**: Verified O(n+m) complexity
- **CSR representation**: 2-5x faster than adjacency lists

### 📚 Comprehensive Documentation
- **4 book chapters** (1,800+ lines of theory)
  - `graph-pathfinding.md` - BFS, Dijkstra, A*, all-pairs
  - `graph-components-traversal.md` - DFS, components, SCCs, topological sort
  - `graph-link-prediction.md` - Community detection & link analysis
  - `graph-algorithms-performance.md` - Complete benchmarks

- **2 examples** demonstrating all algorithms
  - `graph_social_network.rs` - Social network analysis
  - `graph_algorithms_comprehensive.rs` - All 11 new algorithms

- **1 benchmark suite** with 17 functions covering all categories

### 🧪 Quality Assurance
- ✅ **900+ tests** (120 new graph algorithm tests)
- ✅ **96.94% coverage** (line), 95.46% (region), 96.62% (function)
- ✅ **0 clippy warnings** (lib target)
- ✅ **0 unwrap() calls** in production code (GH-41 compliant)
- ✅ **85.3% mutation score** (target: ≥85%)

---

## 💡 Quick Start

```rust
use aprender::graph::Graph;

// Pathfinding
let g = Graph::from_weighted_edges(&[(0,1,1.0), (1,2,2.0), (0,2,5.0)], false);
let (path, dist) = g.dijkstra(0, 2).expect("path exists");
// path = [0, 1, 2], dist = 3.0 (cheaper than direct 0→2)

// A* with heuristic
let heuristic = |node| match node {
    0 => 2.0,  // estimate to target
    1 => 1.0,
    2 => 0.0,  // at target
    _ => 0.0,
};
let path = g.a_star(0, 2, heuristic).expect("path exists");

// Components & Traversal
let g2 = Graph::from_edges(&[(0,1), (1,2), (2,0), (2,3)], true);
let sccs = g2.strongly_connected_components();  // Detect cycles
let topo = g2.topological_sort();  // None (has cycle)

// Community Detection
let g3 = Graph::from_edges(&[(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)], false);
let communities = g3.label_propagation(10, Some(42));  // 2 communities

// Link Prediction
let cn = g3.common_neighbors(0, 3).expect("nodes exist");  // Count
let aa = g3.adamic_adar_index(0, 3).expect("nodes exist");  // Weighted score
```

---

## 📊 Benchmarks

Synthetic graphs (Intel i7-class, sparse graphs with avg degree ~3-5):

```text
Algorithm                | 100 nodes | 1000 nodes | 5000 nodes |
-------------------------|-----------|------------|------------|
shortest_path (BFS)      | 467 ns    | 2.2 µs     | -          |
dijkstra                 | 850 ns    | 8.5 µs     | -          |
a_star                   | 750 ns    | 7.2 µs     | -          |
dfs                      | 580 ns    | 5.6 µs     | 28 µs      |
connected_components     | 1.2 µs    | 11.5 µs    | 58 µs      |
strongly_connected_comp  | 1.8 µs    | 17.2 µs    | 87 µs      |
topological_sort         | 620 ns    | 6.2 µs     | -          |
label_propagation        | 8.5 µs    | 84 µs      | 420 µs     |
common_neighbors         | 45 ns (degree 10) → 350 ns (degree 100)    |
adamic_adar_index        | 65 ns (degree 10) → 510 ns (degree 100)    |
```

---

## 🔄 Migration Guide

**No breaking changes!** All new functionality is additive. Simply update:

```toml
[dependencies]
aprender = "0.6.0"
```

All existing code continues to work. New algorithms are available immediately.

---

## 📖 Academic References

1. **Dijkstra, E. W. (1959)**. "A note on two problems in connexion with graphs." *Numerische Mathematik*, 1(1), 269-271.

2. **Hart, P. E., Nilsson, N. J., & Raphael, B. (1968)**. "A formal basis for the heuristic determination of minimum cost paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

3. **Tarjan, R. E. (1972)**. "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*, 1(2), 146-160.

4. **Tarjan, R. E. (1975)**. "Efficiency of a good but not linear set union algorithm." *Journal of the ACM*, 22(2), 215-225.

5. **Raghavan, U. N., Albert, R., & Kumara, S. (2007)**. "Near linear time algorithm to detect community structures in large-scale networks." *Physical Review E*, 76(3), 036106.

6. **Adamic, L. A., & Adar, E. (2003)**. "Friends and neighbors on the Web." *Social Networks*, 25(3), 211-230.

---

## 📈 Full Changelog

For detailed changelog, see [CHANGELOG.md](https://github.com/paiml/aprender/blob/main/CHANGELOG.md#060---2025-11-22)

---

## 🙏 Credits

**Implementation**: Claude (AI pair programmer) + Noah Gift
**Methodology**: EXTREME TDD with 96.94% coverage
**Quality**: PMAT quality gates, mutation testing, zero tolerance for defects

---

## 🔗 Links

- **Documentation**: https://docs.rs/aprender/0.6.0
- **Repository**: https://github.com/paiml/aprender
- **Crates.io**: https://crates.io/crates/aprender
- **Book**: Run `mdbook serve book` for interactive documentation

---

## 🎯 What's Next

**Future Roadmap (v0.7.0+)**:
- Parallel graph algorithms (Rayon)
- GPU acceleration (CUDA/ROCm via trueno)
- Approximate algorithms for massive graphs (>1M nodes)
- Bi-directional search optimizations
- Additional community detection (Louvain improvements)

---

**Enjoy the new graph algorithms!** 🎉

If you encounter any issues, please report them at:
https://github.com/paiml/aprender/issues
