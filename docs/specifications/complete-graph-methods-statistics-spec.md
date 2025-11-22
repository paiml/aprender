# Complete Graph Methods and Statistics Specification

**Version**: 2.0.0
**Status**: Final
**Date**: 2025-11-22
**Authors**: Pragmatic AI Labs Team, Claude (AI pair programmer)
**Review Status**: UC Berkeley & Neo4j Graph Data Science standards

## Abstract

This specification defines a comprehensive set of graph algorithms and statistical methods for the **Aprender** library, achieving 80%+ coverage of industry-standard graph operations based on UC Berkeley CS curriculum and Neo4j Graph Data Science library. The implementation follows EXTREME TDD methodology with ≥95% test coverage, ≥85% mutation score, and PMAT quality gates.

## 1. Introduction

### 1.1 Motivation

Modern graph analytics requires a comprehensive toolkit spanning centrality measures, pathfinding, community detection, and structural analysis [1,2]. Current Rust graph libraries either lack breadth (petgraph focuses on algorithms, not analytics) or integration with ML workflows. This specification bridges that gap with 25 core graph methods covering:

1. **Centrality algorithms** (7 methods) - Node importance ranking
2. **Pathfinding algorithms** (4 methods) - Shortest path computation
3. **Community detection** (3 methods) - Graph clustering
4. **Structural analysis** (6 methods) - Graph properties
5. **Traversal algorithms** (3 methods) - Graph exploration
6. **Link analysis** (2 methods) - Prediction and similarity

### 1.2 Standards Compliance

**UC Berkeley CS 61B/170**: Graph algorithms curriculum [3]
- BFS/DFS traversal
- Dijkstra's algorithm
- MST (Kruskal/Prim)
- Strongly connected components

**Neo4j Graph Data Science**: Production-grade analytics [4]
- Centrality algorithms (PageRank, betweenness, closeness)
- Community detection (Louvain, label propagation)
- Pathfinding (shortest path, A*)
- Link prediction (common neighbors, Adamic-Adar)

**NetworkX API**: Python de facto standard [5]
- Consistent naming conventions
- Return type patterns
- Error handling semantics

### 1.3 Current Implementation Status (v0.5.0)

**Implemented (15/25 = 60%)**:
```rust
// Centrality (7/7) ✅
degree_centrality() -> HashMap<NodeId, f64>
betweenness_centrality() -> Vec<f64>
closeness_centrality() -> Vec<f64>
harmonic_centrality() -> Vec<f64>
pagerank(damping, max_iter, tol) -> Result<Vec<f64>>
eigenvector_centrality(max_iter, tol) -> Result<Vec<f64>>
katz_centrality(alpha, beta, max_iter, tol) -> Result<Vec<f64>>

// Community Detection (2/3)
louvain() -> Vec<usize>  // ✅
modularity(communities: &[usize]) -> f64  // ✅

// Structural Analysis (4/6)
density() -> f64  // ✅
diameter() -> Option<usize>  // ✅
clustering_coefficient() -> f64  // ✅
assortativity() -> f64  // ✅

// Traversal (2/3)
neighbors(v: NodeId) -> &[NodeId]  // ✅ (internal)
// BFS used internally ✅
```

**Missing (10/25 = 40%)**:
```rust
// Pathfinding (0/4) ❌
shortest_path(source, target) -> Option<Vec<NodeId>>
dijkstra(source, target) -> Option<(Vec<NodeId>, f64)>
all_pairs_shortest_paths() -> Vec<Vec<Option<usize>>>
a_star(source, target, heuristic) -> Option<Vec<NodeId>>

// Community Detection (1/3) ❌
label_propagation(max_iter) -> Vec<usize>

// Structural Analysis (2/6) ❌
connected_components() -> Vec<Vec<NodeId>>
strongly_connected_components() -> Vec<Vec<NodeId>>

// Traversal (1/3) ❌
dfs(start: NodeId) -> Vec<NodeId>

// Link Analysis (0/2) ❌
common_neighbors(u: NodeId, v: NodeId) -> usize
adamic_adar_index(u: NodeId, v: NodeId) -> f64
```

## 2. Algorithm Specifications

### 2.1 Centrality Algorithms (7/7 Complete)

#### 2.1.1 Degree Centrality ✅

**Status**: Implemented (v0.5.0)

**Formula** (Freeman 1978):
```
C_D(v) = deg(v) / (n - 1)
```

**Complexity**: O(n)

**Citation**: Freeman, L. C. (1978). Centrality in social networks conceptual clarification. *Social Networks*, 1(3), 215-239. [1]

#### 2.1.2 Betweenness Centrality ✅

**Status**: Implemented (v0.5.0)

**Formula** (Brandes 2001):
```
C_B(v) = Σ[σ_st(v) / σ_st] for all s ≠ t ≠ v
```

**Algorithm**: Brandes' algorithm with parallel BFS

**Complexity**: O(nm) serial, O(nm/p) parallel

**Citation**: Brandes, U. (2001). A faster algorithm for betweenness centrality. *Journal of Mathematical Sociology*, 25(2), 163-177. [2]

#### 2.1.3-2.1.7 Other Centralities ✅

**Implemented**: closeness, harmonic, PageRank, eigenvector, Katz (see v0.5.0)

### 2.2 Pathfinding Algorithms (0/4 Missing)

#### 2.2.1 Shortest Path (Unweighted) ❌

**Signature**:
```rust
pub fn shortest_path(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>>
```

**Algorithm**: Bidirectional BFS [6]

**Complexity**: O(n + m)

**Returns**:
- `Some(path)` - Shortest path as node sequence
- `None` - No path exists

**Citation**: Pohl, I. (1971). Bi-directional search. *Machine Intelligence*, 6, 127-140. [6]

**Test Requirements**:
- Path graph: verify shortest path length
- Disconnected graph: return None
- Self-loop: single-node path
- Multiple paths: any shortest path valid

#### 2.2.2 Dijkstra's Algorithm ❌

**Signature**:
```rust
pub fn dijkstra(&self, source: NodeId, target: NodeId) -> Option<(Vec<NodeId>, f64)>
```

**Algorithm**: Priority queue with binary heap [3]

**Complexity**: O((n + m) log n)

**Requires**: Weighted graph with non-negative weights

**Returns**:
- `Some((path, distance))` - Shortest path and total weight
- `None` - No path exists

**Citation**: Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269-271. [3]

**Error Cases**:
- Negative weights → panic with descriptive message
- Unweighted graph → use edge weight 1.0

**Test Requirements**:
- Weighted path graph: verify optimal path
- Negative weights: panic test
- Equal-weight paths: verify any optimal path

#### 2.2.3 All-Pairs Shortest Paths ❌

**Signature**:
```rust
pub fn all_pairs_shortest_paths(&self) -> Vec<Vec<Option<usize>>>
```

**Algorithm**: Repeated BFS (unweighted) or Floyd-Warshall (weighted)

**Complexity**:
- Unweighted: O(nm)
- Weighted: O(n³)

**Returns**: Distance matrix where `[i][j]` = distance from node i to node j

**Citation**: Floyd, R. W. (1962). Algorithm 97: shortest path. *Communications of the ACM*, 5(6), 345. [7]

**Optimization**: Use BFS for unweighted (faster), Floyd-Warshall for weighted

#### 2.2.4 A* Search ❌

**Signature**:
```rust
pub fn a_star<F>(&self, source: NodeId, target: NodeId, heuristic: F)
    -> Option<Vec<NodeId>>
where
    F: Fn(NodeId) -> f64
```

**Algorithm**: Priority queue with f(n) = g(n) + h(n) [8]

**Complexity**: O(m log n) with admissible heuristic

**Citation**: Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107. [8]

**Test Requirements**:
- Grid graph with Manhattan distance heuristic
- Verify optimality with admissible heuristic
- Compare path length to Dijkstra

### 2.3 Community Detection (2/3)

#### 2.3.1 Louvain Algorithm ✅

**Status**: Implemented (v0.5.0)

**Citation**: Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, 2008(10), P10008. [9]

#### 2.3.2 Label Propagation ❌

**Signature**:
```rust
pub fn label_propagation(&self, max_iter: usize) -> Vec<usize>
```

**Algorithm**:
1. Initialize: Each node gets unique label
2. Iterate: Each node adopts most common neighbor label
3. Converge: Stop when labels stabilize or max_iter reached

**Complexity**: O(k·m) where k = iterations (typically k < 10)

**Returns**: Community labels (one per node)

**Citation**: Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect community structures in large-scale networks. *Physical Review E*, 76(3), 036106. [10]

**Test Requirements**:
- Two disconnected cliques: verify perfect separation
- Karate club graph: compare to ground truth
- Convergence: verify termination within max_iter

#### 2.3.3 Modularity ✅

**Status**: Implemented (v0.5.0)

### 2.4 Structural Analysis (4/6)

#### 2.4.1-2.4.4 Existing Methods ✅

**Implemented**: density, diameter, clustering_coefficient, assortativity (v0.5.0)

#### 2.4.5 Connected Components ❌

**Signature**:
```rust
pub fn connected_components(&self) -> Vec<Vec<NodeId>>
```

**Algorithm**: Union-find with path compression

**Complexity**: O(m α(n)) where α = inverse Ackermann (near-constant)

**Returns**: Vector of components (each component is vector of node IDs)

**Citation**: Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225. [11]

**Test Requirements**:
- Disconnected graph: verify component count
- Single component: return single vector
- Tree: verify single component

#### 2.4.6 Strongly Connected Components ❌

**Signature**:
```rust
pub fn strongly_connected_components(&self) -> Vec<Vec<NodeId>>
```

**Algorithm**: Tarjan's algorithm (single DFS pass) or Kosaraju's (two DFS passes)

**Complexity**: O(n + m)

**Requires**: Directed graph

**Returns**: Vector of SCCs (topologically sorted)

**Citation**: Tarjan, R. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*, 1(2), 146-160. [12]

**Test Requirements**:
- DAG: each node is separate SCC
- Cycle graph: entire graph is one SCC
- Two cycles connected: verify two SCCs

### 2.5 Traversal Algorithms (2/3)

#### 2.5.1 Breadth-First Search ✅

**Status**: Implemented internally (v0.5.0)

**Promote to public API**:
```rust
pub fn bfs(&self, start: NodeId) -> Vec<NodeId>
```

#### 2.5.2 Depth-First Search ❌

**Signature**:
```rust
pub fn dfs(&self, start: NodeId) -> Vec<NodeId>
```

**Algorithm**: Recursive DFS with visited tracking

**Complexity**: O(n + m)

**Returns**: Nodes in DFS order

**Test Requirements**:
- Tree: verify single path to leaves
- Cycle: verify all nodes visited once
- Disconnected: only reachable nodes

#### 2.5.3 Topological Sort ❌

**Signature**:
```rust
pub fn topological_sort(&self) -> Result<Vec<NodeId>, String>
```

**Algorithm**: DFS with post-order reversal or Kahn's algorithm

**Complexity**: O(n + m)

**Requires**: DAG (directed acyclic graph)

**Returns**:
- `Ok(order)` - Valid topological ordering
- `Err(msg)` - Graph contains cycle

**Test Requirements**:
- DAG: verify valid ordering
- Cycle: return error
- Multiple valid orderings: accept any

### 2.6 Link Analysis (0/2)

#### 2.6.1 Common Neighbors ❌

**Signature**:
```rust
pub fn common_neighbors(&self, u: NodeId, v: NodeId) -> usize
```

**Formula**:
```
CN(u,v) = |N(u) ∩ N(v)|
```

**Complexity**: O(deg(u) + deg(v))

**Use Case**: Link prediction baseline

**Test Requirements**:
- Triangle: 1 common neighbor
- No overlap: 0 common neighbors
- Complete graph: n-2 common neighbors

#### 2.6.2 Adamic-Adar Index ❌

**Signature**:
```rust
pub fn adamic_adar_index(&self, u: NodeId, v: NodeId) -> f64
```

**Formula**:
```
AA(u,v) = Σ[1 / log(deg(w))] for w in N(u) ∩ N(v)
```

**Complexity**: O(deg(u) + deg(v))

**Citation**: Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the web. *Social Networks*, 25(3), 211-230. [13]

**Test Requirements**:
- High-degree common neighbor: lower contribution
- Low-degree common neighbor: higher contribution
- No common neighbors: return 0.0

## 3. Implementation Requirements

### 3.1 CSR Representation (Existing)

**Memory Layout**:
```rust
pub struct Graph {
    row_ptr: Vec<usize>,          // O(n) space
    col_indices: Vec<NodeId>,     // O(m) space
    edge_weights: Vec<f64>,       // O(m) space
    is_directed: bool,
    n_nodes: usize,
    n_edges: usize,
}
```

**Benefits**:
- 50-70% memory reduction vs HashMap
- Cache-friendly sequential access
- SIMD-friendly neighbor iteration

### 3.2 Error Handling

**Pattern**:
```rust
pub enum GraphError {
    NodeNotFound(NodeId),
    InvalidGraph(String),
    NegativeWeight(f64),
    NotDAG,
    Disconnected,
}

pub type GraphResult<T> = Result<T, GraphError>;
```

### 3.3 Testing Requirements

**Coverage Targets**:
- Line coverage: ≥95%
- Branch coverage: ≥90%
- Mutation score: ≥85%

**Test Categories** (60% unit, 30% property, 10% integration):
1. **Unit tests**: Individual algorithm correctness
2. **Property tests**: Graph invariants (proptest with 10K cases)
3. **Integration tests**: Real-world graphs (Karate club, Les Misérables)

**Benchmark Graphs**:
- Small: 10-100 nodes (unit tests)
- Medium: 1K-10K nodes (performance tests)
- Large: 100K+ nodes (scalability tests)

### 3.4 Performance Requirements

**Targets** (Intel i7-8700K, 6 cores):

| Algorithm | 10K nodes | 100K nodes | 1M nodes |
|-----------|-----------|------------|----------|
| Shortest path (BFS) | <1 ms | 10 ms | 150 ms |
| Dijkstra | 5 ms | 80 ms | 1.2 s |
| Connected components | <1 ms | 15 ms | 200 ms |
| Label propagation | 8 ms | 120 ms | 2.0 s |
| DFS/BFS | <1 ms | 8 ms | 100 ms |

**Parallelization**:
- Betweenness centrality: ✅ (Rayon)
- All-pairs shortest paths: ✅ (independent BFS)
- Label propagation: ❌ (inherently sequential)

## 4. Quality Gates (PMAT Integration)

### 4.1 Pre-Commit Checks

```bash
# Tier 1 (<1s)
cargo fmt --check
cargo clippy -- -D warnings

# Tier 2 (<5s)
cargo test --lib
pmat analyze complexity --max-cyclomatic 10

# Tier 3 (<5min)
cargo test --all
make coverage  # Target: ≥95%
```

### 4.2 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
- name: Graph Algorithm Tests
  run: |
    cargo test graph::tests
    cargo test --test graph_integration

- name: Mutation Testing
  run: cargo mutants --file src/graph/mod.rs --timeout 300

- name: Benchmarks
  run: cargo bench --bench graph_benchmarks
```

## 5. Documentation Requirements

### 5.1 Book Chapters

**Required Chapters** (`book/src/ml-fundamentals/`):
- ✅ `graph-algorithms.md` (existing, update for new methods)
- ❌ `graph-pathfinding.md` (new: Dijkstra, A*)
- ❌ `graph-community-detection.md` (expand: add label propagation)
- ❌ `graph-link-prediction.md` (new: common neighbors, Adamic-Adar)

**Example Requirements**:
- Every algorithm MUST have runnable example
- Examples MUST be documented in book
- Examples MUST pass doctests

### 5.2 API Documentation

**Minimum Requirements**:
- Algorithm description
- Time/space complexity
- Mathematical formula (if applicable)
- Example usage
- References to peer-reviewed papers

**Example**:
```rust
/// Compute shortest path using Dijkstra's algorithm.
///
/// # Algorithm
/// Uses priority queue with binary heap for O((n+m) log n) complexity [3].
///
/// # Arguments
/// * `source` - Starting node
/// * `target` - Destination node
///
/// # Returns
/// * `Some((path, distance))` - Shortest path and total weight
/// * `None` - No path exists
///
/// # Panics
/// Panics if graph contains negative edge weights.
///
/// # Examples
/// ```
/// use aprender::graph::Graph;
///
/// let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)];
/// let g = Graph::from_weighted_edges(&edges, false);
/// let (path, dist) = g.dijkstra(0, 2).unwrap();
/// assert_eq!(dist, 3.0); // 0->1->2 is shorter than 0->2
/// ```
///
/// # References
/// [3] Dijkstra, E. W. (1959). A note on two problems...
pub fn dijkstra(&self, source: NodeId, target: NodeId)
    -> Option<(Vec<NodeId>, f64)>
```

## 6. Implementation Roadmap

### Phase 1: Pathfinding (Sprint 1-2, 2 weeks)
- [ ] `shortest_path()` - Bidirectional BFS
- [ ] `dijkstra()` - Priority queue implementation
- [ ] `all_pairs_shortest_paths()` - Repeated BFS
- [ ] `a_star()` - Heuristic search
- [ ] Tests: 100+ unit tests, property tests for path optimality
- [ ] Book chapter: `graph-pathfinding.md`

### Phase 2: Components & Traversal (Sprint 3, 1 week)
- [ ] `connected_components()` - Union-find
- [ ] `strongly_connected_components()` - Tarjan's algorithm
- [ ] `dfs()` - Depth-first search
- [ ] `topological_sort()` - DAG ordering
- [ ] Tests: 80+ unit tests, cycle detection properties
- [ ] Update existing book chapter

### Phase 3: Community & Link Analysis (Sprint 4, 1 week)
- [ ] `label_propagation()` - Iterative label spreading
- [ ] `common_neighbors()` - Set intersection
- [ ] `adamic_adar_index()` - Weighted common neighbors
- [ ] Tests: 60+ unit tests, community quality metrics
- [ ] Book chapter: `graph-link-prediction.md`

### Phase 4: Integration & Optimization (Sprint 5, 1 week)
- [ ] Benchmark suite for all algorithms
- [ ] Parallel versions where applicable
- [ ] Real-world graph examples (Karate, Zachary, etc.)
- [ ] Final documentation pass
- [ ] Coverage validation (≥95%)

**Total Effort**: 5 sprints = 5 weeks (80-100 hours)

**Target Completion**: v0.6.0 (80% graph coverage)

## 7. References

[1] Freeman, L. C. (1978). Centrality in social networks conceptual clarification. *Social Networks*, 1(3), 215-239.

[2] Brandes, U. (2001). A faster algorithm for betweenness centrality. *Journal of Mathematical Sociology*, 25(2), 163-177.

[3] Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269-271.

[4] Neo4j Graph Data Science. (2023). *Neo4j Graph Data Science Library Documentation*. Neo4j, Inc.

[5] Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. *Proceedings of the 7th Python in Science Conference*, pp. 11-15.

[6] Pohl, I. (1971). Bi-directional search. *Machine Intelligence*, 6, 127-140.

[7] Floyd, R. W. (1962). Algorithm 97: shortest path. *Communications of the ACM*, 5(6), 345.

[8] Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

[9] Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, 2008(10), P10008.

[10] Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect community structures in large-scale networks. *Physical Review E*, 76(3), 036106.

[11] Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225.

[12] Tarjan, R. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*, 1(2), 146-160.

[13] Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the web. *Social Networks*, 25(3), 211-230.

## Appendix A: Coverage Matrix

| Category | Method | Status | Tests | Docs | Perf |
|----------|--------|--------|-------|------|------|
| **Centrality** | | **7/7** ✅ | | | |
| | degree_centrality | ✅ | ✅ | ✅ | ✅ |
| | betweenness_centrality | ✅ | ✅ | ✅ | ✅ |
| | closeness_centrality | ✅ | ✅ | ✅ | - |
| | harmonic_centrality | ✅ | ✅ | ✅ | - |
| | pagerank | ✅ | ✅ | ✅ | ✅ |
| | eigenvector_centrality | ✅ | ✅ | ✅ | - |
| | katz_centrality | ✅ | ✅ | ✅ | - |
| **Pathfinding** | | **0/4** ❌ | | | |
| | shortest_path | ❌ | - | - | - |
| | dijkstra | ❌ | - | - | - |
| | all_pairs_shortest_paths | ❌ | - | - | - |
| | a_star | ❌ | - | - | - |
| **Community** | | **2/3** | | | |
| | louvain | ✅ | ✅ | ✅ | - |
| | label_propagation | ❌ | - | - | - |
| | modularity | ✅ | ✅ | ✅ | - |
| **Structural** | | **4/6** | | | |
| | density | ✅ | ✅ | ✅ | - |
| | diameter | ✅ | ✅ | ✅ | - |
| | clustering_coefficient | ✅ | ✅ | ✅ | - |
| | assortativity | ✅ | ✅ | ✅ | - |
| | connected_components | ❌ | - | - | - |
| | strongly_connected_components | ❌ | - | - | - |
| **Traversal** | | **2/3** | | | |
| | bfs | ✅ | ✅ | - | - |
| | dfs | ❌ | - | - | - |
| | topological_sort | ❌ | - | - | - |
| **Link Analysis** | | **0/2** ❌ | | | |
| | common_neighbors | ❌ | - | - | - |
| | adamic_adar_index | ❌ | - | - | - |
| **TOTAL** | | **15/25 (60%)** | | | |

**Target v0.6.0**: 20/25 (80%) - Add pathfinding (4), components (2), traversal (1), link analysis (2), label propagation (1)

## Appendix B: Comparison to Industry Standards

| Library | Language | Algorithms | Performance | ML Integration |
|---------|----------|-----------|-------------|----------------|
| NetworkX | Python | 400+ | Slow (pure Python) | Good (NumPy) |
| igraph | C/Python/R | 200+ | Fast (C core) | Good |
| Neo4j GDS | Java/Cypher | 65+ | Fast (parallel) | Excellent |
| petgraph | Rust | 30+ | Fast | Poor |
| **Aprender v0.6.0** | **Rust** | **20+** | **Fast (SIMD)** | **Excellent (Trueno)** |

**Competitive Advantage**: Only Rust library combining graph analytics with SIMD-optimized ML primitives.
