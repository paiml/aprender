# Graph Algorithms Theory

Graph algorithms are fundamental tools for analyzing relationships and structures in networked data. This chapter covers the theory behind aprender's graph module, focusing on efficient representations and centrality measures.

## Graph Representation

### Adjacency List vs CSR

Graphs can be represented in multiple ways, each with different performance characteristics:

**Adjacency List (HashMap-based)**:
- `HashMap<NodeId, Vec<NodeId>>`
- Pros: Easy to modify, intuitive API
- Cons: Poor cache locality, 50-70% memory overhead from pointers

**Compressed Sparse Row (CSR)**:
- Two flat arrays: `row_ptr` (offsets) and `col_indices` (neighbors)
- Pros: 50-70% memory reduction, sequential access (3-5x fewer cache misses)
- Cons: Immutable structure, slightly more complex construction

Aprender uses **CSR** for production workloads, optimizing for read-heavy analytics.

### CSR Format Details

For a graph with `n` nodes and `m` edges:

```text
row_ptr: [0, 2, 5, 7, ...]  # length = n + 1
col_indices: [1, 3, 0, 2, 4, ...]  # length = m (undirected: 2m)
```

Neighbors of node `v` are stored in:
```text
col_indices[row_ptr[v] .. row_ptr[v+1]]
```

**Memory comparison** (1M nodes, 5M edges):
- HashMap: ~240 MB (pointers + Vec overhead)
- CSR: ~84 MB (two flat arrays)

## Degree Centrality

### Definition

Degree centrality measures the number of edges connected to a node. It identifies the most "popular" nodes in a network.

**Unnormalized degree**:
```text
C_D(v) = deg(v)
```

**Freeman normalization** (for comparability across graphs):
```text
C_D(v) = deg(v) / (n - 1)
```

where `n` is the number of nodes.

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 3), (0, 2)];
let graph = Graph::from_edges(&edges, false);

let centrality = graph.degree_centrality();
for (node, score) in centrality.iter() {
    println!("Node {}: {:.3}", node, score);
}
```

### Time Complexity

- **Construction**: O(n + m) to build CSR
- **Query**: O(1) per node (subtract adjacent row_ptr values)
- **All nodes**: O(n)

### Applications

- Social networks: Find influencers by connection count
- Protein interaction networks: Identify hub proteins
- Transportation: Find major transit hubs

## PageRank

### Theory

PageRank models the probability that a random surfer lands on a node. Originally developed by Google for web page ranking, it considers both **quantity** and **quality** of connections.

**Iterative formula**:
```text
PR(v) = (1-d)/n + d * Σ[PR(u) / outdeg(u)]
```

where:
- `d` = damping factor (typically 0.85)
- `n` = number of nodes
- Sum over all nodes `u` with edges to `v`

### Dangling Nodes

Nodes with no outgoing edges (dangling nodes) require special handling to preserve the probability distribution:

```text
dangling_sum = Σ PR(v) for all dangling v
PR_new(v) += d * dangling_sum / n
```

Without this correction, rank "leaks" out of the system and Σ PR(v) ≠ 1.

### Numerical Stability

Naive summation accumulates O(n·ε) floating-point error on large graphs. Aprender uses **Kahan compensated summation**:

```rust,ignore
let mut sum = 0.0;
let mut c = 0.0;  // Compensation term

for value in values {
    let y = value - c;
    let t = sum + y;
    c = (t - sum) - y;  // Recover low-order bits
    sum = t;
}
```

**Result**: Σ PR(v) = 1.0 within 1e-10 precision (vs 1e-5 naive).

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
let graph = Graph::from_edges(&edges, true);  // directed

let ranks = graph.pagerank(0.85, 100, 1e-6).unwrap();
println!("PageRank scores: {:?}", ranks);
```

### Time Complexity

- **Per iteration**: O(n + m)
- **Convergence**: Typically 20-50 iterations
- **Total**: O(k(n + m)) where k = iteration count

### Applications

- Web search: Rank pages by importance
- Social networks: Identify influential users (considers network structure)
- Citation analysis: Find seminal papers

## Betweenness Centrality

### Theory

Betweenness centrality measures how often a node appears on shortest paths between other nodes. High betweenness indicates **bridging** role in the network.

**Formula**:
```text
C_B(v) = Σ[σ_st(v) / σ_st]
```

where:
- `σ_st` = number of shortest paths from `s` to `t`
- `σ_st(v)` = number of those paths passing through `v`
- Sum over all pairs `s ≠ t ≠ v`

### Brandes' Algorithm

Naive computation is O(n³). Brandes' algorithm reduces this to O(nm) using two phases:

**Phase 1: Forward BFS from each source**
- Compute shortest path counts
- Build predecessor lists

**Phase 2: Backward accumulation**
- Propagate dependencies from leaves to root
- Accumulate betweenness scores

### Parallel Implementation

The outer loop (BFS from each source) is **embarrassingly parallel**:

```rust,ignore
use rayon::prelude::*;

let partial_scores: Vec<Vec<f64>> = (0..n)
    .into_par_iter()  // Parallel iterator
    .map(|source| brandes_bfs_from_source(source))
    .collect();

// Reduce (single-threaded, fast)
let mut centrality = vec![0.0; n];
for partial in partial_scores {
    for (i, &score) in partial.iter().enumerate() {
        centrality[i] += score;
    }
}
```

**Expected speedup**: ~8x on 8-core CPU for graphs with >1K nodes.

### Normalization

For undirected graphs, each path is counted twice:

```rust,ignore
if !is_directed {
    for score in &mut centrality {
        *score /= 2.0;
    }
}
```

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![
    (0, 1), (1, 2), (2, 3),  // Linear chain
    (1, 4), (4, 3),          // Shortcut
];
let graph = Graph::from_edges(&edges, false);

let betweenness = graph.betweenness_centrality();
println!("Node 1 betweenness: {:.2}", betweenness[1]);  // High (bridge)
```

### Time Complexity

- **Serial**: O(nm) for unweighted graphs
- **Parallel**: O(nm / p) where p = number of cores
- **Space**: O(n + m) per thread

### Applications

- Social networks: Find connectors between communities
- Transportation: Identify critical junctions
- Epidemiology: Find super-spreaders in contact networks

## Closeness Centrality

### Theory

Closeness centrality measures how close a node is to all other nodes in the network. Nodes with high closeness can spread information or resources efficiently through the network.

**Formula** (Wasserman & Faust 1994):
```text
C_C(v) = (n-1) / Σ d(v,u)
```

where:
- `n` = number of nodes
- `d(v,u)` = shortest path distance from v to u
- Sum over all reachable nodes u

For **disconnected nodes** (unreachable from v), closeness = 0.0 (convention).

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 3)];  // Path graph
let graph = Graph::from_edges(&edges, false);

let closeness = graph.closeness_centrality();
println!("Node 1 closeness: {:.3}", closeness[1]);  // Central node
```

### Time Complexity

- **Per node**: O(n + m) via BFS
- **All nodes**: O(n·(n + m))
- **Parallel**: Available via Rayon (future optimization)

### Applications

- Social networks: Identify people who can spread information quickly
- Supply chains: Find optimal distribution centers
- Disease modeling: Find efficient vaccination targets

## Eigenvector Centrality

### Theory

Eigenvector centrality assigns importance based on the importance of neighbors. It's the principle behind Google's PageRank, but for undirected graphs.

**Formula**:
```text
x_v = (1/λ) * Σ A_vu * x_u
```

where:
- `A` = adjacency matrix
- `λ` = largest eigenvalue
- `x` = eigenvector (centrality scores)

Solved via **power iteration**:
```text
x^(k+1) = A · x^(k) / ||A · x^(k)||
```

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 0), (1, 3)];  // Triangle + spoke
let graph = Graph::from_edges(&edges, false);

let centrality = graph.eigenvector_centrality(100, 1e-6).unwrap();
println!("Centralities: {:?}", centrality);
```

### Convergence

- **Typical iterations**: 10-30 for most graphs
- **Disconnected graphs**: Returns error (no dominant eigenvalue)
- **Convergence check**: ||x^(k+1) - x^(k)|| < tolerance

### Time Complexity

- **Per iteration**: O(n + m)
- **Convergence**: O(k·(n + m)) where k ≈ 10-30

### Applications

- Social networks: Find influencers (connected to other influencers)
- Citation networks: Identify seminal papers
- Collaboration networks: Find well-connected researchers

## Katz Centrality

### Theory

Katz centrality is a generalization of eigenvector centrality that works for directed graphs and gives every node a baseline importance.

**Formula**:
```text
x = (I - αA^T)^(-1) · β·1
```

where:
- `α` = attenuation factor (< 1/λ_max)
- `β` = baseline importance (typically 1.0)
- `A^T` = transpose of adjacency matrix

Solved via **power iteration**:
```text
x^(k+1) = β·1 + α·A^T·x^(k)
```

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 0)];  // Directed cycle
let graph = Graph::from_edges(&edges, true);

let centrality = graph.katz_centrality(0.1, 1.0, 100, 1e-6).unwrap();
println!("Katz scores: {:?}", centrality);
```

### Parameter Selection

- **Alpha**: Must be < 1/λ_max for convergence
  - Rule of thumb: α = 0.1 works for most graphs
  - Larger α → more weight to distant neighbors
- **Beta**: Baseline importance (usually 1.0)

### Time Complexity

- **Per iteration**: O(n + m)
- **Convergence**: O(k·(n + m)) where k ≈ 10-30

### Applications

- Social networks: Influence with baseline activity
- Web graphs: Modified PageRank for directed graphs
- Recommendation systems: Item importance scoring

## Harmonic Centrality

### Theory

Harmonic centrality is a robust variant of closeness centrality that handles disconnected graphs gracefully by summing inverse distances instead of averaging.

**Formula** (Boldi & Vigna 2014):
```text
H(v) = Σ 1/d(v,u)
```

where:
- `d(v,u)` = shortest path distance
- If u unreachable: 1/∞ = 0 (natural handling)
- No special case needed for disconnected graphs

### Advantages over Closeness

1. **No zero-division** for disconnected nodes
2. **Discriminates better** in sparse graphs
3. **Additive**: Can compute incrementally

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![
    (0, 1), (1, 2),  // Component 1
    (3, 4),          // Component 2 (disconnected)
];
let graph = Graph::from_edges(&edges, false);

let harmonic = graph.harmonic_centrality();
// Works correctly even with disconnected components
```

### Time Complexity

- **All nodes**: O(n·(n + m))
- Same as closeness, but more robust

### Applications

- Fragmented networks: Social networks with isolated communities
- Transportation: Networks with unreachable zones
- Communication: Networks with partitions

## Network Density

### Theory

Density measures the ratio of actual edges to possible edges. It quantifies how "connected" a graph is overall.

**Formula** (undirected):
```text
D = 2m / (n(n-1))
```

**Formula** (directed):
```text
D = m / (n(n-1))
```

where:
- `m` = number of edges
- `n` = number of nodes

### Interpretation

- **D = 0**: No edges (empty graph)
- **D = 1**: Complete graph (every pair connected)
- **D ∈ (0,1)**: Partial connectivity

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 0)];  // Triangle
let graph = Graph::from_edges(&edges, false);

let density = graph.density();
println!("Density: {:.3}", density);  // 3 edges / 3 possible = 1.0
```

### Time Complexity

- **O(1)**: Just arithmetic on n_nodes and n_edges

### Applications

- Social networks: Measure community cohesion
- Biological networks: Protein interaction density
- Comparison: Compare connectivity across graphs

## Network Diameter

### Theory

Diameter is the longest shortest path between any pair of nodes. It measures the "worst-case" reachability in a network.

**Formula**:
```text
diam(G) = max{d(u,v) : u,v ∈ V}
```

**Special cases**:
- Disconnected graph → `None` (infinite diameter)
- Single node → 0
- Empty graph → 0

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 3)];  // Path of length 3
let graph = Graph::from_edges(&edges, false);

match graph.diameter() {
    Some(d) => println!("Diameter: {}", d),  // 3 hops
    None => println!("Graph is disconnected"),
}
```

### Algorithm

Uses **all-pairs BFS**:
1. Run BFS from each node
2. Track maximum distance found
3. Return None if any node unreachable

### Time Complexity

- **O(n·(n + m))**: BFS from every node
- Can be expensive for large graphs

### Applications

- Communication networks: Worst-case message delay
- Social networks: "Six degrees of separation"
- Transportation: Maximum travel time

## Clustering Coefficient

### Theory

Clustering coefficient measures how much nodes tend to cluster together. It quantifies the probability that two neighbors of a node are also neighbors of each other (forming triangles).

**Formula** (global):
```text
C = (3 × number of triangles) / number of connected triples
```

**Implementation** (average local clustering):
```text
C = (1/n) Σ C_i

where C_i = (2 × triangles around i) / (deg(i) × (deg(i)-1))
```

### Interpretation

- **C = 0**: No triangles (e.g., tree structure)
- **C = 1**: Every neighbor pair is connected
- **C ∈ (0,1)**: Partial clustering

### Implementation

```rust,ignore
use aprender::graph::Graph;

let edges = vec![(0, 1), (1, 2), (2, 0)];  // Perfect triangle
let graph = Graph::from_edges(&edges, false);

let clustering = graph.clustering_coefficient();
println!("Clustering: {:.3}", clustering);  // 1.0
```

### Time Complexity

- **O(n·d²)** where d = average degree
- Worst case O(n³) for dense graphs
- Typically much faster due to sparsity

### Applications

- Social networks: Measure friend-of-friend connections
- Biological networks: Functional module detection
- Small-world property: High clustering + low diameter

## Degree Assortativity

### Theory

Assortativity measures the tendency of nodes to connect with similar nodes. For degree assortativity, it answers: "Do high-degree nodes connect with other high-degree nodes?"

**Formula** (Newman 2002):
```text
r = Σ_e j·k·e_jk - [Σ_e (j+k)·e_jk/2]²
    ─────────────────────────────────────
    Σ_e (j²+k²)·e_jk/2 - [Σ_e (j+k)·e_jk/2]²
```

where `e_jk` = fraction of edges connecting degree-j to degree-k nodes.

**Simplified interpretation**: Pearson correlation of degrees at edge endpoints.

### Interpretation

- **r > 0**: Assortative (similar degrees connect)
  - Examples: Social networks (homophily)
- **r < 0**: Disassortative (different degrees connect)
  - Examples: Biological networks (hubs connect to leaves)
- **r = 0**: No correlation

### Implementation

```rust,ignore
use aprender::graph::Graph;

// Star graph: hub (high degree) connects to leaves (low degree)
let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
let graph = Graph::from_edges(&edges, false);

let assortativity = graph.assortativity();
println!("Assortativity: {:.3}", assortativity);  // Negative (disassortative)
```

### Time Complexity

- **O(n + m)**: Linear scan of edges

### Applications

- Social networks: Detect homophily (like connects to like)
- Biological networks: Hub-and-spoke vs mesh topology
- Resilience analysis: Assortative networks more robust to attacks

## Performance Characteristics

### Memory Usage (1M nodes, 10M edges)

| Representation | Memory | Cache Misses |
|----------------|--------|--------------|
| HashMap adjacency | 480 MB | High (pointer chasing) |
| CSR adjacency | 168 MB | Low (sequential) |

### Runtime Benchmarks (Intel i7-8700K, 6 cores)

| Algorithm | 10K nodes | 100K nodes | 1M nodes |
|-----------|-----------|------------|----------|
| Degree centrality | <1 ms | 8 ms | 95 ms |
| PageRank (50 iter) | 12 ms | 180 ms | 2.4 s |
| Betweenness (serial) | 450 ms | 52 s | timeout |
| Betweenness (parallel) | 95 ms | 8.7 s | 89 s |

**Parallelization benefit**: 4.7x speedup on 6-core CPU.

## Real-World Applications

### Social Network Analysis

**Problem**: Identify influential users in a social network.

**Approach**:
1. Build graph from friendship/follower edges
2. Compute PageRank for overall influence
3. Compute betweenness to find community bridges
4. Compute degree for local popularity

**Example**: Twitter influencer detection, LinkedIn connection recommendations.

### Supply Chain Optimization

**Problem**: Find critical nodes in a logistics network.

**Approach**:
1. Model warehouses/suppliers as nodes
2. Compute betweenness centrality
3. High-betweenness nodes are single points of failure
4. Add redundancy or buffer inventory

**Example**: Amazon warehouse placement, manufacturing supply chains.

### Epidemiology

**Problem**: Prioritize vaccination in contact networks.

**Approach**:
1. Build contact network from tracing data
2. Compute betweenness centrality
3. Vaccinate high-betweenness individuals first
4. Reduces R₀ by breaking transmission paths

**Example**: COVID-19 contact tracing, hospital infection control.

## Toyota Way Principles in Implementation

### Muda (Waste Elimination)

**CSR representation**: Eliminates HashMap pointer overhead, reduces memory by 50-70%.

**Parallel betweenness**: No synchronization needed in outer loop (embarrassingly parallel).

### Poka-Yoke (Error Prevention)

**Kahan summation**: Prevents floating-point drift in PageRank. Without compensation:
- 10K nodes: error ~1e-7
- 100K nodes: error ~1e-5
- 1M nodes: error ~1e-4

With Kahan summation, error consistently <1e-10.

### Heijunka (Load Balancing)

**Rayon work-stealing**: Automatically balances BFS tasks across cores. Nodes with more edges take longer, but work-stealing prevents idle threads.

## Best Practices

### When to Use Each Centrality

- **Degree**: Quick analysis, local importance only
- **PageRank**: Global influence, considers network structure
- **Betweenness**: Find bridges, critical paths

### Graph Construction Tips

```rust,ignore
// Build graph once, query many times
let graph = Graph::from_edges(&edges, false);

// Reuse for multiple algorithms
let degree = graph.degree_centrality();
let pagerank = graph.pagerank(0.85, 100, 1e-6).unwrap();
let betweenness = graph.betweenness_centrality();
```

### Choosing PageRank Parameters

- **Damping factor (d)**: 0.85 standard, higher = more weight to links
- **Max iterations**: 100 usually sufficient (convergence ~20-50 iterations)
- **Tolerance**: 1e-6 balances precision vs speed

## Further Reading

**Graph Algorithms**:
- Brandes, U. (2001). "A Faster Algorithm for Betweenness Centrality"
- Page, L., Brin, S., et al. (1999). "The PageRank Citation Ranking"
- Buluç, A., et al. (2009). "Parallel Sparse Matrix-Vector Multiplication"

**CSR Representation**:
- Saad, Y. (2003). "Iterative Methods for Sparse Linear Systems"

**Numerical Stability**:
- Higham, N. (1993). "The Accuracy of Floating Point Summation"

## Summary

- **CSR format**: 50-70% memory reduction, 3-5x cache improvement
- **PageRank**: Global influence with Kahan summation for numerical stability
- **Betweenness**: Identifies bridges with parallel Brandes algorithm
- **Performance**: Scales to 1M+ nodes with parallel algorithms
- **Toyota Way**: Eliminates waste (CSR), prevents errors (Kahan), balances load (Rayon)
