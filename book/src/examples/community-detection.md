# Case Study: Community Detection with Louvain

This chapter documents the EXTREME TDD implementation of community detection using the Louvain algorithm for modularity optimization (Issue #22).

## Background

**GitHub Issue #22**: Implement Community Detection (Louvain/Leiden) for Graphs

**Requirements:**
- Louvain algorithm for modularity optimization
- Modularity computation: Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
- Detect densely connected groups (communities) in networks
- 15+ comprehensive tests

**Initial State:**
- Tests: 667 passing
- Existing graph module with centrality algorithms
- No community detection capabilities

## Implementation Summary

### RED Phase

Created 16 comprehensive tests:
- **Modularity tests** (5): empty graph, single community, two communities, perfect split, bad partition
- **Louvain tests** (11): empty graph, single node, two nodes, triangle, two triangles, disconnected components, karate club, star graph, complete graph, modularity improvement, all nodes assigned

### GREEN Phase

Implemented two core algorithms:

**1. Modularity Computation** (~130 lines):
```rust
pub fn modularity(&self, communities: &[Vec<NodeId>]) -> f64 {
    // Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
    // - Build community membership map
    // - Compute degrees
    // - For each node pair in same community:
    //     Add (A_ij - expected) to Q
    // - Return Q / 2m
}
```

**2. Louvain Algorithm** (~140 lines):
```rust
pub fn louvain(&self) -> Vec<Vec<NodeId>> {
    // Initialize: each node in own community
    // While improved:
    //   For each node:
    //     Try moving to neighbor communities
    //     Accept move if ΔQ > 0
    // Return final communities
}
```

**Key helper**:
```rust
fn modularity_gain(&self, node, from_comm, to_comm, node_to_comm) -> f64 {
    // ΔQ = (k_i_to - k_i_from)/m - k_i*(Σ_to - Σ_from)/(2m²)
}
```

### REFACTOR Phase

- Replaced loops with iterator chains (clippy fixes)
- Simplified edge counting logic
- Used `or_default()` instead of `or_insert_with(Vec::new)`
- Zero clippy warnings

**Final State:**
- Tests: 667 → 683 (+16)
- Zero warnings
- All quality gates passing

## Algorithm Details

### Modularity Formula

Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)

Where:
- m = total edges
- A_ij = 1 if edge exists, 0 otherwise
- k_i = degree of node i
- δ(c_i, c_j) = 1 if nodes i,j in same community

**Interpretation**:
- Q ∈ [-0.5, 1.0]
- Q > 0.3: Significant community structure
- Q ≈ 0: Random graph (no structure)
- Q < 0: Anti-community structure

### Louvain Algorithm

**Phase 1: Node movements**
1. Start: each node in own community
2. For each node v:
   - Calculate ΔQ for moving v to each neighbor's community
   - Move to community with highest ΔQ > 0
3. Repeat until no improvements

**Complexity**:
- Time: O(m·log n) typical
- Space: O(n + m)
- Iterations: Usually 5-10 until convergence

## Example Highlights

The example demonstrates:
1. **Two triangles connected**: Detects 2 communities (Q=0.357)
2. **Social network**: Bridge nodes connect groups (Q=0.357)
3. **Disconnected components**: Perfect separation (Q=0.500)
4. **Modularity comparison**: Good (Q=0.5) vs bad (Q=-0.167) partitions
5. **Complete graph**: Single community (Q≈0)

## Key Takeaways

1. **Modularity Q**: Measures community quality (higher is better)
2. **Greedy optimization**: Louvain finds local optima efficiently
3. **Detects structure**: Works on social networks, biological networks, citation graphs
4. **Handles disconnected graphs**: Correctly separates components
5. **O(m·log n)**: Fast enough for large networks

## Use Cases

### 1. Social Networks
Detect friend groups, communities in Facebook/Twitter graphs.

### 2. Biological Networks
Find protein interaction modules, gene co-expression clusters.

### 3. Citation Networks
Discover research topic communities.

### 4. Web Graphs
Cluster web pages by topic.

### 5. Recommendation Systems
Group users/items with similar preferences.

## Testing Strategy

**Unit Tests** (16 implemented):
- Correctness: Communities match expected structure
- Modularity: Q values in expected ranges
- Edge cases: Empty, single node, complete graphs
- Quality: Louvain improves modularity

## Technical Challenges Solved

### Challenge 1: Efficient Modularity Gain
**Problem**: Naive O(n²) for each potential move.
**Solution**: Incremental calculation using community degrees.

### Challenge 2: Avoiding Redundant Checks
**Problem**: Multiple neighbors in same community.
**Solution**: HashSet to track tried communities.

### Challenge 3: Iterator Chain Optimization
**Problem**: Clippy warnings for indexing loops.
**Solution**: Use `enumerate().filter().map().sum()` chains.

## Related Topics

- [Betweenness Centrality](../ml-fundamentals/graph-algorithms.md)
- [PageRank](../ml-fundamentals/graph-algorithms.md)
- [K-Means Clustering](./kmeans-clustering.md)

## References

1. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. J. Stat. Mech.
2. Newman, M. E. (2006). Modularity and community structure in networks. PNAS.
3. Fortunato, S. (2010). Community detection in graphs. Physics Reports.
