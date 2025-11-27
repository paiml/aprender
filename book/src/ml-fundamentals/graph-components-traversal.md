# Graph Components and Traversal Algorithms

Component analysis and graph traversal are fundamental techniques for understanding graph structure, detecting communities, validating properties, and exploring relationships. This chapter covers the theory and implementation of four essential algorithms in aprender's graph module.

## Overview

Aprender implements four key algorithms for graph exploration and decomposition:

1. **Depth-First Search (DFS)**: Stack-based graph traversal
2. **Connected Components**: Find groups of reachable nodes (undirected graphs)
3. **Strongly Connected Components (SCCs)**: Find mutually reachable groups (directed graphs)
4. **Topological Sort**: Linear ordering of directed acyclic graphs (DAGs)

All algorithms operate on the Compressed Sparse Row (CSR) graph representation for optimal cache locality and memory efficiency.

## Depth-First Search (DFS)

### Algorithm

Depth-First Search explores a graph by going as deep as possible along each branch before backtracking. It uses a stack (explicit or via recursion) to track the exploration path.

**Properties**:
- Time Complexity: O(n + m) where n = nodes, m = edges
- Space Complexity: O(n) for visited tracking and stack
- Explores one branch completely before trying others
- Returns nodes in pre-order visitation

### Implementation

```rust
use aprender::graph::Graph;

let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (1, 4)], false);

// DFS from node 0
let order = g.dfs(0).expect("node should exist");
// Possible result: [0, 1, 2, 3, 4] or [0, 1, 4, 2, 3]
// Order depends on neighbor iteration order

// DFS on disconnected graph only visits reachable nodes
let g2 = Graph::from_edges(&[(0, 1), (2, 3)], false);
let order2 = g2.dfs(0).expect("node should exist");
assert_eq!(order2, vec![0, 1]); // Only component with node 0

// Invalid starting node returns None
assert!(g.dfs(100).is_none());
```

### How It Works

1. **Initialization**: Push source node onto stack, mark as visited
2. **Loop**: While stack is not empty:
   - Pop node from stack
   - If already visited, skip
   - Mark as visited, add to result
   - Push unvisited neighbors onto stack (in reverse order for consistent traversal)
3. **Termination**: Stack is empty when all reachable nodes explored

**Visual Example** (tree):
```text
Graph:      0
           / \
          1   2
         /
        3

DFS from 0:
Stack: [0]           Visited: {}        Order: []
Stack: [2, 1]        Visited: {0}       Order: [0]
Stack: [2, 3]        Visited: {0,1}     Order: [0,1]
Stack: [2]           Visited: {0,1,3}   Order: [0,1,3]
Stack: []            Visited: {0,1,2,3} Order: [0,1,3,2]
```

**Stack-Based vs Recursive**:
- Aprender uses **explicit stack** (not recursion)
- Avoids stack overflow on deep graphs (>10K depth)
- Pre-order traversal: node added to result when first visited
- Neighbors pushed in reverse order for deterministic left-to-right traversal

### Use Cases

- **Cycle Detection**: DFS can detect cycles by tracking in-stack nodes
- **Path Finding**: Find any path between two nodes (not necessarily shortest)
- **Maze Solving**: Explore all paths until exit found
- **Topological Sort**: DFS post-order is foundation for DAG ordering
- **Connected Components**: DFS from each unvisited node finds components

### Comparison with BFS

| Aspect | DFS | BFS |
|--------|-----|-----|
| Data Structure | Stack (LIFO) | Queue (FIFO) |
| Exploration | Deep (branch-first) | Wide (level-first) |
| Path Found | Any path | Shortest path (unweighted) |
| Memory | O(n) worst case | O(n) worst case |
| Use Case | Structure analysis | Distance computation |

```rust
use aprender::graph::Graph;

let g = Graph::from_edges(
    &[(0, 1), (0, 2), (1, 3), (2, 3)],
    false
);

// DFS might visit: 0 → 1 → 3 → 2
let dfs_order = g.dfs(0).expect("node exists");

// BFS (via shortest_path) visits: 0 → 1, 2 → 3 (level-by-level)
let path_to_3 = g.shortest_path(0, 3).expect("path exists");
assert_eq!(path_to_3.len(), 3); // 0 → 1 → 3 (or 0 → 2 → 3)
```

## Connected Components

### Algorithm

Connected Components identifies groups of nodes that are mutually reachable in an **undirected graph**. Aprender uses **Union-Find** (also called Disjoint Set Union) with path compression and union by rank.

**Properties**:
- Time Complexity: O(m α(n)) where α = inverse Ackermann function (effectively constant)
- Space Complexity: O(n) for parent and rank arrays
- Near-linear performance in practice
- Returns component ID for each node

### Implementation

```rust
use aprender::graph::Graph;

// Three components: {0,1}, {2,3,4}, {5}
let g = Graph::from_edges(
    &[(0, 1), (2, 3), (3, 4)],
    false
);

let components = g.connected_components();
assert_eq!(components.len(), 6);

// Nodes in same component have same ID
assert_eq!(components[0], components[1]); // 0 and 1 connected
assert_eq!(components[2], components[3]); // 2 and 3 connected
assert_eq!(components[3], components[4]); // 3 and 4 connected

// Different components have different IDs
assert_ne!(components[0], components[2]);
assert_ne!(components[0], components[5]);

// Count number of components
use std::collections::HashSet;
let num_components: usize = components.iter().collect::<HashSet<_>>().len();
assert_eq!(num_components, 3);
```

### How It Works

Union-Find maintains a forest of trees where each tree represents a component.

**Data Structures**:
- `parent[i]`: Parent of node i (root if parent[i] == i)
- `rank[i]`: Approximate depth of tree rooted at i

**Operations**:

1. **Find(x)**: Find root of x's tree with **path compression**
```rust
fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]); // Path compression
    }
    parent[x]
}
```

2. **Union(x, y)**: Merge trees of x and y with **union by rank**
```rust
fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
    let root_x = find(parent, x);
    let root_y = find(parent, y);

    if root_x == root_y { return; }

    // Attach smaller tree under larger tree
    if rank[root_x] < rank[root_y] {
        parent[root_x] = root_y;
    } else if rank[root_x] > rank[root_y] {
        parent[root_y] = root_x;
    } else {
        parent[root_y] = root_x;
        rank[root_x] += 1;
    }
}
```

**Visual Example**:
```text
Graph: 0---1   2---3---4   5

Initial: parent=[0,1,2,3,4,5], rank=[0,0,0,0,0,0]

Process edge (0,1):
  Union(0,1): parent=[0,0,2,3,4,5], rank=[1,0,0,0,0,0]

Process edge (2,3):
  Union(2,3): parent=[0,0,2,2,4,5], rank=[1,0,1,0,0,0]

Process edge (3,4):
  Union(2,4): parent=[0,0,2,2,2,5], rank=[1,0,2,0,0,0]

Final components:
  Component 0: {0,1}
  Component 2: {2,3,4}
  Component 5: {5}
```

### Path Compression

Path compression flattens trees during find operations, making future queries faster.

**Without path compression**:
```text
Find(4): 4 → 3 → 2  (3 steps)
```

**With path compression**:
```text
After Find(4): 4 → 2, 3 → 2  (all point to root)
Next Find(4): 4 → 2  (1 step)
```

This achieves amortized O(α(n)) ≈ O(1) time per operation.

### Use Cases

- **Network Connectivity**: Identify isolated sub-networks
- **Image Segmentation**: Group connected pixels
- **Social Network Clusters**: Find friend groups
- **Graph Partitioning**: Identify disconnected regions
- **Reachability Queries**: "Can I get from A to B?"

## Strongly Connected Components (SCCs)

### Algorithm

Strongly Connected Components finds groups of nodes in a **directed graph** where every node can reach every other node in the group. Aprender uses **Tarjan's algorithm** (single DFS pass).

**Properties**:
- Time Complexity: O(n + m) - single DFS traversal
- Space Complexity: O(n) for discovery time, low-link values, and stack
- Returns component ID for each node
- Components are returned in reverse topological order

### Implementation

```rust
use aprender::graph::Graph;

// Directed graph with 2 SCCs: {0,1,2} and {3}
//   0 → 1 → 2 → 0 (cycle)
//   2 → 3 (one-way edge to isolated node)
let g = Graph::from_edges(
    &[(0, 1), (1, 2), (2, 0), (2, 3)],
    true  // directed
);

let sccs = g.strongly_connected_components();
assert_eq!(sccs.len(), 4);

// Cycle forms one SCC
assert_eq!(sccs[0], sccs[1]);
assert_eq!(sccs[1], sccs[2]);

// Node 3 is separate SCC (no incoming edges in cycle)
assert_ne!(sccs[0], sccs[3]);

// On DAG, each node is its own SCC
let dag = Graph::from_edges(&[(0, 1), (1, 2)], true);
let dag_sccs = dag.strongly_connected_components();
assert_ne!(dag_sccs[0], dag_sccs[1]);
assert_ne!(dag_sccs[1], dag_sccs[2]);
```

### How It Works

Tarjan's algorithm uses DFS with two timestamps per node:

- **disc[v]**: Discovery time (when v first visited)
- **low[v]**: Lowest discovery time reachable from v

**Key Insight**: If `low[v] == disc[v]`, then v is the root of an SCC.

**Algorithm Steps**:

1. **DFS Traversal**: Visit nodes in DFS order
2. **Discovery Time**: Assign `disc[v] = time++` when visiting v
3. **Low-Link Calculation**:
   - For tree edges: `low[v] = min(low[v], low[w])`
   - For back edges: `low[v] = min(low[v], disc[w])`
4. **SCC Detection**: If `low[v] == disc[v]`, pop stack until v is found
5. **Stack Management**: Maintain stack of nodes in current DFS path

**Visual Example**:
```text
Graph:  0 → 1 → 2
        ↑       ↓
        └───────┘

DFS from 0:
Visit 0: disc[0]=0, low[0]=0, stack=[0]
Visit 1: disc[1]=1, low[1]=1, stack=[0,1]
Visit 2: disc[2]=2, low[2]=2, stack=[0,1,2]
Back edge 2→0: low[2]=min(2,0)=0
               low[1]=min(1,0)=0
               low[0]=min(0,0)=0

SCC detection at 0: low[0]==disc[0]
Pop stack until 0: {2,1,0} form one SCC
```

### Comparison: Tarjan vs Kosaraju

| Aspect | Tarjan | Kosaraju |
|--------|--------|----------|
| DFS Passes | 1 | 2 |
| Transpose Graph | No | Yes |
| Complexity | O(n+m) | O(n+m) |
| Implementation | More complex | Simpler |
| Performance | ~30% faster | Easier to understand |

Aprender uses Tarjan's for better performance.

### Use Cases

- **Dependency Analysis**: Find circular dependencies
- **Compiler Optimization**: Detect infinite loops
- **Web Crawling**: Identify link cycles
- **Database Transactions**: Detect deadlocks
- **Social Network Analysis**: Find tightly-knit groups

## Topological Sort

### Algorithm

Topological Sort produces a linear ordering of nodes in a **directed acyclic graph (DAG)** such that for every edge u → v, u appears before v. This is used for task scheduling, dependency resolution, and build systems.

**Properties**:
- Time Complexity: O(n + m) - DFS-based
- Space Complexity: O(n) for visited and in-stack tracking
- Returns `Some(order)` for DAGs, `None` for graphs with cycles
- Multiple valid orderings may exist

### Implementation

```rust
use aprender::graph::Graph;

// DAG: 0 → 1 → 3
//      ↓    ↓
//      2 ───┘
let g = Graph::from_edges(
    &[(0, 1), (0, 2), (1, 3), (2, 3)],
    true  // directed
);

let order = g.topological_sort().expect("DAG should have valid ordering");
assert_eq!(order.len(), 4);

// Verify ordering: each edge (u,v) has u before v
let pos: std::collections::HashMap<_, _> =
    order.iter().enumerate().map(|(i, &v)| (v, i)).collect();

// Edge 0→1: pos[0] < pos[1]
assert!(pos[&0] < pos[&1]);
assert!(pos[&0] < pos[&2]);
assert!(pos[&1] < pos[&3]);
assert!(pos[&2] < pos[&3]);

// Cycle detection: returns None
let cycle = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
assert!(cycle.topological_sort().is_none());
```

### How It Works

Topological sort uses DFS with **post-order** traversal and cycle detection.

**Algorithm Steps**:

1. **Initialization**: Mark all nodes as unvisited
2. **DFS with Cycle Detection**: For each unvisited node:
   - Mark as in-stack (currently exploring)
   - Recursively visit all unvisited neighbors
   - If neighbor is in-stack, cycle detected → return None
   - Mark as visited (finished exploring)
   - Add to result in post-order (after all descendants)
3. **Reverse**: Reverse post-order to get topological order

**Visual Example**:
```text
Graph:  0 → 1 → 3
        ↓    ↓
        2 ───┘

DFS from 0:
  Visit 0 (in_stack)
    Visit 1 (in_stack)
      Visit 3 (in_stack)
      3 done → post_order=[3]
    1 done → post_order=[3,1]
    Visit 2 (in_stack)
      3 already visited, skip
    2 done → post_order=[3,1,2]
  0 done → post_order=[3,1,2,0]

Reverse: [0,2,1,3] (valid topological order)
```

**Cycle Detection**:
```text
Graph: 0 → 1 → 2 → 0 (cycle)

DFS from 0:
  Visit 0 (in_stack={0})
    Visit 1 (in_stack={0,1})
      Visit 2 (in_stack={0,1,2})
        Visit 0 (in_stack={0,1,2})
        0 is in_stack → CYCLE DETECTED
        Return None
```

### Multiple Valid Orderings

DAGs often have multiple valid topological orderings:

```rust
use aprender::graph::Graph;

// Diamond DAG:  0
//              / \
//             1   2
//              \ /
//               3

let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], true);
let order = g.topological_sort().expect("valid DAG");

// Valid orderings: [0,1,2,3] or [0,2,1,3]
// Both satisfy: 0 before 1,2 and 1,2 before 3
```

### Use Cases

- **Build Systems**: Compile source files in dependency order (Makefile, Cargo)
- **Course Prerequisites**: Schedule classes respecting prerequisites
- **Task Scheduling**: Execute tasks with dependencies (CI/CD pipelines)
- **Package Managers**: Install dependencies before dependents (npm, pip)
- **Spreadsheet Calculations**: Compute cells in formula dependency order

### Kahn's Algorithm (Alternative)

Kahn's algorithm is an alternative using in-degree counting:

1. Find all nodes with in-degree 0
2. Add them to result, remove from graph
3. Repeat until graph is empty (valid) or no zero in-degree nodes (cycle)

**Comparison**:

| Aspect | DFS-based (aprender) | Kahn's Algorithm |
|--------|----------------------|------------------|
| Complexity | O(n+m) | O(n+m) |
| Cycle Detection | Early termination | End of algorithm |
| Output Order | Deterministic | Queue-dependent |
| Implementation | Recursive/stack | Queue-based |

Aprender uses DFS-based for early cycle detection and simpler implementation.

## Performance Comparison

### Complexity Summary

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| DFS | O(n+m) | O(n) | Graph exploration |
| Connected Components | O(m α(n)) | O(n) | Undirected connectivity |
| SCCs (Tarjan) | O(n+m) | O(n) | Directed connectivity |
| Topological Sort | O(n+m) | O(n) | DAG ordering |

All algorithms achieve near-linear performance on sparse graphs (m ≈ n).

### Benchmark Results

Synthetic graphs (average degree ≈ 3):

```text
Algorithm              | 100 nodes | 1000 nodes | 5000 nodes |
-----------------------|-----------|------------|------------|
DFS                    | 580 ns    | 5.6 µs     | 28 µs      |
Connected Components   | 1.2 µs    | 11.5 µs    | 58 µs      |
SCCs (Tarjan)          | 1.8 µs    | 17.2 µs    | 87 µs      |
Topological Sort       | 620 ns    | 6.2 µs     | 31 µs      |
```

**Key Observations**:
- Perfect linear scaling: 10x nodes → ~10x time
- DFS and topological sort have minimal overhead
- SCCs ~1.5x slower than connected components (directed graph complexity)
- All algorithms <100µs for 5000-node graphs

## Advanced Topics

### Bi-Connected Components

Bi-connected components are maximal subgraphs with no articulation points (bridges). Removing any single node doesn't disconnect the component.

**Application**: Network resilience analysis

**Not yet implemented** in aprender (future roadmap).

### Condensation Graph

The condensation graph represents SCCs as nodes, with edges between SCCs.

```text
Original:  0 → 1 ⇄ 2      Condensation:  {0} → {1,2} → {3}
           ↓       ↓
           3 ←─────┘
```

**Property**: Condensation is always a DAG

**Use Case**: Simplify graph analysis by collapsing cycles

### Parallel Algorithms

DFS is inherently sequential (stack-based), but components can be parallelized:

- **Parallel Union-Find**: Use concurrent data structures for find/union
- **Parallel SCCs**: Multiple independent DFS starting points
- **Parallel Topological Sort**: Level-based parallelization

**Not yet implemented** in aprender (future optimization).

## See Also

- [Graph Algorithms](./graph-algorithms.md) - Centrality and structural analysis
- [Graph Pathfinding](./graph-pathfinding.md) - Shortest path algorithms
- [Graph Link Prediction](./graph-link-prediction.md) - Community detection and link analysis
- [Graph Examples](../../../examples/graph_social_network.rs) - Practical usage examples

## References

1. Tarjan, R. E. (1972). "Depth-first search and linear graph algorithms." *SIAM Journal on Computing*, 1(2), 146-160.

2. Tarjan, R. E. (1975). "Efficiency of a good but not linear set union algorithm." *Journal of the ACM*, 22(2), 215-225.

3. Cormen, T. H., et al. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
   - Chapter 22: Elementary Graph Algorithms (DFS, topological sort)
   - Chapter 21: Data Structures for Disjoint Sets (Union-Find)

4. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 1: Fundamental Algorithms* (3rd ed.). Section 2.3.3: Topological Sorting.

5. Sharir, M. (1981). "A strong-connectivity algorithm and its applications in data flow analysis." *Computers & Mathematics with Applications*, 7(1), 67-72.
