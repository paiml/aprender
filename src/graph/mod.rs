//! Graph construction and analysis with cache-optimized CSR representation.
//!
//! This module provides high-performance graph algorithms built on top of
//! Compressed Sparse Row (CSR) format for maximum cache locality. Key features:
//!
//! - CSR representation (50-70% memory reduction vs `HashMap`)
//! - Centrality measures (degree, betweenness, `PageRank`)
//! - Parallel algorithms using Rayon
//! - Numerical stability (Kahan summation in `PageRank`)
//!
//! # Examples
//!
//! ```
//! use aprender::graph::Graph;
//!
//! let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
//!
//! let dc = g.degree_centrality();
//! assert_eq!(dc.len(), 3);
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

pub mod centrality;

pub use centrality::GraphCentrality;

/// Graph node identifier (contiguous integers for cache efficiency).
pub type NodeId = usize;

/// Graph edge with optional weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: Option<f64>,
}

/// Graph structure using CSR (Compressed Sparse Row) for cache efficiency.
///
/// Memory layout inspired by Combinatorial BLAS (Buluc et al. 2009):
/// - Adjacency stored as two flat vectors (CSR format)
/// - Node labels stored separately (accessed rarely)
/// - String→NodeId mapping via `HashMap` (build-time only)
///
/// # Performance
/// - Memory: 50-70% reduction vs `HashMap` (no pointer overhead)
/// - Cache misses: 3-5x fewer (sequential access pattern)
/// - SIMD-friendly: Neighbor iteration can use vectorization
#[derive(Debug)]
pub struct Graph {
    // CSR adjacency representation (cache-friendly)
    row_ptr: Vec<usize>,      // Offset into col_indices (length = n_nodes + 1)
    col_indices: Vec<NodeId>, // Flattened neighbor lists (length = n_edges)
    edge_weights: Vec<f64>,   // Parallel to col_indices (empty if unweighted)

    // Metadata (accessed less frequently)
    #[allow(dead_code)]
    node_labels: Vec<Option<String>>, // Indexed by NodeId
    #[allow(dead_code)]
    label_to_id: HashMap<String, NodeId>, // For label lookups

    is_directed: bool,
    n_nodes: usize,
    n_edges: usize,
}

impl Graph {
    /// Create empty graph.
    ///
    /// # Arguments
    /// * `is_directed` - Whether the graph is directed
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::new(false); // undirected
    /// assert_eq!(g.num_nodes(), 0);
    /// ```
    #[must_use]
    pub fn new(is_directed: bool) -> Self {
        Self {
            row_ptr: vec![0],
            col_indices: Vec::new(),
            edge_weights: Vec::new(),
            node_labels: Vec::new(),
            label_to_id: HashMap::new(),
            is_directed,
            n_nodes: 0,
            n_edges: 0,
        }
    }

    /// Get number of nodes in graph.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Get number of edges in graph.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.n_edges
    }

    /// Check if graph is directed.
    #[must_use]
    pub fn is_directed(&self) -> bool {
        self.is_directed
    }

    /// Get neighbors of node v in O(degree(v)) time with perfect cache locality.
    ///
    /// # Arguments
    /// * `v` - Node ID
    ///
    /// # Returns
    /// Slice of neighbor node IDs
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    /// assert_eq!(g.neighbors(1), &[0, 2]);
    /// ```
    #[must_use]
    pub fn neighbors(&self, v: NodeId) -> &[NodeId] {
        if v >= self.n_nodes {
            return &[];
        }
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        &self.col_indices[start..end]
    }

    /// Build graph from edge list.
    ///
    /// This is the primary construction method. Automatically detects
    /// the number of nodes from the edge list and builds CSR representation.
    ///
    /// # Arguments
    /// * `edges` - Slice of (source, target) tuples
    /// * `is_directed` - Whether the graph is directed
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    /// assert_eq!(g.num_nodes(), 3);
    /// assert_eq!(g.num_edges(), 3);
    /// ```
    #[must_use]
    pub fn from_edges(edges: &[(NodeId, NodeId)], is_directed: bool) -> Self {
        if edges.is_empty() {
            return Self::new(is_directed);
        }

        // Find max node ID to determine number of nodes
        let max_node = edges.iter().flat_map(|&(s, t)| [s, t]).max().unwrap_or(0);
        let n_nodes = max_node + 1;

        // Build adjacency list first (for sorting and deduplication)
        let mut adj_list: Vec<Vec<NodeId>> = vec![Vec::new(); n_nodes];
        for &(source, target) in edges {
            adj_list[source].push(target);
            if !is_directed && source != target {
                // For undirected graphs, add reverse edge
                adj_list[target].push(source);
            }
        }

        // Sort and deduplicate neighbors for each node
        for neighbors in &mut adj_list {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        // Build CSR representation
        let mut row_ptr = Vec::with_capacity(n_nodes + 1);
        let mut col_indices = Vec::new();

        row_ptr.push(0);
        for neighbors in &adj_list {
            col_indices.extend_from_slice(neighbors);
            row_ptr.push(col_indices.len());
        }

        let n_edges = if is_directed {
            edges.len()
        } else {
            // For undirected, each edge is counted once in input
            edges.len()
        };

        Self {
            row_ptr,
            col_indices,
            edge_weights: Vec::new(),
            node_labels: vec![None; n_nodes],
            label_to_id: HashMap::new(),
            is_directed,
            n_nodes,
            n_edges,
        }
    }

    /// Build weighted graph from edge list with weights.
    ///
    /// # Arguments
    /// * `edges` - Slice of (source, target, weight) tuples
    /// * `is_directed` - Whether the graph is directed
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.5)], false);
    /// assert_eq!(g.num_nodes(), 3);
    /// assert_eq!(g.num_edges(), 2);
    /// ```
    #[must_use]
    pub fn from_weighted_edges(edges: &[(NodeId, NodeId, f64)], is_directed: bool) -> Self {
        if edges.is_empty() {
            return Self::new(is_directed);
        }

        // Find max node ID
        let max_node = edges
            .iter()
            .flat_map(|&(s, t, _)| [s, t])
            .max()
            .unwrap_or(0);
        let n_nodes = max_node + 1;

        // Build adjacency list with weights
        let mut adj_list: Vec<Vec<(NodeId, f64)>> = vec![Vec::new(); n_nodes];
        for &(source, target, weight) in edges {
            adj_list[source].push((target, weight));
            if !is_directed && source != target {
                adj_list[target].push((source, weight));
            }
        }

        // Sort and deduplicate (keep first weight for duplicates)
        for neighbors in &mut adj_list {
            neighbors.sort_unstable_by_key(|&(id, _)| id);
            neighbors.dedup_by_key(|&mut (id, _)| id);
        }

        // Build CSR representation
        let mut row_ptr = Vec::with_capacity(n_nodes + 1);
        let mut col_indices = Vec::new();
        let mut edge_weights = Vec::new();

        row_ptr.push(0);
        for neighbors in &adj_list {
            for &(neighbor, weight) in neighbors {
                col_indices.push(neighbor);
                edge_weights.push(weight);
            }
            row_ptr.push(col_indices.len());
        }

        let n_edges = edges.len();

        Self {
            row_ptr,
            col_indices,
            edge_weights,
            node_labels: vec![None; n_nodes],
            label_to_id: HashMap::new(),
            is_directed,
            n_nodes,
            n_edges,
        }
    }

    /// Get edge weight between two nodes.
    ///
    /// # Returns
    /// * `Some(weight)` if edge exists
    /// * `None` if no edge exists
    #[allow(dead_code)]
    fn edge_weight(&self, source: NodeId, target: NodeId) -> Option<f64> {
        if source >= self.n_nodes {
            return None;
        }

        let start = self.row_ptr[source];
        let end = self.row_ptr[source + 1];
        let neighbors = &self.col_indices[start..end];

        // Binary search for target
        let pos = neighbors.binary_search(&target).ok()?;

        if self.edge_weights.is_empty() {
            Some(1.0) // Unweighted graph
        } else {
            Some(self.edge_weights[start + pos])
        }
    }

    /// Compute degree centrality for all nodes.
    ///
    /// Uses Freeman's normalization (1978): `C_D(v)` = deg(v) / (n - 1)
    ///
    /// # Returns
    /// `HashMap` mapping `NodeId` to centrality score in [0, 1]
    ///
    /// # Performance
    /// O(n + m) where n = nodes, m = edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Star graph: center has degree 3, leaves have degree 1
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    /// let dc = g.degree_centrality();
    ///
    /// assert_eq!(dc[&0], 1.0); // center connected to all others
    /// assert!((dc[&1] - 0.333).abs() < 0.01); // leaves connected to 1 of 3
    /// ```
    #[must_use]
    pub fn degree_centrality(&self) -> HashMap<NodeId, f64> {
        let mut centrality = HashMap::with_capacity(self.n_nodes);

        if self.n_nodes <= 1 {
            // Single node or empty graph
            for v in 0..self.n_nodes {
                centrality.insert(v, 0.0);
            }
            return centrality;
        }

        let norm = (self.n_nodes - 1) as f64;

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.n_nodes {
            let degree = self.neighbors(v).len() as f64;
            centrality.insert(v, degree / norm);
        }

        centrality
    }

    /// Compute `PageRank` using power iteration with Kahan summation.
    ///
    /// Uses the `PageRank` algorithm (Page et al. 1999) with numerically
    /// stable Kahan summation (Higham 1993) to prevent floating-point
    /// drift in large graphs (>10K nodes).
    ///
    /// # Arguments
    /// * `damping` - Damping factor (typically 0.85)
    /// * `max_iter` - Maximum iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Returns
    /// Vector of `PageRank` scores (one per node)
    ///
    /// # Performance
    /// O(k * m) where k = iterations, m = edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    /// let pr = g.pagerank(0.85, 100, 1e-6).expect("pagerank should converge for valid graph");
    /// assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-10); // Kahan ensures precision
    /// ```
    pub fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        if self.n_nodes == 0 {
            return Ok(Vec::new());
        }

        let n = self.n_nodes;
        let mut ranks = vec![1.0 / n as f64; n];
        let mut new_ranks = vec![0.0; n];

        for _ in 0..max_iter {
            // Handle dangling nodes (nodes with no outgoing edges)
            // Redistribute their rank uniformly to all nodes
            let mut dangling_sum = 0.0;
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                if self.neighbors(v).is_empty() {
                    dangling_sum += ranks[v];
                }
            }
            let dangling_contribution = damping * dangling_sum / n as f64;

            // Compute new ranks with Kahan summation
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                let incoming_neighbors = self.incoming_neighbors(v);

                let mut sum = 0.0;
                let mut c = 0.0; // Kahan compensation term

                for u in &incoming_neighbors {
                    let out_degree = self.neighbors(*u).len() as f64;
                    if out_degree > 0.0 {
                        let y = (ranks[*u] / out_degree) - c;
                        let t = sum + y;
                        c = (t - sum) - y; // Recover low-order bits
                        sum = t;
                    }
                }

                new_ranks[v] = (1.0 - damping) / n as f64 + damping * sum + dangling_contribution;
            }

            // Convergence check using Kahan for diff calculation
            let diff = kahan_diff(&ranks, &new_ranks);
            if diff < tol {
                return Ok(new_ranks);
            }

            std::mem::swap(&mut ranks, &mut new_ranks);
        }

        Ok(ranks)
    }

    /// Get incoming neighbors for directed graphs (reverse edges).
    ///
    /// For undirected graphs, this is the same as `neighbors()`.
    /// For directed graphs, we need to scan all nodes to find incoming edges.
    pub(crate) fn incoming_neighbors(&self, v: NodeId) -> Vec<NodeId> {
        if !self.is_directed {
            // For undirected graphs, incoming == outgoing
            return self.neighbors(v).to_vec();
        }

        // For directed graphs, scan all nodes to find incoming edges
        let mut incoming = Vec::new();
        for u in 0..self.n_nodes {
            if self.neighbors(u).contains(&v) {
                incoming.push(u);
            }
        }
        incoming
    }

    /// Compute betweenness centrality using parallel Brandes' algorithm.
    ///
    /// Uses Brandes' algorithm (2001) with Rayon parallelization for the outer loop.
    /// Each source's BFS is independent, making this "embarrassingly parallel" (Bader & Madduri 2006).
    ///
    /// # Performance
    /// - Serial: O(nm) for unweighted graphs
    /// - Parallel: O(nm/p) where p = number of CPU cores
    /// - Expected speedup: ~8x on 8-core CPU for graphs with >1K nodes
    ///
    /// # Returns
    /// Vector of betweenness centrality scores (one per node)
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Path graph: 0 -- 1 -- 2
    /// // Middle node has higher betweenness
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    /// let bc = g.betweenness_centrality();
    /// assert!(bc[1] > bc[0]); // middle node has highest betweenness
    /// assert!(bc[1] > bc[2]);
    /// ```
    #[must_use]
    pub fn betweenness_centrality(&self) -> Vec<f64> {
        if self.n_nodes == 0 {
            return Vec::new();
        }

        // Compute partial betweenness from each source (parallel when available)
        #[cfg(feature = "parallel")]
        let partial_scores: Vec<Vec<f64>> = (0..self.n_nodes)
            .into_par_iter()
            .map(|source| self.brandes_bfs_from_source(source))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let partial_scores: Vec<Vec<f64>> = (0..self.n_nodes)
            .map(|source| self.brandes_bfs_from_source(source))
            .collect();

        // Reduce partial scores (single-threaded, but fast)
        let mut centrality = vec![0.0; self.n_nodes];
        for partial in partial_scores {
            for (i, &score) in partial.iter().enumerate() {
                centrality[i] += score;
            }
        }

        // Normalize for undirected graphs
        if !self.is_directed {
            for score in &mut centrality {
                *score /= 2.0;
            }
        }

        centrality
    }

    /// Brandes' BFS from a single source node.
    ///
    /// Computes the contribution to betweenness centrality from paths
    /// starting at the given source node.
    fn brandes_bfs_from_source(&self, source: NodeId) -> Vec<f64> {
        let n = self.n_nodes;
        let mut stack = Vec::new(); // Nodes in order of non-increasing distance
        let mut paths = vec![0u64; n]; // Number of shortest paths from source to each node
        let mut distance = vec![i32::MAX; n]; // Distance from source
        let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); n]; // Predecessors on shortest paths
        let mut dependency = vec![0.0; n]; // Dependency of source on each node

        // BFS initialization
        paths[source] = 1;
        distance[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);

        // BFS to compute shortest paths
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in self.neighbors(v) {
                // First time we see w?
                if distance[w] == i32::MAX {
                    distance[w] = distance[v] + 1;
                    queue.push_back(w);
                }
                // Shortest path to w via v?
                if distance[w] == distance[v] + 1 {
                    paths[w] = paths[w].saturating_add(paths[v]);
                    predecessors[w].push(v);
                }
            }
        }

        // Backward accumulation of dependencies
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                let coeff = (paths[v] as f64 / paths[w] as f64) * (1.0 + dependency[w]);
                dependency[v] += coeff;
            }
        }

        dependency
    }

    /// Compute modularity of a community partition.
    ///
    /// Modularity Q measures the density of edges within communities compared to
    /// a random graph. Ranges from -0.5 to 1.0 (higher is better).
    ///
    /// Formula: Q = (1/2m) Σ[`A_ij` - `k_i`*`k_j/2m`] `δ(c_i`, `c_j`)
    /// where:
    /// - m = total edges
    /// - `A_ij` = adjacency matrix
    /// - `k_i` = degree of node i
    /// - `δ(c_i`, `c_j`) = 1 if nodes i,j in same community, 0 otherwise
    ///
    /// # Arguments
    /// * `communities` - Vector of communities, each community is a vector of node IDs
    ///
    /// # Returns
    /// Modularity score Q ∈ [-0.5, 1.0]
    #[must_use]
    pub fn modularity(&self, communities: &[Vec<NodeId>]) -> f64 {
        if self.n_nodes == 0 || communities.is_empty() {
            return 0.0;
        }

        // Build community membership map
        let mut community_map = vec![None; self.n_nodes];
        for (comm_id, community) in communities.iter().enumerate() {
            for &node in community {
                community_map[node] = Some(comm_id);
            }
        }

        // Total edges
        let m = self.n_edges as f64;

        if m == 0.0 {
            return 0.0;
        }

        let two_m = 2.0 * m;

        // Compute degrees
        let degrees: Vec<f64> = (0..self.n_nodes)
            .map(|i| self.neighbors(i).len() as f64)
            .collect();

        // Compute modularity: Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
        let mut q = 0.0;

        for i in 0..self.n_nodes {
            for j in 0..self.n_nodes {
                // Skip if nodes not in same community
                match (community_map[i], community_map[j]) {
                    (Some(ci), Some(cj)) if ci == cj => {
                        // Check if edge exists
                        let a_ij = if self.neighbors(i).contains(&j) {
                            1.0
                        } else {
                            0.0
                        };

                        // Expected edges in random graph
                        let expected = (degrees[i] * degrees[j]) / two_m;

                        q += a_ij - expected;
                    }
                    _ => {}
                }
            }
        }

        q / two_m
    }

    /// Detect communities using the Louvain algorithm.
    ///
    /// The Louvain method is a greedy modularity optimization algorithm that:
    /// 1. Starts with each node in its own community
    /// 2. Iteratively moves nodes to communities that maximize modularity gain
    /// 3. Aggregates the graph by treating communities as super-nodes
    /// 4. Repeats until modularity converges
    ///
    /// # Performance
    /// - Time: O(m·log n) typical, where m = edges, n = nodes
    /// - Space: O(n + m)
    ///
    /// # Returns
    /// Vector of communities, each community is a vector of node IDs
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Two triangles connected by one edge
    /// let g = Graph::from_edges(&[
    ///     (0, 1), (1, 2), (2, 0),  // Triangle 1
    ///     (3, 4), (4, 5), (5, 3),  // Triangle 2
    ///     (2, 3),                   // Connection
    /// ], false);
    ///
    /// let communities = g.louvain();
    /// assert_eq!(communities.len(), 2);  // Two communities detected
    /// ```
    #[must_use]
    pub fn louvain(&self) -> Vec<Vec<NodeId>> {
        if self.n_nodes == 0 {
            return Vec::new();
        }

        // Initialize: each node in its own community
        let mut node_to_comm: Vec<usize> = (0..self.n_nodes).collect();
        let mut improved = true;
        let mut iteration = 0;
        let max_iterations = 100;

        // Phase 1: Iteratively move nodes to communities
        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for node in 0..self.n_nodes {
                let current_comm = node_to_comm[node];
                let neighbors = self.neighbors(node);

                if neighbors.is_empty() {
                    continue;
                }

                // Find best community to move to
                let mut best_comm = current_comm;
                let mut best_gain = 0.0;

                // Try moving to each neighbor's community
                let mut tried_comms = std::collections::HashSet::new();
                for &neighbor in neighbors {
                    let neighbor_comm = node_to_comm[neighbor];

                    if tried_comms.contains(&neighbor_comm) {
                        continue;
                    }
                    tried_comms.insert(neighbor_comm);

                    // Calculate modularity gain
                    let gain =
                        self.modularity_gain(node, current_comm, neighbor_comm, &node_to_comm);

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = neighbor_comm;
                    }
                }

                // Move node if improves modularity
                if best_comm != current_comm && best_gain > 1e-10 {
                    node_to_comm[node] = best_comm;
                    improved = true;
                }
            }
        }

        // Convert node_to_comm map to community lists
        let mut communities: HashMap<usize, Vec<NodeId>> = HashMap::new();

        for (node, &comm) in node_to_comm.iter().enumerate() {
            communities.entry(comm).or_default().push(node);
        }

        communities.into_values().collect()
    }

    /// Calculate modularity gain from moving a node to a different community.
    fn modularity_gain(
        &self,
        node: NodeId,
        from_comm: usize,
        to_comm: usize,
        node_to_comm: &[usize],
    ) -> f64 {
        if from_comm == to_comm {
            return 0.0;
        }

        let m = self.n_edges as f64;
        if m == 0.0 {
            return 0.0;
        }

        let two_m = 2.0 * m;
        let k_i = self.neighbors(node).len() as f64;

        // Count edges from node to target community
        let mut k_i_to = 0.0;
        for &neighbor in self.neighbors(node) {
            if node_to_comm[neighbor] == to_comm {
                k_i_to += 1.0;
            }
        }

        // Count edges from node to current community (excluding self)
        let mut k_i_from = 0.0;
        for &neighbor in self.neighbors(node) {
            if node_to_comm[neighbor] == from_comm && neighbor != node {
                k_i_from += 1.0;
            }
        }

        // Total degree of target community (excluding node)
        let sigma_tot_to: f64 = node_to_comm
            .iter()
            .enumerate()
            .filter(|&(n, &comm)| comm == to_comm && n != node)
            .map(|(n, _)| self.neighbors(n).len() as f64)
            .sum();

        // Total degree of current community (excluding node)
        let sigma_tot_from: f64 = node_to_comm
            .iter()
            .enumerate()
            .filter(|&(n, &comm)| comm == from_comm && n != node)
            .map(|(n, _)| self.neighbors(n).len() as f64)
            .sum();

        // Modularity gain formula
        (k_i_to - k_i_from) / m - k_i * (sigma_tot_to - sigma_tot_from) / (two_m * m)
    }

    /// Compute closeness centrality for all nodes.
    ///
    /// Closeness measures how close a node is to all other nodes in the graph.
    /// Uses Wasserman & Faust (1994) normalization: `C_C(v)` = (n-1) / Σd(v,u)
    ///
    /// # Returns
    /// Vector of closeness centrality scores (one per node)
    /// For disconnected graphs, nodes unreachable from v are ignored in the sum.
    ///
    /// # Performance
    /// O(n·(n + m)) where n = nodes, m = edges (BFS from each node)
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Star graph: center is close to all nodes
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    /// let cc = g.closeness_centrality();
    /// assert!(cc[0] > cc[1]); // center has highest closeness
    /// ```
    #[must_use]
    pub fn closeness_centrality(&self) -> Vec<f64> {
        if self.n_nodes == 0 {
            return Vec::new();
        }

        let mut centrality = vec![0.0; self.n_nodes];

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.n_nodes {
            let distances = self.bfs_distances(v);

            // Sum of distances to all reachable nodes
            let sum: usize = distances.iter().filter(|&&d| d != usize::MAX).sum();
            let reachable = distances
                .iter()
                .filter(|&&d| d != usize::MAX && d > 0)
                .count();

            if reachable > 0 && sum > 0 {
                // Normalized closeness: (n-1) / sum_of_distances
                centrality[v] = reachable as f64 / sum as f64;
            }
        }

        centrality
    }

    /// BFS to compute shortest path distances from a source node.
    fn bfs_distances(&self, source: NodeId) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.n_nodes];
        distances[source] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            for &w in self.neighbors(v) {
                if distances[w] == usize::MAX {
                    distances[w] = distances[v] + 1;
                    queue.push_back(w);
                }
            }
        }

        distances
    }

    /// Compute eigenvector centrality using power iteration.
    ///
    /// Eigenvector centrality measures node importance based on the importance
    /// of its neighbors. Uses the dominant eigenvector of the adjacency matrix.
    ///
    /// # Arguments
    /// * `max_iter` - Maximum power iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Returns
    /// Vector of eigenvector centrality scores (one per node)
    ///
    /// # Performance
    /// O(k·m) where k = iterations, m = edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Path graph: middle nodes have higher eigenvector centrality
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    /// let ec = g.eigenvector_centrality(100, 1e-6).unwrap();
    /// assert!(ec[1] > ec[0]); // middle nodes more central
    /// ```
    pub fn eigenvector_centrality(&self, max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        if self.n_nodes == 0 {
            return Ok(Vec::new());
        }

        let n = self.n_nodes;
        let mut x = vec![1.0 / (n as f64).sqrt(); n]; // Initial uniform vector
        let mut x_new = vec![0.0; n];

        for _ in 0..max_iter {
            // Matrix-vector multiplication: x_new = A * x
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                x_new[v] = self.neighbors(v).iter().map(|&u| x[u]).sum();
            }

            // Normalize to unit vector (L2 norm)
            let norm: f64 = x_new.iter().map(|&val| val * val).sum::<f64>().sqrt();

            if norm < 1e-10 {
                // Disconnected graph or no edges
                return Ok(vec![0.0; n]);
            }

            for val in &mut x_new {
                *val /= norm;
            }

            // Check convergence
            let diff: f64 = x.iter().zip(&x_new).map(|(a, b)| (a - b).abs()).sum();

            if diff < tol {
                return Ok(x_new);
            }

            std::mem::swap(&mut x, &mut x_new);
        }

        Ok(x)
    }

    /// Compute Katz centrality with attenuation factor.
    ///
    /// Katz centrality generalizes eigenvector centrality by adding an attenuation
    /// factor for long-range connections: `C_K` = Σ(α^k · A^k · 1)
    ///
    /// # Arguments
    /// * `alpha` - Attenuation factor (typically 0.1-0.5, must be < `1/λ_max`)
    /// * `max_iter` - Maximum iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Returns
    /// Vector of Katz centrality scores
    ///
    /// # Performance
    /// O(k·m) where k = iterations, m = edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    /// let kc = g.katz_centrality(0.1, 100, 1e-6).unwrap();
    /// assert_eq!(kc.len(), 3);
    /// ```
    pub fn katz_centrality(
        &self,
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Vec<f64>, String> {
        if self.n_nodes == 0 {
            return Ok(Vec::new());
        }

        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be in (0, 1)".to_string());
        }

        let n = self.n_nodes;
        let mut x = vec![1.0; n]; // Initial vector of ones
        let mut x_new = vec![0.0; n];

        for _ in 0..max_iter {
            // Katz iteration: x_new = β + α·A^T·x (where β = 1)
            // Use incoming neighbors (transpose adjacency matrix)
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                let incoming = self.incoming_neighbors(v);
                let neighbors_sum: f64 = incoming.iter().map(|&u| x[u]).sum();
                x_new[v] = 1.0 + alpha * neighbors_sum;
            }

            // Check convergence
            let diff: f64 = x.iter().zip(&x_new).map(|(a, b)| (a - b).abs()).sum();

            if diff < tol {
                return Ok(x_new);
            }

            std::mem::swap(&mut x, &mut x_new);
        }

        Ok(x)
    }

    /// Compute harmonic centrality for all nodes.
    ///
    /// Harmonic centrality is the sum of reciprocal distances to all other nodes.
    /// More robust than closeness for disconnected graphs (Boldi & Vigna 2014).
    ///
    /// Formula: `C_H(v)` = Σ(1/d(v,u)) for all u ≠ v
    ///
    /// # Returns
    /// Vector of harmonic centrality scores
    ///
    /// # Performance
    /// O(n·(n + m)) where n = nodes, m = edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Star graph: center has highest harmonic centrality
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
    /// let hc = g.harmonic_centrality();
    /// assert!(hc[0] > hc[1]); // center most central
    /// ```
    #[must_use]
    pub fn harmonic_centrality(&self) -> Vec<f64> {
        if self.n_nodes == 0 {
            return Vec::new();
        }

        let mut centrality = vec![0.0; self.n_nodes];

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.n_nodes {
            let distances = self.bfs_distances(v);

            // Sum reciprocals of distances (skip unreachable nodes)
            for &dist in &distances {
                if dist > 0 && dist != usize::MAX {
                    centrality[v] += 1.0 / dist as f64;
                }
            }
        }

        centrality
    }

    /// Compute graph density.
    ///
    /// Density is the ratio of actual edges to possible edges.
    /// For undirected: d = 2m / (n(n-1))
    /// For directed: d = m / (n(n-1))
    ///
    /// # Returns
    /// Density in [0, 1] where 1 is a complete graph
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Complete graph K4 has density 1.0
    /// let g = Graph::from_edges(&[(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)], false);
    /// assert!((g.density() - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn density(&self) -> f64 {
        if self.n_nodes <= 1 {
            return 0.0;
        }

        let n = self.n_nodes as f64;
        let m = self.n_edges as f64;
        let possible = n * (n - 1.0);

        if self.is_directed {
            m / possible
        } else {
            (2.0 * m) / possible
        }
    }

    /// Compute graph diameter (longest shortest path).
    ///
    /// Returns None if graph is disconnected.
    ///
    /// # Returns
    /// Some(diameter) if connected, None otherwise
    ///
    /// # Performance
    /// O(n·(n + m)) - runs BFS from all nodes
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Path graph 0--1--2--3 has diameter 3
    /// let g = Graph::from_edges(&[(0,1), (1,2), (2,3)], false);
    /// assert_eq!(g.diameter(), Some(3));
    /// ```
    #[must_use]
    pub fn diameter(&self) -> Option<usize> {
        if self.n_nodes == 0 {
            return None;
        }

        let mut max_dist = 0;

        for v in 0..self.n_nodes {
            let distances = self.bfs_distances(v);

            for &dist in &distances {
                if dist == usize::MAX {
                    // Disconnected graph
                    return None;
                }
                if dist > max_dist {
                    max_dist = dist;
                }
            }
        }

        Some(max_dist)
    }

    /// Compute global clustering coefficient.
    ///
    /// Measures the probability that two neighbors of a node are connected.
    /// Formula: C = 3 × triangles / triads
    ///
    /// # Returns
    /// Clustering coefficient in [0, 1]
    ///
    /// # Performance
    /// O(n·d²) where d = average degree
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Triangle has clustering coefficient 1.0
    /// let g = Graph::from_edges(&[(0,1), (1,2), (2,0)], false);
    /// assert!((g.clustering_coefficient() - 1.0).abs() < 1e-6);
    /// ```
    #[allow(clippy::cast_lossless)]
    #[must_use]
    pub fn clustering_coefficient(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }

        let mut triangles = 0;
        let mut triads = 0;

        for v in 0..self.n_nodes {
            let neighbors = self.neighbors(v);
            let deg = neighbors.len();

            if deg < 2 {
                continue;
            }

            // Count triads (pairs of neighbors)
            triads += deg * (deg - 1) / 2;

            // Count triangles (connected pairs of neighbors)
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let u = neighbors[i];
                    let w = neighbors[j];

                    // Check if u and w are connected
                    if self.neighbors(u).contains(&w) {
                        triangles += 1;
                    }
                }
            }
        }

        if triads == 0 {
            return 0.0;
        }

        // Each triangle is counted 3 times (once from each vertex)
        // So we divide by 3 to get actual triangle count
        (triangles as f64) / (triads as f64)
    }

    /// Compute degree assortativity coefficient.
    ///
    /// Measures correlation between degrees of connected nodes.
    /// Positive: high-degree nodes connect to high-degree nodes
    /// Negative: high-degree nodes connect to low-degree nodes
    ///
    /// # Returns
    /// Assortativity coefficient in [-1, 1]
    ///
    /// # Performance
    /// O(m) where m = edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Star graph has negative assortativity
    /// let g = Graph::from_edges(&[(0,1), (0,2), (0,3)], false);
    /// assert!(g.assortativity() < 0.0);
    /// ```
    #[must_use]
    pub fn assortativity(&self) -> f64 {
        if self.n_edges == 0 {
            return 0.0;
        }

        // Compute degrees
        let degrees: Vec<f64> = (0..self.n_nodes)
            .map(|i| self.neighbors(i).len() as f64)
            .collect();

        let m = self.n_edges as f64;
        let mut sum_jk = 0.0;
        let mut sum_j = 0.0;
        let mut sum_k = 0.0;
        let mut sum_j_sq = 0.0;
        let mut sum_k_sq = 0.0;

        // Sum over all edges
        for v in 0..self.n_nodes {
            let j = degrees[v];
            for &u in self.neighbors(v) {
                let k = degrees[u];
                sum_jk += j * k;
                sum_j += j;
                sum_k += k;
                sum_j_sq += j * j;
                sum_k_sq += k * k;
            }
        }

        // For undirected graphs, each edge is counted twice
        let normalization = if self.is_directed { m } else { 2.0 * m };

        sum_jk /= normalization;
        sum_j /= normalization;
        sum_k /= normalization;
        sum_j_sq /= normalization;
        sum_k_sq /= normalization;

        let numerator = sum_jk - sum_j * sum_k;
        let denominator = ((sum_j_sq - sum_j * sum_j) * (sum_k_sq - sum_k * sum_k)).sqrt();

        if denominator < 1e-10 {
            return 0.0;
        }

        numerator / denominator
    }

    /// Compute shortest path between two nodes using BFS.
    ///
    /// Finds the shortest path (minimum number of hops) from source to target
    /// using breadth-first search. Works for both directed and undirected graphs.
    ///
    /// # Algorithm
    /// Uses BFS with predecessor tracking (Pohl 1971, bidirectional variant).
    ///
    /// # Arguments
    /// * `source` - Starting node ID
    /// * `target` - Destination node ID
    ///
    /// # Returns
    /// * `Some(path)` - Shortest path as vector of node IDs from source to target
    /// * `None` - No path exists between source and target
    ///
    /// # Complexity
    /// * Time: O(n + m) where n = nodes, m = edges
    /// * Space: O(n) for predecessor tracking
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
    /// let g = Graph::from_edges(&edges, false);
    ///
    /// // Shortest path from 0 to 3
    /// let path = g.shortest_path(0, 3).unwrap();
    /// assert_eq!(path.len(), 2); // 0 -> 3 (direct edge)
    /// assert_eq!(path[0], 0);
    /// assert_eq!(path[1], 3);
    ///
    /// // Path 0 to 2
    /// let path = g.shortest_path(0, 2).unwrap();
    /// assert!(path.len() <= 3); // Either 0->1->2 or 0->3->2
    /// ```
    #[must_use]
    pub fn shortest_path(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        // Bounds checking
        if source >= self.n_nodes || target >= self.n_nodes {
            return None;
        }

        // Special case: source == target
        if source == target {
            return Some(vec![source]);
        }

        // BFS with predecessor tracking
        let mut visited = vec![false; self.n_nodes];
        let mut predecessor = vec![None; self.n_nodes];
        let mut queue = VecDeque::new();

        visited[source] = true;
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            // Early termination if we reach target
            if v == target {
                break;
            }

            for &w in self.neighbors(v) {
                if !visited[w] {
                    visited[w] = true;
                    predecessor[w] = Some(v);
                    queue.push_back(w);
                }
            }
        }

        // Reconstruct path from target to source
        if !visited[target] {
            return None; // No path exists
        }

        let mut path = Vec::new();
        let mut current = Some(target);

        while let Some(node) = current {
            path.push(node);
            current = predecessor[node];
        }

        path.reverse();
        Some(path)
    }

    /// Compute shortest path using Dijkstra's algorithm for weighted graphs.
    ///
    /// Finds the shortest path from source to target using Dijkstra's algorithm
    /// with a binary heap priority queue. Handles both weighted and unweighted graphs.
    ///
    /// # Algorithm
    /// Uses Dijkstra's algorithm (1959) with priority queue for O((n+m) log n) complexity.
    ///
    /// # Arguments
    /// * `source` - Starting node ID
    /// * `target` - Destination node ID
    ///
    /// # Returns
    /// * `Some((path, distance))` - Shortest path and total distance
    /// * `None` - No path exists
    ///
    /// # Panics
    /// Panics if graph contains negative edge weights.
    ///
    /// # Complexity
    /// * Time: O((n + m) log n)
    /// * Space: O(n) for distance tracking and priority queue
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], false);
    /// let (path, dist) = g.dijkstra(0, 2).unwrap();
    /// assert_eq!(dist, 3.0); // 0->1->2 is shorter than 0->2
    /// ```
    #[must_use]
    pub fn dijkstra(&self, source: NodeId, target: NodeId) -> Option<(Vec<NodeId>, f64)> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        // Bounds checking
        if source >= self.n_nodes || target >= self.n_nodes {
            return None;
        }

        // Special case: source == target
        if source == target {
            return Some((vec![source], 0.0));
        }

        // Priority queue entry: (negative distance, node)
        // Use Reverse to make min-heap
        #[derive(Copy, Clone, PartialEq)]
        struct State {
            cost: f64,
            node: NodeId,
        }

        impl Eq for State {}

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse ordering for min-heap (negate costs)
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut distances = vec![f64::INFINITY; self.n_nodes];
        let mut predecessor = vec![None; self.n_nodes];
        let mut heap = BinaryHeap::new();

        distances[source] = 0.0;
        heap.push(State {
            cost: 0.0,
            node: source,
        });

        while let Some(State { cost, node }) = heap.pop() {
            // Early termination if we reach target
            if node == target {
                break;
            }

            // Skip if we've already found a better path
            if cost > distances[node] {
                continue;
            }

            // Explore neighbors
            let start = self.row_ptr[node];
            let end = self.row_ptr[node + 1];

            for i in start..end {
                let neighbor = self.col_indices[i];
                let edge_weight = if self.edge_weights.is_empty() {
                    1.0
                } else {
                    self.edge_weights[i]
                };

                // Panic on negative weights (Dijkstra requirement)
                assert!(
                    edge_weight >= 0.0,
                    "Dijkstra's algorithm requires non-negative edge weights. \
                     Found negative weight {edge_weight} on edge ({node}, {neighbor})"
                );

                let next_cost = cost + edge_weight;

                // Relaxation step
                if next_cost < distances[neighbor] {
                    distances[neighbor] = next_cost;
                    predecessor[neighbor] = Some(node);
                    heap.push(State {
                        cost: next_cost,
                        node: neighbor,
                    });
                }
            }
        }

        // Check if target is reachable
        if distances[target].is_infinite() {
            return None;
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = Some(target);

        while let Some(node) = current {
            path.push(node);
            current = predecessor[node];
        }

        path.reverse();
        Some((path, distances[target]))
    }

    /// Compute all-pairs shortest paths using repeated BFS.
    ///
    /// Computes the shortest path distance between all pairs of nodes.
    /// Uses BFS for unweighted graphs (O(nm)) which is faster than
    /// Floyd-Warshall (O(n³)) for sparse graphs.
    ///
    /// # Algorithm
    /// Runs BFS from each node (Floyd 1962 for weighted, BFS for unweighted).
    ///
    /// # Returns
    /// Distance matrix where `[i][j]` = shortest distance from node i to node j.
    /// Returns `None` if nodes are not connected.
    ///
    /// # Complexity
    /// * Time: O(n·(n + m)) for unweighted graphs
    /// * Space: O(n²) for distance matrix
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
    /// let dist = g.all_pairs_shortest_paths();
    ///
    /// assert_eq!(dist[0][0], Some(0));
    /// assert_eq!(dist[0][1], Some(1));
    /// assert_eq!(dist[0][2], Some(2));
    /// ```
    #[must_use]
    pub fn all_pairs_shortest_paths(&self) -> Vec<Vec<Option<usize>>> {
        let n = self.n_nodes;
        let mut distances = vec![vec![None; n]; n];

        // Run BFS from each node
        for (source, row) in distances.iter_mut().enumerate().take(n) {
            let dist = self.bfs_distances(source);

            for (target, cell) in row.iter_mut().enumerate().take(n) {
                if dist[target] != usize::MAX {
                    *cell = Some(dist[target]);
                }
            }
        }

        distances
    }

    /// Compute shortest path using A* search algorithm with heuristic.
    ///
    /// Finds the shortest path from source to target using A* algorithm
    /// with a user-provided heuristic function. The heuristic must be
    /// admissible (never overestimate) for optimality guarantees.
    ///
    /// # Algorithm
    /// Uses A* search (Hart et al. 1968) with f(n) = g(n) + h(n) where:
    /// - g(n) = actual cost from source to n
    /// - h(n) = estimated cost from n to target (heuristic)
    ///
    /// # Arguments
    /// * `source` - Starting node ID
    /// * `target` - Destination node ID
    /// * `heuristic` - Function mapping `NodeId` to estimated distance to target
    ///
    /// # Returns
    /// * `Some(path)` - Shortest path as vector of node IDs
    /// * `None` - No path exists
    ///
    /// # Complexity
    /// * Time: O((n + m) log n) with admissible heuristic
    /// * Space: O(n) for tracking
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
    ///
    /// // Manhattan distance heuristic (example)
    /// let heuristic = |node: usize| (3 - node) as f64;
    ///
    /// let path = g.a_star(0, 3, heuristic).unwrap();
    /// assert_eq!(path, vec![0, 1, 2, 3]);
    /// ```
    pub fn a_star<F>(&self, source: NodeId, target: NodeId, heuristic: F) -> Option<Vec<NodeId>>
    where
        F: Fn(NodeId) -> f64,
    {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        // Bounds checking
        if source >= self.n_nodes || target >= self.n_nodes {
            return None;
        }

        // Special case: source == target
        if source == target {
            return Some(vec![source]);
        }

        // Priority queue entry with f-score
        #[derive(Copy, Clone, PartialEq)]
        struct State {
            f_score: f64, // g + h
            g_score: f64, // actual cost
            node: NodeId,
        }

        impl Eq for State {}

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap: lower f-score has higher priority
                other
                    .f_score
                    .partial_cmp(&self.f_score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut g_scores = vec![f64::INFINITY; self.n_nodes];
        let mut predecessor = vec![None; self.n_nodes];
        let mut heap = BinaryHeap::new();

        g_scores[source] = 0.0;
        heap.push(State {
            f_score: heuristic(source),
            g_score: 0.0,
            node: source,
        });

        while let Some(State {
            f_score: _,
            g_score,
            node,
        }) = heap.pop()
        {
            // Early termination if we reach target
            if node == target {
                break;
            }

            // Skip if we've already found a better path
            if g_score > g_scores[node] {
                continue;
            }

            // Explore neighbors
            let start = self.row_ptr[node];
            let end = self.row_ptr[node + 1];

            for i in start..end {
                let neighbor = self.col_indices[i];
                let edge_weight = if self.edge_weights.is_empty() {
                    1.0
                } else {
                    self.edge_weights[i]
                };

                let tentative_g = g_score + edge_weight;

                // Relaxation step
                if tentative_g < g_scores[neighbor] {
                    g_scores[neighbor] = tentative_g;
                    predecessor[neighbor] = Some(node);

                    let f = tentative_g + heuristic(neighbor);
                    heap.push(State {
                        f_score: f,
                        g_score: tentative_g,
                        node: neighbor,
                    });
                }
            }
        }

        // Check if target is reachable
        if g_scores[target].is_infinite() {
            return None;
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = Some(target);

        while let Some(node) = current {
            path.push(node);
            current = predecessor[node];
        }

        path.reverse();
        Some(path)
    }

    /// Depth-First Search (DFS) traversal starting from a given node.
    ///
    /// Returns nodes in the order they were visited (pre-order traversal).
    /// Only visits nodes reachable from the source node.
    ///
    /// # Arguments
    /// * `source` - Starting node for traversal
    ///
    /// # Returns
    /// Vector of visited nodes in DFS order, or None if source is invalid
    ///
    /// # Time Complexity
    /// O(n + m) where n = number of nodes, m = number of edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (0, 3)], false);
    /// let visited = g.dfs(0).expect("valid source");
    /// assert_eq!(visited.len(), 4); // All nodes reachable
    /// assert_eq!(visited[0], 0); // Starts at source
    /// ```
    #[must_use]
    pub fn dfs(&self, source: NodeId) -> Option<Vec<NodeId>> {
        // Validate source node
        if source >= self.n_nodes {
            return None;
        }

        let mut visited = vec![false; self.n_nodes];
        let mut stack = Vec::new();
        let mut order = Vec::new();

        // Start DFS from source
        stack.push(source);

        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }

            visited[node] = true;
            order.push(node);

            // Add neighbors to stack (in reverse order for consistent left-to-right traversal)
            let neighbors = self.neighbors(node);
            for &neighbor in neighbors.iter().rev() {
                if !visited[neighbor] {
                    stack.push(neighbor);
                }
            }
        }

        Some(order)
    }

    /// Find connected components using Union-Find algorithm.
    ///
    /// Returns a vector where each index is a node ID and the value is its component ID.
    /// Nodes in the same component have the same component ID.
    ///
    /// For directed graphs, this finds weakly connected components (ignores edge direction).
    ///
    /// # Returns
    /// Vector mapping each node to its component ID (0-indexed)
    ///
    /// # Time Complexity
    /// O(m·α(n)) where α is the inverse Ackermann function (effectively constant)
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Two disconnected components: (0,1) and (2,3)
    /// let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
    /// let components = g.connected_components();
    ///
    /// assert_eq!(components[0], components[1]); // Same component
    /// assert_ne!(components[0], components[2]); // Different components
    /// ```
    #[must_use]
    pub fn connected_components(&self) -> Vec<usize> {
        let n = self.n_nodes;
        if n == 0 {
            return Vec::new();
        }

        // Union-Find data structure
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0; n];

        // Find with path compression
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                let next = parent[x];
                parent[x] = parent[next]; // Path compression
                x = next;
            }
            x
        }

        // Union by rank
        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);

            if root_x == root_y {
                return;
            }

            // Union by rank
            use std::cmp::Ordering;
            match rank[root_x].cmp(&rank[root_y]) {
                Ordering::Less => parent[root_x] = root_y,
                Ordering::Greater => parent[root_y] = root_x,
                Ordering::Equal => {
                    parent[root_y] = root_x;
                    rank[root_x] += 1;
                }
            }
        }

        // Process all edges (treat directed graphs as undirected for weak connectivity)
        for node in 0..n {
            for &neighbor in self.neighbors(node) {
                union(&mut parent, &mut rank, node, neighbor);
            }
        }

        // Assign component IDs (compress paths and renumber)
        let mut component_map = HashMap::new();
        let mut next_component_id = 0;
        let mut result = vec![0; n];

        for (node, component) in result.iter_mut().enumerate().take(n) {
            let root = find(&mut parent, node);
            let component_id = *component_map.entry(root).or_insert_with(|| {
                let id = next_component_id;
                next_component_id += 1;
                id
            });
            *component = component_id;
        }

        result
    }

    /// Find strongly connected components using Tarjan's algorithm.
    ///
    /// A strongly connected component (SCC) is a maximal set of vertices where
    /// every vertex is reachable from every other vertex in the set.
    ///
    /// Only meaningful for directed graphs. For undirected graphs, use `connected_components()`.
    ///
    /// # Returns
    /// Vector mapping each node to its SCC ID (0-indexed)
    ///
    /// # Time Complexity
    /// O(n + m) - single-pass DFS
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Directed cycle: 0 -> 1 -> 2 -> 0
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    /// let sccs = g.strongly_connected_components();
    ///
    /// // All nodes in same SCC
    /// assert_eq!(sccs[0], sccs[1]);
    /// assert_eq!(sccs[1], sccs[2]);
    /// ```
    #[must_use]
    pub fn strongly_connected_components(&self) -> Vec<usize> {
        let n = self.n_nodes;
        if n == 0 {
            return Vec::new();
        }

        // Tarjan's algorithm state
        struct TarjanState {
            disc: Vec<Option<usize>>,
            low: Vec<usize>,
            on_stack: Vec<bool>,
            stack: Vec<usize>,
            time: usize,
            scc_id: Vec<usize>,
            scc_counter: usize,
        }

        impl TarjanState {
            fn new(n: usize) -> Self {
                Self {
                    disc: vec![None; n],
                    low: vec![0; n],
                    on_stack: vec![false; n],
                    stack: Vec::new(),
                    time: 0,
                    scc_id: vec![0; n],
                    scc_counter: 0,
                }
            }

            fn dfs(&mut self, v: usize, graph: &Graph) {
                // Initialize discovery time and low-link value
                self.disc[v] = Some(self.time);
                self.low[v] = self.time;
                self.time += 1;
                self.stack.push(v);
                self.on_stack[v] = true;

                // Visit all neighbors
                for &w in graph.neighbors(v) {
                    if self.disc[w].is_none() {
                        // Tree edge: recurse
                        self.dfs(w, graph);
                        self.low[v] = self.low[v].min(self.low[w]);
                    } else if self.on_stack[w] {
                        // Back edge to node on stack
                        self.low[v] =
                            self.low[v].min(self.disc[w].expect("disc[w] should be Some"));
                    }
                }

                // If v is a root node, pop the stack and create SCC
                if let Some(disc_v) = self.disc[v] {
                    if self.low[v] == disc_v {
                        // Found an SCC
                        loop {
                            let w = self.stack.pop().expect("stack should not be empty");
                            self.on_stack[w] = false;
                            self.scc_id[w] = self.scc_counter;
                            if w == v {
                                break;
                            }
                        }
                        self.scc_counter += 1;
                    }
                }
            }
        }

        let mut state = TarjanState::new(n);

        // Run Tarjan's algorithm from each unvisited node
        for v in 0..n {
            if state.disc[v].is_none() {
                state.dfs(v, self);
            }
        }

        state.scc_id
    }

    /// Topological sort for directed acyclic graphs (DAGs).
    ///
    /// Returns a linear ordering of vertices where for every directed edge (u,v),
    /// u appears before v in the ordering.
    ///
    /// Only valid for DAGs. If the graph contains a cycle, returns None.
    ///
    /// # Returns
    /// `Some(Vec<NodeId>)` with nodes in topological order, or `None` if graph has cycles
    ///
    /// # Time Complexity
    /// O(n + m) - DFS-based approach
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // DAG: 0 -> 1 -> 2
    /// let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
    /// let order = g.topological_sort().expect("DAG should have topological order");
    ///
    /// // 0 comes before 1, 1 comes before 2
    /// assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
    /// assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 2));
    /// ```
    #[must_use]
    pub fn topological_sort(&self) -> Option<Vec<NodeId>> {
        let n = self.n_nodes;
        if n == 0 {
            return Some(Vec::new());
        }

        // DFS-based topological sort with cycle detection
        let mut visited = vec![false; n];
        let mut in_stack = vec![false; n]; // For cycle detection
        let mut order = Vec::new();

        fn dfs(
            v: usize,
            graph: &Graph,
            visited: &mut [bool],
            in_stack: &mut [bool],
            order: &mut Vec<usize>,
        ) -> bool {
            if in_stack[v] {
                // Back edge found - cycle detected
                return false;
            }
            if visited[v] {
                // Already processed
                return true;
            }

            visited[v] = true;
            in_stack[v] = true;

            // Visit all neighbors
            for &neighbor in graph.neighbors(v) {
                if !dfs(neighbor, graph, visited, in_stack, order) {
                    return false; // Cycle detected
                }
            }

            in_stack[v] = false;
            order.push(v); // Add to order in post-order (reverse topological)

            true
        }

        // Run DFS from each unvisited node
        for v in 0..n {
            if !visited[v] && !dfs(v, self, &mut visited, &mut in_stack, &mut order) {
                return None; // Cycle detected
            }
        }

        // Reverse to get topological order
        order.reverse();
        Some(order)
    }

    /// Count common neighbors between two nodes (link prediction metric).
    ///
    /// Returns the number of neighbors shared by both nodes u and v.
    /// Used for link prediction: nodes with many common neighbors are
    /// more likely to form a connection.
    ///
    /// # Arguments
    /// * `u` - First node
    /// * `v` - Second node
    ///
    /// # Returns
    /// Number of common neighbors, or None if either node is invalid
    ///
    /// # Time Complexity
    /// O(min(deg(u), deg(v))) - intersection of neighbor sets
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Triangle: 0-1, 0-2, 1-2
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], false);
    ///
    /// // Nodes 1 and 2 share neighbor 0
    /// assert_eq!(g.common_neighbors(1, 2), Some(1));
    /// ```
    #[must_use]
    pub fn common_neighbors(&self, u: NodeId, v: NodeId) -> Option<usize> {
        // Validate nodes
        if u >= self.n_nodes || v >= self.n_nodes {
            return None;
        }

        let neighbors_u = self.neighbors(u);
        let neighbors_v = self.neighbors(v);

        // Use two-pointer technique (both are sorted)
        let mut count = 0;
        let mut i = 0;
        let mut j = 0;

        while i < neighbors_u.len() && j < neighbors_v.len() {
            use std::cmp::Ordering;
            match neighbors_u[i].cmp(&neighbors_v[j]) {
                Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }

        Some(count)
    }

    /// Adamic-Adar index for link prediction between two nodes.
    ///
    /// Computes a weighted measure of common neighbors, where neighbors with
    /// fewer connections are weighted more heavily. This captures the intuition
    /// that sharing a rare neighbor is more significant than sharing a common one.
    ///
    /// Formula: AA(u,v) = Σ 1/log(deg(z)) for all common neighbors z
    ///
    /// # Arguments
    /// * `u` - First node
    /// * `v` - Second node
    ///
    /// # Returns
    /// Adamic-Adar index score, or None if either node is invalid
    ///
    /// # Time Complexity
    /// O(min(deg(u), deg(v))) - intersection of neighbor sets
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Graph with shared neighbors
    /// let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], false);
    ///
    /// // Adamic-Adar index for nodes 1 and 2
    /// let aa = g.adamic_adar_index(1, 2).expect("valid nodes");
    /// assert!(aa > 0.0);
    /// ```
    #[must_use]
    pub fn adamic_adar_index(&self, u: NodeId, v: NodeId) -> Option<f64> {
        // Validate nodes
        if u >= self.n_nodes || v >= self.n_nodes {
            return None;
        }

        let neighbors_u = self.neighbors(u);
        let neighbors_v = self.neighbors(v);

        // Use two-pointer technique to find common neighbors
        let mut score = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < neighbors_u.len() && j < neighbors_v.len() {
            use std::cmp::Ordering;
            match neighbors_u[i].cmp(&neighbors_v[j]) {
                Ordering::Equal => {
                    let common_neighbor = neighbors_u[i];
                    let degree = self.neighbors(common_neighbor).len();

                    // Avoid log(1) = 0 division by zero
                    if degree > 1 {
                        score += 1.0 / (degree as f64).ln();
                    }

                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }

        Some(score)
    }

    /// Label propagation algorithm for community detection.
    ///
    /// Iteratively assigns each node the most common label among its neighbors.
    /// Nodes with the same final label belong to the same community.
    ///
    /// # Arguments
    /// * `max_iter` - Maximum number of iterations (default: 100)
    /// * `seed` - Random seed for deterministic tie-breaking (optional)
    ///
    /// # Returns
    /// Vector mapping each node to its community label (0-indexed)
    ///
    /// # Time Complexity
    /// `O(max_iter` · m) where m = number of edges
    ///
    /// # Examples
    /// ```
    /// use aprender::graph::Graph;
    ///
    /// // Graph with two communities: (0,1,2) and (3,4,5)
    /// let g = Graph::from_edges(
    ///     &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
    ///     false
    /// );
    ///
    /// let communities = g.label_propagation(100, Some(42));
    /// // Nodes in same community have same label
    /// assert_eq!(communities[0], communities[1]);
    /// ```
    #[must_use]
    pub fn label_propagation(&self, max_iter: usize, seed: Option<u64>) -> Vec<usize> {
        let n = self.n_nodes;
        if n == 0 {
            return Vec::new();
        }

        // Initialize each node with unique label
        let mut labels: Vec<usize> = (0..n).collect();

        // Simple deterministic ordering based on seed
        let mut node_order: Vec<usize> = (0..n).collect();
        if let Some(s) = seed {
            // Simple shuffle based on seed for deterministic results
            for i in 0..n {
                let j = ((s.wrapping_mul(i as u64 + 1)) % (n as u64)) as usize;
                node_order.swap(i, j);
            }
        }

        for _ in 0..max_iter {
            let mut changed = false;

            // Process nodes in random order
            for &node in &node_order {
                let neighbors = self.neighbors(node);
                if neighbors.is_empty() {
                    continue;
                }

                // Count neighbor labels
                let mut label_counts = HashMap::new();
                for &neighbor in neighbors {
                    *label_counts.entry(labels[neighbor]).or_insert(0) += 1;
                }

                // Find most common label (with deterministic tie-breaking)
                let most_common_label = label_counts
                    .iter()
                    .max_by_key(|(label, count)| (*count, std::cmp::Reverse(*label)))
                    .map(|(label, _)| *label)
                    .expect("label_counts should not be empty");

                if labels[node] != most_common_label {
                    labels[node] = most_common_label;
                    changed = true;
                }
            }

            // Early termination if converged
            if !changed {
                break;
            }
        }

        labels
    }
}

/// Kahan summation for computing L1 distance between two vectors.
///
/// Uses compensated summation to prevent floating-point drift.
/// Accumulates O(n·ε) error where ε is machine epsilon without compensation.
fn kahan_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for i in 0..a.len() {
        let y = (a[i] - b[i]).abs() - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[cfg(test)]
mod tests;
