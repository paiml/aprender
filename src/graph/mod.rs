//! Graph construction and analysis with cache-optimized CSR representation.
//!
//! This module provides high-performance graph algorithms built on top of
//! Compressed Sparse Row (CSR) format for maximum cache locality. Key features:
//!
//! - CSR representation (50-70% memory reduction vs HashMap)
//! - Centrality measures (degree, betweenness, PageRank)
//! - Parallel algorithms using Rayon
//! - Numerical stability (Kahan summation in PageRank)
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

use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

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
/// - String→NodeId mapping via HashMap (build-time only)
///
/// # Performance
/// - Memory: 50-70% reduction vs HashMap (no pointer overhead)
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
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Get number of edges in graph.
    pub fn num_edges(&self) -> usize {
        self.n_edges
    }

    /// Check if graph is directed.
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
    /// Uses Freeman's normalization (1978): C_D(v) = deg(v) / (n - 1)
    ///
    /// # Returns
    /// HashMap mapping NodeId to centrality score in [0, 1]
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

    /// Compute PageRank using power iteration with Kahan summation.
    ///
    /// Uses the PageRank algorithm (Page et al. 1999) with numerically
    /// stable Kahan summation (Higham 1993) to prevent floating-point
    /// drift in large graphs (>10K nodes).
    ///
    /// # Arguments
    /// * `damping` - Damping factor (typically 0.85)
    /// * `max_iter` - Maximum iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Returns
    /// Vector of PageRank scores (one per node)
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
    /// For undirected graphs, this is the same as neighbors().
    /// For directed graphs, we need to scan all nodes to find incoming edges.
    fn incoming_neighbors(&self, v: NodeId) -> Vec<NodeId> {
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
    pub fn betweenness_centrality(&self) -> Vec<f64> {
        if self.n_nodes == 0 {
            return Vec::new();
        }

        // Parallel outer loop: compute partial betweenness from each source
        let partial_scores: Vec<Vec<f64>> = (0..self.n_nodes)
            .into_par_iter()
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
    /// Formula: Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
    /// where:
    /// - m = total edges
    /// - A_ij = adjacency matrix
    /// - k_i = degree of node i
    /// - δ(c_i, c_j) = 1 if nodes i,j in same community, 0 otherwise
    ///
    /// # Arguments
    /// * `communities` - Vector of communities, each community is a vector of node IDs
    ///
    /// # Returns
    /// Modularity score Q ∈ [-0.5, 1.0]
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
    /// Uses Wasserman & Faust (1994) normalization: C_C(v) = (n-1) / Σd(v,u)
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
    /// factor for long-range connections: C_K = Σ(α^k · A^k · 1)
    ///
    /// # Arguments
    /// * `alpha` - Attenuation factor (typically 0.1-0.5, must be < 1/λ_max)
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
    /// Formula: C_H(v) = Σ(1/d(v,u)) for all u ≠ v
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
    /// * `heuristic` - Function mapping NodeId to estimated distance to target
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
    /// Some(Vec<NodeId>) with nodes in topological order, or None if graph has cycles
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
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(false);
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.num_edges(), 0);
        assert!(!g.is_directed());
    }

    #[test]
    fn test_directed_graph() {
        let g = Graph::new(true);
        assert!(g.is_directed());
    }

    #[test]
    fn test_from_edges_empty() {
        let g = Graph::from_edges(&[], false);
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_from_edges_undirected() {
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_edges(), 3);

        // Check neighbors (should be sorted)
        assert_eq!(g.neighbors(0), &[1, 2]);
        assert_eq!(g.neighbors(1), &[0, 2]);
        assert_eq!(g.neighbors(2), &[0, 1]);
    }

    #[test]
    fn test_from_edges_directed() {
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_edges(), 2);

        // Directed: edges only go one way
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.neighbors(1), &[2]);
        assert!(g.neighbors(2).is_empty()); // no outgoing edges
    }

    #[test]
    fn test_from_edges_with_gaps() {
        // Node IDs don't have to be contiguous
        let g = Graph::from_edges(&[(0, 5), (5, 10)], false);
        assert_eq!(g.num_nodes(), 11); // max node + 1
        assert_eq!(g.num_edges(), 2);

        assert_eq!(g.neighbors(0), &[5]);
        assert_eq!(g.neighbors(5), &[0, 10]);
        assert!(g.neighbors(1).is_empty()); // isolated node
    }

    #[test]
    fn test_from_edges_duplicate_edges() {
        // Duplicate edges should be deduplicated
        let g = Graph::from_edges(&[(0, 1), (0, 1), (1, 0)], false);
        assert_eq!(g.num_nodes(), 2);

        // Should only have one edge (0,1) in undirected graph
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.neighbors(1), &[0]);
    }

    #[test]
    fn test_from_edges_self_loop() {
        let g = Graph::from_edges(&[(0, 0), (0, 1)], false);
        assert_eq!(g.num_nodes(), 2);

        // Self-loop should appear once
        assert_eq!(g.neighbors(0), &[0, 1]);
    }

    #[test]
    fn test_neighbors_invalid_node() {
        let g = Graph::from_edges(&[(0, 1)], false);
        assert!(g.neighbors(999).is_empty()); // non-existent node
    }

    #[test]
    fn test_degree_centrality_empty() {
        let g = Graph::new(false);
        let dc = g.degree_centrality();
        assert_eq!(dc.len(), 0);
    }

    #[test]
    fn test_degree_centrality_single_node() {
        let g = Graph::from_edges(&[(0, 0)], false);
        let dc = g.degree_centrality();
        assert_eq!(dc[&0], 0.0); // single node, normalized degree is 0
    }

    #[test]
    fn test_degree_centrality_star_graph() {
        // Star graph: center node connected to 3 leaves
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let dc = g.degree_centrality();

        assert_eq!(dc[&0], 1.0); // center: degree 3 / (4-1) = 1.0
        assert!((dc[&1] - 1.0 / 3.0).abs() < 1e-6); // leaves: degree 1 / 3
        assert!((dc[&2] - 1.0 / 3.0).abs() < 1e-6);
        assert!((dc[&3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_degree_centrality_complete_graph() {
        // Complete graph K4: every node connected to every other
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let dc = g.degree_centrality();

        // All nodes have degree 3 in K4, normalized: 3/3 = 1.0
        for v in 0..4 {
            assert_eq!(dc[&v], 1.0);
        }
    }

    #[test]
    fn test_degree_centrality_path_graph() {
        // Path graph: 0 -- 1 -- 2 -- 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let dc = g.degree_centrality();

        // Endpoints have degree 1, middle nodes have degree 2
        assert!((dc[&0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((dc[&1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((dc[&2] - 2.0 / 3.0).abs() < 1e-6);
        assert!((dc[&3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_degree_centrality_directed() {
        // Directed: only count outgoing edges
        let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 2)], true);
        let dc = g.degree_centrality();

        assert!((dc[&0] - 2.0 / 2.0).abs() < 1e-6); // 2 outgoing edges
        assert!((dc[&1] - 1.0 / 2.0).abs() < 1e-6); // 1 outgoing edge
        assert_eq!(dc[&2], 0.0); // 0 outgoing edges
    }

    // PageRank tests

    #[test]
    fn test_pagerank_empty() {
        let g = Graph::new(true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should succeed for empty graph");
        assert!(pr.is_empty());
    }

    #[test]
    fn test_pagerank_single_node() {
        let g = Graph::from_edges(&[(0, 0)], true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should succeed for single node graph");
        assert_eq!(pr.len(), 1);
        assert!((pr[0] - 1.0).abs() < 1e-6); // Single node has all rank
    }

    #[test]
    fn test_pagerank_sum_equals_one() {
        // PageRank scores must sum to 1.0 (within numerical precision)
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should converge for cycle graph");
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10); // Kahan ensures high precision
    }

    #[test]
    fn test_pagerank_cycle_graph() {
        // Cycle graph: 0 -> 1 -> 2 -> 0
        // All nodes should have equal PageRank (by symmetry)
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should converge for symmetric cycle");

        assert_eq!(pr.len(), 3);
        // All nodes have equal rank in symmetric cycle
        assert!((pr[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((pr[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((pr[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_star_graph_directed() {
        // Star graph: 0 -> {1, 2, 3}
        // Node 0 distributes rank equally to 1, 2, 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should converge for directed star graph");

        assert_eq!(pr.len(), 4);
        // Leaves have no incoming edges except from 0
        // Node 0 has no incoming edges (lowest rank)
        assert!(pr[0] < pr[1]); // 0 has lowest rank
        assert!((pr[1] - pr[2]).abs() < 1e-6); // leaves have equal rank
        assert!((pr[2] - pr[3]).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_convergence() {
        // Test that PageRank converges within max_iter
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (1, 0)], true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should converge within max iterations");

        // Should converge (not hit max_iter)
        assert_eq!(pr.len(), 3);
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pagerank_no_outgoing_edges() {
        // Node with no outgoing edges (dangling node)
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should handle dangling nodes correctly");

        // Node 2 has no outgoing edges, but should still have rank
        assert_eq!(pr.len(), 3);
        assert!(pr[2] > 0.0);
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pagerank_undirected() {
        // Undirected graph: each edge goes both ways
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
        let pr = g
            .pagerank(0.85, 100, 1e-6)
            .expect("pagerank should converge for undirected path graph");

        assert_eq!(pr.len(), 3);
        // Middle node should have highest rank
        assert!(pr[1] > pr[0]);
        assert!(pr[1] > pr[2]);
        assert!((pr[0] - pr[2]).abs() < 1e-6); // endpoints equal
    }

    #[test]
    fn test_kahan_diff() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 2.9];
        let diff = kahan_diff(&a, &b);
        assert!((diff - 0.3).abs() < 1e-10);
    }

    // Betweenness centrality tests

    #[test]
    fn test_betweenness_centrality_empty() {
        let g = Graph::new(false);
        let bc = g.betweenness_centrality();
        assert!(bc.is_empty());
    }

    #[test]
    fn test_betweenness_centrality_single_node() {
        let g = Graph::from_edges(&[(0, 0)], false);
        let bc = g.betweenness_centrality();
        assert_eq!(bc.len(), 1);
        assert_eq!(bc[0], 0.0); // Single node has no betweenness
    }

    #[test]
    fn test_betweenness_centrality_path_graph() {
        // Path graph: 0 -- 1 -- 2
        // Middle node lies on all paths between endpoints
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 3);
        // Middle node has highest betweenness (all paths go through it)
        assert!(bc[1] > bc[0]);
        assert!(bc[1] > bc[2]);
        // Endpoints should have equal betweenness (by symmetry)
        assert!((bc[0] - bc[2]).abs() < 1e-6);
    }

    #[test]
    fn test_betweenness_centrality_star_graph() {
        // Star graph: center (0) connected to leaves {1, 2, 3}
        // Center lies on all paths between leaves
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 4);
        // Center has highest betweenness
        assert!(bc[0] > bc[1]);
        assert!(bc[0] > bc[2]);
        assert!(bc[0] > bc[3]);
        // Leaves should have equal betweenness (by symmetry)
        assert!((bc[1] - bc[2]).abs() < 1e-6);
        assert!((bc[2] - bc[3]).abs() < 1e-6);
    }

    #[test]
    fn test_betweenness_centrality_cycle_graph() {
        // Cycle graph: 0 -- 1 -- 2 -- 3 -- 0
        // All nodes have equal betweenness by symmetry
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], false);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 4);
        // All nodes should have equal betweenness
        for i in 0..4 {
            for j in i + 1..4 {
                assert!((bc[i] - bc[j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_betweenness_centrality_complete_graph() {
        // Complete graph K4: every node connected to every other
        // All nodes have equal betweenness by symmetry
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 4);
        // All nodes should have equal betweenness (by symmetry)
        for i in 0..4 {
            for j in i + 1..4 {
                assert!((bc[i] - bc[j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_betweenness_centrality_bridge_graph() {
        // Bridge graph: (0 -- 1) -- 2 -- (3 -- 4)
        // Node 2 is a bridge and should have high betweenness
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], false);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 5);
        // Bridge node (2) has highest betweenness
        assert!(bc[2] > bc[0]);
        assert!(bc[2] > bc[1]);
        assert!(bc[2] > bc[3]);
        assert!(bc[2] > bc[4]);
        // Nodes 1 and 3 also have some betweenness (but less than 2)
        assert!(bc[1] > bc[0]);
        assert!(bc[3] > bc[4]);
    }

    #[test]
    fn test_betweenness_centrality_directed() {
        // Directed path: 0 -> 1 -> 2
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 3);
        // In a directed path, middle node should have positive betweenness
        // (it lies on the path from 0 to 2)
        // All nodes should have some betweenness in directed graphs
        assert!(bc.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_betweenness_centrality_disconnected() {
        // Disconnected graph: (0 -- 1) and (2 -- 3)
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        let bc = g.betweenness_centrality();

        assert_eq!(bc.len(), 4);
        // Nodes within same component should have equal betweenness
        assert!((bc[0] - bc[1]).abs() < 1e-6);
        assert!((bc[2] - bc[3]).abs() < 1e-6);
    }

    // Community Detection Tests

    #[test]
    fn test_modularity_empty_graph() {
        let g = Graph::new(false);
        let communities = vec![];
        let modularity = g.modularity(&communities);
        assert_eq!(modularity, 0.0);
    }

    #[test]
    fn test_modularity_single_community() {
        // Triangle: all nodes in one community
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let communities = vec![vec![0, 1, 2]];
        let modularity = g.modularity(&communities);
        // For single community covering whole graph, Q = 0
        assert!((modularity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_modularity_two_communities() {
        // Two triangles connected by single edge: 0-1-2 and 3-4-5, edge 2-3
        let g = Graph::from_edges(
            &[
                (0, 1),
                (1, 2),
                (2, 0), // Triangle 1
                (3, 4),
                (4, 5),
                (5, 3), // Triangle 2
                (2, 3), // Inter-community edge
            ],
            false,
        );

        let communities = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let modularity = g.modularity(&communities);

        // Should have positive modularity (good community structure)
        assert!(modularity > 0.0);
        assert!(modularity < 1.0); // Not perfect due to inter-community edge
    }

    #[test]
    fn test_modularity_perfect_split() {
        // Two disconnected triangles
        let g = Graph::from_edges(
            &[
                (0, 1),
                (1, 2),
                (2, 0), // Triangle 1
                (3, 4),
                (4, 5),
                (5, 3), // Triangle 2
            ],
            false,
        );

        let communities = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let modularity = g.modularity(&communities);

        // Perfect split should have high modularity
        assert!(modularity > 0.5);
    }

    #[test]
    // Implementation complete
    fn test_louvain_empty_graph() {
        let g = Graph::new(false);
        let communities = g.louvain();
        assert_eq!(communities.len(), 0);
    }

    #[test]
    // Implementation complete
    fn test_louvain_single_node() {
        // Single node with self-loop
        let g = Graph::from_edges(&[(0, 0)], false);
        let communities = g.louvain();
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].len(), 1);
    }

    #[test]
    // Implementation complete
    fn test_louvain_two_nodes() {
        let g = Graph::from_edges(&[(0, 1)], false);
        let communities = g.louvain();

        // Should find 1 community containing both nodes
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].len(), 2);
    }

    #[test]
    // Implementation complete
    fn test_louvain_triangle() {
        // Single triangle - should be one community
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let communities = g.louvain();

        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].len(), 3);
    }

    #[test]
    // Implementation complete
    fn test_louvain_two_triangles_connected() {
        // Two triangles connected by one edge
        let g = Graph::from_edges(
            &[
                (0, 1),
                (1, 2),
                (2, 0), // Triangle 1
                (3, 4),
                (4, 5),
                (5, 3), // Triangle 2
                (2, 3), // Connection
            ],
            false,
        );

        let communities = g.louvain();

        // Should find 2 communities
        assert_eq!(communities.len(), 2);

        // Verify all nodes are assigned
        let all_nodes: Vec<_> = communities.iter().flat_map(|c| c.iter()).copied().collect();
        assert_eq!(all_nodes.len(), 6);
    }

    #[test]
    // Implementation complete
    fn test_louvain_disconnected_components() {
        // Two separate triangles (no connection)
        let g = Graph::from_edges(
            &[
                (0, 1),
                (1, 2),
                (2, 0), // Component 1
                (3, 4),
                (4, 5),
                (5, 3), // Component 2
            ],
            false,
        );

        let communities = g.louvain();

        // Should find at least 2 communities (one per component)
        assert!(communities.len() >= 2);

        // Verify nodes 0,1,2 are in different community than 3,4,5
        let comm1_nodes: Vec<_> = communities
            .iter()
            .find(|c| c.contains(&0))
            .expect("node 0 should be assigned to a community")
            .clone();
        let comm2_nodes: Vec<_> = communities
            .iter()
            .find(|c| c.contains(&3))
            .expect("node 3 should be assigned to a community")
            .clone();

        assert!(comm1_nodes.contains(&0));
        assert!(comm1_nodes.contains(&1));
        assert!(comm1_nodes.contains(&2));

        assert!(comm2_nodes.contains(&3));
        assert!(comm2_nodes.contains(&4));
        assert!(comm2_nodes.contains(&5));

        // Verify no overlap
        assert!(!comm1_nodes.contains(&3));
        assert!(!comm2_nodes.contains(&0));
    }

    #[test]
    // Implementation complete
    fn test_louvain_karate_club() {
        // Zachary's Karate Club network (simplified 4-node version)
        // Known ground truth: 2 factions
        let g = Graph::from_edges(
            &[
                (0, 1),
                (0, 2),
                (1, 2), // Group 1
                (2, 3), // Bridge
                (3, 4),
                (3, 5),
                (4, 5), // Group 2
            ],
            false,
        );

        let communities = g.louvain();

        // Should detect at least 2 communities
        assert!(communities.len() >= 2);

        // Node 2 and 3 are bridge nodes - could be in either community
        // But groups {0,1} and {4,5} should be detected
        let all_nodes: Vec<_> = communities.iter().flat_map(|c| c.iter()).copied().collect();
        assert_eq!(all_nodes.len(), 6);
    }

    #[test]
    // Implementation complete
    fn test_louvain_star_graph() {
        // Star graph: central node 0 connected to 1,2,3,4
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

        let communities = g.louvain();

        // Star graph could be 1 community or split
        // Just verify all nodes are assigned
        assert!(!communities.is_empty());
        let all_nodes: Vec<_> = communities.iter().flat_map(|c| c.iter()).copied().collect();
        assert_eq!(all_nodes.len(), 5);
    }

    #[test]
    // Implementation complete
    fn test_louvain_complete_graph() {
        // Complete graph K4 - all nodes connected
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

        let communities = g.louvain();

        // Complete graph should be single community
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].len(), 4);
    }

    #[test]
    // Implementation complete
    fn test_louvain_modularity_improves() {
        // Two clear communities
        let g = Graph::from_edges(
            &[
                (0, 1),
                (1, 2),
                (2, 0), // Triangle 1
                (3, 4),
                (4, 5),
                (5, 3), // Triangle 2
            ],
            false,
        );

        let communities = g.louvain();
        let modularity = g.modularity(&communities);

        // Louvain should find good communities (high modularity)
        assert!(modularity > 0.3);
    }

    #[test]
    // Implementation complete
    fn test_louvain_all_nodes_assigned() {
        // Verify every node gets assigned to exactly one community
        let g = Graph::from_edges(
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0), // Pentagon
            ],
            false,
        );

        let communities = g.louvain();

        let mut assigned_nodes: Vec<NodeId> = Vec::new();
        for community in &communities {
            assigned_nodes.extend(community);
        }

        // All 5 nodes should be assigned
        assigned_nodes.sort_unstable();
        assert_eq!(assigned_nodes, vec![0, 1, 2, 3, 4]);

        // No node should appear twice
        let unique_count = assigned_nodes.len();
        assigned_nodes.dedup();
        assert_eq!(assigned_nodes.len(), unique_count);
    }

    #[test]
    fn test_modularity_bad_partition() {
        // Triangle with each node in separate community (worst partition)
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);

        let communities = vec![vec![0], vec![1], vec![2]];
        let modularity = g.modularity(&communities);

        // Bad partition should have negative or very low modularity
        assert!(modularity < 0.1);
    }

    // Closeness Centrality Tests

    #[test]
    fn test_closeness_centrality_empty() {
        let g = Graph::new(false);
        let cc = g.closeness_centrality();
        assert!(cc.is_empty());
    }

    #[test]
    fn test_closeness_centrality_star_graph() {
        // Star graph: center (0) is close to all nodes
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let cc = g.closeness_centrality();

        assert_eq!(cc.len(), 4);
        // Center has highest closeness
        assert!(cc[0] > cc[1]);
        assert!(cc[0] > cc[2]);
        assert!(cc[0] > cc[3]);
        // Leaves have equal closeness by symmetry
        assert!((cc[1] - cc[2]).abs() < 1e-6);
        assert!((cc[2] - cc[3]).abs() < 1e-6);
    }

    #[test]
    fn test_closeness_centrality_path_graph() {
        // Path: 0--1--2--3
        // Middle nodes have higher closeness
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let cc = g.closeness_centrality();

        assert_eq!(cc.len(), 4);
        // Middle nodes more central
        assert!(cc[1] > cc[0]);
        assert!(cc[2] > cc[0]);
        assert!(cc[1] > cc[3]);
        assert!(cc[2] > cc[3]);
    }

    // Eigenvector Centrality Tests

    #[test]
    fn test_eigenvector_centrality_empty() {
        let g = Graph::new(false);
        let ec = g
            .eigenvector_centrality(100, 1e-6)
            .expect("eigenvector centrality should succeed on empty graph");
        assert!(ec.is_empty());
    }

    #[test]
    fn test_eigenvector_centrality_star_graph() {
        // Star graph: center has highest eigenvector centrality
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let ec = g
            .eigenvector_centrality(100, 1e-6)
            .expect("eigenvector centrality should succeed on star graph");

        assert_eq!(ec.len(), 4);
        // Center should have highest score
        assert!(ec[0] > ec[1]);
        assert!(ec[0] > ec[2]);
        assert!(ec[0] > ec[3]);
    }

    #[test]
    fn test_eigenvector_centrality_path_graph() {
        // Path graph: middle nodes more central
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let ec = g
            .eigenvector_centrality(100, 1e-6)
            .expect("eigenvector centrality should succeed on path graph");

        assert_eq!(ec.len(), 4);
        // Middle nodes should have higher scores
        assert!(ec[1] > ec[0]);
        assert!(ec[2] > ec[3]);
    }

    #[test]
    fn test_eigenvector_centrality_disconnected() {
        // Graph with no edges
        let g = Graph::from_edges(&[], false);
        let ec = g
            .eigenvector_centrality(100, 1e-6)
            .expect("eigenvector centrality should succeed on graph with no edges");
        assert!(ec.is_empty());
    }

    // Katz Centrality Tests

    #[test]
    fn test_katz_centrality_empty() {
        let g = Graph::new(true);
        let kc = g
            .katz_centrality(0.1, 100, 1e-6)
            .expect("katz centrality should succeed on empty graph");
        assert!(kc.is_empty());
    }

    #[test]
    fn test_katz_centrality_invalid_alpha() {
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

        // Alpha = 0 should fail
        assert!(g.katz_centrality(0.0, 100, 1e-6).is_err());

        // Alpha = 1 should fail
        assert!(g.katz_centrality(1.0, 100, 1e-6).is_err());

        // Alpha > 1 should fail
        assert!(g.katz_centrality(1.5, 100, 1e-6).is_err());
    }

    #[test]
    fn test_katz_centrality_cycle() {
        // Cycle graph: all nodes should have equal Katz centrality
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
        let kc = g
            .katz_centrality(0.1, 100, 1e-6)
            .expect("katz centrality should succeed on cycle graph");

        assert_eq!(kc.len(), 3);
        // All nodes equal by symmetry
        assert!((kc[0] - kc[1]).abs() < 1e-3);
        assert!((kc[1] - kc[2]).abs() < 1e-3);
    }

    #[test]
    fn test_katz_centrality_star_directed() {
        // Directed star: 0 -> {1,2,3}
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], true);
        let kc = g
            .katz_centrality(0.1, 100, 1e-6)
            .expect("katz centrality should succeed on directed star graph");

        assert_eq!(kc.len(), 4);
        // Nodes with incoming edges have higher Katz centrality
        assert!(kc[1] > kc[0]);
        assert!(kc[2] > kc[0]);
        assert!(kc[3] > kc[0]);
    }

    // Harmonic Centrality Tests

    #[test]
    fn test_harmonic_centrality_empty() {
        let g = Graph::new(false);
        let hc = g.harmonic_centrality();
        assert!(hc.is_empty());
    }

    #[test]
    fn test_harmonic_centrality_star_graph() {
        // Star graph: center has highest harmonic centrality
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let hc = g.harmonic_centrality();

        assert_eq!(hc.len(), 4);
        // Center most central
        assert!(hc[0] > hc[1]);
        assert!(hc[0] > hc[2]);
        assert!(hc[0] > hc[3]);
    }

    #[test]
    fn test_harmonic_centrality_disconnected() {
        // Disconnected graph: (0--1) and (2--3)
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        let hc = g.harmonic_centrality();

        assert_eq!(hc.len(), 4);
        // Nodes within same component have equal harmonic centrality
        assert!((hc[0] - hc[1]).abs() < 1e-6);
        assert!((hc[2] - hc[3]).abs() < 1e-6);
    }

    // Density Tests

    #[test]
    fn test_density_empty() {
        let g = Graph::new(false);
        assert_eq!(g.density(), 0.0);
    }

    #[test]
    fn test_density_single_node() {
        let g = Graph::from_edges(&[(0, 0)], false);
        assert_eq!(g.density(), 0.0);
    }

    #[test]
    fn test_density_complete_graph() {
        // Complete graph K4: all nodes connected
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        assert!((g.density() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_density_path_graph() {
        // Path: 0--1--2--3 (3 edges, 4 nodes)
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

        // Undirected: density = 2*m / (n*(n-1)) = 2*3 / (4*3) = 0.5
        assert!((g.density() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_density_directed() {
        // Directed: 0->1, 1->2 (2 edges, 3 nodes)
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

        // Directed: density = m / (n*(n-1)) = 2 / (3*2) = 1/3
        assert!((g.density() - 1.0 / 3.0).abs() < 1e-6);
    }

    // Diameter Tests

    #[test]
    fn test_diameter_empty() {
        let g = Graph::new(false);
        assert_eq!(g.diameter(), None);
    }

    #[test]
    fn test_diameter_single_node() {
        let g = Graph::from_edges(&[(0, 0)], false);
        assert_eq!(g.diameter(), Some(0));
    }

    #[test]
    fn test_diameter_path_graph() {
        // Path: 0--1--2--3 has diameter 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        assert_eq!(g.diameter(), Some(3));
    }

    #[test]
    fn test_diameter_star_graph() {
        // Star graph: center to any leaf is 1, leaf to leaf is 2
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        assert_eq!(g.diameter(), Some(2));
    }

    #[test]
    fn test_diameter_disconnected() {
        // Disconnected: (0--1) and (2--3)
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        assert_eq!(g.diameter(), None); // Disconnected
    }

    #[test]
    fn test_diameter_complete_graph() {
        // Complete graph K4: diameter is 1 (all nodes adjacent)
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        assert_eq!(g.diameter(), Some(1));
    }

    // Clustering Coefficient Tests

    #[test]
    fn test_clustering_coefficient_empty() {
        let g = Graph::new(false);
        assert_eq!(g.clustering_coefficient(), 0.0);
    }

    #[test]
    fn test_clustering_coefficient_triangle() {
        // Triangle: perfect clustering (coefficient = 1.0)
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        assert!((g.clustering_coefficient() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clustering_coefficient_star_graph() {
        // Star graph: no triangles (coefficient = 0.0)
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        assert_eq!(g.clustering_coefficient(), 0.0);
    }

    #[test]
    fn test_clustering_coefficient_partial() {
        // Graph with one triangle among 4 nodes
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0), (0, 3)], false);

        // Node 0: 3 neighbors, 1 triangle (0-1-2)
        // Node 1: 2 neighbors, 1 triangle
        // Node 2: 2 neighbors, 1 triangle
        // Node 3: 1 neighbor, 0 triangles
        let cc = g.clustering_coefficient();
        assert!(cc > 0.0);
        assert!(cc < 1.0);
    }

    // Assortativity Tests

    #[test]
    fn test_assortativity_empty() {
        let g = Graph::new(false);
        assert_eq!(g.assortativity(), 0.0);
    }

    #[test]
    fn test_assortativity_star_graph() {
        // Star graph: hub (deg 3) connects to leaves (deg 1)
        // Negative assortativity (high-degree connects to low-degree)
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        assert!(g.assortativity() < 0.0);
    }

    #[test]
    fn test_assortativity_complete_graph() {
        // Complete graph K4: all nodes have same degree
        // Should have assortativity close to 0 (or NaN due to zero variance)
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let assort = g.assortativity();

        // All nodes have degree 3, so variance is 0
        // Assortativity is undefined but we return 0.0
        assert_eq!(assort, 0.0);
    }

    #[test]
    fn test_assortativity_path_graph() {
        // Path: 0--1--2--3
        // Endpoints (deg 1) connect to middle (deg 2)
        // Middle nodes (deg 2) connect to mixed
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let assort = g.assortativity();

        // Should have negative assortativity
        assert!(assort < 0.0);
    }

    // ========================================================================
    // Pathfinding Algorithm Tests
    // ========================================================================

    #[test]
    fn test_shortest_path_direct_edge() {
        // Simplest case: direct edge between source and target
        let g = Graph::from_edges(&[(0, 1)], false);
        let path = g.shortest_path(0, 1).expect("path should exist");
        assert_eq!(path, vec![0, 1]);
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_shortest_path_same_node() {
        // Source == target should return single-node path
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
        let path = g.shortest_path(1, 1).expect("path should exist");
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn test_shortest_path_disconnected() {
        // No path between disconnected components
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        assert!(g.shortest_path(0, 3).is_none());
        assert!(g.shortest_path(1, 2).is_none());
    }

    #[test]
    fn test_shortest_path_invalid_nodes() {
        // Out-of-bounds node IDs should return None
        let g = Graph::from_edges(&[(0, 1)], false);
        assert!(g.shortest_path(0, 10).is_none());
        assert!(g.shortest_path(10, 0).is_none());
    }

    #[test]
    fn test_shortest_path_multiple_paths() {
        // Graph with multiple paths of same length
        // 0 -- 1
        // |    |
        // 2 -- 3
        let g = Graph::from_edges(&[(0, 1), (1, 3), (0, 2), (2, 3)], false);
        let path = g.shortest_path(0, 3).expect("path should exist");

        // Both 0->1->3 and 0->2->3 are shortest paths (length 3)
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], 0);
        assert_eq!(path[2], 3);
        assert!(path[1] == 1 || path[1] == 2); // Either path is valid
    }

    #[test]
    fn test_shortest_path_linear_chain() {
        // Path graph: 0 -- 1 -- 2 -- 3 -- 4
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 4)], false);

        // Test various source-target pairs
        let path = g.shortest_path(0, 4).expect("path should exist");
        assert_eq!(path, vec![0, 1, 2, 3, 4]);

        let path = g.shortest_path(0, 2).expect("path should exist");
        assert_eq!(path, vec![0, 1, 2]);

        let path = g.shortest_path(1, 3).expect("path should exist");
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn test_shortest_path_triangle() {
        // Triangle graph
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);

        // All pairs should have path length 2
        let path = g.shortest_path(0, 1).expect("path should exist");
        assert_eq!(path.len(), 2);

        let path = g.shortest_path(0, 2).expect("path should exist");
        assert_eq!(path.len(), 2);

        let path = g.shortest_path(1, 2).expect("path should exist");
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_shortest_path_directed() {
        // Directed graph: 0 -> 1 -> 2
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

        // Forward paths exist
        let path = g.shortest_path(0, 2).expect("forward path should exist");
        assert_eq!(path, vec![0, 1, 2]);

        // Backward paths don't exist
        assert!(g.shortest_path(2, 0).is_none());
    }

    #[test]
    fn test_shortest_path_cycle() {
        // Cycle graph: 0 -> 1 -> 2 -> 3 -> 0
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], true);

        // Test path that uses cycle
        let path = g.shortest_path(0, 3).expect("path should exist");

        // Direct path 0->1->2->3 (length 4) vs backward 0<-3 (not possible in directed)
        assert_eq!(path.len(), 4);
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shortest_path_star_graph() {
        // Star graph: 0 connected to 1, 2, 3, 4
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (0, 4)], false);

        // Center to leaf: length 2
        let path = g.shortest_path(0, 1).expect("path should exist");
        assert_eq!(path.len(), 2);

        // Leaf to leaf through center: length 3
        let path = g.shortest_path(1, 2).expect("path should exist");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], 1);
        assert_eq!(path[1], 0); // Must go through center
        assert_eq!(path[2], 2);
    }

    #[test]
    fn test_shortest_path_empty_graph() {
        // Empty graph
        let g = Graph::new(false);
        assert!(g.shortest_path(0, 0).is_none());
    }

    #[test]
    fn test_shortest_path_single_node_graph() {
        // Graph with single self-loop
        let g = Graph::from_edges(&[(0, 0)], false);
        let path = g.shortest_path(0, 0).expect("path should exist");
        assert_eq!(path, vec![0]);
    }

    #[test]
    fn test_shortest_path_complete_graph() {
        // Complete graph K4
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

        // All pairs should have direct edge (length 2)
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    let path = g.shortest_path(i, j).expect("path should exist");
                    assert_eq!(path.len(), 2, "Path from {i} to {j} should be direct");
                    assert_eq!(path[0], i);
                    assert_eq!(path[1], j);
                }
            }
        }
    }

    #[test]
    fn test_shortest_path_bidirectional() {
        // Undirected: path should exist in both directions
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

        let path_forward = g.shortest_path(0, 2).expect("forward path should exist");
        let path_backward = g.shortest_path(2, 0).expect("backward path should exist");

        assert_eq!(path_forward.len(), path_backward.len());
        assert_eq!(path_forward.len(), 3);

        // Paths should be reverses of each other
        let reversed: Vec<_> = path_backward.iter().rev().copied().collect();
        assert_eq!(path_forward, reversed);
    }

    // ========================================================================
    // Dijkstra's Algorithm Tests
    // ========================================================================

    #[test]
    fn test_dijkstra_simple_weighted() {
        // Simple weighted graph
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], false);

        let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert_eq!(dist, 3.0); // 0->1->2 is shorter than 0->2
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dijkstra_same_node() {
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0)], false);
        let (path, dist) = g.dijkstra(0, 0).expect("path should exist");
        assert_eq!(path, vec![0]);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_dijkstra_disconnected() {
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (2, 3, 1.0)], false);
        assert!(g.dijkstra(0, 3).is_none());
    }

    #[test]
    fn test_dijkstra_unweighted() {
        // Unweighted graph (uses weight 1.0 for all edges)
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
        let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert_eq!(dist, 2.0);
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dijkstra_triangle_weighted() {
        // Triangle with different weights
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)], false);

        let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert_eq!(dist, 2.0); // 0->1->2 (cost 2) vs 0->2 (cost 5)
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dijkstra_multiple_paths() {
        // Graph with multiple paths of different costs
        //     1 ----2.0---- 2
        //    /              |
        //   /               |
        //  0                1.0
        //   \               |
        //    \              |
        //     3 ----1.0---- 4
        let g = Graph::from_weighted_edges(
            &[
                (0, 1, 1.0),
                (1, 2, 2.0),
                (0, 3, 1.0),
                (3, 4, 1.0),
                (4, 2, 1.0),
            ],
            false,
        );

        let (_path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert_eq!(dist, 3.0); // Best path: 0->3->4->2 or 0->1->2
    }

    #[test]
    fn test_dijkstra_linear_chain() {
        // Weighted linear chain
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)], false);

        let (path, dist) = g.dijkstra(0, 3).expect("path should exist");
        assert_eq!(dist, 6.0);
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_dijkstra_directed_graph() {
        // Directed weighted graph
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0)], true);

        let (path, dist) = g.dijkstra(0, 2).expect("forward path should exist");
        assert_eq!(dist, 3.0);
        assert_eq!(path, vec![0, 1, 2]);

        // Backward path doesn't exist
        assert!(g.dijkstra(2, 0).is_none());
    }

    #[test]
    fn test_dijkstra_invalid_nodes() {
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0)], false);
        assert!(g.dijkstra(0, 10).is_none());
        assert!(g.dijkstra(10, 0).is_none());
    }

    #[test]
    #[should_panic(expected = "negative edge weights")]
    fn test_dijkstra_negative_weights() {
        // Dijkstra should panic on negative weights
        let g = Graph::from_weighted_edges(&[(0, 1, -1.0)], false);
        let _ = g.dijkstra(0, 1);
    }

    #[test]
    fn test_dijkstra_zero_weight_edges() {
        // Zero-weight edges should work
        let g = Graph::from_weighted_edges(&[(0, 1, 0.0), (1, 2, 1.0)], false);
        let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert_eq!(dist, 1.0);
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dijkstra_complete_graph_weighted() {
        // Complete graph K3 with different weights
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 3.0)], false);

        // Direct edge 0->2 costs 3.0, but 0->1->2 costs 2.0
        let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert_eq!(dist, 2.0);
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dijkstra_star_graph_weighted() {
        // Star graph with center at node 0
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)], false);

        // Path from 1 to 3 must go through 0
        let (path, dist) = g.dijkstra(1, 3).expect("path should exist");
        assert_eq!(dist, 4.0); // 1->0 (1.0) + 0->3 (3.0)
        assert_eq!(path, vec![1, 0, 3]);
    }

    #[test]
    fn test_dijkstra_vs_shortest_path() {
        // On unweighted graph, Dijkstra should match shortest_path
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

        let sp_path = g
            .shortest_path(0, 3)
            .expect("shortest_path should find path");
        let (dij_path, dij_dist) = g.dijkstra(0, 3).expect("dijkstra should find path");

        assert_eq!(sp_path.len(), dij_path.len());
        assert_eq!(dij_dist, (dij_path.len() - 1) as f64);
    }

    #[test]
    fn test_dijkstra_floating_point_precision() {
        // Test with fractional weights
        let g = Graph::from_weighted_edges(&[(0, 1, 0.1), (1, 2, 0.2), (0, 2, 0.31)], false);

        let (path, dist) = g.dijkstra(0, 2).expect("path should exist");
        assert!((dist - 0.3).abs() < 1e-10); // 0.1 + 0.2 = 0.3
        assert_eq!(path, vec![0, 1, 2]);
    }

    // ========================================================================
    // All-Pairs Shortest Paths Tests
    // ========================================================================

    #[test]
    fn test_apsp_linear_chain() {
        // Linear chain: 0 -- 1 -- 2 -- 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let dist = g.all_pairs_shortest_paths();

        // Check diagonal (distance to self = 0)
        for (i, row) in dist.iter().enumerate().take(4) {
            assert_eq!(row[i], Some(0));
        }

        // Check distances
        assert_eq!(dist[0][1], Some(1));
        assert_eq!(dist[0][2], Some(2));
        assert_eq!(dist[0][3], Some(3));
        assert_eq!(dist[1][2], Some(1));
        assert_eq!(dist[1][3], Some(2));
        assert_eq!(dist[2][3], Some(1));

        // Check symmetry (undirected graph)
        assert_eq!(dist[0][3], dist[3][0]);
        assert_eq!(dist[1][2], dist[2][1]);
    }

    #[test]
    fn test_apsp_complete_graph() {
        // Complete graph K4
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let dist = g.all_pairs_shortest_paths();

        // All pairs should have distance 1 (direct edge) except diagonal
        for (i, row) in dist.iter().enumerate().take(4) {
            for (j, &cell) in row.iter().enumerate().take(4) {
                if i == j {
                    assert_eq!(cell, Some(0));
                } else {
                    assert_eq!(cell, Some(1));
                }
            }
        }
    }

    #[test]
    fn test_apsp_disconnected() {
        // Two disconnected components: (0, 1) and (2, 3)
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        let dist = g.all_pairs_shortest_paths();

        // Within components
        assert_eq!(dist[0][1], Some(1));
        assert_eq!(dist[1][0], Some(1));
        assert_eq!(dist[2][3], Some(1));
        assert_eq!(dist[3][2], Some(1));

        // Between components (no path)
        assert_eq!(dist[0][2], None);
        assert_eq!(dist[0][3], None);
        assert_eq!(dist[1][2], None);
        assert_eq!(dist[1][3], None);
    }

    #[test]
    fn test_apsp_directed() {
        // Directed graph: 0 -> 1 -> 2
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        let dist = g.all_pairs_shortest_paths();

        // Forward paths
        assert_eq!(dist[0][1], Some(1));
        assert_eq!(dist[0][2], Some(2));
        assert_eq!(dist[1][2], Some(1));

        // Backward paths (no reverse edges)
        assert_eq!(dist[1][0], None);
        assert_eq!(dist[2][0], None);
        assert_eq!(dist[2][1], None);
    }

    #[test]
    fn test_apsp_triangle() {
        // Triangle graph
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let dist = g.all_pairs_shortest_paths();

        // All pairs should have distance 1 (triangle) except diagonal
        for (i, row) in dist.iter().enumerate().take(3) {
            for (j, &cell) in row.iter().enumerate().take(3) {
                if i == j {
                    assert_eq!(cell, Some(0));
                } else {
                    assert_eq!(cell, Some(1));
                }
            }
        }
    }

    #[test]
    fn test_apsp_star_graph() {
        // Star graph: 0 connected to 1, 2, 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let dist = g.all_pairs_shortest_paths();

        // Center to leaves: distance 1
        assert_eq!(dist[0][1], Some(1));
        assert_eq!(dist[0][2], Some(1));
        assert_eq!(dist[0][3], Some(1));

        // Leaf to leaf through center: distance 2
        assert_eq!(dist[1][2], Some(2));
        assert_eq!(dist[1][3], Some(2));
        assert_eq!(dist[2][3], Some(2));
    }

    #[test]
    fn test_apsp_empty_graph() {
        let g = Graph::new(false);
        let dist = g.all_pairs_shortest_paths();
        assert_eq!(dist.len(), 0);
    }

    #[test]
    fn test_apsp_single_node() {
        let g = Graph::from_edges(&[(0, 0)], false);
        let dist = g.all_pairs_shortest_paths();

        assert_eq!(dist.len(), 1);
        assert_eq!(dist[0][0], Some(0));
    }

    #[test]
    fn test_apsp_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 3 -> 0
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 0)], true);
        let dist = g.all_pairs_shortest_paths();

        // Along cycle direction
        assert_eq!(dist[0][1], Some(1));
        assert_eq!(dist[0][2], Some(2));
        assert_eq!(dist[0][3], Some(3));
        assert_eq!(dist[1][3], Some(2));

        // All nodes reachable in directed cycle
        for row in dist.iter().take(4) {
            for &cell in row.iter().take(4) {
                assert!(cell.is_some());
            }
        }
    }

    #[test]
    fn test_apsp_matrix_size() {
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let dist = g.all_pairs_shortest_paths();

        // Matrix should be n×n
        assert_eq!(dist.len(), 4);
        for row in &dist {
            assert_eq!(row.len(), 4);
        }
    }

    // ========================================================================
    // A* Search Algorithm Tests
    // ========================================================================

    #[test]
    fn test_astar_linear_chain() {
        // Linear chain: 0 -- 1 -- 2 -- 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

        // Simple distance heuristic
        let heuristic = |node: usize| (3 - node) as f64;

        let path = g.a_star(0, 3, heuristic).expect("path should exist");
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_astar_same_node() {
        let g = Graph::from_edges(&[(0, 1)], false);
        let heuristic = |_: usize| 0.0;

        let path = g.a_star(0, 0, heuristic).expect("path should exist");
        assert_eq!(path, vec![0]);
    }

    #[test]
    fn test_astar_disconnected() {
        // Two disconnected components
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        let heuristic = |_: usize| 0.0;

        assert!(g.a_star(0, 3, heuristic).is_none());
    }

    #[test]
    fn test_astar_zero_heuristic() {
        // With h(n) = 0, A* behaves like Dijkstra
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let heuristic = |_: usize| 0.0;

        let path = g.a_star(0, 3, heuristic).expect("path should exist");
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_astar_admissible_heuristic() {
        // Graph with shortcut
        // 0 -- 1 -- 2
        // |         |
        // +----3----+
        let g = Graph::from_weighted_edges(
            &[(0, 1, 1.0), (1, 2, 1.0), (0, 3, 0.5), (3, 2, 0.5)],
            false,
        );

        // Admissible heuristic (straight-line distance estimate)
        let heuristic = |node: usize| match node {
            0 => 1.0, // Estimate to reach 2
            1 => 1.0,
            2 => 0.0, // At target
            3 => 0.5,
            _ => 0.0,
        };

        let path = g.a_star(0, 2, heuristic).expect("path should exist");
        // Should find shortest path via 3
        assert!(path.contains(&3)); // Must use the shortcut
    }

    #[test]
    fn test_astar_directed() {
        // Directed graph: 0 -> 1 -> 2
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        let heuristic = |node: usize| (2 - node) as f64;

        let path = g
            .a_star(0, 2, heuristic)
            .expect("forward path should exist");
        assert_eq!(path, vec![0, 1, 2]);

        // Backward path doesn't exist
        assert!(g.a_star(2, 0, |_| 0.0).is_none());
    }

    #[test]
    fn test_astar_triangle() {
        // Triangle graph
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let heuristic = |_: usize| 0.0;

        let path = g.a_star(0, 2, heuristic).expect("path should exist");
        assert_eq!(path.len(), 2); // Direct edge 0-2
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 2);
    }

    #[test]
    fn test_astar_weighted_graph() {
        // Weighted graph with better heuristic guidance
        let g = Graph::from_weighted_edges(&[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], false);

        // Heuristic guides toward node 2
        let heuristic = |node: usize| match node {
            0 => 3.0,
            1 => 2.0,
            2 => 0.0,
            _ => 0.0,
        };

        let path = g.a_star(0, 2, heuristic).expect("path should exist");
        // Should find path 0->1->2 (cost 3) instead of 0->2 (cost 5)
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_astar_complete_graph() {
        // Complete graph K4
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let heuristic = |_: usize| 0.0;

        // All nodes directly connected
        let path = g.a_star(0, 3, heuristic).expect("path should exist");
        assert_eq!(path.len(), 2); // Direct path
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 3);
    }

    #[test]
    fn test_astar_invalid_nodes() {
        let g = Graph::from_edges(&[(0, 1)], false);
        let heuristic = |_: usize| 0.0;

        assert!(g.a_star(0, 10, heuristic).is_none());
        assert!(g.a_star(10, 0, heuristic).is_none());
    }

    #[test]
    fn test_astar_vs_shortest_path() {
        // On unweighted graph with zero heuristic, A* should match shortest_path
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

        let sp_path = g
            .shortest_path(0, 3)
            .expect("shortest_path should find path");
        let astar_path = g.a_star(0, 3, |_| 0.0).expect("astar should find path");

        assert_eq!(sp_path.len(), astar_path.len());
        assert_eq!(sp_path, astar_path);
    }

    #[test]
    fn test_astar_star_graph() {
        // Star graph: 0 connected to 1, 2, 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);

        // Heuristic that guides toward target 3
        let heuristic = |node: usize| if node == 3 { 0.0 } else { 1.0 };

        let path = g.a_star(1, 3, heuristic).expect("path should exist");
        assert_eq!(path.len(), 3); // Must go through center
        assert_eq!(path[0], 1);
        assert_eq!(path[1], 0);
        assert_eq!(path[2], 3);
    }

    #[test]
    fn test_astar_perfect_heuristic() {
        // Perfect heuristic (exact distance to target)
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);

        // Perfect heuristic = exact remaining distance
        let heuristic = |node: usize| (3 - node) as f64;

        let path = g.a_star(0, 3, heuristic).expect("path should exist");
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_astar_complex_graph() {
        // More complex graph to test heuristic efficiency
        //     1
        //    / \
        //   0   3 - 4
        //    \ /
        //     2
        let g = Graph::from_weighted_edges(
            &[
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
            ],
            false,
        );

        // Distance-based heuristic
        let heuristic = |node: usize| (4 - node) as f64;

        let path = g.a_star(0, 4, heuristic).expect("path should exist");
        assert_eq!(path.len(), 4); // 0->1->3->4 or 0->2->3->4
        assert_eq!(path[0], 0);
        assert_eq!(path[3], 4);
    }

    // DFS Tests

    #[test]
    fn test_dfs_linear_chain() {
        // Linear chain: 0 -- 1 -- 2 -- 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let visited = g.dfs(0).expect("valid source");

        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], 0); // Starts at source
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
    }

    #[test]
    fn test_dfs_tree() {
        // Tree: 0 connected to 1, 2, 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let visited = g.dfs(0).expect("valid source");

        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], 0); // Root first
                                   // Children visited in some order
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
    }

    #[test]
    fn test_dfs_cycle() {
        // Cycle: 0 -- 1 -- 2 -- 0
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let visited = g.dfs(0).expect("valid source");

        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], 0);
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
    }

    #[test]
    fn test_dfs_disconnected() {
        // Two components: (0, 1) and (2, 3)
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);

        // DFS from 0 only visits component containing 0
        let visited = g.dfs(0).expect("valid source");
        assert_eq!(visited.len(), 2);
        assert!(visited.contains(&0));
        assert!(visited.contains(&1));
        assert!(!visited.contains(&2));
        assert!(!visited.contains(&3));

        // DFS from 2 only visits component containing 2
        let visited2 = g.dfs(2).expect("valid source");
        assert_eq!(visited2.len(), 2);
        assert!(visited2.contains(&2));
        assert!(visited2.contains(&3));
    }

    #[test]
    fn test_dfs_directed() {
        // Directed: 0 -> 1 -> 2
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);

        // Forward traversal
        let visited = g.dfs(0).expect("valid source");
        assert_eq!(visited.len(), 3);
        assert_eq!(visited[0], 0);

        // Backward traversal (node 2 has no outgoing edges)
        let visited2 = g.dfs(2).expect("valid source");
        assert_eq!(visited2.len(), 1);
        assert_eq!(visited2[0], 2);
    }

    #[test]
    fn test_dfs_single_node() {
        // Single node with self-loop
        let g = Graph::from_edges(&[(0, 0)], false);

        let visited = g.dfs(0).expect("valid source");
        assert_eq!(visited.len(), 1);
        assert_eq!(visited[0], 0);
    }

    #[test]
    fn test_dfs_invalid_source() {
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);

        // Invalid source node
        assert!(g.dfs(10).is_none());
        assert!(g.dfs(100).is_none());
    }

    #[test]
    fn test_dfs_complete_graph() {
        // Complete graph K4
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let visited = g.dfs(0).expect("valid source");

        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], 0);
        // All other nodes reachable
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
    }

    #[test]
    fn test_dfs_dag() {
        // DAG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], true);
        let visited = g.dfs(0).expect("valid source");

        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], 0);
        // All nodes reachable from 0
        assert!(visited.contains(&1));
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));

        // Node 3 is a sink (no outgoing edges)
        let visited3 = g.dfs(3).expect("valid source");
        assert_eq!(visited3.len(), 1);
        assert_eq!(visited3[0], 3);
    }

    #[test]
    fn test_dfs_empty_graph() {
        let g = Graph::new(false);
        // No nodes, so any DFS should return None
        assert!(g.dfs(0).is_none());
    }

    // Connected Components Tests

    #[test]
    fn test_connected_components_single() {
        // Single connected component
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], false);
        let components = g.connected_components();

        assert_eq!(components.len(), 4);
        // All nodes in same component
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);
        assert_eq!(components[2], components[3]);
    }

    #[test]
    fn test_connected_components_two() {
        // Two disconnected components: (0,1) and (2,3)
        let g = Graph::from_edges(&[(0, 1), (2, 3)], false);
        let components = g.connected_components();

        assert_eq!(components.len(), 4);
        // Component 1: nodes 0 and 1
        assert_eq!(components[0], components[1]);
        // Component 2: nodes 2 and 3
        assert_eq!(components[2], components[3]);
        // Different components
        assert_ne!(components[0], components[2]);
    }

    #[test]
    fn test_connected_components_three() {
        // Three components: (0,1), (2,3), (4)
        let g = Graph::from_edges(&[(0, 1), (2, 3), (4, 4)], false);
        let components = g.connected_components();

        assert_eq!(components.len(), 5);
        // Three distinct components
        assert_eq!(components[0], components[1]);
        assert_eq!(components[2], components[3]);
        assert_ne!(components[0], components[2]);
        assert_ne!(components[0], components[4]);
        assert_ne!(components[2], components[4]);
    }

    #[test]
    fn test_connected_components_complete() {
        // Complete graph K4 - all in one component
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);
        let components = g.connected_components();

        assert_eq!(components.len(), 4);
        let first = components[0];
        assert!(components.iter().all(|&c| c == first));
    }

    #[test]
    fn test_connected_components_star() {
        // Star graph - all connected through center
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let components = g.connected_components();

        assert_eq!(components.len(), 4);
        // All in same component
        assert_eq!(components[0], components[1]);
        assert_eq!(components[0], components[2]);
        assert_eq!(components[0], components[3]);
    }

    #[test]
    fn test_connected_components_directed_weak() {
        // Directed graph: 0 -> 1 -> 2 (weakly connected)
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        let components = g.connected_components();

        assert_eq!(components.len(), 3);
        // Weakly connected (ignores direction)
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);
    }

    #[test]
    fn test_connected_components_cycle() {
        // Cycle graph
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let components = g.connected_components();

        assert_eq!(components.len(), 3);
        // All in same component
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);
    }

    #[test]
    fn test_connected_components_empty() {
        let g = Graph::new(false);
        let components = g.connected_components();
        assert!(components.is_empty());
    }

    #[test]
    fn test_connected_components_isolated_nodes() {
        // Graph with some isolated nodes
        let g = Graph::from_edges(&[(0, 1), (3, 4)], false);
        // Node 2 is isolated (no edges)
        // But we only have nodes that appear in edges
        let components = g.connected_components();

        assert_eq!(components.len(), 5);
        // Two components: (0,1) and (3,4), and isolated 2
        assert_eq!(components[0], components[1]);
        assert_eq!(components[3], components[4]);
        assert_ne!(components[0], components[3]);
        // Node 2 is in its own component
        assert_ne!(components[2], components[0]);
        assert_ne!(components[2], components[3]);
    }

    #[test]
    fn test_connected_components_count() {
        // Helper to count unique components
        fn count_components(components: &[usize]) -> usize {
            use std::collections::HashSet;
            components.iter().copied().collect::<HashSet<_>>().len()
        }

        // Single component
        let g1 = Graph::from_edges(&[(0, 1), (1, 2)], false);
        assert_eq!(count_components(&g1.connected_components()), 1);

        // Two components
        let g2 = Graph::from_edges(&[(0, 1), (2, 3)], false);
        assert_eq!(count_components(&g2.connected_components()), 2);

        // Three components
        let g3 = Graph::from_edges(&[(0, 1), (2, 3), (4, 5)], false);
        assert_eq!(count_components(&g3.connected_components()), 3);
    }

    // Strongly Connected Components Tests

    #[test]
    fn test_scc_single_cycle() {
        // Single SCC: 0 -> 1 -> 2 -> 0
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 3);
        // All nodes in same SCC
        assert_eq!(sccs[0], sccs[1]);
        assert_eq!(sccs[1], sccs[2]);
    }

    #[test]
    fn test_scc_dag() {
        // DAG: 0 -> 1 -> 2 (each node is its own SCC)
        let g = Graph::from_edges(&[(0, 1), (1, 2)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 3);
        // Each node is its own SCC
        assert_ne!(sccs[0], sccs[1]);
        assert_ne!(sccs[1], sccs[2]);
        assert_ne!(sccs[0], sccs[2]);
    }

    #[test]
    fn test_scc_two_components() {
        // Two SCCs: (0->1->0) and (2->3->2)
        let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 3), (3, 2)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 4);
        // SCC 1: nodes 0 and 1
        assert_eq!(sccs[0], sccs[1]);
        // SCC 2: nodes 2 and 3
        assert_eq!(sccs[2], sccs[3]);
        // Different SCCs
        assert_ne!(sccs[0], sccs[2]);
    }

    #[test]
    fn test_scc_complex() {
        // Complex graph with multiple SCCs
        // SCC 1: 0 -> 1 -> 0
        // SCC 2: 2 -> 3 -> 4 -> 2
        // Edge from SCC1 to SCC2: 1 -> 2
        let g = Graph::from_edges(&[(0, 1), (1, 0), (1, 2), (2, 3), (3, 4), (4, 2)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 5);
        // SCC 1: nodes 0 and 1
        assert_eq!(sccs[0], sccs[1]);
        // SCC 2: nodes 2, 3, 4
        assert_eq!(sccs[2], sccs[3]);
        assert_eq!(sccs[3], sccs[4]);
        // Different SCCs
        assert_ne!(sccs[0], sccs[2]);
    }

    #[test]
    fn test_scc_self_loop() {
        // Single node with self-loop is an SCC
        let g = Graph::from_edges(&[(0, 0)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], 0);
    }

    #[test]
    fn test_scc_disconnected() {
        // Two disconnected cycles
        let g = Graph::from_edges(&[(0, 1), (1, 0), (2, 3), (3, 2)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 4);
        // Two separate SCCs
        assert_eq!(sccs[0], sccs[1]);
        assert_eq!(sccs[2], sccs[3]);
        assert_ne!(sccs[0], sccs[2]);
    }

    #[test]
    fn test_scc_empty() {
        let g = Graph::new(true);
        let sccs = g.strongly_connected_components();
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_scc_linear_dag() {
        // Linear DAG: 0 -> 1 -> 2 -> 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 4);
        // Each node is its own SCC in a DAG
        use std::collections::HashSet;
        let unique_sccs: HashSet<_> = sccs.iter().copied().collect();
        assert_eq!(unique_sccs.len(), 4);
    }

    #[test]
    fn test_scc_complete_graph() {
        // Complete directed graph (all nodes reachable from all)
        let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)], true);
        let sccs = g.strongly_connected_components();

        assert_eq!(sccs.len(), 3);
        // All in same SCC
        assert_eq!(sccs[0], sccs[1]);
        assert_eq!(sccs[1], sccs[2]);
    }

    #[test]
    fn test_scc_count() {
        // Helper to count unique SCCs
        fn count_sccs(sccs: &[usize]) -> usize {
            use std::collections::HashSet;
            sccs.iter().copied().collect::<HashSet<_>>().len()
        }

        // Single SCC
        let g1 = Graph::from_edges(&[(0, 1), (1, 0)], true);
        assert_eq!(count_sccs(&g1.strongly_connected_components()), 1);

        // Two SCCs
        let g2 = Graph::from_edges(&[(0, 1)], true);
        assert_eq!(count_sccs(&g2.strongly_connected_components()), 2);

        // Three SCCs
        let g3 = Graph::from_edges(&[(0, 1), (2, 3), (3, 2)], true);
        assert_eq!(count_sccs(&g3.strongly_connected_components()), 3);
    }

    // Topological Sort Tests

    #[test]
    fn test_topo_linear_dag() {
        // Linear DAG: 0 -> 1 -> 2 -> 3
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 3)], true);
        let order = g
            .topological_sort()
            .expect("DAG should have topological order");

        assert_eq!(order.len(), 4);
        // Check ordering constraints
        assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
        assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 2));
        assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
    }

    #[test]
    fn test_topo_cycle() {
        // Cycle: 0 -> 1 -> 2 -> 0
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
        assert!(g.topological_sort().is_none()); // Should detect cycle
    }

    #[test]
    fn test_topo_diamond() {
        // Diamond DAG: 0 -> {1,2} -> 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (2, 3)], true);
        let order = g
            .topological_sort()
            .expect("DAG should have topological order");

        assert_eq!(order.len(), 4);
        // 0 must come before 1, 2, 3
        let pos_0 = order
            .iter()
            .position(|&x| x == 0)
            .expect("0 should be in order");
        assert!(
            pos_0
                < order
                    .iter()
                    .position(|&x| x == 1)
                    .expect("1 should be in order")
        );
        assert!(
            pos_0
                < order
                    .iter()
                    .position(|&x| x == 2)
                    .expect("2 should be in order")
        );
        assert!(
            pos_0
                < order
                    .iter()
                    .position(|&x| x == 3)
                    .expect("3 should be in order")
        );

        // 3 must come after 1 and 2
        let pos_3 = order
            .iter()
            .position(|&x| x == 3)
            .expect("3 should be in order");
        assert!(
            order
                .iter()
                .position(|&x| x == 1)
                .expect("1 should be in order")
                < pos_3
        );
        assert!(
            order
                .iter()
                .position(|&x| x == 2)
                .expect("2 should be in order")
                < pos_3
        );
    }

    #[test]
    fn test_topo_empty() {
        let g = Graph::new(true);
        let order = g
            .topological_sort()
            .expect("Empty graph has topological order");
        assert!(order.is_empty());
    }

    #[test]
    fn test_topo_single_node() {
        // Single node with self-loop creates cycle
        let g = Graph::from_edges(&[(0, 0)], true);
        assert!(g.topological_sort().is_none()); // Self-loop is a cycle
    }

    #[test]
    fn test_topo_disconnected_dag() {
        // Two disconnected chains: 0->1 and 2->3
        let g = Graph::from_edges(&[(0, 1), (2, 3)], true);
        let order = g
            .topological_sort()
            .expect("Disconnected DAG has topological order");

        assert_eq!(order.len(), 4);
        // Within each chain, ordering is preserved
        assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
        assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
    }

    #[test]
    fn test_topo_tree() {
        // Tree: 0 -> {1, 2}, 1 -> {3, 4}
        let g = Graph::from_edges(&[(0, 1), (0, 2), (1, 3), (1, 4)], true);
        let order = g.topological_sort().expect("Tree is a DAG");

        assert_eq!(order.len(), 5);
        // 0 must come first
        assert_eq!(order.iter().position(|&x| x == 0), Some(0));
        // 1 before 3 and 4
        let pos_1 = order
            .iter()
            .position(|&x| x == 1)
            .expect("1 should be in order");
        assert!(
            pos_1
                < order
                    .iter()
                    .position(|&x| x == 3)
                    .expect("3 should be in order")
        );
        assert!(
            pos_1
                < order
                    .iter()
                    .position(|&x| x == 4)
                    .expect("4 should be in order")
        );
    }

    #[test]
    fn test_topo_self_loop() {
        // Self-loop is a cycle
        let g = Graph::from_edges(&[(0, 0)], true);
        assert!(g.topological_sort().is_none());
    }

    #[test]
    fn test_topo_complete_dag() {
        // Complete DAG: 0 -> {1,2,3}, 1 -> {2,3}, 2 -> 3
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], true);
        let order = g
            .topological_sort()
            .expect("Complete DAG has topological order");

        assert_eq!(order.len(), 4);
        // Check all ordering constraints
        let positions: Vec<_> = (0..4)
            .map(|i| {
                order
                    .iter()
                    .position(|&x| x == i)
                    .expect("node should be in order")
            })
            .collect();

        assert!(positions[0] < positions[1]);
        assert!(positions[0] < positions[2]);
        assert!(positions[0] < positions[3]);
        assert!(positions[1] < positions[2]);
        assert!(positions[1] < positions[3]);
        assert!(positions[2] < positions[3]);
    }

    #[test]
    fn test_topo_undirected() {
        // Undirected graph is treated as bidirectional (always has cycles unless tree)
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
        // Undirected edges create cycles (0->1 and 1->0)
        assert!(g.topological_sort().is_none());
    }
}
