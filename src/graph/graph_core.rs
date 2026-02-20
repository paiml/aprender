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

    // Centrality methods moved to graph::centrality module (GraphCentrality trait)
    // See centrality.rs for: degree_centrality, pagerank, betweenness_centrality,
    // closeness_centrality, eigenvector_centrality, katz_centrality, harmonic_centrality

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
}
