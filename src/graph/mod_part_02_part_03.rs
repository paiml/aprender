impl Graph {

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
    /// let path = g.shortest_path(0, 3).expect("path from 0 to 3 should exist");
    /// assert_eq!(path.len(), 2); // 0 -> 3 (direct edge)
    /// assert_eq!(path[0], 0);
    /// assert_eq!(path[1], 3);
    ///
    /// // Path 0 to 2
    /// let path = g.shortest_path(0, 2).expect("path from 0 to 2 should exist");
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
}
