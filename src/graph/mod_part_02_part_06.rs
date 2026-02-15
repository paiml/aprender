impl Graph {

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
