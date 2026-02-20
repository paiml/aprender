impl Graph {

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
    /// let (path, dist) = g.dijkstra(0, 2).expect("dijkstra path should exist");
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
    /// let path = g.a_star(0, 3, heuristic).expect("a_star path should exist");
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
}
