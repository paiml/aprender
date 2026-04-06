impl Graph {

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
}
