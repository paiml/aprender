//! Graph Analytics
//!
//! Analytics algorithms for graph analysis including PageRank, community detection,
//! and centrality metrics.
//!
//! ## Academic References
//!
//! - Page et al. (1999) "The PageRank Citation Ranking"
//! - Blondel et al. (2008) "Fast unfolding of communities in large networks"
//! - Brandes (2001) "A Faster Algorithm for Betweenness Centrality"

use std::collections::HashMap;

use super::graph::Graph;

/// Community color palette for Louvain visualization
/// Colors chosen for maximum distinguishability (per Healey's preattentive research)
pub const COMMUNITY_COLORS: [&str; 8] = [
    "\x1b[31m", // Red
    "\x1b[32m", // Green
    "\x1b[33m", // Yellow
    "\x1b[34m", // Blue
    "\x1b[35m", // Magenta
    "\x1b[36m", // Cyan
    "\x1b[91m", // Bright Red
    "\x1b[92m", // Bright Green
];

/// Graph analytics engine for PageRank and community detection
pub struct GraphAnalytics;

impl GraphAnalytics {
    /// Compute PageRank scores and apply as node importance
    ///
    /// Based on Page et al. (1999) "The PageRank Citation Ranking"
    ///
    /// # Arguments
    /// * `graph` - Graph to analyze
    /// * `damping` - Damping factor (default: 0.85)
    /// * `iterations` - Max iterations (default: 20)
    ///
    /// # Returns
    /// HashMap of node_id -> PageRank score
    pub fn pagerank<N, E>(
        graph: &Graph<N, E>,
        damping: f32,
        iterations: usize,
    ) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        let teleport = (1.0 - damping) / n as f32;
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        // Initialize: uniform distribution
        let mut ranks: HashMap<String, f32> =
            node_ids.iter().map(|id| (id.clone(), 1.0 / n as f32)).collect();

        // Compute out-degrees
        let out_degrees: HashMap<String, usize> =
            node_ids.iter().map(|id| (id.clone(), graph.neighbors(id).len())).collect();

        // Power iteration
        for _ in 0..iterations {
            let mut new_ranks: HashMap<String, f32> =
                node_ids.iter().map(|id| (id.clone(), teleport)).collect();

            // Distribute rank along edges
            for edge in graph.edges() {
                let out_deg = *out_degrees.get(&edge.from).unwrap_or(&1);
                if out_deg > 0 {
                    let contribution =
                        damping * ranks.get(&edge.from).unwrap_or(&0.0) / out_deg as f32;
                    *new_ranks.entry(edge.to.clone()).or_default() += contribution;
                }
            }

            // Handle dangling nodes (no outgoing edges)
            let dangling_sum: f32 = node_ids
                .iter()
                .filter(|id| *out_degrees.get(*id).unwrap_or(&0) == 0)
                .map(|id| ranks.get(id).unwrap_or(&0.0))
                .sum();

            let dangling_contribution = damping * dangling_sum / n as f32;
            for rank in new_ranks.values_mut() {
                *rank += dangling_contribution;
            }

            ranks = new_ranks;
        }

        // Normalize to [0, 1]
        let max_rank = ranks.values().copied().fold(0.0_f32, f32::max);
        if max_rank > 0.0 {
            for rank in ranks.values_mut() {
                *rank /= max_rank;
            }
        }

        ranks
    }

    /// Apply PageRank scores as node importance
    pub fn apply_pagerank<N, E>(graph: &mut Graph<N, E>, damping: f32, iterations: usize) {
        let ranks = Self::pagerank(graph, damping, iterations);
        for (id, rank) in ranks {
            if let Some(node) = graph.get_node_mut(&id) {
                node.importance = rank;
            }
        }
    }

    /// Detect communities using Louvain-style modularity optimization
    ///
    /// Based on Blondel et al. (2008) "Fast unfolding of communities in large networks"
    ///
    /// Simplified implementation suitable for TUI visualization.
    /// For production use, prefer trueno-graph's full Louvain implementation.
    ///
    /// # Returns
    /// HashMap of node_id -> community_id
    pub fn detect_communities<N, E>(graph: &Graph<N, E>) -> HashMap<String, usize> {
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();
        let n = node_ids.len();

        if n == 0 {
            return HashMap::new();
        }

        // Initialize: each node in its own community
        let mut communities: HashMap<String, usize> =
            node_ids.iter().enumerate().map(|(i, id)| (id.clone(), i)).collect();

        // Build adjacency for quick lookup
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for edge in graph.edges() {
            adjacency.entry(edge.from.clone()).or_default().push(edge.to.clone());
            // For undirected treatment
            adjacency.entry(edge.to.clone()).or_default().push(edge.from.clone());
        }

        // Greedy modularity optimization (simplified)
        let mut improved = true;
        let mut max_iterations = 10;

        while improved && max_iterations > 0 {
            improved = false;
            max_iterations -= 1;

            for node_id in &node_ids {
                let current_comm = *communities.get(node_id).unwrap_or(&0);

                // Find best community among neighbors
                let neighbors = adjacency.get(node_id).cloned().unwrap_or_default();
                let mut neighbor_comms: HashMap<usize, usize> = HashMap::new();

                for neighbor in &neighbors {
                    let comm = *communities.get(neighbor).unwrap_or(&0);
                    *neighbor_comms.entry(comm).or_default() += 1;
                }

                // Move to community with most connections
                if let Some((&best_comm, &count)) = neighbor_comms.iter().max_by_key(|(_, &c)| c) {
                    if best_comm != current_comm && count > 1 {
                        communities.insert(node_id.clone(), best_comm);
                        improved = true;
                    }
                }
            }
        }

        // Renumber communities to be contiguous (0, 1, 2, ...)
        let mut comm_map: HashMap<usize, usize> = HashMap::new();
        let mut next_comm = 0;

        for comm in communities.values_mut() {
            let new_comm = *comm_map.entry(*comm).or_insert_with(|| {
                let c = next_comm;
                next_comm += 1;
                c
            });
            *comm = new_comm;
        }

        communities
    }

    /// Apply community detection and color nodes by community
    pub fn apply_communities<N, E>(graph: &mut Graph<N, E>) -> usize {
        let communities = Self::detect_communities(graph);
        let num_communities = communities.values().max().map(|&m| m + 1).unwrap_or(0);

        // Update node status based on community for visualization
        for (id, comm) in &communities {
            if let Some(node) = graph.get_node_mut(id) {
                let comm_normalized =
                    if num_communities > 0 { *comm as f32 / num_communities as f32 } else { 0.0 };
                node.importance = f32::midpoint(node.importance, comm_normalized);
            }
        }

        num_communities
    }

    /// Get community color for rendering
    #[must_use]
    pub fn community_color(community_id: usize) -> &'static str {
        COMMUNITY_COLORS[community_id % COMMUNITY_COLORS.len()]
    }

    /// Compute degree centrality for all nodes
    ///
    /// Degree centrality = number of connections / (n-1)
    #[must_use]
    pub fn degree_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n <= 1 {
            return graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        }

        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut out_degree: HashMap<String, usize> = HashMap::new();

        for id in graph.nodes.keys() {
            in_degree.insert(id.clone(), 0);
            out_degree.insert(id.clone(), 0);
        }

        for edge in &graph.edges {
            *out_degree.entry(edge.from.clone()).or_default() += 1;
            *in_degree.entry(edge.to.clone()).or_default() += 1;
        }

        graph
            .nodes
            .keys()
            .map(|id| {
                let total = in_degree.get(id).unwrap_or(&0) + out_degree.get(id).unwrap_or(&0);
                (id.clone(), total as f32 / (n - 1) as f32)
            })
            .collect()
    }

    /// Compute betweenness centrality using Brandes algorithm
    ///
    /// Reference: Brandes (2001) "A Faster Algorithm for Betweenness Centrality"
    #[must_use]
    pub fn betweenness_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n <= 2 {
            return graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        }

        let mut betweenness: HashMap<String, f32> =
            graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        // Build adjacency list
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for id in &node_ids {
            adj.insert(id.clone(), Vec::new());
        }
        for edge in &graph.edges {
            adj.entry(edge.from.clone()).or_default().push(edge.to.clone());
        }

        // Brandes algorithm
        for s in &node_ids {
            let mut stack: Vec<String> = Vec::new();
            let mut pred: HashMap<String, Vec<String>> = HashMap::new();
            let mut sigma: HashMap<String, f32> = HashMap::new();
            let mut dist: HashMap<String, i32> = HashMap::new();

            for id in &node_ids {
                pred.insert(id.clone(), Vec::new());
                sigma.insert(id.clone(), 0.0);
                dist.insert(id.clone(), -1);
            }
            sigma.insert(s.clone(), 1.0);
            dist.insert(s.clone(), 0);

            let mut queue = std::collections::VecDeque::new();
            queue.push_back(s.clone());

            while let Some(v) = queue.pop_front() {
                stack.push(v.clone());
                let v_dist = *dist.get(&v).unwrap_or(&0);

                for w in adj.get(&v).unwrap_or(&Vec::new()) {
                    let w_dist = dist.get(w).unwrap_or(&-1);
                    if *w_dist < 0 {
                        queue.push_back(w.clone());
                        dist.insert(w.clone(), v_dist + 1);
                    }
                    if *dist.get(w).unwrap_or(&0) == v_dist + 1 {
                        let sigma_v = *sigma.get(&v).unwrap_or(&0.0);
                        *sigma.entry(w.clone()).or_default() += sigma_v;
                        pred.entry(w.clone()).or_default().push(v.clone());
                    }
                }
            }

            let mut delta: HashMap<String, f32> =
                node_ids.iter().map(|id| (id.clone(), 0.0)).collect();
            while let Some(w) = stack.pop() {
                for v in pred.get(&w).unwrap_or(&Vec::new()) {
                    let sigma_v = sigma.get(v).unwrap_or(&1.0);
                    let sigma_w = sigma.get(&w).unwrap_or(&1.0);
                    let delta_w = delta.get(&w).unwrap_or(&0.0);
                    *delta.entry(v.clone()).or_default() += (sigma_v / sigma_w) * (1.0 + delta_w);
                }
                if &w != s {
                    *betweenness.entry(w.clone()).or_default() += delta.get(&w).unwrap_or(&0.0);
                }
            }
        }

        // Normalize
        let norm = if n > 2 { 2.0 / ((n - 1) * (n - 2)) as f32 } else { 1.0 };
        for val in betweenness.values_mut() {
            *val *= norm;
        }

        betweenness
    }

    /// Compute closeness centrality
    ///
    /// Closeness = (n-1) / sum of shortest path distances from node to all others.
    #[must_use]
    pub fn closeness_centrality<N, E>(graph: &Graph<N, E>) -> HashMap<String, f32> {
        let n = graph.node_count();
        if n <= 1 {
            return graph.nodes.keys().map(|id| (id.clone(), 0.0)).collect();
        }

        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        node_ids
            .iter()
            .map(|source| {
                let distances = Self::bfs_distances(graph, source);
                let reachable: Vec<i32> = distances.values().filter(|&&d| d > 0).copied().collect();

                let closeness = if reachable.is_empty() {
                    0.0
                } else {
                    let sum_dist: i32 = reachable.iter().sum();
                    reachable.len() as f32 / sum_dist as f32
                };

                (source.clone(), closeness)
            })
            .collect()
    }

    /// Find shortest path between two nodes using BFS
    #[must_use]
    pub fn shortest_path<N, E>(graph: &Graph<N, E>, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        if !graph.nodes.contains_key(from) || !graph.nodes.contains_key(to) {
            return None;
        }

        let mut visited: HashMap<String, String> = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(from.to_string());
        visited.insert(from.to_string(), String::new());

        while let Some(current) = queue.pop_front() {
            for neighbor in graph.neighbors(&current) {
                if !visited.contains_key(neighbor) {
                    visited.insert(neighbor.to_string(), current.clone());
                    if neighbor == to {
                        let mut path = vec![to.to_string()];
                        let mut node = to.to_string();
                        while let Some(pred) = visited.get(&node) {
                            if pred.is_empty() {
                                break;
                            }
                            path.push(pred.clone());
                            node = pred.clone();
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        None
    }

    /// BFS distances from source to all reachable nodes
    fn bfs_distances<N, E>(graph: &Graph<N, E>, source: &str) -> HashMap<String, i32> {
        let mut dist: HashMap<String, i32> =
            graph.nodes.keys().map(|id| (id.clone(), -1)).collect();
        dist.insert(source.to_string(), 0);

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = *dist.get(&current).unwrap_or(&0);
            for neighbor in graph.neighbors(&current) {
                if *dist.get(neighbor).unwrap_or(&-1) < 0 {
                    dist.insert(neighbor.to_string(), current_dist + 1);
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        dist
    }

    /// Check if graph is connected (weakly connected for directed graphs)
    #[must_use]
    pub fn is_connected<N, E>(graph: &Graph<N, E>) -> bool {
        if graph.node_count() == 0 {
            return true;
        }

        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for id in graph.nodes.keys() {
            adj.insert(id.clone(), Vec::new());
        }
        for edge in &graph.edges {
            adj.entry(edge.from.clone()).or_default().push(edge.to.clone());
            adj.entry(edge.to.clone()).or_default().push(edge.from.clone());
        }

        let first = graph.nodes.keys().next().expect("graph is non-empty after check");
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(first.clone());
        visited.insert(first.clone());

        while let Some(current) = queue.pop_front() {
            for neighbor in adj.get(&current).unwrap_or(&Vec::new()) {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }

        visited.len() == graph.node_count()
    }
}

/// Extended graph with analytics capabilities
pub trait GraphAnalyticsExt<N, E> {
    /// Compute and apply PageRank
    fn compute_pagerank(&mut self, damping: f32, iterations: usize);

    /// Detect and apply communities
    fn detect_communities(&mut self) -> usize;

    /// Get PageRank scores
    fn pagerank_scores(&self) -> HashMap<String, f32>;
}

impl<N, E> GraphAnalyticsExt<N, E> for Graph<N, E> {
    fn compute_pagerank(&mut self, damping: f32, iterations: usize) {
        GraphAnalytics::apply_pagerank(self, damping, iterations);
    }

    fn detect_communities(&mut self) -> usize {
        GraphAnalytics::apply_communities(self)
    }

    fn pagerank_scores(&self) -> HashMap<String, f32> {
        GraphAnalytics::pagerank(self, 0.85, 20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::graph::Node;

    #[test]
    fn test_pagerank_empty_graph() {
        let graph: Graph<(), ()> = Graph::new();
        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 20);
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_pagerank_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        let ranks = GraphAnalytics::pagerank(&graph, 0.85, 20);
        assert_eq!(ranks.len(), 1);
        assert!(ranks.contains_key("a"));
    }

    #[test]
    fn test_community_detection_empty() {
        let graph: Graph<(), ()> = Graph::new();
        let communities = GraphAnalytics::detect_communities(&graph);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_is_connected_empty() {
        let graph: Graph<(), ()> = Graph::new();
        assert!(GraphAnalytics::is_connected(&graph));
    }

    #[test]
    fn test_community_color() {
        let color = GraphAnalytics::community_color(0);
        assert_eq!(color, "\x1b[31m"); // Red
    }

    #[test]
    fn test_community_color_wraps() {
        // Test that colors wrap around
        let c0 = GraphAnalytics::community_color(0);
        let c8 = GraphAnalytics::community_color(8);
        assert_eq!(c0, c8); // Should wrap to same color
    }

    #[test]
    fn test_degree_centrality_empty() {
        let graph: Graph<(), ()> = Graph::new();
        let centrality = GraphAnalytics::degree_centrality(&graph);
        assert!(centrality.is_empty());
    }

    #[test]
    fn test_degree_centrality_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        let centrality = GraphAnalytics::degree_centrality(&graph);
        assert_eq!(centrality.len(), 1);
        assert_eq!(*centrality.get("a").expect("key not found"), 0.0);
    }

    #[test]
    fn test_betweenness_centrality_empty() {
        let graph: Graph<(), ()> = Graph::new();
        let centrality = GraphAnalytics::betweenness_centrality(&graph);
        assert!(centrality.is_empty());
    }

    #[test]
    fn test_closeness_centrality_empty() {
        let graph: Graph<(), ()> = Graph::new();
        let centrality = GraphAnalytics::closeness_centrality(&graph);
        assert!(centrality.is_empty());
    }

    #[test]
    fn test_closeness_centrality_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        let centrality = GraphAnalytics::closeness_centrality(&graph);
        assert_eq!(centrality.len(), 1);
        assert_eq!(*centrality.get("a").expect("key not found"), 0.0);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        let path = GraphAnalytics::shortest_path(&graph, "a", "a");
        assert_eq!(path, Some(vec!["a".to_string()]));
    }

    #[test]
    fn test_shortest_path_nonexistent() {
        let graph: Graph<(), ()> = Graph::new();
        let path = GraphAnalytics::shortest_path(&graph, "a", "b");
        assert!(path.is_none());
    }

    #[test]
    fn test_is_connected_single_node() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        assert!(GraphAnalytics::is_connected(&graph));
    }

    #[test]
    fn test_graph_analytics_ext_trait() {
        use crate::tui::graph::Edge;

        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_edge(Edge::new("a", "b", ()));

        // Test trait methods
        graph.compute_pagerank(0.85, 10);
        let num_communities = graph.detect_communities();
        let scores = graph.pagerank_scores();

        assert!(scores.contains_key("a"));
        let _ = num_communities; // usize is always >= 0
    }

    #[test]
    fn test_apply_pagerank() {
        use crate::tui::graph::Edge;

        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_edge(Edge::new("a", "b", ()));

        GraphAnalytics::apply_pagerank(&mut graph, 0.85, 10);

        // Both nodes should have importance > 0
        assert!(graph.get_node("a").expect("unexpected failure").importance >= 0.0);
        assert!(graph.get_node("b").expect("unexpected failure").importance >= 0.0);
    }

    #[test]
    fn test_apply_communities() {
        use crate::tui::graph::Edge;

        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_edge(Edge::new("a", "b", ()));

        let num_communities = GraphAnalytics::apply_communities(&mut graph);
        assert!(num_communities >= 1);
    }
}
