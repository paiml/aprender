//! Graph Layout Algorithms
//!
//! Layout algorithms for graph visualization in the TUI.
//!
//! ## Academic References
//!
//! - Fruchterman & Reingold (1991) - Force-directed layout
//! - Kamada & Kawai (1989) - Spring-based layout
//! - Sugiyama et al. (1981) - Hierarchical DAG layout

use std::collections::HashMap;

use super::graph::{Graph, Position};

/// Convert usize to f32 for layout math. Graph node counts are small
/// enough that f32 precision (24-bit mantissa) is always sufficient.
#[inline]
fn f(n: usize) -> f32 {
    debug_assert!(n <= (1 << 24), "layout value {n} exceeds f32 mantissa precision");
    n as f32
}

/// Layout algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutAlgorithm {
    /// Simple grid layout (O(n))
    #[default]
    Grid,
    /// Force-directed Fruchterman-Reingold (O(n²) per iteration)
    ForceDirected,
    /// Hierarchical Sugiyama for DAGs (O(nm log m))
    Hierarchical,
    /// Radial layout for trees (O(n + m))
    Radial,
    /// Circular layout - nodes in a circle (O(n)) - Cytoscape pattern
    Circular,
    /// Concentric layout - nodes in rings by metric (O(n)) - Cytoscape/Gephi pattern
    Concentric,
}

/// Layout configuration
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Algorithm to use
    pub algorithm: LayoutAlgorithm,
    /// Width of layout area
    pub width: f32,
    /// Height of layout area
    pub height: f32,
    /// Iterations for iterative algorithms
    pub iterations: usize,
    /// Optimal edge length for force-directed
    pub optimal_distance: f32,
    /// Cooling factor for force-directed
    pub cooling: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            algorithm: LayoutAlgorithm::Grid,
            width: 80.0,
            height: 24.0,
            iterations: 50,
            optimal_distance: 3.0,
            cooling: 0.95,
        }
    }
}

/// Layout engine
pub struct LayoutEngine;

impl LayoutEngine {
    /// Compute layout for graph
    pub fn compute<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        match config.algorithm {
            LayoutAlgorithm::Grid => Self::grid_layout(graph, config),
            LayoutAlgorithm::ForceDirected => Self::force_directed_layout(graph, config),
            LayoutAlgorithm::Hierarchical => Self::hierarchical_layout(graph, config),
            LayoutAlgorithm::Circular => Self::circular_layout(graph, config),
            LayoutAlgorithm::Concentric => Self::concentric_layout(graph, config),
            LayoutAlgorithm::Radial => Self::radial_layout(graph, config),
        }
    }

    /// Grid layout - O(n)
    fn grid_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let cols = f(n).sqrt().ceil() as usize;
        let rows = n.div_ceil(cols);

        let cell_width = config.width / f(cols);
        let cell_height = config.height / f(rows);

        for (i, node) in graph.nodes_mut().enumerate() {
            let col = i % cols;
            let row = i / cols;
            node.position =
                Position::new((f(col) + 0.5) * cell_width, (f(row) + 0.5) * cell_height);
        }
    }

    /// Force-directed layout - Fruchterman-Reingold
    /// O(n²) per iteration, with software fallback for non-SIMD (Jidoka per peer review #4)
    fn force_directed_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        // Initialize with grid layout
        Self::grid_layout(graph, config);

        let k = config.optimal_distance;
        let k_squared = k * k;
        let mut temperature = config.width.min(config.height) / 10.0;

        // Collect node IDs for iteration
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();

        for _ in 0..config.iterations {
            // Compute forces
            let mut forces: HashMap<String, (f32, f32)> = HashMap::new();
            for id in &node_ids {
                forces.insert(id.clone(), (0.0, 0.0));
            }

            // Repulsive forces between all pairs
            for i in 0..node_ids.len() {
                for j in (i + 1)..node_ids.len() {
                    let id_i = &node_ids[i];
                    let id_j = &node_ids[j];

                    let pos_i = graph.nodes.get(id_i).map(|n| n.position).unwrap_or_default();
                    let pos_j = graph.nodes.get(id_j).map(|n| n.position).unwrap_or_default();

                    let dx = pos_i.x - pos_j.x;
                    let dy = pos_i.y - pos_j.y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    // Repulsive force: k² / d
                    let force = k_squared / dist;
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;

                    if let Some(f) = forces.get_mut(id_i) {
                        f.0 += fx;
                        f.1 += fy;
                    }
                    if let Some(f) = forces.get_mut(id_j) {
                        f.0 -= fx;
                        f.1 -= fy;
                    }
                }
            }

            // Attractive forces along edges
            for edge in &graph.edges {
                let pos_from = graph.nodes.get(&edge.from).map(|n| n.position).unwrap_or_default();
                let pos_to = graph.nodes.get(&edge.to).map(|n| n.position).unwrap_or_default();

                let dx = pos_to.x - pos_from.x;
                let dy = pos_to.y - pos_from.y;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                // Attractive force: d² / k
                let force = (dist * dist) / k;
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;

                if let Some(f) = forces.get_mut(&edge.from) {
                    f.0 += fx;
                    f.1 += fy;
                }
                if let Some(f) = forces.get_mut(&edge.to) {
                    f.0 -= fx;
                    f.1 -= fy;
                }
            }

            // Apply forces with temperature limiting
            for id in &node_ids {
                if let (Some(node), Some(&(fx, fy))) = (graph.nodes.get_mut(id), forces.get(id)) {
                    let mag = (fx * fx + fy * fy).sqrt().max(0.01);
                    let capped_mag = mag.min(temperature);
                    node.position.x += (fx / mag) * capped_mag;
                    node.position.y += (fy / mag) * capped_mag;

                    // Clamp to bounds
                    node.position.x = node.position.x.clamp(1.0, config.width - 1.0);
                    node.position.y = node.position.y.clamp(1.0, config.height - 1.0);
                }
            }

            // Cool down
            temperature *= config.cooling;
        }
    }

    /// Hierarchical layout - simplified Sugiyama
    fn hierarchical_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        // Compute in-degrees for layer assignment
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in graph.nodes.keys() {
            in_degree.insert(id.clone(), 0);
        }
        for edge in &graph.edges {
            *in_degree.entry(edge.to.clone()).or_default() += 1;
        }

        // Assign layers based on longest path
        let mut layers: HashMap<String, usize> = HashMap::new();
        let mut queue: Vec<String> =
            in_degree.iter().filter(|(_, &d)| d == 0).map(|(id, _)| id.clone()).collect();

        for id in &queue {
            layers.insert(id.clone(), 0);
        }

        let mut max_layer = 0;
        while let Some(id) = queue.pop() {
            let current_layer = *layers.get(&id).unwrap_or(&0);
            for neighbor in graph.neighbors(&id) {
                let new_layer = current_layer + 1;
                let entry = layers.entry(neighbor.to_string()).or_insert(0);
                if new_layer > *entry {
                    *entry = new_layer;
                    max_layer = max_layer.max(new_layer);
                }
                queue.push(neighbor.to_string());
            }
        }

        // Position nodes by layer
        let layer_height = config.height / f(max_layer + 1);
        let mut layer_counts: HashMap<usize, usize> = HashMap::new();
        let mut layer_positions: HashMap<usize, usize> = HashMap::new();

        // Count nodes per layer
        for &layer in layers.values() {
            *layer_counts.entry(layer).or_default() += 1;
        }

        // Position nodes
        for node in graph.nodes_mut() {
            let layer = *layers.get(&node.id).unwrap_or(&0);
            let count = *layer_counts.get(&layer).unwrap_or(&1);
            let pos_in_layer = *layer_positions.entry(layer).or_default();
            layer_positions.insert(layer, pos_in_layer + 1);

            let layer_width = config.width / f(count);
            node.position = Position::new(
                (f(pos_in_layer) + 0.5) * layer_width,
                (f(layer) + 0.5) * layer_height,
            );
        }
    }

    /// Radial layout - for tree structures
    fn radial_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let center_x = config.width / 2.0;
        let center_y = config.height / 2.0;
        let radius = config.width.min(config.height) / 2.5;

        // Find root (node with no incoming edges)
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in graph.nodes.keys() {
            in_degree.insert(id.clone(), 0);
        }
        for edge in &graph.edges {
            *in_degree.entry(edge.to.clone()).or_default() += 1;
        }

        let root = in_degree.iter().find(|(_, &d)| d == 0).map(|(id, _)| id.clone());

        if let Some(root_id) = root {
            // Place root at center
            if let Some(node) = graph.nodes.get_mut(&root_id) {
                node.position = Position::new(center_x, center_y);
            }

            // BFS to assign radial positions
            let mut visited: HashMap<String, usize> = HashMap::new();
            visited.insert(root_id.clone(), 0);
            let root_id_clone = root_id.clone();
            let mut queue = vec![root_id];
            let mut depth_counts: HashMap<usize, usize> = HashMap::new();
            let mut depth_positions: HashMap<usize, usize> = HashMap::new();

            while let Some(id) = queue.pop() {
                let depth = *visited.get(&id).unwrap_or(&0);
                for neighbor in graph.neighbors(&id) {
                    if !visited.contains_key(neighbor) {
                        visited.insert(neighbor.to_string(), depth + 1);
                        *depth_counts.entry(depth + 1).or_default() += 1;
                        queue.push(neighbor.to_string());
                    }
                }
            }

            // Position nodes radially
            let max_depth = visited.values().max().copied().unwrap_or(1);
            let mut unvisited_idx = 0;
            let unvisited_count = graph.node_count() - visited.len();

            for node in graph.nodes_mut() {
                if node.id == root_id_clone {
                    continue;
                }

                if let Some(&depth) = visited.get(&node.id) {
                    // Node was visited by BFS - position radially
                    let count = *depth_counts.get(&depth).unwrap_or(&1);
                    let pos = *depth_positions.entry(depth).or_default();
                    depth_positions.insert(depth, pos + 1);

                    let angle = 2.0 * std::f32::consts::PI * (f(pos) / f(count));
                    let r = radius * (f(depth) / f(max_depth));

                    node.position = Position::new(
                        (center_x + r * angle.cos()).clamp(1.0, config.width - 1.0),
                        (center_y + r * angle.sin()).clamp(1.0, config.height - 1.0),
                    );
                } else {
                    // Unvisited/disconnected node - place on outer ring
                    let angle =
                        2.0 * std::f32::consts::PI * (f(unvisited_idx) / f(unvisited_count.max(1)));
                    let r = radius * 1.2; // Slightly outside main graph
                    unvisited_idx += 1;

                    node.position = Position::new(
                        (center_x + r * angle.cos()).clamp(1.0, config.width - 1.0),
                        (center_y + r * angle.sin()).clamp(1.0, config.height - 1.0),
                    );
                }
            }
        } else {
            // Fallback to grid if no root found
            Self::grid_layout(graph, config);
        }
    }

    /// Circular layout - nodes arranged in a circle (Cytoscape pattern)
    fn circular_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let center_x = config.width / 2.0;
        let center_y = config.height / 2.0;
        let radius = config.width.min(config.height) / 2.5;

        for (i, node) in graph.nodes_mut().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * (f(i) / f(n));
            node.position = Position::new(
                (center_x + radius * angle.cos()).clamp(1.0, config.width - 1.0),
                (center_y + radius * angle.sin()).clamp(1.0, config.height - 1.0),
            );
        }
    }

    /// Concentric layout - nodes in rings by importance/degree (Cytoscape/Gephi pattern)
    fn concentric_layout<N, E>(graph: &mut Graph<N, E>, config: &LayoutConfig) {
        let n = graph.node_count();
        if n == 0 {
            return;
        }

        let center_x = config.width / 2.0;
        let center_y = config.height / 2.0;
        let max_radius = config.width.min(config.height) / 2.5;

        // Sort nodes by importance (highest importance = center)
        let mut node_order: Vec<(String, f32)> =
            graph.nodes().map(|n| (n.id.clone(), n.importance)).collect();
        node_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign to concentric rings (higher importance = smaller ring)
        let num_rings = 4.min(n);
        let nodes_per_ring = n.div_ceil(num_rings);

        for (idx, (node_id, _)) in node_order.iter().enumerate() {
            let ring = idx / nodes_per_ring;
            let pos_in_ring = idx % nodes_per_ring;
            let nodes_in_this_ring =
                if ring == num_rings - 1 { n - ring * nodes_per_ring } else { nodes_per_ring };

            // Ring 0 = center, ring N = outer
            let radius = if ring == 0 && nodes_in_this_ring == 1 {
                0.0 // Single node at center
            } else {
                max_radius * (f(ring + 1) / f(num_rings))
            };

            let angle = 2.0 * std::f32::consts::PI * (f(pos_in_ring) / f(nodes_in_this_ring));

            if let Some(node) = graph.nodes.get_mut(node_id) {
                node.position = Position::new(
                    (center_x + radius * angle.cos()).clamp(1.0, config.width - 1.0),
                    (center_y + radius * angle.sin()).clamp(1.0, config.height - 1.0),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::graph::Node;

    #[test]
    fn test_layout_algorithm_default() {
        assert_eq!(LayoutAlgorithm::default(), LayoutAlgorithm::Grid);
    }

    #[test]
    fn test_layout_config_default() {
        let config = LayoutConfig::default();
        assert_eq!(config.algorithm, LayoutAlgorithm::Grid);
        assert_eq!(config.width, 80.0);
        assert_eq!(config.height, 24.0);
        assert_eq!(config.iterations, 50);
    }

    #[test]
    fn test_grid_layout_empty() {
        let mut graph: Graph<(), ()> = Graph::new();
        let config = LayoutConfig::default();
        LayoutEngine::compute(&mut graph, &config);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_grid_layout_single() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        let config = LayoutConfig::default();
        LayoutEngine::compute(&mut graph, &config);

        let node = graph.nodes.get("a").expect("key not found");
        assert!(node.position.x > 0.0);
        assert!(node.position.y > 0.0);
    }

    #[test]
    fn test_circular_layout() {
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_node(Node::new("c", ()));

        let config = LayoutConfig { algorithm: LayoutAlgorithm::Circular, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be positioned
        for node in graph.nodes() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }
    }

    #[test]
    fn test_force_directed_layout() {
        use crate::tui::graph::Edge;
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        graph.add_edge(Edge::new("a", "b", ()));

        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::ForceDirected,
            iterations: 5,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be within bounds
        for node in graph.nodes() {
            assert!(node.position.x >= 1.0);
            assert!(node.position.y >= 1.0);
            assert!(node.position.x <= config.width - 1.0);
            assert!(node.position.y <= config.height - 1.0);
        }
    }

    #[test]
    fn test_hierarchical_layout() {
        use crate::tui::graph::Edge;
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("root", ()));
        graph.add_node(Node::new("child1", ()));
        graph.add_node(Node::new("child2", ()));
        graph.add_edge(Edge::new("root", "child1", ()));
        graph.add_edge(Edge::new("root", "child2", ()));

        let config =
            LayoutConfig { algorithm: LayoutAlgorithm::Hierarchical, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be positioned
        for node in graph.nodes() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }
    }

    #[test]
    fn test_radial_layout() {
        use crate::tui::graph::Edge;
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("center", ()));
        graph.add_node(Node::new("leaf1", ()));
        graph.add_node(Node::new("leaf2", ()));
        graph.add_edge(Edge::new("center", "leaf1", ()));
        graph.add_edge(Edge::new("center", "leaf2", ()));

        let config = LayoutConfig { algorithm: LayoutAlgorithm::Radial, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be positioned
        for node in graph.nodes() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }
    }

    #[test]
    fn test_concentric_layout() {
        let mut graph: Graph<(), ()> = Graph::new();
        let mut node1 = Node::new("important", ());
        node1.importance = 1.0;
        let mut node2 = Node::new("less", ());
        node2.importance = 0.5;
        let mut node3 = Node::new("least", ());
        node3.importance = 0.1;
        graph.add_node(node1);
        graph.add_node(node2);
        graph.add_node(node3);

        let config = LayoutConfig { algorithm: LayoutAlgorithm::Concentric, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should be positioned
        for node in graph.nodes() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }
    }

    #[test]
    fn test_radial_layout_no_root() {
        use crate::tui::graph::Edge;
        let mut graph: Graph<(), ()> = Graph::new();
        graph.add_node(Node::new("a", ()));
        graph.add_node(Node::new("b", ()));
        // Create a cycle - no clear root
        graph.add_edge(Edge::new("a", "b", ()));
        graph.add_edge(Edge::new("b", "a", ()));

        let config = LayoutConfig { algorithm: LayoutAlgorithm::Radial, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);

        // Should fallback to grid layout
        for node in graph.nodes() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }
    }

    #[test]
    fn test_layout_algorithm_variants() {
        assert_ne!(LayoutAlgorithm::Grid, LayoutAlgorithm::ForceDirected);
        assert_ne!(LayoutAlgorithm::Hierarchical, LayoutAlgorithm::Radial);
        assert_ne!(LayoutAlgorithm::Circular, LayoutAlgorithm::Concentric);
    }

    #[test]
    fn test_layout_config_custom() {
        let config = LayoutConfig {
            algorithm: LayoutAlgorithm::ForceDirected,
            width: 100.0,
            height: 50.0,
            iterations: 100,
            optimal_distance: 5.0,
            cooling: 0.9,
        };
        assert_eq!(config.iterations, 100);
        assert_eq!(config.optimal_distance, 5.0);
        assert_eq!(config.cooling, 0.9);
    }

    #[test]
    fn test_grid_layout_multiple_rows() {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..6 {
            graph.add_node(Node::new(&format!("n{}", i), ()));
        }

        let config = LayoutConfig::default();
        LayoutEngine::compute(&mut graph, &config);

        // All nodes should have different positions
        let positions: Vec<_> = graph.nodes().map(|n| (n.position.x, n.position.y)).collect();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                assert!(positions[i] != positions[j] || i == j);
            }
        }
    }

    #[test]
    fn test_hierarchical_layout_empty() {
        let mut graph: Graph<(), ()> = Graph::new();
        let config =
            LayoutConfig { algorithm: LayoutAlgorithm::Hierarchical, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_radial_layout_empty() {
        let mut graph: Graph<(), ()> = Graph::new();
        let config = LayoutConfig { algorithm: LayoutAlgorithm::Radial, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_force_directed_layout_empty() {
        let mut graph: Graph<(), ()> = Graph::new();
        let config =
            LayoutConfig { algorithm: LayoutAlgorithm::ForceDirected, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_circular_layout_empty() {
        let mut graph: Graph<(), ()> = Graph::new();
        let config = LayoutConfig { algorithm: LayoutAlgorithm::Circular, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_concentric_layout_empty() {
        let mut graph: Graph<(), ()> = Graph::new();
        let config = LayoutConfig { algorithm: LayoutAlgorithm::Concentric, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);
        assert_eq!(graph.node_count(), 0);
    }
}
