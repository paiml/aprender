//! Layout tests for graph TUI visualization
//!
//! Tests for circular/concentric layout algorithms (Cytoscape patterns).

use super::*;
use crate::tui::graph_layout::{LayoutAlgorithm, LayoutConfig, LayoutEngine};

// -------------------------------------------------------------------------
// Circular/Concentric Layout Tests (Cytoscape patterns)
// -------------------------------------------------------------------------

#[test]
fn test_circular_layout() {
    let mut graph: Graph<(), ()> = Graph::new();
    for i in 0..8 {
        graph.add_node(Node::new(format!("n{}", i), ()));
    }

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Circular,
        width: 80.0,
        height: 24.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    // All nodes should be positioned
    for node in graph.nodes() {
        assert!(node.position.x >= 0.0 && node.position.x <= 80.0);
        assert!(node.position.y >= 0.0 && node.position.y <= 24.0);
    }

    // Nodes should be roughly equidistant from center
    let center_x = 40.0;
    let center_y = 12.0;
    let distances: Vec<f32> = graph
        .nodes()
        .map(|n| ((n.position.x - center_x).powi(2) + (n.position.y - center_y).powi(2)).sqrt())
        .collect();

    let avg_dist: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
    for d in &distances {
        assert!((d - avg_dist).abs() < 1.0, "Nodes should be roughly equidistant from center");
    }
}

#[test]
fn test_concentric_layout() {
    let mut graph: Graph<(), ()> = Graph::new();
    // Add nodes with varying importance
    for i in 0..6 {
        let mut node = Node::new(format!("n{}", i), ());
        node.importance = 1.0 - (i as f32 * 0.15);
        graph.add_node(node);
    }

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Concentric,
        width: 80.0,
        height: 24.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    // All nodes should be positioned within bounds
    for node in graph.nodes() {
        assert!(node.position.x >= 0.0 && node.position.x <= 80.0);
        assert!(node.position.y >= 0.0 && node.position.y <= 24.0);
    }
}
