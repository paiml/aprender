//! Core tests for graph TUI visualization
//!
//! Tests for node shapes, positions, graph structure, layout, and rendering.

use super::*;
use crate::tui::graph_layout::{LayoutAlgorithm, LayoutConfig, LayoutEngine};

// -------------------------------------------------------------------------
// Node Shape Tests
// -------------------------------------------------------------------------

#[test]
fn test_node_shape_unicode_characters() {
    assert_eq!(NodeShape::Circle.unicode(), '●');
    assert_eq!(NodeShape::Diamond.unicode(), '◆');
    assert_eq!(NodeShape::Square.unicode(), '■');
    assert_eq!(NodeShape::Triangle.unicode(), '▲');
    assert_eq!(NodeShape::Star.unicode(), '★');
}

#[test]
fn test_node_shape_ascii_fallback() {
    // Per peer review #5: ASCII fallback for legacy terminals
    assert_eq!(NodeShape::Circle.ascii(), 'o');
    assert_eq!(NodeShape::Diamond.ascii(), '<');
    assert_eq!(NodeShape::Square.ascii(), '#');
    assert_eq!(NodeShape::Triangle.ascii(), '^');
    assert_eq!(NodeShape::Star.ascii(), '*');
}

#[test]
fn test_status_shapes_for_accessibility() {
    // Per peer review #6: Shapes differentiate status, not just color
    assert_eq!(NodeStatus::Healthy.shape(), NodeShape::Circle);
    assert_eq!(NodeStatus::Warning.shape(), NodeShape::Triangle);
    assert_eq!(NodeStatus::Error.shape(), NodeShape::Diamond);
    assert_eq!(NodeStatus::Info.shape(), NodeShape::Star);
    assert_eq!(NodeStatus::Neutral.shape(), NodeShape::Square);
}

// -------------------------------------------------------------------------
// Position Tests
// -------------------------------------------------------------------------

#[test]
fn test_position_distance() {
    let p1 = Position::new(0.0, 0.0);
    let p2 = Position::new(3.0, 4.0);
    assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
}

#[test]
fn test_position_distance_same_point() {
    let p = Position::new(5.0, 5.0);
    assert!((p.distance(&p) - 0.0).abs() < 0.001);
}

// -------------------------------------------------------------------------
// Graph Structure Tests
// -------------------------------------------------------------------------

#[test]
fn test_graph_creation() {
    let graph: Graph<(), ()> = Graph::new();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_graph_add_nodes() {
    let mut graph: Graph<i32, ()> = Graph::new();
    graph.add_node(Node::new("a", 1));
    graph.add_node(Node::new("b", 2));
    assert_eq!(graph.node_count(), 2);
}

#[test]
fn test_graph_add_edges() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("a", ()));
    graph.add_node(Node::new("b", ()));
    graph.add_edge(Edge::new("a", "b", ()));
    assert_eq!(graph.edge_count(), 1);
}

#[test]
fn test_graph_neighbors() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("a", ()));
    graph.add_node(Node::new("b", ()));
    graph.add_node(Node::new("c", ()));
    graph.add_edge(Edge::new("a", "b", ()));
    graph.add_edge(Edge::new("a", "c", ()));

    let neighbors = graph.neighbors("a");
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&"b"));
    assert!(neighbors.contains(&"c"));
}

#[test]
fn test_graph_tui_limit() {
    // Per peer review #3: Hard limit of 500 nodes
    let mut graph: Graph<(), ()> = Graph::new();
    for i in 0..MAX_TUI_NODES {
        graph.add_node(Node::new(format!("n{}", i), ()));
    }
    assert!(!graph.exceeds_tui_limit());

    graph.add_node(Node::new("overflow", ()));
    assert!(graph.exceeds_tui_limit());
}

#[test]
fn test_top_nodes_by_importance() {
    // Per peer review #9: Mieruka - show top N nodes
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("low", ()).with_importance(0.1));
    graph.add_node(Node::new("medium", ()).with_importance(0.5));
    graph.add_node(Node::new("high", ()).with_importance(0.9));

    let top = graph.top_nodes_by_importance(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].id, "high");
    assert_eq!(top[1].id, "medium");
}

// -------------------------------------------------------------------------
// Layout Algorithm Tests
// -------------------------------------------------------------------------

#[test]
fn test_grid_layout_positions_all_nodes() {
    let mut graph: Graph<(), ()> = Graph::new();
    for i in 0..9 {
        graph.add_node(Node::new(format!("n{}", i), ()));
    }

    let config = LayoutConfig::default();
    LayoutEngine::compute(&mut graph, &config);

    // All nodes should have non-default positions
    for node in graph.nodes() {
        assert!(node.position.x > 0.0 || node.position.y > 0.0);
    }
}

#[test]
fn test_grid_layout_no_overlaps() {
    let mut graph: Graph<(), ()> = Graph::new();
    for i in 0..16 {
        graph.add_node(Node::new(format!("n{}", i), ()));
    }

    let config = LayoutConfig::default();
    LayoutEngine::compute(&mut graph, &config);

    // Check no two nodes at exact same position
    let positions: Vec<_> = graph.nodes().map(|n| n.position).collect();
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dist = positions[i].distance(&positions[j]);
            assert!(dist > 0.1, "Nodes {} and {} overlap", i, j);
        }
    }
}

#[test]
fn test_force_directed_convergence() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("a", ()));
    graph.add_node(Node::new("b", ()));
    graph.add_node(Node::new("c", ()));
    graph.add_edge(Edge::new("a", "b", ()));
    graph.add_edge(Edge::new("b", "c", ()));

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::ForceDirected,
        iterations: 100,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    // Connected nodes should be closer than unconnected
    let pos_a = graph.get_node("a").unwrap().position;
    let pos_b = graph.get_node("b").unwrap().position;
    let pos_c = graph.get_node("c").unwrap().position;

    let ab_dist = pos_a.distance(&pos_b);
    let bc_dist = pos_b.distance(&pos_c);
    let ac_dist = pos_a.distance(&pos_c);

    // a-b and b-c are connected, a-c is not
    assert!(ab_dist < ac_dist || bc_dist < ac_dist);
}

#[test]
fn test_hierarchical_layout_layers() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("root", ()));
    graph.add_node(Node::new("child1", ()));
    graph.add_node(Node::new("child2", ()));
    graph.add_node(Node::new("grandchild", ()));
    graph.add_edge(Edge::new("root", "child1", ()));
    graph.add_edge(Edge::new("root", "child2", ()));
    graph.add_edge(Edge::new("child1", "grandchild", ()));

    let config = LayoutConfig { algorithm: LayoutAlgorithm::Hierarchical, ..Default::default() };
    LayoutEngine::compute(&mut graph, &config);

    // Root should be at top (lowest y)
    let root_y = graph.get_node("root").unwrap().position.y;
    let child1_y = graph.get_node("child1").unwrap().position.y;
    let grandchild_y = graph.get_node("grandchild").unwrap().position.y;

    assert!(root_y < child1_y);
    assert!(child1_y < grandchild_y);
}

#[test]
fn test_radial_layout_center() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("center", ()));
    graph.add_node(Node::new("leaf1", ()));
    graph.add_node(Node::new("leaf2", ()));
    graph.add_edge(Edge::new("center", "leaf1", ()));
    graph.add_edge(Edge::new("center", "leaf2", ()));

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Radial,
        width: 80.0,
        height: 24.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    // Center node should be near center of layout
    let center_pos = graph.get_node("center").unwrap().position;
    assert!((center_pos.x - 40.0).abs() < 5.0);
    assert!((center_pos.y - 12.0).abs() < 5.0);
}

// -------------------------------------------------------------------------
// Rendering Tests
// -------------------------------------------------------------------------

#[test]
fn test_rendered_graph_creation() {
    let rg = RenderedGraph::new(80, 24);
    assert_eq!(rg.width, 80);
    assert_eq!(rg.height, 24);
    assert_eq!(rg.buffer.len(), 24);
    assert_eq!(rg.buffer[0].len(), 80);
}

#[test]
fn test_rendered_graph_set() {
    let mut rg = RenderedGraph::new(10, 10);
    rg.set(5, 5, 'X', None);
    assert_eq!(rg.buffer[5][5], 'X');
}

#[test]
fn test_renderer_unicode_mode() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("test", ()).with_status(NodeStatus::Healthy));

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Grid,
        width: 80.0,
        height: 24.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    let renderer = GraphRenderer::new().with_mode(RenderMode::Unicode);
    let output = renderer.render(&graph, 80, 24);

    // Should contain circle character
    let plain = output.to_string_plain();
    assert!(plain.contains('●'));
}

#[test]
fn test_renderer_ascii_mode() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("test", ()).with_status(NodeStatus::Healthy));

    let config = LayoutConfig::default();
    LayoutEngine::compute(&mut graph, &config);

    let renderer = GraphRenderer::new().with_mode(RenderMode::Ascii);
    let output = renderer.render(&graph, 80, 24);

    // Should contain ASCII 'o' not Unicode circle
    let plain = output.to_string_plain();
    assert!(plain.contains('o'));
    assert!(!plain.contains('●'));
}

#[test]
fn test_renderer_with_edges() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("a", ()));
    graph.add_node(Node::new("b", ()));
    graph.add_edge(Edge::new("a", "b", ()));

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Grid,
        width: 80.0,
        height: 24.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    let renderer = GraphRenderer::new();
    let output = renderer.render(&graph, 80, 24);

    // Should contain edge characters
    let plain = output.to_string_plain();
    assert!(plain.contains('·') || plain.contains('.'));
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[test]
fn test_full_pipeline() {
    // Create graph
    let mut graph: Graph<&str, &str> = Graph::new();
    graph.add_node(
        Node::new("trueno", "core").with_status(NodeStatus::Healthy).with_label("trueno"),
    );
    graph.add_node(
        Node::new("aprender", "ml").with_status(NodeStatus::Healthy).with_label("aprender"),
    );
    graph.add_node(
        Node::new("batuta", "orch").with_status(NodeStatus::Warning).with_label("batuta"),
    );
    graph.add_edge(Edge::new("trueno", "aprender", "depends"));
    graph.add_edge(Edge::new("aprender", "batuta", "depends"));

    // Compute layout
    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Hierarchical,
        width: 80.0,
        height: 24.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    // Render
    let renderer = GraphRenderer::new();
    let output = renderer.render(&graph, 80, 24);

    // Verify output contains labels
    let plain = output.to_string_plain();
    assert!(plain.contains("trueno") || plain.contains("aprender") || plain.contains("batuta"));
}

// -------------------------------------------------------------------------
// Property-Based Tests
// -------------------------------------------------------------------------

#[test]
fn test_layout_bounds() {
    // All layouts should keep nodes within bounds
    for algo in [
        LayoutAlgorithm::Grid,
        LayoutAlgorithm::ForceDirected,
        LayoutAlgorithm::Hierarchical,
        LayoutAlgorithm::Radial,
    ] {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..20 {
            graph.add_node(Node::new(format!("n{}", i), ()));
        }
        for i in 0..15 {
            graph.add_edge(Edge::new(format!("n{}", i), format!("n{}", i + 1), ()));
        }

        let config = LayoutConfig {
            algorithm: algo,
            width: 80.0,
            height: 24.0,
            iterations: 50,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        for node in graph.nodes() {
            assert!(node.position.x >= 0.0, "{:?} x < 0", algo);
            assert!(node.position.y >= 0.0, "{:?} y < 0", algo);
            assert!(node.position.x <= 80.0, "{:?} x > width", algo);
            assert!(node.position.y <= 24.0, "{:?} y > height", algo);
        }
    }
}

#[test]
fn test_empty_graph_handling() {
    let mut graph: Graph<(), ()> = Graph::new();

    for algo in [
        LayoutAlgorithm::Grid,
        LayoutAlgorithm::ForceDirected,
        LayoutAlgorithm::Hierarchical,
        LayoutAlgorithm::Radial,
    ] {
        let config = LayoutConfig { algorithm: algo, ..Default::default() };
        // Should not panic
        LayoutEngine::compute(&mut graph, &config);
    }
}

#[test]
fn test_single_node_graph() {
    let mut graph: Graph<(), ()> = Graph::new();
    graph.add_node(Node::new("solo", ()));

    for algo in [
        LayoutAlgorithm::Grid,
        LayoutAlgorithm::ForceDirected,
        LayoutAlgorithm::Hierarchical,
        LayoutAlgorithm::Radial,
    ] {
        let config = LayoutConfig { algorithm: algo, ..Default::default() };
        LayoutEngine::compute(&mut graph, &config);

        let pos = graph.get_node("solo").unwrap().position;
        assert!(pos.x > 0.0);
        assert!(pos.y > 0.0);
    }
}
