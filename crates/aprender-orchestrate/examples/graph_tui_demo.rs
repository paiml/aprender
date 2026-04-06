//! Graph TUI Visualization Demo
//!
//! Demonstrates the terminal-based graph visualization system.
//!
//! ## Features
//!
//! - **Layout Algorithms**: Grid, Force-Directed, Hierarchical, Radial
//! - **Accessibility**: Shape-based status (not just color)
//! - **ASCII Fallback**: Legacy terminal support
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example graph_tui_demo --features native
//! ```
//!
//! ## Toyota Way Principles
//!
//! - **Mieruka**: Visual management via semantic colors and shapes
//! - **Respect for People**: Accessibility for color-blind users
//! - **Muri Prevention**: Hard limit of 500 nodes

#[cfg(feature = "native")]
use batuta::tui::{
    Edge, Graph, GraphRenderer, LayoutAlgorithm, LayoutConfig, LayoutEngine, Node, NodeStatus,
    RenderMode, DEFAULT_VISIBLE_NODES, MAX_TUI_NODES,
};

#[cfg(feature = "native")]
fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("     Graph TUI Visualization Demo");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Phase 1: Basic Graph Creation
    // =========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 1: Graph Data Structure                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    demo_graph_creation();

    // =========================================================================
    // Phase 2: Layout Algorithms
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 2: Layout Algorithms                                      │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    demo_layout_algorithms();

    // =========================================================================
    // Phase 3: Status and Accessibility
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 3: Status & Accessibility (Respect for People)            │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    demo_status_accessibility();

    // =========================================================================
    // Phase 4: Render Modes
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 4: Render Modes (Unicode, ASCII, Plain)                   │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    demo_render_modes();

    // =========================================================================
    // Phase 5: Publish Status Visualization (Use Case)
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 5: Publish Status Dependency Graph                        │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    demo_publish_status_graph();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("     Graph TUI Demo Completed!");
    println!("═══════════════════════════════════════════════════════════════════\n");
}

#[cfg(feature = "native")]
fn demo_graph_creation() {
    let mut graph: Graph<&str, &str> = Graph::new();

    // Add nodes
    graph.add_node(Node::new("A", "Node A").with_label("A"));
    graph.add_node(Node::new("B", "Node B").with_label("B"));
    graph.add_node(Node::new("C", "Node C").with_label("C"));

    // Add edges
    graph.add_edge(Edge::new("A", "B", "edge1"));
    graph.add_edge(Edge::new("B", "C", "edge2"));

    println!("  Graph Statistics:");
    println!("    Nodes: {}", graph.node_count());
    println!("    Edges: {}", graph.edge_count());
    println!("    Exceeds TUI limit ({}): {}", MAX_TUI_NODES, graph.exceeds_tui_limit());
    println!("    Default visible nodes: {}", DEFAULT_VISIBLE_NODES);
}

#[cfg(feature = "native")]
fn demo_layout_algorithms() {
    let algorithms = [
        ("Grid", LayoutAlgorithm::Grid),
        ("Force-Directed", LayoutAlgorithm::ForceDirected),
        ("Hierarchical", LayoutAlgorithm::Hierarchical),
        ("Radial", LayoutAlgorithm::Radial),
    ];

    println!("  Available Layout Algorithms:");
    println!();
    println!("  {:<20} {:<15} {:<30}", "Algorithm", "Complexity", "Best For");
    println!("  {}", "─".repeat(65));

    let descriptions = [
        ("O(n)", "Simple visualization"),
        ("O(n²/iter)", "General graphs"),
        ("O(nm log m)", "DAGs, pipelines"),
        ("O(n + m)", "Trees, hierarchies"),
    ];

    for ((name, _algo), (complexity, best_for)) in algorithms.iter().zip(descriptions.iter()) {
        println!("  {:<20} {:<15} {:<30}", name, complexity, best_for);
    }

    // Demo each layout
    println!("\n  Layout Demonstrations:\n");

    for (name, algo) in algorithms {
        let mut graph: Graph<(), ()> = Graph::new();
        for i in 0..6 {
            graph.add_node(Node::new(format!("n{}", i), ()));
        }
        graph.add_edge(Edge::new("n0", "n1", ()));
        graph.add_edge(Edge::new("n0", "n2", ()));
        graph.add_edge(Edge::new("n1", "n3", ()));
        graph.add_edge(Edge::new("n2", "n4", ()));
        graph.add_edge(Edge::new("n3", "n5", ()));
        graph.add_edge(Edge::new("n4", "n5", ()));

        let config = LayoutConfig {
            algorithm: algo,
            width: 40.0,
            height: 10.0,
            iterations: 50,
            ..Default::default()
        };
        LayoutEngine::compute(&mut graph, &config);

        let renderer = GraphRenderer::new();
        let output = renderer.render(&graph, 40, 10);

        println!("  {}:", name);
        println!("  ┌{}┐", "─".repeat(40));
        for line in output.to_string_plain().lines() {
            println!("  │{}│", line);
        }
        println!("  └{}┘", "─".repeat(40));
        println!();
    }
}

#[cfg(feature = "native")]
fn demo_status_accessibility() {
    println!("  Node Status with Shapes (Accessibility):");
    println!();
    println!("  Per peer review #6: Shapes differentiate status, not just color.");
    println!("  This ensures color-blind users can distinguish node states.\n");

    let statuses = [
        (NodeStatus::Healthy, "Healthy", "Green"),
        (NodeStatus::Warning, "Warning", "Yellow"),
        (NodeStatus::Error, "Error", "Red"),
        (NodeStatus::Info, "Info", "Cyan"),
        (NodeStatus::Neutral, "Neutral", "Gray"),
    ];

    println!("  {:<12} {:<10} {:<10} {:<10}", "Status", "Shape", "Unicode", "ASCII");
    println!("  {}", "─".repeat(45));

    for (status, name, _color) in statuses {
        let shape = status.shape();
        println!(
            "  {}{:<12}\x1b[0m {:<10} {:<10} {:<10}",
            status.color_code(),
            name,
            format!("{:?}", shape),
            shape.unicode(),
            shape.ascii()
        );
    }
}

#[cfg(feature = "native")]
fn demo_render_modes() {
    let mut graph: Graph<&str, ()> = Graph::new();
    graph.add_node(Node::new("healthy", "ok").with_status(NodeStatus::Healthy).with_label("OK"));
    graph
        .add_node(Node::new("warning", "warn").with_status(NodeStatus::Warning).with_label("WARN"));
    graph.add_node(Node::new("error", "err").with_status(NodeStatus::Error).with_label("ERR"));
    graph.add_edge(Edge::new("healthy", "warning", ()));
    graph.add_edge(Edge::new("warning", "error", ()));

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Hierarchical,
        width: 50.0,
        height: 8.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    let modes = [
        ("Unicode (default)", RenderMode::Unicode),
        ("ASCII (legacy)", RenderMode::Ascii),
        ("Plain (no color)", RenderMode::Plain),
    ];

    for (name, mode) in modes {
        let renderer = GraphRenderer::new().with_mode(mode);
        let output = renderer.render(&graph, 50, 8);

        println!("  {}:", name);
        if mode == RenderMode::Plain {
            println!("{}", output.to_string_plain());
        } else {
            println!("{}", output.to_string_colored());
        }
        println!();
    }
}

#[cfg(feature = "native")]
fn demo_publish_status_graph() {
    println!("  Simulating PAIML Stack Dependency Graph:\n");

    let mut graph: Graph<&str, &str> = Graph::new();

    // Core layer
    graph.add_node(
        Node::new("trueno", "core")
            .with_status(NodeStatus::Healthy)
            .with_label("trueno")
            .with_importance(1.0),
    );

    // ML layer
    graph.add_node(
        Node::new("aprender", "ml")
            .with_status(NodeStatus::Healthy)
            .with_label("aprender")
            .with_importance(0.9),
    );

    // Orchestration layer
    graph.add_node(
        Node::new("batuta", "orch")
            .with_status(NodeStatus::Warning)
            .with_label("batuta")
            .with_importance(0.8),
    );

    // Needs publish
    graph.add_node(
        Node::new("depyler", "transpiler")
            .with_status(NodeStatus::Error)
            .with_label("depyler")
            .with_importance(0.7),
    );

    // Not published
    graph.add_node(
        Node::new("certeza", "quality")
            .with_status(NodeStatus::Info)
            .with_label("certeza")
            .with_importance(0.6),
    );

    // Dependencies
    graph.add_edge(Edge::new("trueno", "aprender", "depends"));
    graph.add_edge(Edge::new("aprender", "batuta", "depends"));
    graph.add_edge(Edge::new("trueno", "depyler", "depends"));
    graph.add_edge(Edge::new("batuta", "certeza", "depends"));

    let config = LayoutConfig {
        algorithm: LayoutAlgorithm::Hierarchical,
        width: 60.0,
        height: 12.0,
        ..Default::default()
    };
    LayoutEngine::compute(&mut graph, &config);

    let renderer = GraphRenderer::new();
    let output = renderer.render(&graph, 60, 12);

    println!("{}", output.to_string_colored());

    println!("  Legend:");
    println!("    {} Healthy (up to date)", NodeStatus::Healthy.shape().unicode());
    println!("    {} Warning (needs commit)", NodeStatus::Warning.shape().unicode());
    println!("    {} Error (needs publish)", NodeStatus::Error.shape().unicode());
    println!("    {} Info (not published)", NodeStatus::Info.shape().unicode());

    println!("\n  Top nodes by importance:");
    for node in graph.top_nodes_by_importance(3) {
        println!("    {} (importance: {:.1})", node.id, node.importance);
    }
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example graph_tui_demo --features native");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn test_demo_graph_creation() {
        let mut graph: Graph<&str, &str> = Graph::new();
        graph.add_node(Node::new("A", "test"));
        graph.add_node(Node::new("B", "test"));
        graph.add_edge(Edge::new("A", "B", "edge"));

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_all_layouts_work() {
        for algo in [
            LayoutAlgorithm::Grid,
            LayoutAlgorithm::ForceDirected,
            LayoutAlgorithm::Hierarchical,
            LayoutAlgorithm::Radial,
        ] {
            let mut graph: Graph<(), ()> = Graph::new();
            for i in 0..5 {
                graph.add_node(Node::new(format!("n{}", i), ()));
            }
            graph.add_edge(Edge::new("n0", "n1", ()));
            graph.add_edge(Edge::new("n1", "n2", ()));

            let config = LayoutConfig { algorithm: algo, ..Default::default() };
            LayoutEngine::compute(&mut graph, &config);

            let renderer = GraphRenderer::new();
            let output = renderer.render(&graph, 40, 10);

            assert!(!output.to_string_plain().is_empty());
        }
    }

    #[test]
    fn test_status_shapes_unique() {
        let statuses = [
            NodeStatus::Healthy,
            NodeStatus::Warning,
            NodeStatus::Error,
            NodeStatus::Info,
            NodeStatus::Neutral,
        ];

        let shapes: Vec<_> = statuses.iter().map(|s| s.shape()).collect();

        // All shapes should be unique
        for i in 0..shapes.len() {
            for j in (i + 1)..shapes.len() {
                assert_ne!(shapes[i], shapes[j], "Shapes must be unique for accessibility");
            }
        }
    }
}
