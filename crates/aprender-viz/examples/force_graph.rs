#![allow(clippy::expect_used, clippy::unwrap_used, clippy::needless_range_loop)]
//! Force-Directed Graph Layout Example
//!
//! Demonstrates creating network graph visualizations using
//! the Fruchterman-Reingold force-directed layout algorithm.
//!
//! Run with: `cargo run --example force_graph`

use trueno_viz::color::Rgba;
use trueno_viz::output::PngEncoder;
use trueno_viz::plots::{ForceGraph, GraphEdge, GraphNode};
use trueno_viz::prelude::WithDimensions;

fn main() {
    println!("Force-Directed Graph Layout Example");
    println!("====================================\n");

    // Example 1: Simple triangle
    println!("Example 1: Triangle Graph");
    println!("-------------------------");

    let triangle = ForceGraph::new()
        .add_node(GraphNode::new(0).color(Rgba::new(234, 67, 53, 255)))
        .add_node(GraphNode::new(1).color(Rgba::new(66, 133, 244, 255)))
        .add_node(GraphNode::new(2).color(Rgba::new(52, 168, 83, 255)))
        .edge(0, 1)
        .edge(1, 2)
        .edge(2, 0)
        .dimensions(300, 300)
        .iterations(100)
        .build()
        .expect("Failed to build graph");

    println!("  Nodes: {}", triangle.num_nodes());
    println!("  Edges: {}", triangle.num_edges());

    let fb = triangle.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "graph_triangle.png").expect("Failed to write PNG");
    println!("  Saved: graph_triangle.png\n");

    // Example 2: Star topology
    println!("Example 2: Star Topology");
    println!("------------------------");

    let mut star = ForceGraph::new()
        .add_node(GraphNode::new(0).color(Rgba::new(255, 193, 7, 255)).radius(12.0)) // Center
        .dimensions(400, 400)
        .iterations(150);

    // Add outer nodes connected to center
    for i in 1..=6 {
        star = star.add_node(GraphNode::new(i).color(Rgba::new(156, 39, 176, 255))).edge(0, i);
    }

    let star_graph = star.build().expect("Failed to build star");
    println!("  Nodes: {} (1 center + 6 outer)", star_graph.num_nodes());
    println!("  Edges: {}", star_graph.num_edges());

    let fb = star_graph.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "graph_star.png").expect("Failed to write PNG");
    println!("  Saved: graph_star.png\n");

    // Example 3: Social network-like graph
    println!("Example 3: Social Network Graph");
    println!("-------------------------------");

    let social = create_social_network();
    println!("  Nodes: {}", social.num_nodes());
    println!("  Edges: {}", social.num_edges());

    let fb = social.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "graph_social.png").expect("Failed to write PNG");
    println!("  Saved: graph_social.png\n");

    // Example 4: Grid graph
    println!("Example 4: Grid Graph");
    println!("---------------------");

    let grid = create_grid_graph(4, 3);
    println!("  Nodes: {} (4x3 grid)", grid.num_nodes());
    println!("  Edges: {}", grid.num_edges());

    let fb = grid.to_framebuffer().expect("Failed to render");
    PngEncoder::write_to_file(&fb, "graph_grid.png").expect("Failed to write PNG");
    println!("  Saved: graph_grid.png\n");

    // Print positions for the simple triangle
    println!("--- Node Positions (Triangle) ---");
    for (i, (x, y)) in triangle.positions().iter().enumerate() {
        println!("  Node {i}: ({x:.1}, {y:.1})");
    }

    println!("\n--- Summary ---");
    println!("Force-directed layout uses physics simulation:");
    println!("  - Nodes repel each other (like charged particles)");
    println!("  - Edges attract connected nodes (like springs)");
    println!("  - Result: visually pleasing graph layouts");
    println!("\nGraph visualizations successfully generated!");
}

/// Create a social network-like graph with clusters.
fn create_social_network() -> trueno_viz::plots::BuiltForceGraph {
    let colors = [
        Rgba::new(66, 133, 244, 255), // Blue cluster
        Rgba::new(234, 67, 53, 255),  // Red cluster
        Rgba::new(52, 168, 83, 255),  // Green cluster
    ];

    let mut graph =
        ForceGraph::new().dimensions(500, 400).iterations(200).repulsion(8000.0).attraction(0.02);

    // Create 3 clusters of 4 nodes each
    for cluster in 0..3 {
        let base = cluster * 4;
        let color = colors[cluster];

        // Add nodes
        for i in 0..4 {
            let node =
                GraphNode::new(base + i).color(color).radius(if i == 0 { 10.0 } else { 6.0 }); // Leader is bigger
            graph = graph.add_node(node);
        }

        // Connect within cluster (all to leader)
        for i in 1..4 {
            graph = graph.add_edge(GraphEdge::new(base, base + i).width(2.0));
        }
    }

    // Connect clusters through leaders
    graph = graph
        .add_edge(GraphEdge::new(0, 4).color(Rgba::new(150, 150, 150, 150)))
        .add_edge(GraphEdge::new(4, 8).color(Rgba::new(150, 150, 150, 150)))
        .add_edge(GraphEdge::new(8, 0).color(Rgba::new(150, 150, 150, 150)));

    graph.build().expect("Failed to build social network")
}

/// Create a grid graph.
fn create_grid_graph(cols: usize, rows: usize) -> trueno_viz::plots::BuiltForceGraph {
    let mut graph =
        ForceGraph::new().dimensions(400, 300).iterations(150).repulsion(5000.0).attraction(0.03);

    // Add nodes
    for i in 0..(cols * rows) {
        let node = GraphNode::new(i).color(Rgba::new(100, 149, 237, 255)).radius(8.0);
        graph = graph.add_node(node);
    }

    // Add horizontal edges
    for row in 0..rows {
        for col in 0..(cols - 1) {
            let from = row * cols + col;
            let to = from + 1;
            graph = graph.edge(from, to);
        }
    }

    // Add vertical edges
    for row in 0..(rows - 1) {
        for col in 0..cols {
            let from = row * cols + col;
            let to = from + cols;
            graph = graph.edge(from, to);
        }
    }

    graph.build().expect("Failed to build grid graph")
}
