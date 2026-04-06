#![allow(clippy::expect_used, clippy::unwrap_used)]
//! trueno-graph Integration Example
//!
//! Demonstrates using trueno-viz visualization extensions with trueno-graph
//! for graph visualization, `PageRank` analysis, and community detection.
//!
//! Run with: `cargo run --example trueno_graph_integration --features graph`

use trueno_graph::{louvain, pagerank, CsrGraph, NodeId};
use trueno_viz::interop::trueno_graph::{CommunityViz, GraphViz, PageRankViz};
use trueno_viz::output::PngEncoder;

fn main() {
    println!("trueno-graph Integration Example");
    println!("=================================\n");

    // Create a sample graph
    println!("Creating sample graph...");
    let graph = create_sample_graph();
    println!("   Graph: {} nodes, {} edges\n", graph.num_nodes(), graph.num_edges());

    // Example 1: Basic force-directed layout
    println!("1. Creating force-directed graph visualization...");
    let fb = graph.to_force_graph().expect("Failed to create force graph");
    PngEncoder::write_to_file(&fb, "trueno_force_graph.png").expect("Failed to write PNG");
    println!("   Saved: trueno_force_graph.png ({}x{})\n", fb.width(), fb.height());

    // Example 2: Custom dimensions
    println!("2. Creating force graph with custom dimensions...");
    let fb = graph.to_force_graph_with(800, 600).expect("Failed to create force graph");
    PngEncoder::write_to_file(&fb, "trueno_force_graph_large.png").expect("Failed to write PNG");
    println!("   Saved: trueno_force_graph_large.png ({}x{})\n", fb.width(), fb.height());

    // Example 3: Community detection visualization
    println!("3. Creating community-colored graph...");
    let fb = graph.to_community_graph().expect("Failed to create community graph");
    PngEncoder::write_to_file(&fb, "trueno_communities.png").expect("Failed to write PNG");
    println!("   Saved: trueno_communities.png ({}x{})\n", fb.width(), fb.height());

    // Example 4: PageRank-sized nodes
    println!("4. Creating PageRank-sized graph...");
    let fb = graph.to_pagerank_graph().expect("Failed to create pagerank graph");
    PngEncoder::write_to_file(&fb, "trueno_pagerank.png").expect("Failed to write PNG");
    println!("   Saved: trueno_pagerank.png ({}x{})\n", fb.width(), fb.height());

    // Example 5: Full analysis (communities + PageRank)
    println!("5. Creating full analysis graph (communities + PageRank)...");
    let fb = graph.to_analysis_graph().expect("Failed to create analysis graph");
    PngEncoder::write_to_file(&fb, "trueno_analysis.png").expect("Failed to write PNG");
    println!("   Saved: trueno_analysis.png ({}x{})\n", fb.width(), fb.height());

    // Example 6: Degree histogram
    println!("6. Creating degree distribution histogram...");
    let fb = graph.degree_histogram().expect("Failed to create degree histogram");
    PngEncoder::write_to_file(&fb, "trueno_degree_hist.png").expect("Failed to write PNG");
    println!("   Saved: trueno_degree_hist.png ({}x{})\n", fb.width(), fb.height());

    // Example 7: In-degree vs Out-degree scatter
    println!("7. Creating in-degree vs out-degree scatter plot...");
    let fb = graph.degree_scatter().expect("Failed to create degree scatter");
    PngEncoder::write_to_file(&fb, "trueno_degree_scatter.png").expect("Failed to write PNG");
    println!("   Saved: trueno_degree_scatter.png ({}x{})\n", fb.width(), fb.height());

    // Example 8: PageRank score histogram
    println!("8. Creating PageRank score histogram...");
    let scores = pagerank(&graph, 20, 1e-6).expect("PageRank failed");
    let fb = scores.to_histogram().expect("Failed to create histogram");
    PngEncoder::write_to_file(&fb, "trueno_pr_hist.png").expect("Failed to write PNG");
    println!("   Saved: trueno_pr_hist.png ({}x{})\n", fb.width(), fb.height());

    // Example 9: Top PageRank nodes
    println!("9. Creating top-5 PageRank nodes visualization...");
    let fb = scores.top_n_bar(5).expect("Failed to create bar chart");
    PngEncoder::write_to_file(&fb, "trueno_pr_top5.png").expect("Failed to write PNG");
    println!("   Saved: trueno_pr_top5.png ({}x{})\n", fb.width(), fb.height());

    // Example 10: Community size histogram
    println!("10. Creating community size histogram...");
    let communities = louvain(&graph).expect("Louvain failed");
    println!(
        "    Found {} communities (modularity: {:.3})",
        communities.num_communities,
        communities.modularity_score()
    );
    let fb = communities.size_histogram().expect("Failed to create histogram");
    PngEncoder::write_to_file(&fb, "trueno_comm_sizes.png").expect("Failed to write PNG");
    println!("   Saved: trueno_comm_sizes.png ({}x{})\n", fb.width(), fb.height());

    println!("--- Summary ---");
    println!("Generated 10 visualization files demonstrating trueno-graph integration:");
    println!("  - Force-directed layouts: basic, custom size");
    println!("  - Community detection: colored nodes");
    println!("  - PageRank analysis: sized nodes, histograms, top-N");
    println!("  - Degree analysis: histogram, scatter plot");
    println!("\nAll visualizations successfully generated!");
}

/// Create a sample graph with communities.
fn create_sample_graph() -> CsrGraph {
    let mut graph = CsrGraph::new();

    // Community 1: Triangle cluster (nodes 0, 1, 2)
    graph.add_edge(NodeId(0), NodeId(1), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(1), NodeId(0), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(1), NodeId(2), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(2), NodeId(1), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(2), NodeId(0), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(0), NodeId(2), 1.0).expect("operation should succeed");

    // Community 2: Triangle cluster (nodes 3, 4, 5)
    graph.add_edge(NodeId(3), NodeId(4), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(4), NodeId(3), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(4), NodeId(5), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(5), NodeId(4), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(5), NodeId(3), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(3), NodeId(5), 1.0).expect("operation should succeed");

    // Community 3: Star around node 6 (nodes 6, 7, 8, 9)
    graph.add_edge(NodeId(7), NodeId(6), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(8), NodeId(6), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(9), NodeId(6), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(6), NodeId(7), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(6), NodeId(8), 1.0).expect("operation should succeed");
    graph.add_edge(NodeId(6), NodeId(9), 1.0).expect("operation should succeed");

    // Bridge edges between communities
    graph.add_edge(NodeId(2), NodeId(3), 0.5).expect("operation should succeed");
    graph.add_edge(NodeId(3), NodeId(2), 0.5).expect("operation should succeed");
    graph.add_edge(NodeId(5), NodeId(6), 0.5).expect("operation should succeed");
    graph.add_edge(NodeId(6), NodeId(5), 0.5).expect("operation should succeed");

    // Set node names
    graph.set_node_name(NodeId(0), "A".to_string());
    graph.set_node_name(NodeId(1), "B".to_string());
    graph.set_node_name(NodeId(2), "C".to_string());
    graph.set_node_name(NodeId(3), "D".to_string());
    graph.set_node_name(NodeId(4), "E".to_string());
    graph.set_node_name(NodeId(5), "F".to_string());
    graph.set_node_name(NodeId(6), "Hub".to_string());
    graph.set_node_name(NodeId(7), "G".to_string());
    graph.set_node_name(NodeId(8), "H".to_string());
    graph.set_node_name(NodeId(9), "I".to_string());

    graph
}
