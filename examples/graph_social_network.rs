//! Social Network Analysis Example
//!
//! This example demonstrates graph algorithms on a social network:
//! - Building a friendship graph
//! - Finding most connected people (degree centrality)
//! - Identifying influential users (PageRank)
//! - Discovering bridges between communities (betweenness centrality)
//!
//! Run with: `cargo run --example graph_social_network`

use aprender::graph::Graph;

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Social Network Analysis with Aprender");
    println!("═══════════════════════════════════════════════════════\n");

    // Build a social network with 10 people
    // Network structure:
    //   Tech Community: Alice (0) - Bob (1) - Charlie (2) - Diana (3)
    //   Art Community:  Eve (4) - Frank (5) - Grace (6)
    //   Bridge: Diana (3) connects to Eve (4)
    //   Isolated group: Henry (7) - Iris (8) - Jack (9)
    //   Bridge: Grace (6) connects to Henry (7)

    let edges = vec![
        // Tech community (densely connected)
        (0, 1), // Alice - Bob
        (1, 2), // Bob - Charlie
        (2, 3), // Charlie - Diana
        (0, 2), // Alice - Charlie (shortcut)
        (1, 3), // Bob - Diana (shortcut)
        // Art community (moderately connected)
        (4, 5), // Eve - Frank
        (5, 6), // Frank - Grace
        (4, 6), // Eve - Grace (shortcut)
        // Bridge between tech and art
        (3, 4), // Diana - Eve (BRIDGE)
        // Isolated group
        (7, 8), // Henry - Iris
        (8, 9), // Iris - Jack
        (7, 9), // Henry - Jack (triangle)
        // Bridge to isolated group
        (6, 7), // Grace - Henry (BRIDGE)
    ];

    let names = vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
    ];

    println!("📊 Building social network with {} people...", names.len());
    println!("   - Tech Community: Alice, Bob, Charlie, Diana");
    println!("   - Art Community: Eve, Frank, Grace");
    println!("   - Isolated Group: Henry, Iris, Jack");
    println!("   - {} friendships total\n", edges.len());

    let graph = Graph::from_edges(&edges, false);

    // ========================================================================
    // Degree Centrality: Who has the most friends?
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  1. Degree Centrality (Most Connected People)");
    println!("═══════════════════════════════════════════════════════\n");

    let degree_scores = graph.degree_centrality();
    let mut degree_with_names: Vec<(&str, f64)> = Vec::new();
    for (node_id, score) in &degree_scores {
        degree_with_names.push((names[*node_id], *score));
    }
    degree_with_names.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    println!("Top 5 Most Connected People:");
    for (i, (name, score)) in degree_with_names.iter().take(5).enumerate() {
        println!(
            "  {}. {} - {:.3} (normalized degree centrality)",
            i + 1,
            name,
            score
        );
    }
    println!("\n💡 Interpretation:");
    println!("   Higher scores = more direct friendships");
    println!("   These people are the \"social butterflies\"\n");

    // ========================================================================
    // PageRank: Who is most influential?
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  2. PageRank (Influence Scores)");
    println!("═══════════════════════════════════════════════════════\n");

    let pagerank_scores = graph.pagerank(0.85, 100, 1e-6).expect("PageRank failed");
    let mut pagerank_with_names: Vec<(&str, f64)> = Vec::new();
    for (i, &score) in pagerank_scores.iter().enumerate() {
        pagerank_with_names.push((names[i], score));
    }
    pagerank_with_names
        .sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    println!("Top 5 Most Influential People:");
    for (i, (name, score)) in pagerank_with_names.iter().take(5).enumerate() {
        println!("  {}. {} - {:.4} (PageRank score)", i + 1, name, score);
    }
    println!("\n💡 Interpretation:");
    println!("   PageRank considers both quantity and quality of connections");
    println!("   Being connected to influential people boosts your score");
    println!("   Similar to how Google ranks web pages\n");

    // ========================================================================
    // Betweenness Centrality: Who bridges communities?
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  3. Betweenness Centrality (Bridges Between Groups)");
    println!("═══════════════════════════════════════════════════════\n");

    let betweenness_scores = graph.betweenness_centrality();
    let mut betweenness_with_names: Vec<(&str, f64)> = Vec::new();
    for (i, &score) in betweenness_scores.iter().enumerate() {
        betweenness_with_names.push((names[i], score));
    }
    betweenness_with_names
        .sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    println!("Top 5 Bridge People:");
    for (i, (name, score)) in betweenness_with_names.iter().take(5).enumerate() {
        println!(
            "  {}. {} - {:.2} (betweenness centrality)",
            i + 1,
            name,
            score
        );
    }
    println!("\n💡 Interpretation:");
    println!("   High scores = many shortest paths pass through this person");
    println!("   These people connect different communities");
    println!("   Removing them would fragment the network\n");

    // ========================================================================
    // Analysis Summary
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  Analysis Summary");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Key Insights:");
    println!(
        "  • {} is well-connected in their community",
        degree_with_names[0].0
    );
    println!(
        "  • {} is the most influential person overall",
        pagerank_with_names[0].0
    );
    println!(
        "  • {} is a critical bridge between groups",
        betweenness_with_names[0].0
    );
    println!("\nNetwork Properties:");
    println!("  • Nodes: {}", graph.num_nodes());
    println!("  • Edges: {}", graph.num_edges());
    println!(
        "  • Average degree: {:.2}",
        2.0 * graph.num_edges() as f64 / graph.num_nodes() as f64
    );

    println!("\n🚀 Performance Notes:");
    println!("  • CSR representation: 50-70% memory reduction vs HashMap");
    println!("  • PageRank with Kahan summation: prevents floating-point drift");
    println!("  • Betweenness with Rayon: ~8x speedup on 8-core CPU");

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Example Complete!");
    println!("═══════════════════════════════════════════════════════");
}
