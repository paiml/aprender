//! Social Network Analysis Example
//!
//! This example demonstrates graph algorithms on a social network:
//! - Building a friendship graph
//! - Finding most connected people (degree centrality)
//! - Identifying influential users (PageRank, eigenvector, Katz)
//! - Discovering bridges between communities (betweenness centrality)
//! - Measuring network closeness (closeness and harmonic centrality)
//! - Analyzing network structure (density, diameter, clustering, assortativity)
//!
//! Run with: `cargo run --example graph_social_network`

use aprender::graph::Graph;

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Social Network Analysis with Aprender");
    println!("═══════════════════════════════════════════════════════\n");

    let (graph, names) = build_social_network();

    let degree_with_names = analyze_degree_centrality(&graph, &names);
    let pagerank_with_names = analyze_pagerank(&graph, &names);
    let betweenness_with_names = analyze_betweenness(&graph, &names);
    analyze_closeness(&graph, &names);
    analyze_eigenvector(&graph, &names);
    analyze_network_structure(&graph);
    print_summary(
        &degree_with_names,
        &pagerank_with_names,
        &betweenness_with_names,
        &graph,
    );
}

fn build_social_network() -> (Graph, Vec<&'static str>) {
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
    (graph, names)
}

fn analyze_degree_centrality<'a>(graph: &Graph, names: &[&'a str]) -> Vec<(&'a str, f64)> {
    println!("═══════════════════════════════════════════════════════");
    println!("  1. Degree Centrality (Most Connected People)");
    println!("═══════════════════════════════════════════════════════\n");

    let degree_scores = graph.degree_centrality();
    let mut result: Vec<(&str, f64)> = degree_scores
        .iter()
        .map(|(id, score)| (names[*id], *score))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    print_top_5(
        &result,
        "Most Connected People",
        "normalized degree centrality",
    );
    println!("\n💡 Interpretation:");
    println!("   Higher scores = more direct friendships");
    println!("   These people are the \"social butterflies\"\n");
    result
}

fn analyze_pagerank<'a>(graph: &Graph, names: &[&'a str]) -> Vec<(&'a str, f64)> {
    println!("═══════════════════════════════════════════════════════");
    println!("  2. PageRank (Influence Scores)");
    println!("═══════════════════════════════════════════════════════\n");

    let pagerank_scores = graph.pagerank(0.85, 100, 1e-6).expect("PageRank failed");
    let mut result: Vec<(&str, f64)> = pagerank_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (names[i], s))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    print_top_5(&result, "Most Influential People", "PageRank score");
    println!("\n💡 Interpretation:");
    println!("   PageRank considers both quantity and quality of connections");
    println!("   Being connected to influential people boosts your score");
    println!("   Similar to how Google ranks web pages\n");
    result
}

fn analyze_betweenness<'a>(graph: &Graph, names: &[&'a str]) -> Vec<(&'a str, f64)> {
    println!("═══════════════════════════════════════════════════════");
    println!("  3. Betweenness Centrality (Bridges Between Groups)");
    println!("═══════════════════════════════════════════════════════\n");

    let betweenness_scores = graph.betweenness_centrality();
    let mut result: Vec<(&str, f64)> = betweenness_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (names[i], s))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    print_top_5(&result, "Bridge People", "betweenness centrality");
    println!("\n💡 Interpretation:");
    println!("   High scores = many shortest paths pass through this person");
    println!("   These people connect different communities");
    println!("   Removing them would fragment the network\n");
    result
}

fn analyze_closeness(graph: &Graph, names: &[&str]) {
    println!("═══════════════════════════════════════════════════════");
    println!("  4. Closeness Centrality (Reachability)");
    println!("═══════════════════════════════════════════════════════\n");

    let closeness_scores = graph.closeness_centrality();
    let mut result: Vec<(&str, f64)> = closeness_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (names[i], s))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    print_top_5(&result, "Most Reachable People", "closeness centrality");
    println!("\n💡 Interpretation:");
    println!("   High scores = can reach others via short paths");
    println!("   These people can spread information quickly");
    println!("   Based on average shortest path distance\n");
}

fn analyze_eigenvector(graph: &Graph, names: &[&str]) {
    println!("═══════════════════════════════════════════════════════");
    println!("  5. Eigenvector Centrality (Quality of Connections)");
    println!("═══════════════════════════════════════════════════════\n");

    let eigenvector_scores = graph
        .eigenvector_centrality(100, 1e-6)
        .expect("Eigenvector computation failed");
    let mut result: Vec<(&str, f64)> = eigenvector_scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (names[i], s))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Example data should be valid"));

    print_top_5(&result, "by Connection Quality", "eigenvector centrality");
    println!("\n💡 Interpretation:");
    println!("   High scores = connected to other well-connected people");
    println!("   Quality matters more than quantity of connections");
    println!("   Uses power iteration method\n");
}

fn print_top_5(data: &[(&str, f64)], label: &str, metric: &str) {
    println!("Top 5 {label}:");
    for (i, (name, score)) in data.iter().take(5).enumerate() {
        println!("  {}. {} - {:.4} ({metric})", i + 1, name, score);
    }
}

fn analyze_network_structure(graph: &Graph) {
    println!("═══════════════════════════════════════════════════════");
    println!("  6. Network Structural Statistics");
    println!("═══════════════════════════════════════════════════════\n");

    let density = graph.density();
    let diameter = graph.diameter();
    let clustering = graph.clustering_coefficient();
    let assortativity = graph.assortativity();

    println!("Network Metrics:");
    println!("  • Density: {density:.4}");
    println!("    (Ratio of actual to possible edges)");
    match diameter {
        Some(d) => println!("  • Diameter: {d} hops"),
        None => println!("  • Diameter: ∞ (disconnected graph)"),
    }
    println!("    (Longest shortest path in network)");
    println!("  • Clustering Coefficient: {clustering:.4}");
    println!("    (Probability that your friends are friends)");
    println!("  • Degree Assortativity: {assortativity:.4}");
    println!("    (Do popular people connect with popular people?)");

    println!("\n💡 Interpretation:");
    println!(
        "   Density: {:.1}% of possible connections exist",
        density * 100.0
    );
    if assortativity > 0.0 {
        println!("   Assortativity > 0: Popular people prefer popular people");
    } else {
        println!("   Assortativity < 0: Popular people connect with less popular");
    }
    if clustering > 0.5 {
        println!("   High clustering: Strong community structure");
    } else {
        println!("   Low clustering: Sparse friend-of-friend connections");
    }
    println!();
}

fn print_summary(
    degree: &[(&str, f64)],
    pagerank: &[(&str, f64)],
    betweenness: &[(&str, f64)],
    graph: &Graph,
) {
    println!("═══════════════════════════════════════════════════════");
    println!("  Analysis Summary");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Key Insights:");
    println!("  • {} is well-connected in their community", degree[0].0);
    println!(
        "  • {} is the most influential person overall",
        pagerank[0].0
    );
    println!(
        "  • {} is a critical bridge between groups",
        betweenness[0].0
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
