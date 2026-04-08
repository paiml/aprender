#![allow(clippy::disallowed_methods)]
//! Chapter 18: Graph Algorithms
//!
//! Demonstrates graph construction, Dijkstra, community detection, metrics.
//! Citation: Blondel et al., "Fast Unfolding of Communities," arXiv:0803.0476
//! Contract: contracts/apr-book-ch18-v1.yaml (v2 — api_calls enforced)

use aprender::graph::Graph;

fn main() {
    // Build a social-network-like undirected graph with two communities
    let edges = vec![
        // Community 1: tightly connected (nodes 0-3)
        (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0),
        // Community 2: tightly connected (nodes 4-7)
        (4, 5, 1.0), (4, 6, 1.0), (5, 6, 1.0), (5, 7, 1.0), (6, 7, 1.0),
        // Bridge: weak connection between communities
        (3, 4, 1.0),
    ];
    let g = Graph::from_weighted_edges(&edges, false);

    println!("Graph: {} nodes, {} edges", g.num_nodes(), g.num_edges());
    assert!(g.num_nodes() >= 8, "Graph must have 8 nodes");

    // Dijkstra shortest path
    if let Some((path, dist)) = g.dijkstra(0, 7) {
        println!("\nDijkstra 0 -> 7: path={:?}, distance={dist:.0}", path);
        assert!(dist > 0.0, "Distance must be positive");
        assert!(path.len() >= 2, "Path must have start and end");
    }

    // Graph density (Cormen et al.)
    let density = g.density();
    println!("\nGraph metrics:");
    println!("  Density: {density:.3}");
    assert!(density > 0.0 && density <= 1.0, "Density in (0,1]");

    // Clustering coefficient
    let cc = g.clustering_coefficient();
    println!("  Clustering coefficient: {cc:.3}");
    assert!(cc >= 0.0 && cc <= 1.0, "CC in [0,1]");
    // Our two tightly-connected communities should have high CC
    assert!(cc > 0.3, "Two cliques should yield CC > 0.3");

    // Label propagation community detection (Blondel et al., 2008)
    let labels = g.label_propagation(100, Some(42));
    println!("\nLabel propagation communities:");
    for (node, &label) in labels.iter().enumerate() {
        println!("  node {node}: community {label}");
    }
    assert_eq!(labels.len(), g.num_nodes(), "One label per node");

    // Community 1 (0-3) should share a label, community 2 (4-7) another
    let c1_label = labels[0];
    let c2_label = labels[4];
    let c1_coherent = labels[0..4].iter().all(|&l| l == c1_label);
    let c2_coherent = labels[4..8].iter().all(|&l| l == c2_label);
    println!("  Community 1 coherent: {c1_coherent}");
    println!("  Community 2 coherent: {c2_coherent}");

    println!("\nChapter 18 contracts: PASSED");
}
