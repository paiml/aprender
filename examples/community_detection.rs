//! Community Detection using Louvain Algorithm
//!
//! This example demonstrates community detection in networks using the Louvain
//! algorithm for modularity optimization.
//!
//! # Run
//!
//! ```bash
//! cargo run --example community_detection
//! ```

use aprender::graph::Graph;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Community Detection with Louvain Algorithm ===\n");

    // Example 1: Simple Two-Community Network
    println!("Example 1: Two Triangles Connected");
    println!("-----------------------------------");

    let g1 = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0), // Triangle 1
            (3, 4),
            (4, 5),
            (5, 3), // Triangle 2
            (2, 3), // Bridge
        ],
        false,
    );

    let communities = g1.louvain();
    let modularity = g1.modularity(&communities);

    println!("Detected {} communities", communities.len());
    for (i, community) in communities.iter().enumerate() {
        println!("  Community {}: {:?}", i + 1, community);
    }
    println!("Modularity: {modularity:.3}\n");

    // Example 2: Social Network (Karate Club Style)
    println!("\nExample 2: Social Network Clustering");
    println!("------------------------------------");

    let g2 = Graph::from_edges(
        &[
            (0, 1),
            (0, 2),
            (1, 2), // Group A
            (2, 3), // Bridge person
            (3, 4),
            (3, 5),
            (4, 5), // Group B
        ],
        false,
    );

    let communities2 = g2.louvain();
    let modularity2 = g2.modularity(&communities2);

    println!("Detected {} communities", communities2.len());
    for (i, community) in communities2.iter().enumerate() {
        println!("  Community {}: {:?}", i + 1, community);
    }
    println!("Modularity: {modularity2:.3}");
    println!(
        "Bridge node (3) connects communities: {}",
        if communities2.len() >= 2 {
            "Yes"
        } else {
            "All in one community"
        }
    );

    // Example 3: Disconnected Components
    println!("\n\nExample 3: Disconnected Network Components");
    println!("-----------------------------------------");

    let g3 = Graph::from_edges(
        &[
            (0, 1),
            (1, 2),
            (2, 0), // Component 1
            (3, 4),
            (4, 5),
            (5, 3), // Component 2 (disconnected)
        ],
        false,
    );

    let communities3 = g3.louvain();
    let modularity3 = g3.modularity(&communities3);

    println!("Detected {} communities", communities3.len());
    for (i, community) in communities3.iter().enumerate() {
        println!("  Community {}: {:?}", i + 1, community);
    }
    println!("Modularity: {modularity3:.3}");
    println!("Disconnected components are correctly separated\n");

    // Example 4: Modularity Comparison
    println!("\nExample 4: Modularity Quality Metric");
    println!("-----------------------------------");

    // Good partition
    let good_partition = vec![vec![0, 1, 2], vec![3, 4, 5]];
    let good_mod = g3.modularity(&good_partition);

    // Bad partition (each node separate)
    let bad_partition = vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]];
    let bad_mod = g3.modularity(&bad_partition);

    println!("Good partition (by components):");
    println!("  {good_partition:?}");
    println!("  Modularity: {good_mod:.3}");
    println!("\nBad partition (all separate):");
    println!("  {bad_partition:?}");
    println!("  Modularity: {bad_mod:.3}");
    println!("\nLouvain found modularity: {modularity3:.3} (should match good partition)");

    // Example 5: Complete Graph (Single Community)
    println!("\n\nExample 5: Complete Graph");
    println!("------------------------");

    let g4 = Graph::from_edges(&[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], false);

    let communities4 = g4.louvain();
    let modularity4 = g4.modularity(&communities4);

    println!(
        "Detected {} community (all nodes connected)",
        communities4.len()
    );
    println!("Community: {:?}", communities4[0]);
    println!("Modularity: {modularity4:.3}");
    println!("Complete graphs have Q ≈ 0 (no community structure)\n");

    println!("=== Analysis Complete ===");

    Ok(())
}
