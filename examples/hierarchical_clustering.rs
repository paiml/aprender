//! Hierarchical clustering example
//!
//! Demonstrates agglomerative clustering with different linkage methods.
//!
//! Run with:
//! ```bash
//! cargo run --example hierarchical_clustering
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hierarchical Clustering Example ===\n");

    // Create dataset with 3 natural clusters
    let data = Matrix::from_vec(
        9,
        2,
        vec![
            // Cluster 1 (bottom-left): 3 points
            1.0, 1.0, 1.1, 1.1, 1.2, 1.0, // Cluster 2 (top-right): 3 points
            5.0, 5.0, 5.1, 5.1, 5.0, 5.2, // Cluster 3 (middle): 3 points
            3.0, 3.0, 3.1, 3.0, 3.0, 3.1,
        ],
    )?;

    println!(
        "Dataset: {} samples, {} features",
        data.shape().0,
        data.shape().1
    );
    println!("\nData points:");
    for i in 0..data.shape().0 {
        println!(
            "  Point {}: ({:.1}, {:.1})",
            i,
            data.get(i, 0),
            data.get(i, 1)
        );
    }

    // Example 1: Standard hierarchical clustering with Average linkage
    println!("\n--- Example 1: Average Linkage Clustering ---");
    println!("Parameters: n_clusters=3, linkage=Average");

    let mut hc = AgglomerativeClustering::new(3, Linkage::Average);
    hc.fit(&data)?;

    let labels = hc.labels();
    println!("\nCluster assignments:");
    for (i, &label) in labels.iter().enumerate() {
        println!("  Point {}: Cluster {}", i, label);
    }

    // Count points per cluster
    let mut cluster_counts = [0; 3];
    for &label in labels {
        cluster_counts[label] += 1;
    }
    println!("\nCluster sizes:");
    for (i, count) in cluster_counts.iter().enumerate() {
        println!("  Cluster {}: {} points", i, count);
    }

    // Example 2: Dendrogram (merge history)
    println!("\n--- Example 2: Dendrogram (Merge History) ---");
    let dendrogram = hc.dendrogram();
    println!("Number of merges: {}", dendrogram.len());
    println!("\nMerge history:");
    for (step, merge) in dendrogram.iter().enumerate() {
        println!(
            "  Step {}: Merged clusters ({}, {}) at distance {:.3}, new size: {}",
            step + 1,
            merge.clusters.0,
            merge.clusters.1,
            merge.distance,
            merge.size
        );
    }

    // Example 3: Comparing linkage methods
    println!("\n--- Example 3: Comparing Linkage Methods ---");

    println!("\n3a. Single Linkage (minimum distance):");
    let mut hc_single = AgglomerativeClustering::new(3, Linkage::Single);
    hc_single.fit(&data)?;
    let labels_single = hc_single.labels();
    print_cluster_summary(labels_single, 3);

    println!("\n3b. Complete Linkage (maximum distance):");
    let mut hc_complete = AgglomerativeClustering::new(3, Linkage::Complete);
    hc_complete.fit(&data)?;
    let labels_complete = hc_complete.labels();
    print_cluster_summary(labels_complete, 3);

    println!("\n3c. Average Linkage (mean distance):");
    let mut hc_average = AgglomerativeClustering::new(3, Linkage::Average);
    hc_average.fit(&data)?;
    let labels_average = hc_average.labels();
    print_cluster_summary(labels_average, 3);

    println!("\n3d. Ward Linkage (minimize variance):");
    let mut hc_ward = AgglomerativeClustering::new(3, Linkage::Ward);
    hc_ward.fit(&data)?;
    let labels_ward = hc_ward.labels();
    print_cluster_summary(labels_ward, 3);

    println!("\nObservation: Different linkage methods may produce different clusterings");
    println!("  - Single: tends to create chain-like clusters");
    println!("  - Complete: tends to create compact clusters");
    println!("  - Average: balanced between single and complete");
    println!("  - Ward: minimizes within-cluster variance");

    // Example 4: Effect of n_clusters parameter
    println!("\n--- Example 4: Effect of n_clusters Parameter ---");

    println!("\nn_clusters=2 (two large groups):");
    let mut hc2 = AgglomerativeClustering::new(2, Linkage::Average);
    hc2.fit(&data)?;
    let labels2 = hc2.labels();
    print_cluster_summary(labels2, 2);

    println!("\nn_clusters=5 (more granular):");
    let mut hc5 = AgglomerativeClustering::new(5, Linkage::Average);
    hc5.fit(&data)?;
    let labels5 = hc5.labels();
    print_cluster_summary(labels5, 5);

    println!("\nn_clusters=9 (each point is its own cluster):");
    let mut hc9 = AgglomerativeClustering::new(9, Linkage::Average);
    hc9.fit(&data)?;
    let labels9 = hc9.labels();
    print_cluster_summary(labels9, 9);

    println!("\nObservation: n_clusters controls granularity of clustering");

    // Example 5: Practical use case - Building a taxonomy
    println!("\n--- Example 5: Practical Use Case - Document Taxonomy ---");
    println!("\nHierarchical clustering is ideal for:");
    println!("  1. Building taxonomies (biology, document organization)");
    println!("  2. Customer segmentation with hierarchy");
    println!("  3. Gene expression analysis");
    println!("  4. Phylogenetic tree construction");
    println!("\nKey advantages:");
    println!("  - No need to pre-specify exact number of clusters");
    println!("  - Can examine dendrogram to choose optimal cut point");
    println!("  - Provides hierarchy of relationships");
    println!("  - Deterministic results (same input → same output)");

    // Example 6: Reproducibility
    println!("\n--- Example 6: Reproducibility ---");
    let mut hc_test1 = AgglomerativeClustering::new(3, Linkage::Average);
    hc_test1.fit(&data)?;
    let labels_test1 = hc_test1.labels().clone();

    let mut hc_test2 = AgglomerativeClustering::new(3, Linkage::Average);
    hc_test2.fit(&data)?;
    let labels_test2 = hc_test2.labels().clone();

    let reproducible = labels_test1 == labels_test2;
    println!("Results are reproducible: {}", reproducible);
    println!("(Same data and parameters always produce same clustering)");

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Helper function to print cluster summary
fn print_cluster_summary(labels: &[usize], n_clusters: usize) {
    let mut cluster_counts = vec![0; n_clusters];
    for &label in labels {
        if label < n_clusters {
            cluster_counts[label] += 1;
        }
    }
    print!("  Cluster sizes: ");
    for (i, count) in cluster_counts.iter().enumerate() {
        if *count > 0 {
            print!("C{}={} ", i, count);
        }
    }
    println!();
}
