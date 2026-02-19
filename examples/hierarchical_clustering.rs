#![allow(clippy::disallowed_methods)]
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

    let data = create_sample_data()?;
    print_dataset_info(&data);

    example_average_linkage(&data)?;
    example_dendrogram(&data)?;
    example_compare_linkages(&data)?;
    example_n_clusters_effect(&data)?;
    example_practical_use_cases();
    example_reproducibility(&data)?;

    println!("\n=== Example Complete ===");
    Ok(())
}

fn create_sample_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        9,
        2,
        vec![
            // Cluster 1 (bottom-left): 3 points
            1.0, 1.0, 1.1, 1.1, 1.2, 1.0, // Cluster 2 (top-right): 3 points
            5.0, 5.0, 5.1, 5.1, 5.0, 5.2, // Cluster 3 (middle): 3 points
            3.0, 3.0, 3.1, 3.0, 3.0, 3.1,
        ],
    )
    .map_err(Into::into)
}

fn print_dataset_info(data: &Matrix<f32>) {
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
}

fn example_average_linkage(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 1: Average Linkage Clustering ---");
    println!("Parameters: n_clusters=3, linkage=Average");

    let mut hc = AgglomerativeClustering::new(3, Linkage::Average);
    hc.fit(data)?;

    let labels = hc.labels();
    println!("\nCluster assignments:");
    for (i, &label) in labels.iter().enumerate() {
        println!("  Point {i}: Cluster {label}");
    }

    let mut cluster_counts = [0; 3];
    for &label in labels {
        cluster_counts[label] += 1;
    }
    println!("\nCluster sizes:");
    for (i, count) in cluster_counts.iter().enumerate() {
        println!("  Cluster {i}: {count} points");
    }
    Ok(())
}

fn example_dendrogram(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 2: Dendrogram (Merge History) ---");

    let mut hc = AgglomerativeClustering::new(3, Linkage::Average);
    hc.fit(data)?;

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
    Ok(())
}

fn example_compare_linkages(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: Comparing Linkage Methods ---");

    for (name, linkage) in [
        ("3a. Single Linkage (minimum distance)", Linkage::Single),
        ("3b. Complete Linkage (maximum distance)", Linkage::Complete),
        ("3c. Average Linkage (mean distance)", Linkage::Average),
        ("3d. Ward Linkage (minimize variance)", Linkage::Ward),
    ] {
        println!("\n{name}:");
        let mut hc = AgglomerativeClustering::new(3, linkage);
        hc.fit(data)?;
        print_cluster_summary(hc.labels(), 3);
    }

    println!("\nObservation: Different linkage methods may produce different clusterings");
    println!("  - Single: tends to create chain-like clusters");
    println!("  - Complete: tends to create compact clusters");
    println!("  - Average: balanced between single and complete");
    println!("  - Ward: minimizes within-cluster variance");
    Ok(())
}

fn example_n_clusters_effect(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 4: Effect of n_clusters Parameter ---");

    for (n, desc) in [
        (2, "two large groups"),
        (5, "more granular"),
        (9, "each point is its own cluster"),
    ] {
        println!("\nn_clusters={n} ({desc}):");
        let mut hc = AgglomerativeClustering::new(n, Linkage::Average);
        hc.fit(data)?;
        print_cluster_summary(hc.labels(), n);
    }

    println!("\nObservation: n_clusters controls granularity of clustering");
    Ok(())
}

fn example_practical_use_cases() {
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
}

fn example_reproducibility(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 6: Reproducibility ---");

    let mut hc_test1 = AgglomerativeClustering::new(3, Linkage::Average);
    hc_test1.fit(data)?;
    let labels_test1 = hc_test1.labels().clone();

    let mut hc_test2 = AgglomerativeClustering::new(3, Linkage::Average);
    hc_test2.fit(data)?;
    let labels_test2 = hc_test2.labels().clone();

    let reproducible = labels_test1 == labels_test2;
    println!("Results are reproducible: {reproducible}");
    println!("(Same data and parameters always produce same clustering)");
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
            print!("C{i}={count} ");
        }
    }
    println!();
}
