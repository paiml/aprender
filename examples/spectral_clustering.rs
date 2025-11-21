//! Spectral Clustering example
//!
//! Demonstrates graph-based clustering using eigendecomposition.
//!
//! Run with:
//! ```bash
//! cargo run --example spectral_clustering
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Spectral Clustering Example ===\n");

    // Example 1: Basic clustering with RBF affinity
    println!("--- Example 1: Basic RBF Affinity Clustering ---");

    let data = Matrix::from_vec(
        6,
        2,
        vec![
            // Cluster 1: around (1, 1)
            1.0, 1.0, 1.1, 1.0, 0.9, 1.1, // Cluster 2: around (5, 5)
            5.0, 5.0, 5.1, 5.0, 4.9, 5.1,
        ],
    )?;

    println!(
        "Dataset: {} samples, {} features",
        data.shape().0,
        data.shape().1
    );

    let mut sc = SpectralClustering::new(2);
    sc.fit(&data)?;

    let labels = sc.predict(&data);
    println!("Cluster labels: {labels:?}");

    // Verify clustering
    let cluster_0_points = labels.iter().filter(|&&l| l == 0).count();
    let cluster_1_points = labels.iter().filter(|&&l| l == 1).count();
    println!(
        "Cluster 0: {cluster_0_points} points, Cluster 1: {cluster_1_points} points\n"
    );

    // Example 2: K-NN Affinity for graph-based clustering
    println!("--- Example 2: K-NN Affinity ---");
    println!("K-NN creates a graph by connecting each point to its k nearest neighbors\n");

    let data2 = Matrix::from_vec(
        8,
        2,
        vec![
            // Chain-like cluster 1
            0.0, 0.0, 0.5, 0.1, 1.0, 0.0, 1.5, 0.1, // Chain-like cluster 2
            0.0, 2.0, 0.5, 2.1, 1.0, 2.0, 1.5, 2.1,
        ],
    )?;

    let mut sc_knn = SpectralClustering::new(2)
        .with_affinity(Affinity::KNN)
        .with_n_neighbors(2);
    sc_knn.fit(&data2)?;

    let labels_knn = sc_knn.predict(&data2);
    println!("K-NN Cluster labels: {labels_knn:?}");

    // Example 3: Gamma parameter effects (RBF affinity)
    println!("\n--- Example 3: Gamma Parameter Effects ---");
    println!("Gamma controls the scale of the RBF kernel:");
    println!("  - Small gamma: More global similarity");
    println!("  - Large gamma: More local similarity\n");

    // Small gamma
    let mut sc_small = SpectralClustering::new(2).with_gamma(0.1);
    sc_small.fit(&data)?;
    println!("Small gamma (0.1): {:?}", sc_small.predict(&data));

    // Default gamma
    let mut sc_default = SpectralClustering::new(2).with_gamma(1.0);
    sc_default.fit(&data)?;
    println!("Default gamma (1.0): {:?}", sc_default.predict(&data));

    // Large gamma
    let mut sc_large = SpectralClustering::new(2).with_gamma(5.0);
    sc_large.fit(&data)?;
    println!("Large gamma (5.0): {:?}", sc_large.predict(&data));

    // Example 4: Multiple clusters (3 clusters)
    println!("\n--- Example 4: Multiple Clusters (k=3) ---");

    let data3 = Matrix::from_vec(
        9,
        2,
        vec![
            // Cluster 1
            0.0, 0.0, 0.1, 0.1, -0.1, -0.1, // Cluster 2
            5.0, 5.0, 5.1, 5.1, 4.9, 4.9, // Cluster 3
            10.0, 10.0, 10.1, 10.1, 9.9, 9.9,
        ],
    )?;

    let mut sc3 = SpectralClustering::new(3);
    sc3.fit(&data3)?;

    let labels3 = sc3.predict(&data3);
    println!("Three-cluster labels: {labels3:?}");

    // Count points per cluster
    for cluster in 0..3 {
        let count = labels3.iter().filter(|&&l| l == cluster).count();
        println!("  Cluster {cluster}: {count} points");
    }

    // Example 5: Spectral Clustering vs K-Means on non-convex data
    println!("\n--- Example 5: Spectral Clustering vs K-Means ---");
    println!("Testing on non-convex (chain-like) clusters:\n");

    // Create elongated clusters
    let elongated = Matrix::from_vec(
        10,
        2,
        vec![
            // Horizontal chain
            0.0, 0.0, 0.5, 0.0, 1.0, 0.1, 1.5, 0.0, 2.0, 0.1, // Vertical chain
            5.0, 0.0, 5.0, 0.5, 5.1, 1.0, 5.0, 1.5, 5.1, 2.0,
        ],
    )?;

    // Spectral Clustering with K-NN
    let mut sc_knn_test = SpectralClustering::new(2)
        .with_affinity(Affinity::KNN)
        .with_n_neighbors(2);
    sc_knn_test.fit(&elongated)?;
    let sc_labels = sc_knn_test.predict(&elongated);

    // K-Means for comparison
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&elongated)?;
    let km_labels = kmeans.predict(&elongated);

    println!("Spectral (K-NN): {sc_labels:?}");
    println!("K-Means:         {km_labels:?}");
    println!("\nSpectral Clustering works better for:");
    println!("  ✓ Non-convex cluster shapes");
    println!("  ✓ Clusters with varying densities");
    println!("  ✓ Graph-structured data");
    println!("\nK-Means works better for:");
    println!("  ✓ Convex, spherical clusters");
    println!("  ✓ Large datasets (faster)");
    println!("  ✓ When cluster sizes are similar");

    // Example 6: Affinity matrix demonstration
    println!("\n--- Example 6: Understanding Affinity Matrices ---");
    println!("RBF Affinity: W[i,j] = exp(-gamma * ||x_i - x_j||^2)");
    println!("  - Nearby points have high similarity (close to 1)");
    println!("  - Distant points have low similarity (close to 0)");
    println!("\nK-NN Affinity: W[i,j] = 1 if j in k-nearest neighbors of i");
    println!("  - Creates sparse graph");
    println!("  - Better for non-convex shapes");

    println!("\n=== Example Complete ===");
    println!("\nKey takeaways:");
    println!("✓ Spectral Clustering uses graph Laplacian eigendecomposition");
    println!("✓ RBF affinity: Good for globular clusters");
    println!("✓ K-NN affinity: Good for non-convex clusters");
    println!("✓ Gamma controls locality in RBF kernel");
    println!("✓ Works well for graph-structured data");

    Ok(())
}
