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

    let data = create_basic_data()?;
    example_basic_rbf(&data)?;
    example_knn_affinity()?;
    example_gamma_effects(&data)?;
    example_multiple_clusters()?;
    example_vs_kmeans()?;
    print_affinity_info();

    Ok(())
}

fn create_basic_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        6,
        2,
        vec![
            // Cluster 1: around (1, 1)
            1.0, 1.0, 1.1, 1.0, 0.9, 1.1, // Cluster 2: around (5, 5)
            5.0, 5.0, 5.1, 5.0, 4.9, 5.1,
        ],
    )
    .map_err(Into::into)
}

fn example_basic_rbf(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Example 1: Basic RBF Affinity Clustering ---");
    println!(
        "Dataset: {} samples, {} features",
        data.shape().0,
        data.shape().1
    );

    let mut sc = SpectralClustering::new(2);
    sc.fit(data)?;

    let labels = sc.predict(data);
    println!("Cluster labels: {labels:?}");

    let cluster_0_points = labels.iter().filter(|&&l| l == 0).count();
    let cluster_1_points = labels.iter().filter(|&&l| l == 1).count();
    println!("Cluster 0: {cluster_0_points} points, Cluster 1: {cluster_1_points} points\n");
    Ok(())
}

fn example_knn_affinity() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

fn example_gamma_effects(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: Gamma Parameter Effects ---");
    println!("Gamma controls the scale of the RBF kernel:");
    println!("  - Small gamma: More global similarity");
    println!("  - Large gamma: More local similarity\n");

    for (gamma, desc) in [(0.1, "Small"), (1.0, "Default"), (5.0, "Large")] {
        let mut sc = SpectralClustering::new(2).with_gamma(gamma);
        sc.fit(data)?;
        println!("{desc} gamma ({gamma}): {:?}", sc.predict(data));
    }
    Ok(())
}

fn example_multiple_clusters() -> Result<(), Box<dyn std::error::Error>> {
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

    for cluster in 0..3 {
        let count = labels3.iter().filter(|&&l| l == cluster).count();
        println!("  Cluster {cluster}: {count} points");
    }
    Ok(())
}

fn example_vs_kmeans() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 5: Spectral Clustering vs K-Means ---");
    println!("Testing on non-convex (chain-like) clusters:\n");

    let elongated = Matrix::from_vec(
        10,
        2,
        vec![
            // Horizontal chain
            0.0, 0.0, 0.5, 0.0, 1.0, 0.1, 1.5, 0.0, 2.0, 0.1, // Vertical chain
            5.0, 0.0, 5.0, 0.5, 5.1, 1.0, 5.0, 1.5, 5.1, 2.0,
        ],
    )?;

    let mut sc_knn_test = SpectralClustering::new(2)
        .with_affinity(Affinity::KNN)
        .with_n_neighbors(2);
    sc_knn_test.fit(&elongated)?;
    let sc_labels = sc_knn_test.predict(&elongated);

    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&elongated)?;
    let km_labels = kmeans.predict(&elongated);

    println!("Spectral (K-NN): {sc_labels:?}");
    println!("K-Means:         {km_labels:?}");
    print_comparison_notes();
    Ok(())
}

fn print_comparison_notes() {
    println!("\nSpectral Clustering works better for:");
    println!("  ✓ Non-convex cluster shapes");
    println!("  ✓ Clusters with varying densities");
    println!("  ✓ Graph-structured data");
    println!("\nK-Means works better for:");
    println!("  ✓ Convex, spherical clusters");
    println!("  ✓ Large datasets (faster)");
    println!("  ✓ When cluster sizes are similar");
}

fn print_affinity_info() {
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
}
