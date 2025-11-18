//! Iris Clustering example - K-Means
//!
//! Demonstrates K-Means clustering using simulated iris data.

use aprender::prelude::*;

fn main() {
    println!("Iris Clustering - K-Means Example");
    println!("==================================\n");

    // Simulated iris-like data
    // Features: [sepal_length, sepal_width, petal_length, petal_width]
    // Three distinct species clusters
    let x = Matrix::from_vec(
        15,
        4,
        vec![
            // Setosa-like (cluster 0)
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, // Versicolor-like (cluster 1)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, // Virginica-like (cluster 2)
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2,
        ],
    )
    .unwrap();

    // True labels for comparison
    let true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    // Fit K-Means with 3 clusters
    println!("Fitting K-Means with 3 clusters...");
    let mut kmeans = KMeans::new(3).with_max_iter(100).with_random_state(42);

    kmeans.fit(&x).expect("Failed to fit K-Means");

    // Get cluster assignments
    let predicted_labels = kmeans.predict(&x);

    // Print cluster assignments
    println!("\nCluster Assignments:");
    println!("{:>6} {:>10} {:>12}", "Sample", "True", "Predicted");
    println!("{}", "-".repeat(30));

    for i in 0..15 {
        println!(
            "{:>6} {:>10} {:>12}",
            i, true_labels[i], predicted_labels[i]
        );
    }

    // Print cluster centroids
    let centroids = kmeans.centroids();
    println!("\nCluster Centroids:");
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>8}",
        "Cluster", "Sepal L", "Sepal W", "Petal L", "Petal W"
    );
    println!("{}", "-".repeat(44));

    for k in 0..3 {
        let centroid = centroids.row(k);
        println!(
            "{:>8} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            k,
            centroid.as_slice()[0],
            centroid.as_slice()[1],
            centroid.as_slice()[2],
            centroid.as_slice()[3]
        );
    }

    // Print metrics
    let inertia_val = kmeans.inertia();
    let silhouette = silhouette_score(&x, &predicted_labels);

    println!("\nClustering Metrics:");
    println!("  Inertia:         {:.4}", inertia_val);
    println!("  Silhouette:      {:.4}", silhouette);
    println!("  Iterations:      {}", kmeans.n_iter());

    // Evaluate cluster quality interpretation
    println!("\nInterpretation:");
    if silhouette > 0.5 {
        println!("  ✓ Good cluster separation (silhouette > 0.5)");
    } else if silhouette > 0.25 {
        println!("  ~ Moderate cluster separation");
    } else {
        println!("  ✗ Poor cluster separation");
    }
}
