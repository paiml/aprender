//! DBSCAN clustering example
//!
//! Demonstrates density-based clustering with outlier detection.
//!
//! Run with:
//! ```bash
//! cargo run --example dbscan_clustering
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DBSCAN Clustering Example ===\n");

    // Create dataset with 2 clusters and noise points
    let data = Matrix::from_vec(
        15,
        2,
        vec![
            // Cluster 1 (bottom-left): dense group (5 points)
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.2, 1.0,
            // Cluster 2 (top-right): dense group (5 points)
            5.0, 5.0, 5.1, 5.2, 5.2, 5.1, 5.0, 5.1, 5.1, 5.0,
            // Noise points (outliers) (5 points)
            10.0, 10.0, // far away
            0.0, 5.0, // between clusters
            5.0, 0.0, // between clusters
            2.5, 2.5, // between clusters
            7.5, 2.5, // isolated
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

    // Example 1: Standard DBSCAN
    println!("\n--- Example 1: Standard DBSCAN ---");
    println!("Parameters: eps=0.5, min_samples=3");

    let mut dbscan = DBSCAN::new(0.5, 3);
    dbscan.fit(&data)?;

    let labels = dbscan.labels();
    println!("\nCluster assignments:");
    for (i, &label) in labels.iter().enumerate() {
        let status = if label == -1 { "Noise" } else { "Cluster" };
        println!(
            "  Point {}: {} {}",
            i,
            status,
            if label == -1 {
                "".to_string()
            } else {
                label.to_string()
            }
        );
    }

    // Count clusters and noise
    let n_clusters = labels
        .iter()
        .filter(|&&l| l != -1)
        .copied()
        .max()
        .map_or(0, |m| (m + 1) as usize);
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    println!("\nSummary:");
    println!("  Clusters found: {}", n_clusters);
    println!("  Noise points: {}", n_noise);

    // Example 2: Effect of eps parameter
    println!("\n--- Example 2: Effect of eps (neighborhood size) ---");

    // Small eps: tight neighborhoods
    let mut dbscan_small = DBSCAN::new(0.2, 3);
    dbscan_small.fit(&data)?;
    let labels_small = dbscan_small.labels();
    let clusters_small = labels_small
        .iter()
        .filter(|&&l| l != -1)
        .copied()
        .max()
        .map_or(0, |m| (m + 1) as usize);
    let noise_small = labels_small.iter().filter(|&&l| l == -1).count();

    println!("eps=0.2, min_samples=3:");
    println!("  Clusters: {}, Noise: {}", clusters_small, noise_small);

    // Medium eps: balanced
    let mut dbscan_medium = DBSCAN::new(0.5, 3);
    dbscan_medium.fit(&data)?;
    let labels_medium = dbscan_medium.labels();
    let clusters_medium = labels_medium
        .iter()
        .filter(|&&l| l != -1)
        .copied()
        .max()
        .map_or(0, |m| (m + 1) as usize);
    let noise_medium = labels_medium.iter().filter(|&&l| l == -1).count();

    println!("eps=0.5, min_samples=3:");
    println!("  Clusters: {}, Noise: {}", clusters_medium, noise_medium);

    // Large eps: loose neighborhoods
    let mut dbscan_large = DBSCAN::new(2.0, 3);
    dbscan_large.fit(&data)?;
    let labels_large = dbscan_large.labels();
    let clusters_large = labels_large
        .iter()
        .filter(|&&l| l != -1)
        .copied()
        .max()
        .map_or(0, |m| (m + 1) as usize);
    let noise_large = labels_large.iter().filter(|&&l| l == -1).count();

    println!("eps=2.0, min_samples=3:");
    println!("  Clusters: {}, Noise: {}", clusters_large, noise_large);

    println!("\nObservation: Smaller eps → more noise, larger eps → fewer clusters");

    // Example 3: Effect of min_samples parameter
    println!("\n--- Example 3: Effect of min_samples (density threshold) ---");

    // Low min_samples: more points are core
    let mut dbscan_low = DBSCAN::new(0.5, 2);
    dbscan_low.fit(&data)?;
    let labels_low = dbscan_low.labels();
    let noise_low = labels_low.iter().filter(|&&l| l == -1).count();

    println!("eps=0.5, min_samples=2:");
    println!("  Noise: {}", noise_low);

    // High min_samples: stricter density requirement
    let mut dbscan_high = DBSCAN::new(0.5, 5);
    dbscan_high.fit(&data)?;
    let labels_high = dbscan_high.labels();
    let noise_high = labels_high.iter().filter(|&&l| l == -1).count();

    println!("eps=0.5, min_samples=5:");
    println!("  Noise: {}", noise_high);

    println!("\nObservation: Higher min_samples → stricter density → more noise");

    // Example 4: Comparison with K-Means
    println!("\n--- Example 4: DBSCAN vs K-Means ---");

    // K-Means (requires specifying k)
    let mut kmeans = KMeans::new(3).with_random_state(42);
    kmeans.fit(&data)?;
    let _kmeans_labels = kmeans.predict(&data);

    println!("K-Means (k=3):");
    println!("  Assigns all {} points to 3 clusters", data.shape().0);
    println!("  Cannot detect outliers");

    println!("\nDBSCAN (eps=0.5, min_samples=3):");
    println!("  Finds {} clusters automatically", n_clusters);
    println!("  Identifies {} outliers as noise", n_noise);

    println!("\nKey differences:");
    println!("  - K-Means: must specify k, assigns all points");
    println!("  - DBSCAN: discovers k, identifies outliers");

    // Example 5: Practical use case - Anomaly detection
    println!("\n--- Example 5: Anomaly Detection ---");

    println!("\nUse DBSCAN for outlier detection:");
    println!("  1. Fit DBSCAN with appropriate eps and min_samples");
    println!("  2. Points labeled as -1 are anomalies");
    println!("  3. Useful for fraud detection, sensor anomalies, etc.");

    println!("\nDetected anomalies in this dataset:");
    for (i, &label) in labels.iter().enumerate() {
        if label == -1 {
            println!(
                "  Point {} at ({:.1}, {:.1}) - ANOMALY",
                i,
                data.get(i, 0),
                data.get(i, 1)
            );
        }
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
