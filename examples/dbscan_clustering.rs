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

    let data = create_sample_data()?;
    print_dataset_info(&data);

    let labels = example_standard_dbscan(&data)?;
    let (n_clusters, n_noise) = count_clusters_and_noise(&labels);
    example_eps_effects(&data)?;
    example_min_samples_effects(&data)?;
    example_vs_kmeans(&data, n_clusters, n_noise)?;
    example_anomaly_detection(&data, &labels);

    println!("\n=== Example Complete ===");
    Ok(())
}

fn create_sample_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
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

fn example_standard_dbscan(data: &Matrix<f32>) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    println!("\n--- Example 1: Standard DBSCAN ---");
    println!("Parameters: eps=0.5, min_samples=3");

    let mut dbscan = DBSCAN::new(0.5, 3);
    dbscan.fit(data)?;

    let labels = dbscan.labels().clone();
    println!("\nCluster assignments:");
    for (i, &label) in labels.iter().enumerate() {
        let status = if label == -1 { "Noise" } else { "Cluster" };
        let label_str = if label == -1 {
            String::new()
        } else {
            label.to_string()
        };
        println!("  Point {}: {} {}", i, status, label_str);
    }
    Ok(labels)
}

fn count_clusters_and_noise(labels: &[i32]) -> (usize, usize) {
    let n_clusters = labels
        .iter()
        .filter(|&&l| l != -1)
        .copied()
        .max()
        .map_or(0, |m| (m + 1) as usize);
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    println!("\nSummary:");
    println!("  Clusters found: {n_clusters}");
    println!("  Noise points: {n_noise}");
    (n_clusters, n_noise)
}

fn example_eps_effects(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 2: Effect of eps (neighborhood size) ---");

    for (eps, desc) in [(0.2, "small"), (0.5, "medium"), (2.0, "large")] {
        let mut dbscan = DBSCAN::new(eps, 3);
        dbscan.fit(data)?;
        let labels = dbscan.labels();
        let clusters = labels
            .iter()
            .filter(|&&l| l != -1)
            .copied()
            .max()
            .map_or(0, |m| (m + 1) as usize);
        let noise = labels.iter().filter(|&&l| l == -1).count();
        println!("eps={eps} ({desc}), min_samples=3: Clusters: {clusters}, Noise: {noise}");
    }
    println!("\nObservation: Smaller eps → more noise, larger eps → fewer clusters");
    Ok(())
}

fn example_min_samples_effects(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 3: Effect of min_samples (density threshold) ---");

    for (min_samples, desc) in [(2, "low"), (5, "high")] {
        let mut dbscan = DBSCAN::new(0.5, min_samples);
        dbscan.fit(data)?;
        let noise = dbscan.labels().iter().filter(|&&l| l == -1).count();
        println!("eps=0.5, min_samples={min_samples} ({desc}): Noise: {noise}");
    }
    println!("\nObservation: Higher min_samples → stricter density → more noise");
    Ok(())
}

fn example_vs_kmeans(
    data: &Matrix<f32>,
    n_clusters: usize,
    n_noise: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 4: DBSCAN vs K-Means ---");

    let mut kmeans = KMeans::new(3).with_random_state(42);
    kmeans.fit(data)?;
    let _kmeans_labels = kmeans.predict(data);

    println!("K-Means (k=3):");
    println!("  Assigns all {} points to 3 clusters", data.shape().0);
    println!("  Cannot detect outliers");

    println!("\nDBSCAN (eps=0.5, min_samples=3):");
    println!("  Finds {n_clusters} clusters automatically");
    println!("  Identifies {n_noise} outliers as noise");

    println!("\nKey differences:");
    println!("  - K-Means: must specify k, assigns all points");
    println!("  - DBSCAN: discovers k, identifies outliers");
    Ok(())
}

fn example_anomaly_detection(data: &Matrix<f32>, labels: &[i32]) {
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
}
