//! Gaussian Mixture Model (GMM) clustering example
//!
//! Demonstrates probabilistic clustering with soft assignments using the EM algorithm.
//!
//! Run with:
//! ```bash
//! cargo run --example gmm_clustering
//! ```

use aprender::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gaussian Mixture Model (GMM) Clustering Example ===\n");

    // Create dataset with 2 overlapping clusters
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            // Cluster 1 (around 2,2)
            1.8, 2.0, 2.0, 2.2, 2.1, 1.9, 2.2, 2.0, // Cluster 2 (around 5,5)
            5.0, 5.0, 5.1, 5.2, 5.2, 4.9, 4.8, 5.1,
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

    // Example 1: Basic GMM with Full covariance
    println!("\n--- Example 1: GMM with Full Covariance ---");
    let mut gmm = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm.fit(&data)?;

    let labels = gmm.predict(&data);
    println!("\nHard cluster assignments:");
    for (i, &label) in labels.iter().enumerate() {
        println!("  Point {}: Cluster {}", i, label);
    }

    // Example 2: Soft assignments (probabilities)
    println!("\n--- Example 2: Soft Assignments (Probabilities) ---");
    let proba = gmm.predict_proba(&data);
    println!("\nProbability of belonging to each cluster:");
    for i in 0..data.shape().0 {
        let p0 = proba.get(i, 0);
        let p1 = proba.get(i, 1);
        println!("  Point {}: Cluster 0: {:.3}, Cluster 1: {:.3}", i, p0, p1);
    }

    // Example 3: Model parameters
    println!("\n--- Example 3: Model Parameters ---");
    let means = gmm.means();
    println!("\nComponent means:");
    for k in 0..2 {
        println!(
            "  Component {}: ({:.2}, {:.2})",
            k,
            means.get(k, 0),
            means.get(k, 1)
        );
    }

    let weights = gmm.weights();
    println!("\nMixing weights (sum to 1):");
    for (k, &weight) in weights.as_slice().iter().enumerate() {
        println!("  Component {}: {:.3}", k, weight);
    }

    // Example 4: Log-likelihood (model fit quality)
    println!("\n--- Example 4: Model Quality ---");
    let log_likelihood = gmm.score(&data);
    println!("Average log-likelihood: {:.3}", log_likelihood);
    println!("(Higher is better)");

    // Example 5: Different covariance types
    println!("\n--- Example 5: Covariance Types Comparison ---");

    println!("\nFull covariance (most flexible):");
    let mut gmm_full = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm_full.fit(&data)?;
    println!("  Log-likelihood: {:.3}", gmm_full.score(&data));

    println!("\nDiagonal covariance (assumes independence):");
    let mut gmm_diag = GaussianMixture::new(2, CovarianceType::Diag).with_random_state(42);
    gmm_diag.fit(&data)?;
    println!("  Log-likelihood: {:.3}", gmm_diag.score(&data));

    println!("\nSpherical covariance (like K-Means):");
    let mut gmm_spher = GaussianMixture::new(2, CovarianceType::Spherical).with_random_state(42);
    gmm_spher.fit(&data)?;
    println!("  Log-likelihood: {:.3}", gmm_spher.score(&data));

    println!("\nTied covariance (shared across components):");
    let mut gmm_tied = GaussianMixture::new(2, CovarianceType::Tied).with_random_state(42);
    gmm_tied.fit(&data)?;
    println!("  Log-likelihood: {:.3}", gmm_tied.score(&data));

    // Example 6: GMM vs K-Means
    println!("\n--- Example 6: GMM vs K-Means ---");

    println!("\nK-Means (hard assignments only):");
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(&data)?;
    let kmeans_labels = kmeans.predict(&data);
    println!("  Hard assignments: {:?}", kmeans_labels);

    println!("\nGMM (soft + hard assignments):");
    println!("  Hard assignments: {:?}", labels);
    println!("  Soft assignments available via predict_proba()");
    println!("  Provides uncertainty estimates!");

    println!("\nKey advantages of GMM:");
    println!("  - Soft clustering (probability distributions)");
    println!("  - Handles elliptical clusters");
    println!("  - Provides uncertainty quantification");
    println!("  - Probabilistic framework");

    // Example 7: Reproducibility
    println!("\n--- Example 7: Reproducibility ---");
    let mut gmm1 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm1.fit(&data)?;
    let labels1 = gmm1.predict(&data);

    let mut gmm2 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm2.fit(&data)?;
    let labels2 = gmm2.predict(&data);

    println!("Results are reproducible: {}", labels1 == labels2);
    println!("(Same random seed produces same clustering)");

    println!("\n=== Example Complete ===");
    Ok(())
}
