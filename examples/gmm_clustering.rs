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

    let data = create_sample_data()?;
    print_dataset_info(&data);

    let (gmm, labels) = example_basic_gmm(&data)?;
    example_soft_assignments(&gmm, &data);
    example_model_parameters(&gmm);
    example_model_quality(&gmm, &data);
    example_covariance_types(&data)?;
    example_gmm_vs_kmeans(&data, &labels)?;
    example_reproducibility(&data)?;

    println!("\n=== Example Complete ===");
    Ok(())
}

fn create_sample_data() -> Result<Matrix<f32>, Box<dyn std::error::Error>> {
    Matrix::from_vec(
        8,
        2,
        vec![
            // Cluster 1 (around 2,2)
            1.8, 2.0, 2.0, 2.2, 2.1, 1.9, 2.2, 2.0, // Cluster 2 (around 5,5)
            5.0, 5.0, 5.1, 5.2, 5.2, 4.9, 4.8, 5.1,
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

fn example_basic_gmm(
    data: &Matrix<f32>,
) -> Result<(GaussianMixture, Vec<usize>), Box<dyn std::error::Error>> {
    println!("\n--- Example 1: GMM with Full Covariance ---");
    let mut gmm = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm.fit(data)?;

    let labels = gmm.predict(data);
    println!("\nHard cluster assignments:");
    for (i, &label) in labels.iter().enumerate() {
        println!("  Point {i}: Cluster {label}");
    }
    Ok((gmm, labels))
}

fn example_soft_assignments(gmm: &GaussianMixture, data: &Matrix<f32>) {
    println!("\n--- Example 2: Soft Assignments (Probabilities) ---");
    let proba = gmm.predict_proba(data);
    println!("\nProbability of belonging to each cluster:");
    for i in 0..data.shape().0 {
        let p0 = proba.get(i, 0);
        let p1 = proba.get(i, 1);
        println!("  Point {i}: Cluster 0: {p0:.3}, Cluster 1: {p1:.3}");
    }
}

fn example_model_parameters(gmm: &GaussianMixture) {
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
        println!("  Component {k}: {weight:.3}");
    }
}

fn example_model_quality(gmm: &GaussianMixture, data: &Matrix<f32>) {
    println!("\n--- Example 4: Model Quality ---");
    let log_likelihood = gmm.score(data);
    println!("Average log-likelihood: {log_likelihood:.3}");
    println!("(Higher is better)");
}

fn example_covariance_types(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 5: Covariance Types Comparison ---");

    for (name, cov_type) in [
        ("Full covariance (most flexible)", CovarianceType::Full),
        (
            "Diagonal covariance (assumes independence)",
            CovarianceType::Diag,
        ),
        (
            "Spherical covariance (like K-Means)",
            CovarianceType::Spherical,
        ),
        (
            "Tied covariance (shared across components)",
            CovarianceType::Tied,
        ),
    ] {
        println!("\n{name}:");
        let mut gmm = GaussianMixture::new(2, cov_type).with_random_state(42);
        gmm.fit(data)?;
        println!("  Log-likelihood: {:.3}", gmm.score(data));
    }
    Ok(())
}

fn example_gmm_vs_kmeans(
    data: &Matrix<f32>,
    labels: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 6: GMM vs K-Means ---");

    println!("\nK-Means (hard assignments only):");
    let mut kmeans = KMeans::new(2).with_random_state(42);
    kmeans.fit(data)?;
    let kmeans_labels = kmeans.predict(data);
    println!("  Hard assignments: {kmeans_labels:?}");

    println!("\nGMM (soft + hard assignments):");
    println!("  Hard assignments: {labels:?}");
    println!("  Soft assignments available via predict_proba()");
    println!("  Provides uncertainty estimates!");

    println!("\nKey advantages of GMM:");
    println!("  - Soft clustering (probability distributions)");
    println!("  - Handles elliptical clusters");
    println!("  - Provides uncertainty quantification");
    println!("  - Probabilistic framework");
    Ok(())
}

fn example_reproducibility(data: &Matrix<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Example 7: Reproducibility ---");
    let mut gmm1 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm1.fit(data)?;
    let labels1 = gmm1.predict(data);

    let mut gmm2 = GaussianMixture::new(2, CovarianceType::Full).with_random_state(42);
    gmm2.fit(data)?;
    let labels2 = gmm2.predict(data);

    println!("Results are reproducible: {}", labels1 == labels2);
    println!("(Same random seed produces same clustering)");
    Ok(())
}
