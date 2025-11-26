//! AutoML Clustering Example - Finding Optimal K with TPE
//!
//! Demonstrates using TPE (Tree-structured Parzen Estimator) to automatically
//! find the optimal number of clusters for K-Means using silhouette score.
//!
//! # Key Concepts
//!
//! - Type-safe parameter enums (Poka-Yoke design)
//! - TPE-based Bayesian optimization
//! - AutoTuner with early stopping
//! - Silhouette score as objective function
//!
//! # Running
//!
//! ```bash
//! cargo run --example automl_clustering
//! ```

use aprender::automl::params::ParamKey;
use aprender::automl::{AutoTuner, SearchSpace, TPE};
use aprender::prelude::*;

/// Custom parameter enum for K-Means tuning (Poka-Yoke: compile-time safety)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KMeansParam {
    NClusters,
}

impl ParamKey for KMeansParam {
    fn name(&self) -> &'static str {
        match self {
            KMeansParam::NClusters => "n_clusters",
        }
    }
}

impl std::fmt::Display for KMeansParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    println!("AutoML Clustering - TPE Optimization");
    println!("=====================================\n");

    // Generate synthetic data with 4 true clusters
    let (data, true_k) = generate_clustered_data();
    println!(
        "Generated {} samples with {} true clusters\n",
        data.n_rows(),
        true_k
    );

    // Define search space: K from 2 to 10
    let space: SearchSpace<KMeansParam> = SearchSpace::new().add(KMeansParam::NClusters, 2..11); // Search K in [2, 10]

    println!("Search Space: K ∈ [2, 10]");
    println!("Objective: Maximize silhouette score\n");

    // Track results for final report
    let mut results: Vec<(usize, f64)> = Vec::new();

    println!("═══════════════════════════════════════════");
    println!(" Trial │   K   │ Silhouette │   Status   ");
    println!("═══════╪═══════╪════════════╪════════════");

    // Use TPE optimizer
    let tpe = TPE::new(15)
        .with_seed(42)
        .with_startup_trials(3)  // Random exploration first
        .with_gamma(0.25); // Top 25% as "good"

    let tune_result = AutoTuner::new(tpe)
        .early_stopping(5)  // Stop if no improvement for 5 trials
        .maximize(&space, |trial| {
            let k = trial.get_usize(&KMeansParam::NClusters).unwrap_or(3);

            // Run K-Means multiple times and average (reduce variance)
            let mut scores = Vec::new();
            for seed in [42, 123, 456] {
                let mut kmeans = KMeans::new(k)
                    .with_max_iter(100)
                    .with_random_state(seed);

                if kmeans.fit(&data).is_ok() {
                    let labels = kmeans.predict(&data);
                    let score = silhouette_score(&data, &labels);
                    if score.is_finite() {
                        scores.push(score);
                    }
                }
            }

            let avg_score = if scores.is_empty() {
                -1.0  // Penalty for failed clustering
            } else {
                f64::from(scores.iter().sum::<f32>() / scores.len() as f32)
            };

            // Determine status
            let status = if k == true_k {
                "← TRUE K"
            } else if avg_score > 0.5 {
                "good"
            } else if avg_score > 0.25 {
                "moderate"
            } else {
                "poor"
            };

            results.push((k, avg_score));

            println!(
                "  {:>3}  │  {:>3}  │   {:>6.3}   │ {}",
                results.len(),
                k,
                avg_score,
                status
            );

            avg_score
        });

    println!("═══════════════════════════════════════════\n");

    // Summary by K value
    println!("📊 Summary by K:");
    for k in 2..=10 {
        let k_results: Vec<_> = results.iter().filter(|(kk, _)| *kk == k).collect();
        if !k_results.is_empty() {
            let avg: f64 = k_results.iter().map(|(_, s)| s).sum::<f64>() / k_results.len() as f64;
            let best_marker =
                if tune_result.best_trial.get_usize(&KMeansParam::NClusters) == Some(k) {
                    " ★ BEST"
                } else if k == true_k {
                    " (true)"
                } else {
                    ""
                };
            println!(
                "   K={:>2}: silhouette={:.3} ({} trials){}",
                k,
                avg,
                k_results.len(),
                best_marker
            );
        }
    }

    // Final results
    let best_k = tune_result
        .best_trial
        .get_usize(&KMeansParam::NClusters)
        .unwrap_or(0);

    println!("\n🏆 TPE Optimization Results:");
    println!("   Best K:          {best_k}");
    println!("   Best silhouette: {:.4}", tune_result.best_score);
    println!("   True K:          {true_k}");
    println!("   Trials run:      {}", tune_result.n_trials);
    println!(
        "   Time elapsed:    {:.2}s",
        tune_result.elapsed.as_secs_f64()
    );

    // Verify with final model
    println!("\n🔍 Final Model Verification:");
    let mut final_kmeans = KMeans::new(best_k).with_max_iter(100).with_random_state(42);
    final_kmeans.fit(&data).expect("Final fit should succeed");
    let final_labels = final_kmeans.predict(&data);
    let final_silhouette = silhouette_score(&data, &final_labels);
    let final_inertia = final_kmeans.inertia();

    println!("   Silhouette score: {final_silhouette:.4}");
    println!("   Inertia:          {final_inertia:.2}");
    println!("   Iterations:       {}", final_kmeans.n_iter());

    // Interpretation
    println!("\n📈 Interpretation:");
    if best_k == true_k {
        println!("   ✅ TPE found the true number of clusters!");
    } else if (best_k as i32 - true_k as i32).abs() <= 1 {
        println!("   ✓ TPE found a close approximation (within ±1)");
    } else {
        println!("   ⚠ TPE found a different K (data may have ambiguous structure)");
    }

    if final_silhouette > 0.5 {
        println!("   ✅ Excellent cluster separation (silhouette > 0.5)");
    } else if final_silhouette > 0.25 {
        println!("   ✓ Reasonable cluster structure");
    } else {
        println!("   ⚠ Weak cluster structure - consider different features");
    }
}

/// Generate synthetic clustered data with known structure
fn generate_clustered_data() -> (Matrix<f32>, usize) {
    // 4 clusters with distinct centers
    let true_k = 4;

    // Cluster centers
    let centers = [
        (2.0, 2.0), // Cluster 0: bottom-left
        (8.0, 2.0), // Cluster 1: bottom-right
        (2.0, 8.0), // Cluster 2: top-left
        (8.0, 8.0), // Cluster 3: top-right
    ];

    // Simple LCG for reproducible "random" noise
    let mut seed: u64 = 12345;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed >> 16) & 0x7FFF) as f32 / 32768.0 - 0.5 // [-0.5, 0.5]
    };

    // Generate 25 points per cluster (100 total)
    let points_per_cluster = 25;
    let mut data = Vec::with_capacity(points_per_cluster * true_k * 2);

    for (cx, cy) in &centers {
        for _ in 0..points_per_cluster {
            data.push(cx + rand() * 2.0); // x with noise
            data.push(cy + rand() * 2.0); // y with noise
        }
    }

    let matrix = Matrix::from_vec(points_per_cluster * true_k, 2, data)
        .expect("Data generation should succeed");

    (matrix, true_k)
}
