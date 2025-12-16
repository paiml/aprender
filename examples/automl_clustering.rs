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

fn main() {
    println!("AutoML Clustering - TPE Optimization");
    println!("=====================================\n");

    let (data, true_k) = generate_clustered_data();
    println!(
        "Generated {} samples with {} true clusters\n",
        data.n_rows(),
        true_k
    );

    let space: SearchSpace<KMeansParam> = SearchSpace::new().add(KMeansParam::NClusters, 2..11);
    println!("Search Space: K ∈ [2, 10]");
    println!("Objective: Maximize silhouette score\n");

    let (tune_result, results) = run_tpe_optimization(&data, &space, true_k);

    print_summary_by_k(&results, &tune_result, true_k);
    print_optimization_results(&tune_result, true_k);
    verify_final_model(&data, &tune_result, true_k);
}

fn run_tpe_optimization(
    data: &Matrix<f32>,
    space: &SearchSpace<KMeansParam>,
    true_k: usize,
) -> (aprender::automl::TuneResult<KMeansParam>, Vec<(usize, f64)>) {
    let mut results: Vec<(usize, f64)> = Vec::new();

    println!("═══════════════════════════════════════════");
    println!(" Trial │   K   │ Silhouette │   Status   ");
    println!("═══════╪═══════╪════════════╪════════════");

    let tpe = TPE::new(15)
        .with_seed(42)
        .with_startup_trials(3)
        .with_gamma(0.25);

    let tune_result = AutoTuner::new(tpe)
        .early_stopping(5)
        .maximize(space, |trial| {
            let k = trial.get_usize(&KMeansParam::NClusters).unwrap_or(3);
            let avg_score = evaluate_kmeans(data, k);
            let status = determine_status(k, avg_score, true_k);

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
    (tune_result, results)
}

fn evaluate_kmeans(data: &Matrix<f32>, k: usize) -> f64 {
    let scores: Vec<f32> = [42, 123, 456]
        .iter()
        .filter_map(|&seed| {
            let mut kmeans = KMeans::new(k).with_max_iter(100).with_random_state(seed);
            kmeans.fit(data).ok()?;
            let labels = kmeans.predict(data);
            let score = silhouette_score(data, &labels);
            score.is_finite().then_some(score)
        })
        .collect();

    if scores.is_empty() {
        -1.0
    } else {
        f64::from(scores.iter().sum::<f32>() / scores.len() as f32)
    }
}

fn determine_status(k: usize, score: f64, true_k: usize) -> &'static str {
    if k == true_k {
        "← TRUE K"
    } else if score > 0.5 {
        "good"
    } else if score > 0.25 {
        "moderate"
    } else {
        "poor"
    }
}

fn print_summary_by_k(
    results: &[(usize, f64)],
    tune_result: &aprender::automl::TuneResult<KMeansParam>,
    true_k: usize,
) {
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
}

fn print_optimization_results(
    tune_result: &aprender::automl::TuneResult<KMeansParam>,
    true_k: usize,
) {
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
}

fn verify_final_model(
    data: &Matrix<f32>,
    tune_result: &aprender::automl::TuneResult<KMeansParam>,
    true_k: usize,
) {
    let best_k = tune_result
        .best_trial
        .get_usize(&KMeansParam::NClusters)
        .unwrap_or(0);

    println!("\n🔍 Final Model Verification:");
    let mut final_kmeans = KMeans::new(best_k).with_max_iter(100).with_random_state(42);
    final_kmeans.fit(data).expect("Final fit should succeed");
    let final_labels = final_kmeans.predict(data);
    let final_silhouette = silhouette_score(data, &final_labels);
    let final_inertia = final_kmeans.inertia();

    println!("   Silhouette score: {final_silhouette:.4}");
    println!("   Inertia:          {final_inertia:.2}");
    println!("   Iterations:       {}", final_kmeans.n_iter());

    print_interpretation(best_k, true_k, final_silhouette);
}

fn print_interpretation(best_k: usize, true_k: usize, silhouette: f32) {
    println!("\n📈 Interpretation:");
    if best_k == true_k {
        println!("   ✅ TPE found the true number of clusters!");
    } else if (best_k as i32 - true_k as i32).abs() <= 1 {
        println!("   ✓ TPE found a close approximation (within ±1)");
    } else {
        println!("   ⚠ TPE found a different K (data may have ambiguous structure)");
    }

    if silhouette > 0.5 {
        println!("   ✅ Excellent cluster separation (silhouette > 0.5)");
    } else if silhouette > 0.25 {
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
