//! Evolutionary Merge Optimization (GH-444)
//!
//! Demonstrates CMA-ES optimization of merge weights to find the best
//! combination of multiple model checkpoints.
//!
//! Run: cargo run --example evolutionary_merge

use aprender::format::converter::evolutionary_merge::{
    decode_params, evolutionary_merge, merge_tensors_in_memory, softmax_normalize,
    EvolutionaryMergeConfig,
};
use aprender::format::MergeStrategy;
use std::collections::BTreeMap;

fn make_model(values: &[f32]) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut m = BTreeMap::new();
    m.insert("weight".to_string(), (values.to_vec(), vec![2, 2]));
    m.insert("bias".to_string(), (vec![0.1, 0.2], vec![2]));
    m
}

fn main() {
    println!("=== Evolutionary Merge Optimization (GH-444) ===\n");

    // Create 3 "model checkpoints" with different weight distributions
    let models = vec![
        make_model(&[1.0, 2.0, 3.0, 4.0]),
        make_model(&[4.0, 3.0, 2.0, 1.0]),
        make_model(&[2.0, 2.0, 2.0, 2.0]),
    ];

    // ── 1. Softmax Normalization ──
    println!("── 1. Softmax Weight Normalization ──");
    let raw = [1.0, 0.5, -0.5];
    let normalized = softmax_normalize(&raw);
    println!("  Raw params:  {:?}", raw);
    println!("  Normalized:  {:?}", normalized);
    println!("  Sum:         {:.4}", normalized.iter().sum::<f32>());

    // ── 2. Parameter Decoding ──
    println!("\n── 2. CMA-ES Parameter Decoding ──");
    let config = EvolutionaryMergeConfig {
        num_models: 3,
        optimize_density: true,
        ..Default::default()
    };
    let params = [1.0, 0.5, -0.5, 0.0]; // 3 weights + 1 density
    let (weights, density, drop_rate) = decode_params(&params, &config);
    println!("  Decoded weights: {:?}", weights);
    println!("  Density:         {:.4}", density);
    println!("  Drop rate:       {:.4} (default)", drop_rate);

    // ── 3. In-Memory Merge ──
    println!("\n── 3. In-Memory Tensor Merge ──");
    let equal_weights = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let merged = merge_tensors_in_memory(&models, &equal_weights, MergeStrategy::Average);
    let (w, _) = &merged["weight"];
    println!("  Equal-weight merge: {:?}", w);

    // ── 4. CMA-ES Optimization ──
    println!("\n── 4. CMA-ES Evolutionary Optimization ──");
    let target = vec![3.0_f32, 2.5, 2.5, 2.0];
    println!("  Target weights:    {:?}", target);

    let config = EvolutionaryMergeConfig {
        num_models: 3,
        max_evaluations: 100,
        strategy: MergeStrategy::Weighted,
        seed: 42,
        sigma: 0.5,
        ..Default::default()
    };

    let result = evolutionary_merge(&models, &config, |merged| {
        let (w, _) = &merged["weight"];
        w.iter()
            .zip(target.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    });

    println!("  Optimized weights: {:?}", result.weights);
    println!("  Best score (MSE):  {:.6}", result.best_score);
    println!("  Evaluations:       {}", result.evaluations);

    let (merged_w, _) =
        &merge_tensors_in_memory(&models, &result.weights, MergeStrategy::Weighted)["weight"];
    println!("  Merged result:     {:?}", merged_w);
    println!("  Target:            {:?}", target);

    // ── 5. Verify Constraints ──
    println!("\n── 5. Constraint Verification ──");
    let weight_sum: f32 = result.weights.iter().sum();
    println!("  Weights sum to 1:  {:.6} ✓", weight_sum);
    assert!((weight_sum - 1.0).abs() < 1e-4);
    println!("  Score is finite:   {} ✓", result.best_score.is_finite());
    println!(
        "  MergeOptions OK:   strategy={:?} ✓",
        result.merge_options.strategy
    );

    println!("\n=== Evolutionary merge optimization complete ===");
}
