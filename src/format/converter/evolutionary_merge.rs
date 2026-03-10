//! Evolutionary merge optimization using CMA-ES (GH-444)
//!
//! Optimizes merge weights and hyperparameters (density, drop_rate) using
//! Covariance Matrix Adaptation Evolution Strategy. The objective function
//! evaluates merged model quality (e.g., perplexity on calibration data).

use crate::metaheuristics::{Budget, CmaEs, PerturbativeMetaheuristic, SearchSpace};
use std::collections::BTreeMap;

use super::merge::{MergeOptions, MergeStrategy};

/// Configuration for evolutionary merge optimization.
#[derive(Debug, Clone)]
pub struct EvolutionaryMergeConfig {
    /// Base merge strategy (Average, Weighted, Slerp, Ties, Dare)
    pub strategy: MergeStrategy,
    /// Maximum objective function evaluations
    pub max_evaluations: usize,
    /// CMA-ES initial step size
    pub sigma: f64,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of models being merged
    pub num_models: usize,
    /// Whether to optimize density (for TIES)
    pub optimize_density: bool,
    /// Whether to optimize drop_rate (for DARE)
    pub optimize_drop_rate: bool,
}

impl Default for EvolutionaryMergeConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Weighted,
            max_evaluations: 100,
            sigma: 0.3,
            seed: 42,
            num_models: 2,
            optimize_density: false,
            optimize_drop_rate: false,
        }
    }
}

/// Result of evolutionary merge optimization.
#[derive(Debug, Clone)]
pub struct EvolutionaryMergeResult {
    /// Optimized per-model weights (sum to 1.0)
    pub weights: Vec<f32>,
    /// Optimized TIES density (if optimize_density=true)
    pub density: f32,
    /// Optimized DARE drop_rate (if optimize_drop_rate=true)
    pub drop_rate: f32,
    /// Best objective value found
    pub best_score: f64,
    /// Total evaluations used
    pub evaluations: usize,
    /// Constructed MergeOptions ready for use
    pub merge_options: MergeOptions,
}

/// Dimension count for the CMA-ES search space.
fn param_dim(config: &EvolutionaryMergeConfig) -> usize {
    let mut dim = config.num_models; // weights
    if config.optimize_density {
        dim += 1;
    }
    if config.optimize_drop_rate {
        dim += 1;
    }
    dim
}

/// Decode CMA-ES parameters into merge weights and hyperparameters.
///
/// The first `num_models` parameters are raw weights (softmax-normalized).
/// Optional trailing parameters are density and drop_rate (sigmoid-mapped to (0,1)).
pub fn decode_params(params: &[f64], config: &EvolutionaryMergeConfig) -> (Vec<f32>, f32, f32) {
    let raw_weights = &params[..config.num_models];
    let weights = softmax_normalize(raw_weights);

    let mut idx = config.num_models;
    let density = if config.optimize_density {
        let v = sigmoid(params[idx]);
        idx += 1;
        v as f32
    } else {
        0.2 // default
    };

    let drop_rate = if config.optimize_drop_rate {
        sigmoid(params[idx]) as f32
    } else {
        0.9 // default
    };

    (weights, density, drop_rate)
}

/// Softmax normalization: ensures weights are positive and sum to 1.
pub fn softmax_normalize(raw: &[f64]) -> Vec<f32> {
    let max_val = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = raw.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| (v / sum) as f32).collect()
}

/// Sigmoid function: maps (-inf, inf) to (0, 1).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Build MergeOptions from optimized parameters.
pub fn build_merge_options(
    config: &EvolutionaryMergeConfig,
    weights: Vec<f32>,
    density: f32,
    drop_rate: f32,
) -> MergeOptions {
    MergeOptions {
        strategy: config.strategy,
        weights: Some(weights),
        base_model: None,
        drop_rate,
        density,
        seed: config.seed,
        scales: None,
        outlier_k: 3.0,
    }
}

/// L2 norm of a vector (f64 precision).
fn vector_norm_f64(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// Merge tensors from multiple models in-memory using given weights.
///
/// Supports Average, Weighted, and Slerp strategies.
pub fn merge_tensors_in_memory(
    models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    weights: &[f32],
    strategy: MergeStrategy,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let reference = &models[0];
    let mut merged = BTreeMap::new();

    for (name, (_, shape)) in reference {
        let merged_data = merge_single_tensor(models, name, weights, strategy);
        merged.insert(name.clone(), (merged_data, shape.clone()));
    }
    merged
}

/// Merge a single tensor across models.
fn merge_single_tensor(
    models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    name: &str,
    weights: &[f32],
    strategy: MergeStrategy,
) -> Vec<f32> {
    match strategy {
        MergeStrategy::Slerp if models.len() == 2 => {
            let (a, _) = &models[0][name];
            let (b, _) = &models[1][name];
            slerp_vectors(a, b, weights[1])
        }
        _ => {
            // Weighted average (also works for Average with equal weights)
            let data_len = models[0][name].0.len();
            let mut result = vec![0.0f32; data_len];
            for (model_idx, model) in models.iter().enumerate() {
                let (data, _) = &model[name];
                let w = weights[model_idx];
                for (i, &val) in data.iter().enumerate() {
                    result[i] += val * w;
                }
            }
            result
        }
    }
}

/// SLERP between two flat f32 vectors.
fn slerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let norm_a = vector_norm_f64(a);
    let norm_b = vector_norm_f64(b);

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return lerp_vectors(a, b, t);
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let cos_omega = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < 1e-6 {
        return lerp_vectors(a, b, t);
    }

    let sin_omega = omega.sin();
    let t64 = f64::from(t);
    let coeff_a = ((1.0 - t64) * omega).sin() / sin_omega;
    let coeff_b = (t64 * omega).sin() / sin_omega;

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (coeff_a * f64::from(x) + coeff_b * f64::from(y)) as f32)
        .collect()
}

/// Linear interpolation: (1-t)*a + t*b.
fn lerp_vectors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * (1.0 - t) + y * t)
        .collect()
}

/// Run CMA-ES to find optimal merge parameters.
///
/// The `objective_fn` receives merged tensors and returns a score to MINIMIZE
/// (e.g., perplexity). CMA-ES explores the weight space to find the combination
/// that produces the best merged model.
pub fn evolutionary_merge<F>(
    models: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    config: &EvolutionaryMergeConfig,
    objective_fn: F,
) -> EvolutionaryMergeResult
where
    F: Fn(&BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> f64,
{
    let dim = param_dim(config);
    let space = SearchSpace::continuous(dim, -3.0, 3.0);

    let mut cma = CmaEs::new(dim)
        .with_seed(config.seed)
        .with_sigma(config.sigma);

    let objective = |params: &[f64]| -> f64 {
        let (weights, _density, _drop_rate) = decode_params(params, config);
        let merged = merge_tensors_in_memory(models, &weights, config.strategy);
        objective_fn(&merged)
    };

    let result = cma.optimize(
        &objective,
        &space,
        Budget::Evaluations(config.max_evaluations),
    );

    let (weights, density, drop_rate) = decode_params(&result.solution, config);
    let merge_options = build_merge_options(config, weights.clone(), density, drop_rate);

    EvolutionaryMergeResult {
        weights,
        density,
        drop_rate,
        best_score: result.objective_value,
        evaluations: result.evaluations,
        merge_options,
    }
}

#[cfg(test)]
#[path = "evolutionary_merge_tests.rs"]
mod tests;
