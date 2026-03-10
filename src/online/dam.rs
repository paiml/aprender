//! Differentiable Adaptive Merging (DAM) (GH-446)
//!
//! Optimizes per-tensor merge coefficients for combining multiple models.
//! Unlike uniform or SLERP merging, DAM learns task-specific weights
//! that minimize loss on a held-out calibration set.
//!
//! # Algorithm
//!
//! Given N source models and a target (calibration) signal, DAM finds
//! coefficients `w_i` such that `sum(softmax(w) * tensors)` minimizes
//! reconstruction loss. Uses Nelder-Mead simplex optimization (gradient-free)
//! to handle non-smooth loss landscapes.
//!
//! # References
//!
//! - Nelder & Mead 1965: "A Simplex Method for Function Minimization"
//! - Wortsman et al. 2022: "Model Soups: Averaging Weights of Multiple
//!   Fine-tuned Models Improves Accuracy without Increasing Inference Time"

use std::collections::BTreeMap;

use crate::error::{AprenderError, Result};

/// Configuration for DAM optimization.
#[derive(Debug, Clone)]
pub struct DamConfig {
    /// Learning rate for gradient-based updates
    pub learning_rate: f64,
    /// Maximum number of optimization iterations
    pub num_iterations: usize,
    /// L2 regularization strength to prevent extreme coefficients
    pub regularization: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for DamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            num_iterations: 100,
            regularization: 0.01,
            seed: 42,
        }
    }
}

impl DamConfig {
    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "learning_rate must be positive finite, got {}",
                    self.learning_rate
                ),
            });
        }
        if self.num_iterations == 0 {
            return Err(AprenderError::FormatError {
                message: "num_iterations must be > 0".to_string(),
            });
        }
        if self.regularization < 0.0 || !self.regularization.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "regularization must be non-negative finite, got {}",
                    self.regularization
                ),
            });
        }
        Ok(())
    }
}

/// Per-tensor merge coefficients produced by DAM optimization.
///
/// Each tensor name maps to a vector of weights (one per source model).
/// Weights are in logit space — apply [`softmax`] before merging.
#[derive(Debug, Clone)]
pub struct DamCoefficients {
    /// Per-tensor weights keyed by tensor name.
    /// Each value has length equal to the number of source models.
    pub per_tensor: BTreeMap<String, Vec<f64>>,
}

impl DamCoefficients {
    /// Create uniform coefficients for `num_models` models across the given tensors.
    #[must_use]
    pub fn uniform(tensor_names: &[String], num_models: usize) -> Self {
        let init = vec![0.0; num_models]; // uniform in logit space
        let per_tensor = tensor_names
            .iter()
            .map(|name| (name.clone(), init.clone()))
            .collect();
        Self { per_tensor }
    }

    /// Get the softmax-normalized weights for a given tensor.
    #[must_use]
    pub fn normalized_weights(&self, tensor_name: &str) -> Option<Vec<f64>> {
        self.per_tensor.get(tensor_name).map(|w| softmax(w))
    }
}

/// DAM loss calculator.
///
/// Wraps a [`DamConfig`] and provides methods for computing merge loss,
/// regularization penalty, and performing gradient updates.
#[derive(Debug, Clone)]
pub struct DamLoss {
    config: DamConfig,
}

impl DamLoss {
    /// Create a new DAM loss calculator.
    #[must_use]
    pub fn new(config: DamConfig) -> Self {
        Self { config }
    }

    /// Compute mean squared error between merged output and target.
    ///
    /// MSE = (1/n) * sum((merged_i - target_i)^2)
    #[must_use]
    pub fn compute_merge_loss(merged: &[f64], target: &[f64]) -> f64 {
        if merged.is_empty() || target.is_empty() {
            return 0.0;
        }
        let n = merged.len().min(target.len());
        let sum_sq: f64 = merged[..n]
            .iter()
            .zip(&target[..n])
            .map(|(m, t)| {
                let d = m - t;
                d * d
            })
            .sum();
        sum_sq / n as f64
    }

    /// Compute L2 regularization penalty on coefficients.
    ///
    /// reg = (lambda / n) * sum(c_i^2)
    ///
    /// Prevents coefficients from growing to extreme values, which would
    /// make the softmax degenerate to a one-hot selector.
    #[must_use]
    pub fn compute_regularization(coefficients: &[f64]) -> f64 {
        if coefficients.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = coefficients.iter().map(|c| c * c).sum();
        sum_sq / coefficients.len() as f64
    }

    /// Perform a single SGD gradient step on coefficients.
    ///
    /// `coefficients[i] -= lr * gradients[i]`
    pub fn gradient_step(coefficients: &mut [f64], gradients: &[f64], lr: f64) {
        let n = coefficients.len().min(gradients.len());
        for i in 0..n {
            coefficients[i] -= lr * gradients[i];
        }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &DamConfig {
        &self.config
    }
}

/// Softmax normalization: converts logits to a probability distribution.
///
/// Uses the numerically stable form: `exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.
#[must_use]
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    let max_val = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&xi| (xi - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum == 0.0 {
        // Degenerate case: return uniform
        return vec![1.0 / x.len() as f64; x.len()];
    }

    exps.iter().map(|&e| e / sum).collect()
}

/// Run gradient-free optimization (Nelder-Mead simplex) to find optimal
/// per-model merge coefficients that minimize the given loss function.
///
/// # Arguments
///
/// * `num_models` - Number of source models (dimension of coefficient vector)
/// * `loss_fn` - Objective function mapping coefficients to scalar loss
/// * `config` - Optimization configuration
///
/// # Returns
///
/// Optimal coefficient vector of length `num_models`
pub fn optimize_coefficients(
    num_models: usize,
    loss_fn: impl Fn(&[f64]) -> f64,
    config: &DamConfig,
) -> Vec<f64> {
    if num_models == 0 {
        return vec![];
    }
    if num_models == 1 {
        return vec![1.0];
    }

    // Initialize simplex: n+1 vertices in n-dimensional space
    // Start from uniform (all zeros in logit space) with perturbations
    let n = num_models;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);

    // First vertex: uniform initialization
    simplex.push(vec![0.0; n]);

    // Remaining vertices: perturb one dimension each
    // Use seed for deterministic perturbation magnitude
    let perturbation = 0.5;
    for i in 0..n {
        let mut vertex = vec![0.0; n];
        // LCG-based deterministic perturbation sign
        let state = config
            .seed
            .wrapping_add(i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let sign = if (state >> 33).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        vertex[i] = sign * perturbation;
        simplex.push(vertex);
    }

    // Evaluate loss at each vertex
    let total_loss = |coeffs: &[f64]| -> f64 {
        loss_fn(coeffs) + config.regularization * DamLoss::compute_regularization(coeffs)
    };

    let mut losses: Vec<f64> = simplex.iter().map(|v| total_loss(v)).collect();

    // Nelder-Mead parameters
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    for _iter in 0..config.num_iterations {
        // Sort vertices by loss
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| {
            losses[a]
                .partial_cmp(&losses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Convergence check: if spread is tiny, stop
        let spread = losses[worst_idx] - losses[best_idx];
        if spread.abs() < 1e-12 {
            break;
        }

        // Compute centroid of all vertices except the worst
        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = centroid
            .iter()
            .zip(&simplex[worst_idx])
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();
        let reflected_loss = total_loss(&reflected);

        if reflected_loss < losses[second_worst_idx] && reflected_loss >= losses[best_idx] {
            // Accept reflection
            simplex[worst_idx] = reflected;
            losses[worst_idx] = reflected_loss;
            continue;
        }

        if reflected_loss < losses[best_idx] {
            // Try expansion
            let expanded: Vec<f64> = centroid
                .iter()
                .zip(&reflected)
                .map(|(&c, &r)| c + gamma * (r - c))
                .collect();
            let expanded_loss = total_loss(&expanded);

            if expanded_loss < reflected_loss {
                simplex[worst_idx] = expanded;
                losses[worst_idx] = expanded_loss;
            } else {
                simplex[worst_idx] = reflected;
                losses[worst_idx] = reflected_loss;
            }
            continue;
        }

        // Contraction
        let contracted: Vec<f64> = centroid
            .iter()
            .zip(&simplex[worst_idx])
            .map(|(&c, &w)| c + rho * (w - c))
            .collect();
        let contracted_loss = total_loss(&contracted);

        if contracted_loss < losses[worst_idx] {
            simplex[worst_idx] = contracted;
            losses[worst_idx] = contracted_loss;
            continue;
        }

        // Shrink: move all vertices toward the best
        let best = simplex[best_idx].clone();
        for i in 0..=n {
            if i == best_idx {
                continue;
            }
            for j in 0..n {
                simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
            }
            losses[i] = total_loss(&simplex[i]);
        }
    }

    // Return the best vertex
    let best_idx = losses
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    simplex[best_idx].clone()
}

/// Report summarizing a DAM optimization run.
#[derive(Debug, Clone)]
pub struct DamReport {
    /// Final loss value achieved
    pub final_loss: f64,
    /// Number of iterations executed
    pub num_iterations: usize,
    /// Optimized coefficients (in logit space)
    pub coefficients: Vec<f64>,
    /// Whether optimization converged (loss spread below threshold)
    pub converged: bool,
}

impl DamReport {
    /// Get the softmax-normalized coefficients.
    #[must_use]
    pub fn normalized_coefficients(&self) -> Vec<f64> {
        softmax(&self.coefficients)
    }
}

#[cfg(test)]
#[path = "dam_tests.rs"]
mod tests;
