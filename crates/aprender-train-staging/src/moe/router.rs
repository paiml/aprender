//! Gating/routing mechanisms for Mixture of Experts
//!
//! Provides `TopKRouter` (deterministic) and `NoisyTopKRouter` (with exploration noise)
//! for selecting which experts process each input token.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Result of routing a batch of tokens to experts.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Expert indices selected per token: shape [batch_size, top_k]
    pub expert_indices: Vec<Vec<usize>>,
    /// Gating weights per token for selected experts: shape [batch_size, top_k]
    pub expert_weights: Vec<Vec<f32>>,
    /// Full probability distribution over experts per token: shape [batch_size, num_experts]
    pub routing_probs: Array2<f32>,
}

/// Deterministic top-k router: linear projection followed by softmax, then top-k selection.
#[derive(Debug, Clone)]
pub struct TopKRouter {
    /// Gating weight matrix: [input_dim, num_experts]
    pub gate_weight: Array2<f32>,
    /// Number of experts to route each token to
    pub top_k: usize,
    /// Maximum fraction of tokens each expert can process
    pub capacity_factor: f32,
}

/// Router configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub input_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl TopKRouter {
    /// Create a new top-k router with Xavier-initialized gate weights.
    pub fn new(config: &RouterConfig) -> Self {
        let scale = (2.0 / (config.input_dim + config.num_experts) as f32).sqrt();
        let gate_weight =
            Array2::from_shape_fn((config.input_dim, config.num_experts), |(i, j)| {
                ((i * config.num_experts + j) as f32 * 0.4567).sin() * scale
            });

        Self { gate_weight, top_k: config.top_k, capacity_factor: config.capacity_factor }
    }

    /// Route a batch of input tokens to the top-k experts.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// `RoutingResult` containing expert assignments and weights.
    pub fn route(&self, input: &Array2<f32>) -> RoutingResult {
        let batch_size = input.nrows();
        let num_experts = self.gate_weight.ncols();

        // Compute logits: [batch_size, num_experts] = input @ gate_weight
        let logits = input.dot(&self.gate_weight);

        // Softmax per row to get routing probabilities
        let routing_probs = softmax_rows(&logits);

        // Apply capacity factor to determine max tokens per expert
        let capacity = capacity_limit(batch_size, self.top_k, num_experts, self.capacity_factor);

        // Select top-k experts per token, respecting capacity
        let (expert_indices, expert_weights) =
            select_top_k_with_capacity(&routing_probs, self.top_k, capacity);

        RoutingResult { expert_indices, expert_weights, routing_probs }
    }
}

/// Noisy top-k router: adds Gaussian noise to logits before routing for exploration.
///
/// Based on the Switch Transformer / GShard approach where noise encourages
/// balanced expert utilization during training.
#[derive(Debug, Clone)]
pub struct NoisyTopKRouter {
    /// Underlying deterministic router
    pub inner: TopKRouter,
    /// Standard deviation of Gaussian noise added to logits
    pub noise_std: f32,
}

impl NoisyTopKRouter {
    /// Create a new noisy top-k router.
    pub fn new(config: &RouterConfig, noise_std: f32) -> Self {
        Self { inner: TopKRouter::new(config), noise_std }
    }

    /// Route with added Gaussian noise for exploration.
    pub fn route(&self, input: &Array2<f32>) -> RoutingResult {
        let batch_size = input.nrows();
        let num_experts = self.inner.gate_weight.ncols();

        // Compute logits
        let mut logits = input.dot(&self.inner.gate_weight);

        // Add Gaussian noise
        let mut rng = rand::rng();
        for val in &mut logits {
            let noise: f32 = rng.random::<f32>() * 2.0 - 1.0; // Uniform approximation
            *val += noise * self.noise_std;
        }

        let routing_probs = softmax_rows(&logits);
        let capacity =
            capacity_limit(batch_size, self.inner.top_k, num_experts, self.inner.capacity_factor);
        let (expert_indices, expert_weights) =
            select_top_k_with_capacity(&routing_probs, self.inner.top_k, capacity);

        RoutingResult { expert_indices, expert_weights, routing_probs }
    }
}

/// Compute row-wise softmax of a 2D array.
pub(crate) fn softmax_rows(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = logits.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            row.mapv_inplace(|v| v / sum);
        }
    }
    result
}

/// Compute the capacity limit per expert.
///
/// capacity = ceil(capacity_factor * batch_size * top_k / num_experts)
pub(crate) fn capacity_limit(
    batch_size: usize,
    top_k: usize,
    num_experts: usize,
    capacity_factor: f32,
) -> usize {
    let raw = capacity_factor * (batch_size * top_k) as f32 / num_experts as f32;
    raw.ceil().max(1.0) as usize
}

/// Select top-k experts per token, enforcing a per-expert capacity limit.
///
/// Returns (expert_indices, expert_weights) where each inner Vec has length top_k.
/// When an expert is at capacity, the token's assignment falls through to the next
/// highest-scoring expert.
fn select_top_k_with_capacity(
    probs: &Array2<f32>,
    top_k: usize,
    capacity: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let batch_size = probs.nrows();
    let num_experts = probs.ncols();
    let mut expert_counts = vec![0usize; num_experts];
    let mut all_indices = Vec::with_capacity(batch_size);
    let mut all_weights = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let row: Vec<f32> = probs.row(i).to_vec();
        let (indices, weights) = assign_token_experts(&row, top_k, capacity, &mut expert_counts);
        all_indices.push(indices);
        all_weights.push(weights);
    }

    (all_indices, all_weights)
}

/// Assign top-k experts for a single token, respecting capacity limits.
fn assign_token_experts(
    row: &[f32],
    top_k: usize,
    capacity: usize,
    expert_counts: &mut [usize],
) -> (Vec<usize>, Vec<f32>) {
    let mut sorted: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut indices = Vec::with_capacity(top_k);
    let mut weights = Vec::with_capacity(top_k);

    for &(expert_idx, weight) in &sorted {
        if indices.len() >= top_k {
            break;
        }
        if expert_counts[expert_idx] < capacity {
            indices.push(expert_idx);
            weights.push(weight);
            expert_counts[expert_idx] += 1;
        }
    }

    pad_assignments(&mut indices, &mut weights, top_k);
    renormalize_weights(&mut weights);
    (indices, weights)
}

/// Pad assignments to top_k if capacity prevented full assignment.
fn pad_assignments(indices: &mut Vec<usize>, weights: &mut Vec<f32>, top_k: usize) {
    while indices.len() < top_k {
        if let Some(&last_idx) = indices.last() {
            indices.push(last_idx);
            weights.push(0.0);
        } else {
            indices.push(0);
            weights.push(1.0 / top_k as f32);
        }
    }
}

/// Renormalize weights to sum to 1.0.
fn renormalize_weights(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}

/// Compute the fraction of tokens routed to each expert.
///
/// Returns an Array1 of length num_experts with the fraction of total routing
/// probability assigned to each expert.
pub(crate) fn expert_load_fractions(routing_probs: &Array2<f32>) -> Array1<f32> {
    let num_experts = routing_probs.ncols();
    let batch_size = routing_probs.nrows();
    if batch_size == 0 {
        return Array1::zeros(num_experts);
    }
    let col_sums = routing_probs.sum_axis(ndarray::Axis(0));
    col_sums / batch_size as f32
}
