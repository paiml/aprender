//! Mixture of Experts (MoE) layer
//!
//! Provides a sparse MoE layer where each input token is routed to a subset of
//! expert networks via a learned gating mechanism. This enables scaling model
//! capacity without proportionally increasing computation.
//!
//! ## Architecture
//!
//! - **Router**: Linear gating network with softmax that selects top-k experts per token
//! - **Experts**: Independent feed-forward networks (weight + bias)
//! - **MoeLayer**: Combines router and experts into a single forward pass
//!
//! ## Load Balancing
//!
//! The `balance_loss()` method computes a Switch Transformer-style auxiliary loss
//! that penalizes uneven expert utilization, encouraging the router to distribute
//! tokens uniformly across experts.
//!
//! ## References
//!
//! - Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers. JMLR.
//! - Lepikhin, D., et al. (2021). GShard: Scaling Giant Models. ICLR.

pub mod router;

#[cfg(test)]
mod tests;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

pub use router::{NoisyTopKRouter, RoutingResult, TopKRouter};

/// Configuration for a Mixture of Experts layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of expert networks
    pub num_experts: usize,
    /// Number of experts each token is routed to
    pub top_k: usize,
    /// Capacity factor controlling max tokens per expert (typically 1.0-1.5)
    pub capacity_factor: f32,
    /// Standard deviation of noise for exploration (0.0 = deterministic)
    pub noise_std: f32,
    /// Input/output dimension of each expert
    pub input_dim: usize,
    /// Hidden dimension within each expert
    pub hidden_dim: usize,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            noise_std: 0.0,
            input_dim: 64,
            hidden_dim: 128,
        }
    }
}

/// A single expert network: a two-layer feed-forward with ReLU activation.
///
/// Computes: output = ReLU(input @ W1 + b1) @ W2 + b2
#[derive(Debug, Clone)]
pub struct Expert {
    /// First layer weights: [input_dim, hidden_dim]
    pub w1: Array2<f32>,
    /// First layer bias: [hidden_dim]
    pub b1: Array1<f32>,
    /// Second layer weights: [hidden_dim, input_dim]
    pub w2: Array2<f32>,
    /// Second layer bias: [input_dim]
    pub b2: Array1<f32>,
}

impl Expert {
    /// Create a new expert with Xavier-initialized weights and zero biases.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let scale1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale2 = (2.0 / (hidden_dim + input_dim) as f32).sqrt();

        Self {
            w1: Array2::from_shape_fn((input_dim, hidden_dim), |(i, j)| {
                ((i * hidden_dim + j) as f32 * 0.3141).sin() * scale1
            }),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
                ((i * input_dim + j) as f32 * 0.2718).sin() * scale2
            }),
            b2: Array1::zeros(input_dim),
        }
    }

    /// Forward pass through this expert for a single token.
    ///
    /// # Arguments
    /// * `input` - Input vector of length input_dim
    ///
    /// # Returns
    /// Output vector of length input_dim
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        // hidden = ReLU(input @ W1 + b1)
        let hidden = input.dot(&self.w1) + &self.b1;
        let hidden = hidden.mapv(|v| v.max(0.0)); // ReLU

        // output = hidden @ W2 + b2
        hidden.dot(&self.w2) + &self.b2
    }

    /// Forward pass for a batch of tokens.
    ///
    /// # Arguments
    /// * `input` - Input matrix of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// Output matrix of shape [batch_size, input_dim]
    pub fn forward_batch(&self, input: &Array2<f32>) -> Array2<f32> {
        let hidden = input.dot(&self.w1) + &self.b1;
        let hidden = hidden.mapv(|v| v.max(0.0));
        hidden.dot(&self.w2) + &self.b2
    }
}

/// Router variant: either deterministic or noisy.
#[derive(Debug, Clone)]
pub enum Router {
    /// Deterministic top-k routing
    Deterministic(TopKRouter),
    /// Noisy top-k routing (adds Gaussian noise for exploration)
    Noisy(NoisyTopKRouter),
}

impl Router {
    /// Route input tokens to experts.
    pub fn route(&self, input: &Array2<f32>) -> RoutingResult {
        match self {
            Router::Deterministic(r) => r.route(input),
            Router::Noisy(r) => r.route(input),
        }
    }
}

/// Mixture of Experts layer combining a router with a set of expert networks.
#[derive(Debug, Clone)]
pub struct MoeLayer {
    /// Configuration
    pub config: MoeConfig,
    /// Router (gating network)
    pub router: Router,
    /// Expert networks
    pub experts: Vec<Expert>,
}

impl MoeLayer {
    /// Create a new MoE layer from configuration.
    pub fn new(config: MoeConfig) -> Self {
        let router_config = router::RouterConfig {
            input_dim: config.input_dim,
            num_experts: config.num_experts,
            top_k: config.top_k,
            capacity_factor: config.capacity_factor,
        };

        let router = if config.noise_std > 0.0 {
            Router::Noisy(NoisyTopKRouter::new(&router_config, config.noise_std))
        } else {
            Router::Deterministic(TopKRouter::new(&router_config))
        };

        let experts = (0..config.num_experts)
            .map(|_| Expert::new(config.input_dim, config.hidden_dim))
            .collect();

        Self { config, router, experts }
    }

    /// Forward pass: route each token to top-k experts and combine outputs.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// Tuple of (output tensor [batch_size, input_dim], routing result for loss computation)
    pub fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, RoutingResult) {
        let batch_size = input.nrows();
        let input_dim = input.ncols();
        let routing = self.router.route(input);

        let mut output = Array2::zeros((batch_size, input_dim));

        for i in 0..batch_size {
            let token = input.row(i).to_owned();
            let mut combined = Array1::zeros(input_dim);

            for (k, &expert_idx) in routing.expert_indices[i].iter().enumerate() {
                let weight = routing.expert_weights[i][k];
                if weight > 0.0 {
                    let expert_output = self.experts[expert_idx].forward(&token);
                    combined += &(expert_output * weight);
                }
            }

            output.row_mut(i).assign(&combined);
        }

        (output, routing)
    }

    /// Compute the Switch Transformer-style load balancing auxiliary loss.
    ///
    /// The balance loss encourages uniform expert utilization:
    ///
    ///   L_balance = num_experts * sum_i(f_i * P_i)
    ///
    /// where:
    /// - f_i = fraction of tokens dispatched to expert i
    /// - P_i = mean routing probability for expert i
    ///
    /// A perfectly balanced router produces L_balance = 1.0.
    /// Unbalanced routing produces L_balance > 1.0.
    ///
    /// # Arguments
    /// * `routing` - The routing result from a forward pass
    ///
    /// # Returns
    /// Scalar auxiliary loss value
    pub fn balance_loss(&self, routing: &RoutingResult) -> f32 {
        let num_experts = self.config.num_experts;
        let batch_size = routing.routing_probs.nrows();

        if batch_size == 0 {
            return 0.0;
        }

        // f_i: fraction of tokens actually dispatched to each expert
        let mut dispatch_counts = vec![0usize; num_experts];
        for token_experts in &routing.expert_indices {
            for &expert_idx in token_experts {
                dispatch_counts[expert_idx] += 1;
            }
        }
        let total_dispatches: usize = dispatch_counts.iter().sum();
        let f: Vec<f32> = dispatch_counts
            .iter()
            .map(|&c| if total_dispatches > 0 { c as f32 / total_dispatches as f32 } else { 0.0 })
            .collect();

        // P_i: mean routing probability for each expert across the batch
        let p = router::expert_load_fractions(&routing.routing_probs);

        // L_balance = N * sum(f_i * P_i)
        let dot: f32 = f.iter().zip(p.iter()).map(|(fi, pi)| fi * pi).sum();
        num_experts as f32 * dot
    }
}
