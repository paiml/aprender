//! Mixture of Experts implementation
//!
//! `MoE` enables specialized expert models with a learnable gating network
//! that routes inputs to the most appropriate expert(s).
//!
//! # Architecture
//!
//! ```text
//! Input -> [Gating Network] -> Expert Weights
//!     |
//!     +---> [Expert 1] --+
//!     +---> [Expert 2] --+--> Weighted Sum -> Output
//!     +---> [Expert N] --+
//! ```
//!
//! # References
//!
//! - Shazeer et al. (2017): Outrageously Large Neural Networks
//! - Fedus et al. (2021): Switch Transformers

use super::gating::GatingNetwork;
use crate::traits::Estimator;
use crate::{Matrix, Result, Vector};
use serde::{Deserialize, Serialize};

/// `MoE` routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub top_k: usize,
    pub capacity_factor: f32,
    pub expert_dropout: f32,
    pub load_balance_weight: f32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            top_k: 1,
            capacity_factor: 1.0,
            expert_dropout: 0.0,
            load_balance_weight: 0.01,
        }
    }
}

impl MoeConfig {
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    #[must_use]
    pub fn with_capacity_factor(mut self, factor: f32) -> Self {
        self.capacity_factor = factor;
        self
    }

    #[must_use]
    pub fn with_expert_dropout(mut self, dropout: f32) -> Self {
        self.expert_dropout = dropout;
        self
    }

    #[must_use]
    pub fn with_load_balance_weight(mut self, weight: f32) -> Self {
        self.load_balance_weight = weight;
        self
    }
}

/// Mixture of Experts ensemble
pub struct MixtureOfExperts<E: Estimator, G: GatingNetwork> {
    experts: Vec<E>,
    gating: G,
    config: MoeConfig,
}

impl<E: Estimator + std::fmt::Debug, G: GatingNetwork + std::fmt::Debug> std::fmt::Debug
    for MixtureOfExperts<E, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixtureOfExperts")
            .field("experts", &self.experts)
            .field("gating", &self.gating)
            .field("config", &self.config)
            .finish()
    }
}

impl<E: Estimator, G: GatingNetwork> MixtureOfExperts<E, G> {
    #[must_use]
    pub fn builder() -> MoeBuilder<E, G> {
        MoeBuilder::new()
    }

    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    #[must_use]
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    pub fn predict(&self, input: &[f32]) -> f32 {
        let weights = self.gating.forward(input);
        let mut indexed_weights: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(self.experts.len());
        let top_weights: Vec<(usize, f32)> = indexed_weights.into_iter().take(top_k).collect();
        let weight_sum: f32 = top_weights.iter().map(|(_, w)| w).sum();

        let mut output = 0.0f32;
        for (expert_idx, weight) in top_weights {
            let x = Matrix::from_vec(1, input.len(), input.to_vec()).expect("valid input matrix");
            let pred = self.experts[expert_idx].predict(&x);
            let expert_output = pred.as_slice()[0];
            output += (weight / weight_sum) * expert_output;
        }
        output
    }

    /// Predict for a batch of inputs.
    ///
    /// Returns predictions and optionally the expert routing decisions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of shape [`n_samples`, `n_features`]
    ///
    /// # Returns
    ///
    /// Vector of predictions, one per input sample.
    pub fn predict_batch(&self, inputs: &Matrix<f32>) -> Vector<f32> {
        let n_samples = inputs.n_rows();
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let row = inputs.row(i);
            predictions.push(self.predict(row.as_slice()));
        }

        Vector::from_slice(&predictions)
    }

    /// Compute load balancing auxiliary loss.
    ///
    /// Encourages even distribution of inputs across experts to prevent
    /// expert collapse (all inputs routed to single expert).
    ///
    /// Loss = `sum_i(f_i` * `P_i`) where:
    /// - `f_i` = fraction of inputs routed to expert i
    /// - `P_i` = average gate probability for expert i
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of shape [`n_samples`, `n_features`]
    ///
    /// # Returns
    ///
    /// Load balance loss value (lower is more balanced)
    pub fn compute_load_balance_loss(&self, inputs: &Matrix<f32>) -> f32 {
        let n_samples = inputs.n_rows();
        let n_experts = self.experts.len();

        if n_samples == 0 || n_experts == 0 {
            return 0.0;
        }

        // Count expert assignments and accumulate gate probabilities
        let mut expert_counts = vec![0usize; n_experts];
        let mut expert_probs = vec![0.0f32; n_experts];

        for i in 0..n_samples {
            let row = inputs.row(i);
            let weights = self.gating.forward(row.as_slice());

            // Find top-k experts
            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k = self.config.top_k.min(n_experts);
            for (idx, prob) in indexed.iter().take(top_k) {
                expert_counts[*idx] += 1;
                expert_probs[*idx] += prob;
            }
        }

        // Compute load balance loss
        let n_tokens = (n_samples * self.config.top_k.min(n_experts)) as f32;
        let mut loss = 0.0f32;

        for (count, prob_sum) in expert_counts.iter().zip(expert_probs.iter()) {
            let f_i = *count as f32 / n_tokens.max(1.0);
            let p_i = *prob_sum / n_samples as f32;
            loss += f_i * p_i;
        }

        // Scale by number of experts (Switch Transformer formulation)
        loss * n_experts as f32 * self.config.load_balance_weight
    }

    /// Get expert usage statistics for a batch of inputs.
    ///
    /// Returns the fraction of inputs routed to each expert.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix of shape [`n_samples`, `n_features`]
    ///
    /// # Returns
    ///
    /// Vector of usage fractions, one per expert (sums to `top_k`).
    pub fn expert_usage(&self, inputs: &Matrix<f32>) -> Vec<f32> {
        let n_samples = inputs.n_rows();
        let n_experts = self.experts.len();

        if n_samples == 0 || n_experts == 0 {
            return vec![0.0; n_experts];
        }

        let mut counts = vec![0usize; n_experts];

        for i in 0..n_samples {
            let row = inputs.row(i);
            let weights = self.gating.forward(row.as_slice());

            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k = self.config.top_k.min(n_experts);
            for (idx, _) in indexed.iter().take(top_k) {
                counts[*idx] += 1;
            }
        }

        let total = counts.iter().sum::<usize>() as f32;
        counts
            .iter()
            .map(|&c| c as f32 / total.max(1.0) * self.config.top_k.min(n_experts) as f32)
            .collect()
    }

    /// Get routing weights for a single input (useful for debugging/visualization).
    ///
    /// # Arguments
    ///
    /// * `input` - Feature vector
    ///
    /// # Returns
    ///
    /// Vector of gating weights for each expert.
    #[must_use]
    pub fn get_routing_weights(&self, input: &[f32]) -> Vec<f32> {
        self.gating.forward(input)
    }
}

impl<E, G> MixtureOfExperts<E, G>
where
    E: Estimator + Clone,
    G: GatingNetwork,
{
    /// Fit `MoE` using pre-trained experts.
    ///
    /// This is a simple two-stage training approach:
    /// 1. Experts are assumed to be pre-trained (passed in via builder)
    /// 2. No gating training is performed (uses initial weights)
    ///
    /// For more sophisticated training, use separate expert training
    /// followed by `MoE` construction.
    ///
    /// # Arguments
    ///
    /// * `_x` - Training features (unused in this simple implementation)
    /// * `_y` - Training labels (unused in this simple implementation)
    #[allow(clippy::unused_self)]
    pub fn fit(&mut self, _x: &Matrix<f32>, _y: &Vector<f32>) -> Result<()> {
        // In the simple two-stage approach, experts are pre-trained
        // and gating uses fixed random weights.
        // More sophisticated implementations would:
        // 1. Route inputs to experts based on gating
        // 2. Train experts on their routed inputs
        // 3. Update gating based on expert performance
        Ok(())
    }
}

impl<E, G> MixtureOfExperts<E, G>
where
    E: Estimator + Serialize + serde::de::DeserializeOwned,
    G: GatingNetwork + Serialize + serde::de::DeserializeOwned,
{
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let bytes = bincode::serialize(&SerializableMoe {
            experts: &self.experts,
            gating: &self.gating,
            config: &self.config,
        })
        .map_err(|e| crate::AprenderError::FormatError {
            message: format!("MoE serialization failed: {e}"),
        })?;
        std::fs::write(path, bytes).map_err(crate::AprenderError::Io)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(crate::AprenderError::Io)?;
        let data: DeserializableMoe<E, G> =
            bincode::deserialize(&bytes).map_err(|e| crate::AprenderError::FormatError {
                message: format!("MoE deserialization failed: {e}"),
            })?;
        Ok(Self {
            experts: data.experts,
            gating: data.gating,
            config: data.config,
        })
    }

    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        use crate::format::{ModelType, SaveOptions};
        let data = SerializableMoe {
            experts: &self.experts,
            gating: &self.gating,
            config: &self.config,
        };
        crate::format::save(
            &data,
            ModelType::MixtureOfExperts,
            path,
            SaveOptions::default(),
        )
    }
}

#[derive(Serialize)]
struct SerializableMoe<'a, E, G> {
    experts: &'a Vec<E>,
    gating: &'a G,
    config: &'a MoeConfig,
}

#[derive(Deserialize)]
struct DeserializableMoe<E, G> {
    experts: Vec<E>,
    gating: G,
    config: MoeConfig,
}

/// Builder for `MixtureOfExperts`
pub struct MoeBuilder<E: Estimator, G: GatingNetwork> {
    experts: Vec<E>,
    gating: Option<G>,
    config: MoeConfig,
}

impl<E: Estimator + std::fmt::Debug, G: GatingNetwork + std::fmt::Debug> std::fmt::Debug
    for MoeBuilder<E, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeBuilder")
            .field("experts", &self.experts)
            .field("gating", &self.gating)
            .field("config", &self.config)
            .finish()
    }
}

impl<E: Estimator, G: GatingNetwork> MoeBuilder<E, G> {
    fn new() -> Self {
        Self {
            experts: Vec::new(),
            gating: None,
            config: MoeConfig::default(),
        }
    }

    #[must_use]
    pub fn gating(mut self, g: G) -> Self {
        self.gating = Some(g);
        self
    }

    #[must_use]
    pub fn expert(mut self, e: E) -> Self {
        self.experts.push(e);
        self
    }

    #[must_use]
    pub fn config(mut self, c: MoeConfig) -> Self {
        self.config = c;
        self
    }

    pub fn build(self) -> Result<MixtureOfExperts<E, G>> {
        let gating = self
            .gating
            .ok_or_else(|| crate::AprenderError::InvalidHyperparameter {
                param: "gating".into(),
                value: "None".into(),
                constraint: "GatingNetwork required".into(),
            })?;
        Ok(MixtureOfExperts {
            experts: self.experts,
            gating,
            config: self.config,
        })
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
