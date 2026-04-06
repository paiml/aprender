//! Calibration context for activation-based pruning methods.
//!
//! # Toyota Way: Genchi Genbutsu
//! Real activation patterns from calibration data, not synthetic estimates.
//!
//! # References
//! - Sun, M., et al. (2023). Wanda: A simple and effective pruning approach.

use super::error::PruningError;
use crate::autograd::Tensor;
use std::collections::HashMap;

/// Per-layer activation statistics.
///
/// Collects input activation norms during calibration forward passes,
/// which are used by methods like Wanda to weight importance scores.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// L2 norm of input activations per channel.
    /// Shape: \[`input_features`\]
    pub input_norms: Tensor,
    /// Running mean of squared activations per channel.
    /// Shape: \[`input_features`\]
    pub squared_mean: Tensor,
    /// Number of samples processed.
    pub count: usize,
}

impl ActivationStats {
    /// Create new stats for given input dimension.
    ///
    /// # Arguments
    /// * `input_features` - Number of input channels/features
    #[must_use]
    pub fn new(input_features: usize) -> Self {
        Self {
            input_norms: Tensor::zeros(&[input_features]),
            squared_mean: Tensor::zeros(&[input_features]),
            count: 0,
        }
    }

    /// Update statistics with a new batch using Welford's algorithm.
    ///
    /// # Arguments
    /// * `activations` - Tensor of shape \[`batch_size`, `input_features`\]
    ///
    /// # Algorithm
    /// Uses Welford's online algorithm for numerical stability when
    /// computing running statistics over many batches.
    pub fn update(&mut self, activations: &Tensor) {
        let shape = activations.shape();
        if shape.is_empty() || shape[0] == 0 {
            return; // Empty batch - no-op
        }

        let batch_size = shape[0];
        let input_features = if shape.len() > 1 { shape[1] } else { shape[0] };

        // Handle single-sample case (shape = [features])
        let act_data = activations.data();
        if shape.len() == 1 {
            // Single sample: activations is [input_features]
            let new_count = self.count + 1;

            let old_norms = self.input_norms.data();
            let old_sq_mean = self.squared_mean.data();

            let mut new_norms = vec![0.0f32; input_features];
            let mut new_sq_mean = vec![0.0f32; input_features];

            for i in 0..input_features {
                let val = act_data[i];
                let sq = val * val;

                // Welford's update
                let delta_norm = val.abs() - old_norms[i];
                new_norms[i] = old_norms[i] + delta_norm / new_count as f32;

                let delta_sq = sq - old_sq_mean[i];
                new_sq_mean[i] = old_sq_mean[i] + delta_sq / new_count as f32;
            }

            self.input_norms = Tensor::new(&new_norms, &[input_features]);
            self.squared_mean = Tensor::new(&new_sq_mean, &[input_features]);
            self.count = new_count;
            return;
        }

        // Batch case: activations is [batch_size, input_features]
        let new_count = self.count + batch_size;

        // Compute batch statistics per feature (column)
        let mut batch_sq_mean = vec![0.0f32; input_features];
        let mut batch_norms = vec![0.0f32; input_features];

        for col in 0..input_features {
            let mut sum_sq = 0.0f32;
            for row in 0..batch_size {
                let val = act_data[row * input_features + col];
                sum_sq += val * val;
            }
            batch_sq_mean[col] = sum_sq / batch_size as f32;
            batch_norms[col] = (sum_sq / batch_size as f32).sqrt();
        }

        // Welford's online update
        let old_norms = self.input_norms.data();
        let old_sq_mean = self.squared_mean.data();

        let mut new_norms = vec![0.0f32; input_features];
        let mut new_sq_mean = vec![0.0f32; input_features];

        for i in 0..input_features {
            // Running average update
            let delta_norm = batch_norms[i] - old_norms[i];
            new_norms[i] = old_norms[i] + delta_norm * (batch_size as f32 / new_count as f32);

            let delta_sq = batch_sq_mean[i] - old_sq_mean[i];
            new_sq_mean[i] = old_sq_mean[i] + delta_sq * (batch_size as f32 / new_count as f32);
        }

        self.input_norms = Tensor::new(&new_norms, &[input_features]);
        self.squared_mean = Tensor::new(&new_sq_mean, &[input_features]);
        self.count = new_count;
    }

    /// Get the number of input features.
    #[must_use]
    pub fn input_features(&self) -> usize {
        self.input_norms.data().len()
    }
}

/// Calibration context holding activation statistics for all layers.
///
/// Collects activation statistics during calibration forward passes
/// on a small set of representative samples.
#[derive(Debug, Clone)]
pub struct CalibrationContext {
    /// Per-layer statistics.
    pub activation_stats: HashMap<String, ActivationStats>,
    /// Number of calibration samples processed.
    pub num_samples: usize,
    /// Dataset identifier.
    pub dataset: String,
}

impl CalibrationContext {
    /// Create new calibration context.
    ///
    /// # Arguments
    /// * `dataset` - Identifier for the calibration dataset (e.g., "c4", "wikitext")
    #[must_use]
    pub fn new(dataset: String) -> Self {
        Self {
            activation_stats: HashMap::new(),
            num_samples: 0,
            dataset,
        }
    }

    /// Add statistics for a layer.
    ///
    /// # Arguments
    /// * `layer_name` - Unique identifier for the layer (e.g., "model.layers.0.mlp")
    /// * `stats` - Activation statistics for this layer
    pub fn add_layer_stats(&mut self, layer_name: String, stats: ActivationStats) {
        self.activation_stats.insert(layer_name, stats);
    }

    /// Get statistics for a layer if available.
    #[must_use]
    pub fn get_stats(&self, layer_name: &str) -> Option<&ActivationStats> {
        self.activation_stats.get(layer_name)
    }

    /// Get mutable statistics for a layer if available.
    pub fn get_stats_mut(&mut self, layer_name: &str) -> Option<&mut ActivationStats> {
        self.activation_stats.get_mut(layer_name)
    }

    /// Get statistics or return error.
    ///
    /// # Arguments
    /// * `layer_name` - Layer to look up
    ///
    /// # Returns
    /// Reference to stats, or `MissingActivationStats` error
    pub fn require_stats(&self, layer_name: &str) -> Result<&ActivationStats, PruningError> {
        self.get_stats(layer_name)
            .ok_or_else(|| PruningError::MissingActivationStats {
                layer: layer_name.to_string(),
            })
    }

    /// Check if stats exist for a layer.
    #[must_use]
    pub fn has_stats(&self, layer_name: &str) -> bool {
        self.activation_stats.contains_key(layer_name)
    }

    /// Get list of all layer names with stats.
    pub fn layer_names(&self) -> Vec<&str> {
        self.activation_stats.keys().map(String::as_str).collect()
    }

    /// Increment sample count.
    pub fn increment_samples(&mut self, count: usize) {
        self.num_samples += count;
    }
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
