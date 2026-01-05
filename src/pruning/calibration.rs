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
    /// Shape: [`input_features`]
    pub input_norms: Tensor,
    /// Running mean of squared activations per channel.
    /// Shape: [`input_features`]
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
    /// * `activations` - Tensor of shape [`batch_size`, `input_features`]
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
mod tests {
    use super::*;

    // ==========================================================================
    // FALSIFICATION: ActivationStats initialization
    // ==========================================================================
    #[test]
    fn test_activation_stats_new() {
        let stats = ActivationStats::new(4);

        assert_eq!(
            stats.count, 0,
            "CAL-01 FALSIFIED: initial count should be 0"
        );
        assert_eq!(
            stats.input_norms.data().len(),
            4,
            "CAL-01 FALSIFIED: input_norms should have 4 elements"
        );
        assert!(
            stats.input_norms.data().iter().all(|&v| v == 0.0),
            "CAL-01 FALSIFIED: initial norms should be all zeros"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Empty batch doesn't corrupt stats
    // ==========================================================================
    #[test]
    fn test_activation_stats_empty_batch_noop() {
        let mut stats = ActivationStats::new(4);

        // First update with actual data
        stats.update(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]));
        let count_before = stats.count;
        let norms_before = stats.input_norms.data().to_vec();

        // Empty update (0 batch size)
        stats.update(&Tensor::new(&[], &[0, 4]));

        assert_eq!(
            stats.count, count_before,
            "CAL-02 FALSIFIED: empty batch should not change count"
        );
        assert_eq!(
            stats.input_norms.data(),
            &norms_before[..],
            "CAL-02 FALSIFIED: empty batch should not change norms"
        );
    }

    // ==========================================================================
    // FALSIFICATION: CalibrationContext layer lookup
    // ==========================================================================
    #[test]
    fn test_calibration_context_get_stats() {
        let mut ctx = CalibrationContext::new("test_dataset".to_string());

        let stats = ActivationStats::new(512);
        ctx.add_layer_stats("model.layer0".to_string(), stats);

        assert!(
            ctx.get_stats("model.layer0").is_some(),
            "CAL-03 FALSIFIED: should find added layer"
        );
        assert!(
            ctx.get_stats("nonexistent").is_none(),
            "CAL-03 FALSIFIED: should not find non-existent layer"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Missing stats returns error
    // ==========================================================================
    #[test]
    fn test_calibration_context_missing_stats_error() {
        let ctx = CalibrationContext::new("test".to_string());

        let result = ctx.require_stats("missing_layer");
        assert!(
            result.is_err(),
            "CAL-04 FALSIFIED: should error on missing stats"
        );

        match result.unwrap_err() {
            PruningError::MissingActivationStats { layer } => {
                assert_eq!(
                    layer, "missing_layer",
                    "CAL-04 FALSIFIED: error should contain layer name"
                );
            }
            _ => panic!("CAL-04 FALSIFIED: expected MissingActivationStats error"),
        }
    }

    // ==========================================================================
    // FALSIFICATION: Input norm computation (single sample)
    // ==========================================================================
    #[test]
    fn test_activation_stats_single_sample() {
        let mut stats = ActivationStats::new(2);

        // Single sample: [3.0, 4.0]
        let batch = Tensor::new(&[3.0, 4.0], &[1, 2]);
        stats.update(&batch);

        assert_eq!(stats.count, 1, "CAL-05 FALSIFIED: count should be 1");

        // input_norms should be sqrt(mean(x^2)) = sqrt(x^2) = |x| for single sample
        // But we're computing running RMS, so for single sample:
        // norm[0] = sqrt(9/1) = 3.0
        // norm[1] = sqrt(16/1) = 4.0
        let norms = stats.input_norms.data();
        assert!(
            (norms[0] - 3.0).abs() < 1e-5,
            "CAL-05 FALSIFIED: norm[0] should be 3.0, got {}",
            norms[0]
        );
        assert!(
            (norms[1] - 4.0).abs() < 1e-5,
            "CAL-05 FALSIFIED: norm[1] should be 4.0, got {}",
            norms[1]
        );
    }

    // ==========================================================================
    // FALSIFICATION: Multi-batch Welford update
    // ==========================================================================
    #[test]
    fn test_activation_stats_multi_batch() {
        let mut stats = ActivationStats::new(2);

        // Batch 1: [[1, 2], [3, 4]] - 2 samples
        let batch1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        stats.update(&batch1);
        assert_eq!(
            stats.count, 2,
            "CAL-06 FALSIFIED: count should be 2 after first batch"
        );

        // Batch 2: [[5, 6]] - 1 sample
        let batch2 = Tensor::new(&[5.0, 6.0], &[1, 2]);
        stats.update(&batch2);
        assert_eq!(
            stats.count, 3,
            "CAL-06 FALSIFIED: count should be 3 after second batch"
        );
    }

    // ==========================================================================
    // FALSIFICATION: CalibrationContext dataset name
    // ==========================================================================
    #[test]
    fn test_calibration_context_dataset() {
        let ctx = CalibrationContext::new("c4".to_string());
        assert_eq!(
            ctx.dataset, "c4",
            "CAL-07 FALSIFIED: dataset should be 'c4'"
        );
        assert_eq!(
            ctx.num_samples, 0,
            "CAL-07 FALSIFIED: initial samples should be 0"
        );
    }

    // ==========================================================================
    // FALSIFICATION: has_stats helper
    // ==========================================================================
    #[test]
    fn test_calibration_context_has_stats() {
        let mut ctx = CalibrationContext::new("test".to_string());

        assert!(
            !ctx.has_stats("layer0"),
            "CAL-08 FALSIFIED: should not have stats initially"
        );

        ctx.add_layer_stats("layer0".to_string(), ActivationStats::new(10));
        assert!(
            ctx.has_stats("layer0"),
            "CAL-08 FALSIFIED: should have stats after adding"
        );
    }

    // ==========================================================================
    // FALSIFICATION: layer_names helper
    // ==========================================================================
    #[test]
    fn test_calibration_context_layer_names() {
        let mut ctx = CalibrationContext::new("test".to_string());

        ctx.add_layer_stats("layer_a".to_string(), ActivationStats::new(10));
        ctx.add_layer_stats("layer_b".to_string(), ActivationStats::new(20));

        let names = ctx.layer_names();
        assert_eq!(
            names.len(),
            2,
            "CAL-09 FALSIFIED: should have 2 layer names"
        );
        assert!(
            names.contains(&"layer_a"),
            "CAL-09 FALSIFIED: should contain layer_a"
        );
        assert!(
            names.contains(&"layer_b"),
            "CAL-09 FALSIFIED: should contain layer_b"
        );
    }

    // ==========================================================================
    // FALSIFICATION: increment_samples
    // ==========================================================================
    #[test]
    fn test_calibration_context_increment_samples() {
        let mut ctx = CalibrationContext::new("test".to_string());

        ctx.increment_samples(10);
        assert_eq!(ctx.num_samples, 10, "CAL-10 FALSIFIED: should be 10");

        ctx.increment_samples(5);
        assert_eq!(ctx.num_samples, 15, "CAL-10 FALSIFIED: should be 15");
    }

    // ==========================================================================
    // FALSIFICATION: input_features helper
    // ==========================================================================
    #[test]
    fn test_activation_stats_input_features() {
        let stats = ActivationStats::new(128);
        assert_eq!(
            stats.input_features(),
            128,
            "CAL-11 FALSIFIED: input_features should be 128"
        );
    }

    // ==========================================================================
    // FALSIFICATION: get_stats_mut helper
    // ==========================================================================
    #[test]
    fn test_calibration_context_get_stats_mut() {
        let mut ctx = CalibrationContext::new("test".to_string());
        ctx.add_layer_stats("layer0".to_string(), ActivationStats::new(10));

        // Get mutable stats and modify
        let stats = ctx.get_stats_mut("layer0").unwrap();
        stats.count = 42;

        // Verify modification
        assert_eq!(
            ctx.get_stats("layer0").unwrap().count,
            42,
            "CAL-12 FALSIFIED: Mutable access should allow modification"
        );

        // Non-existent layer returns None
        assert!(
            ctx.get_stats_mut("nonexistent").is_none(),
            "CAL-12 FALSIFIED: Non-existent layer should return None"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug traits
    // ==========================================================================
    #[test]
    fn test_activation_stats_clone() {
        let mut stats = ActivationStats::new(4);
        stats.update(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]));

        let cloned = stats.clone();
        assert_eq!(
            stats.count, cloned.count,
            "CAL-13 FALSIFIED: Clone should preserve count"
        );
        assert_eq!(
            stats.input_norms.data(),
            cloned.input_norms.data(),
            "CAL-13 FALSIFIED: Clone should preserve input_norms"
        );
    }

    #[test]
    fn test_calibration_context_clone() {
        let mut ctx = CalibrationContext::new("test".to_string());
        ctx.add_layer_stats("layer0".to_string(), ActivationStats::new(10));
        ctx.increment_samples(5);

        let cloned = ctx.clone();
        assert_eq!(
            ctx.num_samples, cloned.num_samples,
            "CAL-14 FALSIFIED: Clone should preserve sample count"
        );
        assert_eq!(
            ctx.dataset, cloned.dataset,
            "CAL-14 FALSIFIED: Clone should preserve dataset name"
        );
        assert!(cloned.has_stats("layer0"));
    }

    #[test]
    fn test_calibration_context_debug() {
        let ctx = CalibrationContext::new("debug_test".to_string());
        let debug_str = format!("{:?}", ctx);
        assert!(
            debug_str.contains("CalibrationContext"),
            "CAL-15 FALSIFIED: Debug should show type name"
        );
        assert!(
            debug_str.contains("debug_test"),
            "CAL-15 FALSIFIED: Debug should show dataset"
        );
    }

    #[test]
    fn test_activation_stats_debug() {
        let stats = ActivationStats::new(4);
        let debug_str = format!("{:?}", stats);
        assert!(
            debug_str.contains("ActivationStats"),
            "CAL-16 FALSIFIED: Debug should show type name"
        );
    }
}
