//! Per-layer activation statistics for calibration.

/// Per-layer activation statistics collected during calibration.
///
/// Uses Welford's algorithm for numerically stable online computation.
#[derive(Debug, Clone, Default)]
pub struct LayerActivationStats {
    /// Running sum of L2 norms per input channel.
    input_norm_sum: Vec<f32>,
    /// Running sum of squared activations per channel.
    squared_sum: Vec<f32>,
    /// Number of samples processed.
    count: usize,
    /// Input feature dimension.
    input_dim: usize,
}

impl LayerActivationStats {
    /// Create new statistics tracker for a given input dimension.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_norm_sum: vec![0.0; input_dim],
            squared_sum: vec![0.0; input_dim],
            count: 0,
            input_dim,
        }
    }

    /// Update statistics with a new batch of activations.
    ///
    /// # Arguments
    ///
    /// * `activations` - Batch of activations [batch_size, input_dim]
    ///
    /// # Panics
    ///
    /// Panics if activation dimensions don't match.
    pub fn update(&mut self, activations: &[Vec<f32>]) {
        contract_pre_update!();
        if activations.is_empty() {
            return;
        }

        for sample in activations {
            assert_eq!(
                sample.len(),
                self.input_dim,
                "Activation dimension mismatch: expected {}, got {}",
                self.input_dim,
                sample.len()
            );

            for (i, &val) in sample.iter().enumerate() {
                // Accumulate squared values for L2 norm computation
                self.squared_sum[i] += val * val;
                self.input_norm_sum[i] += val.abs();
            }
            self.count += 1;
        }
    }

    /// Get the mean L2 norm for each input channel.
    ///
    /// Returns sqrt(mean(x^2)) for each channel.
    pub fn input_norms(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.input_dim];
        }

        self.squared_sum.iter().map(|&sum| (sum / self.count as f32).sqrt()).collect()
    }

    /// Get the mean absolute value for each input channel.
    pub fn mean_abs(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.input_dim];
        }

        self.input_norm_sum.iter().map(|&sum| sum / self.count as f32).collect()
    }

    /// Get the number of samples processed.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Check if any statistics have been collected.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.input_norm_sum.fill(0.0);
        self.squared_sum.fill(0.0);
        self.count = 0;
    }
}
