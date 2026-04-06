//! Calibration statistics for quantization
//!
//! Following Kaizen: continuously improved from query distribution.
//! Uses Welford's algorithm for numerical stability (Higham, 2002).

use super::error::{validate_embedding, QuantizationError};
use super::params::QuantizationParams;

/// Calibration statistics for quantization
///
/// Following Kaizen: continuously improved from query distribution.
/// Uses Welford's algorithm for numerical stability (Higham, 2002).
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    /// Maximum absolute value across calibration set
    pub absmax: f32,
    /// Running mean for each dimension
    pub mean: Vec<f32>,
    /// Running M2 for variance calculation (Welford's)
    pub(crate) m2: Vec<f32>,
    /// Number of samples seen
    pub n_samples: usize,
    /// Embedding dimensions
    pub dims: usize,
}

impl CalibrationStats {
    /// Create new calibration stats for given dimensions
    pub fn new(dims: usize) -> Self {
        Self { absmax: 0.0, mean: vec![0.0; dims], m2: vec![0.0; dims], n_samples: 0, dims }
    }

    /// Update calibration with new embedding (Kaizen loop)
    ///
    /// Uses Welford's online algorithm for numerical stability.
    pub fn update(&mut self, embedding: &[f32]) -> Result<(), QuantizationError> {
        // Validate dimensions and finite values upfront
        validate_embedding(embedding, self.dims)?;

        self.n_samples += 1;
        let n = self.n_samples as f32;

        for (i, &v) in embedding.iter().enumerate() {
            // Update absmax
            self.absmax = self.absmax.max(v.abs());

            // Welford's algorithm for mean and variance
            let delta = v - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = v - self.mean[i];
            self.m2[i] += delta * delta2;
        }

        Ok(())
    }

    /// Update with batch of embeddings (Heijunka batching)
    pub fn update_batch(&mut self, embeddings: &[Vec<f32>]) -> Result<(), QuantizationError> {
        for embedding in embeddings {
            self.update(embedding)?;
        }
        Ok(())
    }

    /// Get variance for dimension i
    pub fn variance(&self, i: usize) -> f32 {
        if self.n_samples < 2 || i >= self.dims {
            return 0.0;
        }
        self.m2[i] / (self.n_samples - 1) as f32
    }

    /// Get standard deviation for dimension i
    pub fn std_dev(&self, i: usize) -> f32 {
        self.variance(i).sqrt()
    }

    /// Convert calibration to quantization parameters
    pub fn to_quant_params(&self) -> Result<QuantizationParams, QuantizationError> {
        if self.n_samples == 0 {
            return Err(QuantizationError::CalibrationNotInitialized);
        }
        let absmax = if self.absmax == 0.0 { 1.0 } else { self.absmax };
        QuantizationParams::from_absmax(absmax, self.dims)
    }

    /// Check if calibration has sufficient samples
    pub fn is_sufficient(&self, min_samples: usize) -> bool {
        self.n_samples >= min_samples
    }
}
