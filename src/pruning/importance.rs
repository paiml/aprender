//! Importance scoring infrastructure for neural network pruning.
//!
//! # Toyota Way: Genchi Genbutsu
//! Importance scores reflect actual parameter contributions, not estimates.
//!
//! # References
//! - Han, S., et al. (2015). Learning both weights and connections. `NeurIPS`.
//! - Sun, M., et al. (2023). A simple and effective pruning approach. arXiv:2306.11695.

use super::calibration::CalibrationContext;
use super::error::PruningError;
use crate::autograd::Tensor;
use crate::nn::Module;

/// Statistical summary of importance scores.
///
/// Provides min, max, mean, std for analyzing the distribution
/// of importance values before thresholding.
#[derive(Debug, Clone)]
pub struct ImportanceStats {
    /// Minimum importance value
    pub min: f32,
    /// Maximum importance value
    pub max: f32,
    /// Mean importance value
    pub mean: f32,
    /// Standard deviation of importance values
    pub std: f32,
    /// Sparsity achieved at various thresholds: (threshold, `sparsity_ratio`)
    pub sparsity_at_threshold: Vec<(f32, f32)>,
}

impl ImportanceStats {
    /// Compute statistics from a tensor of importance values.
    ///
    /// # Arguments
    /// * `values` - Tensor containing importance scores
    ///
    /// # Returns
    /// Statistics computed over all values in the tensor.
    #[must_use]
    pub fn from_tensor(values: &Tensor) -> Self {
        let data = values.data();

        if data.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                sparsity_at_threshold: vec![],
            };
        }

        // Compute min and max
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;

        for &v in data {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }

        let mean = sum / data.len() as f32;

        // Compute standard deviation (population std)
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();

        Self {
            min,
            max,
            mean,
            std,
            sparsity_at_threshold: vec![], // Computed lazily if needed
        }
    }

    /// Compute sparsity at given threshold.
    ///
    /// Sparsity is the fraction of values below the threshold.
    #[must_use]
    pub fn sparsity_at(&self, values: &Tensor, threshold: f32) -> f32 {
        let data = values.data();
        if data.is_empty() {
            return 0.0;
        }
        let below = data.iter().filter(|&&v| v < threshold).count();
        below as f32 / data.len() as f32
    }
}

impl Default for ImportanceStats {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            sparsity_at_threshold: vec![],
        }
    }
}

/// Importance scores with metadata.
///
/// Contains the raw importance values and statistical summary
/// for analysis and visualization.
#[derive(Debug, Clone)]
pub struct ImportanceScores {
    /// Raw importance values, same shape as parameter tensor
    pub values: Tensor,
    /// Statistical summary
    pub stats: ImportanceStats,
    /// Method that produced these scores
    pub method: String,
}

impl ImportanceScores {
    /// Create new importance scores, computing stats automatically.
    ///
    /// # Arguments
    /// * `values` - Tensor of importance scores
    /// * `method` - Name of the method that computed these scores
    #[must_use]
    pub fn new(values: Tensor, method: String) -> Self {
        let stats = ImportanceStats::from_tensor(&values);
        Self {
            values,
            stats,
            method,
        }
    }

    /// Get the shape of the importance tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.values.shape()
    }

    /// Get the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.data().len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.data().is_empty()
    }
}

/// Core trait for importance estimation algorithms.
///
/// # Toyota Way: Jidoka
/// All implementations must validate numerical stability.
/// NaN or Inf values trigger immediate failure (Andon cord).
///
/// # Object Safety
/// This trait is object-safe and can be used with `dyn Importance`.
pub trait Importance: Send + Sync {
    /// Compute importance scores for a module's parameters.
    ///
    /// # Arguments
    /// * `module` - Neural network module to analyze
    /// * `context` - Optional calibration context with activation stats
    ///
    /// # Returns
    /// * `Ok(ImportanceScores)` - Computed importance values
    /// * `Err(PruningError)` - If computation fails or numerical issues detected
    fn compute(
        &self,
        module: &dyn Module,
        context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError>;

    /// Name of this importance method.
    ///
    /// Used for logging and error messages.
    fn name(&self) -> &'static str;

    /// Whether this method requires calibration data.
    ///
    /// Methods like Wanda require activation statistics from calibration.
    fn requires_calibration(&self) -> bool;
}

#[cfg(test)]
#[path = "importance_tests.rs"]
mod tests;
