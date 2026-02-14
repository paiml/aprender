//! Wanda (Weights and Activations) importance scoring.
//!
//! # Toyota Way: Genchi Genbutsu
//! Uses real activation patterns from calibration data, not estimates.
//!
//! # References
//! - Sun, M., et al. (2023). A simple and effective pruning approach for large language models.
//!   arXiv:2306.11695.

use super::calibration::CalibrationContext;
use super::error::PruningError;
use super::importance::{Importance, ImportanceScores};
use super::mask::SparsityPattern;
use crate::autograd::Tensor;
use crate::nn::Module;

/// Wanda (Weights and Activations) importance estimator.
///
/// Combines weight magnitudes with input activation norms to identify
/// important weights. This method is from Sun et al. (2023) and
/// achieves strong results with no retraining needed.
///
/// # Formula
/// `importance = |w| * sqrt(activation_norm)`
///
/// Where `activation_norm` is the L2 norm of input activations across
/// calibration samples for each input channel.
///
/// # Advantages
/// - No gradient computation needed
/// - No retraining required after pruning
/// - Works well at moderate sparsity (50%)
/// - Very fast (single forward pass for calibration)
///
/// # Requirements
/// - Calibration data (128 samples typically sufficient)
/// - Activation statistics for target layer
#[derive(Debug, Clone)]
pub struct WandaImportance {
    /// Layer name to look up in calibration context
    layer_name: String,
    /// Optional pattern constraint for N:M pruning
    pattern: Option<SparsityPattern>,
    /// Small epsilon to prevent division by zero
    eps: f32,
}

impl WandaImportance {
    /// Create Wanda importance estimator for a specific layer.
    ///
    /// # Arguments
    /// * `layer_name` - Layer identifier to look up in `CalibrationContext`
    pub fn new(layer_name: impl Into<String>) -> Self {
        Self {
            layer_name: layer_name.into(),
            pattern: None,
            eps: 1e-8,
        }
    }

    /// Set sparsity pattern constraint.
    ///
    /// # Arguments
    /// * `pattern` - N:M pattern or other structural constraint
    #[must_use]
    pub fn with_pattern(mut self, pattern: SparsityPattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set epsilon for numerical stability.
    ///
    /// # Arguments
    /// * `eps` - Small value to prevent division by zero (default: 1e-8)
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Get the layer name.
    #[must_use]
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    /// Get the pattern if set.
    #[must_use]
    pub fn pattern(&self) -> Option<SparsityPattern> {
        self.pattern
    }

    /// Compute Wanda importance scores.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor of shape \[`out_features`, `in_features`\]
    /// * `activation_norms` - L2 norms of input activations \[`in_features`\]
    ///
    /// # Returns
    /// Importance scores with same shape as weights.
    ///
    /// # Formula
    /// `importance[i,j] = |weights[i,j]| * sqrt(activation_norms[j])`
    pub fn compute_from_tensors(
        &self,
        weights: &Tensor,
        activation_norms: &Tensor,
    ) -> Result<Tensor, PruningError> {
        let weight_shape = weights.shape();
        let norm_shape = activation_norms.shape();

        // Validate shapes
        if weight_shape.len() < 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0], // Indicates 2D expected
                got: weight_shape.to_vec(),
            });
        }

        let out_features = weight_shape[0];
        let in_features = weight_shape[1];

        if norm_shape.is_empty() || norm_shape[0] != in_features {
            return Err(PruningError::ShapeMismatch {
                expected: vec![in_features],
                got: norm_shape.to_vec(),
            });
        }

        let weight_data = weights.data();
        let norm_data = activation_norms.data();

        // Jidoka: Check for NaN/Inf
        for (i, &w) in weight_data.iter().enumerate() {
            if w.is_nan() {
                return Err(PruningError::NumericalInstability {
                    method: self.name().to_string(),
                    details: format!("NaN detected in weight at index {i}"),
                });
            }
            if w.is_infinite() {
                return Err(PruningError::NumericalInstability {
                    method: self.name().to_string(),
                    details: format!("Inf detected in weight at index {i}"),
                });
            }
        }

        for (i, &n) in norm_data.iter().enumerate() {
            if n.is_nan() {
                return Err(PruningError::NumericalInstability {
                    method: self.name().to_string(),
                    details: format!("NaN detected in activation norm at index {i}"),
                });
            }
            if n.is_infinite() {
                return Err(PruningError::NumericalInstability {
                    method: self.name().to_string(),
                    details: format!("Inf detected in activation norm at index {i}"),
                });
            }
        }

        // Compute importance: |w| * sqrt(activation_norm)
        let mut importance = vec![0.0f32; out_features * in_features];

        for i in 0..out_features {
            for j in 0..in_features {
                let idx = i * in_features + j;
                let w = weight_data[idx];
                let norm = norm_data[j];

                // Handle zero/negative norms gracefully
                let sqrt_norm = if norm <= 0.0 {
                    self.eps.sqrt() // Use epsilon for zero activations
                } else {
                    norm.sqrt()
                };

                importance[idx] = w.abs() * sqrt_norm;
            }
        }

        Ok(Tensor::new(&importance, weight_shape))
    }
}

impl Importance for WandaImportance {
    fn compute(
        &self,
        module: &dyn Module,
        context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError> {
        // Require calibration context
        let ctx = context.ok_or(PruningError::CalibrationRequired {
            method: self.name().to_string(),
        })?;

        // Get activation stats for this layer
        let stats = ctx.require_stats(&self.layer_name)?;

        // Get module parameters
        let params = module.parameters();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: self.layer_name.clone(),
            });
        }

        let weights = params[0];

        // Compute importance using activation norms
        let importance = self.compute_from_tensors(weights, &stats.input_norms)?;

        Ok(ImportanceScores::new(importance, "wanda".to_string()))
    }

    fn name(&self) -> &'static str {
        "wanda"
    }

    fn requires_calibration(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "wanda_tests.rs"]
mod tests;
