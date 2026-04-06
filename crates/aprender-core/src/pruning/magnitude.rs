//! Magnitude-based importance scoring.
//!
//! # Toyota Way: Genchi Genbutsu
//! Importance scores directly reflect parameter magnitudes - no approximations.
//!
//! # References
//! - Han, S., et al. (2015). Learning both weights and connections. `NeurIPS`.

use super::calibration::CalibrationContext;
use super::error::PruningError;
use super::importance::{Importance, ImportanceScores};
use crate::autograd::Tensor;
use crate::nn::Module;

/// Norm type for magnitude computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// L1 norm: |w| (absolute value)
    L1,
    /// L2 norm: w² (squared value)
    L2,
}

/// Magnitude-based importance estimator.
///
/// The simplest importance metric: larger weights are more important.
/// This is the original method from Han et al. (2015).
///
/// # Formula
/// - L1: `importance = |w|`
/// - L2: `importance = w²`
///
/// # Advantages
/// - No calibration data required
/// - Very fast to compute
/// - Good baseline for comparison
///
/// # Limitations
/// - Ignores activation patterns
/// - Can underestimate importance of small but critical weights
#[derive(Debug, Clone)]
pub struct MagnitudeImportance {
    /// Norm type (L1 or L2)
    norm: NormType,
}

impl MagnitudeImportance {
    /// Create L1 magnitude importance estimator.
    ///
    /// # Formula
    /// `importance = |w|`
    #[must_use]
    pub fn l1() -> Self {
        Self { norm: NormType::L1 }
    }

    /// Create L2 magnitude importance estimator.
    ///
    /// # Formula
    /// `importance = w²`
    #[must_use]
    pub fn l2() -> Self {
        Self { norm: NormType::L2 }
    }

    /// Create magnitude importance with specified norm.
    #[must_use]
    pub fn with_norm(norm: NormType) -> Self {
        Self { norm }
    }

    /// Get the norm type.
    #[must_use]
    pub fn norm(&self) -> NormType {
        self.norm
    }

    /// Compute importance for a single weight tensor.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor to compute importance for
    ///
    /// # Returns
    /// Importance scores with same shape as weights.
    ///
    /// # Toyota Way: Jidoka
    /// Validates for NaN/Inf values and returns error immediately.
    pub fn compute_for_weights(&self, weights: &Tensor) -> Result<Tensor, PruningError> {
        let data = weights.data();

        // Jidoka: Check for NaN/Inf in weights
        for (i, &w) in data.iter().enumerate() {
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

        // Compute importance based on norm type
        let importance: Vec<f32> = match self.norm {
            NormType::L1 => data.iter().map(|&w| w.abs()).collect(),
            NormType::L2 => data.iter().map(|&w| w * w).collect(),
        };

        Ok(Tensor::new(&importance, weights.shape()))
    }
}

impl Importance for MagnitudeImportance {
    fn compute(
        &self,
        module: &dyn Module,
        _context: Option<&CalibrationContext>,
    ) -> Result<ImportanceScores, PruningError> {
        let params = module.parameters();

        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "unknown".to_string(),
            });
        }

        // Compute importance for first weight tensor
        // (typically the main weight matrix in a layer)
        let weights = params[0];
        let importance = self.compute_for_weights(weights)?;

        Ok(ImportanceScores::new(
            importance,
            format!(
                "magnitude_{}",
                match self.norm {
                    NormType::L1 => "l1",
                    NormType::L2 => "l2",
                }
            ),
        ))
    }

    fn name(&self) -> &'static str {
        match self.norm {
            NormType::L1 => "magnitude_l1",
            NormType::L2 => "magnitude_l2",
        }
    }

    fn requires_calibration(&self) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "magnitude_tests.rs"]
mod tests;
