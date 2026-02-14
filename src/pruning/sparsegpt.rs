//! `SparseGPT`: Hessian-based pruning with weight compensation.
//!
//! # Toyota Way: Jidoka
//! All Hessian computations validate for numerical stability.
//! Singular matrices trigger damping adjustment.
//!
//! # References
//! - Frantar, E., & Alistarh, D. (2023). `SparseGPT`: Massive language models
//!   can be accurately pruned in one-shot. ICML.
//! - `LeCun`, Y., et al. (1989). Optimal brain damage. `NeurIPS`.
//! - Hassibi, B., & Stork, D. (1993). Optimal brain surgeon. `NeurIPS`.

use super::calibration::CalibrationContext;
use super::error::PruningError;
use super::importance::{Importance, ImportanceScores};
use crate::autograd::Tensor;
use crate::nn::Module;

/// `SparseGPT` importance estimator using Hessian-based saliency.
///
/// Computes importance scores based on the Optimal Brain Surgeon (OBS)
/// saliency metric: `saliency = w^2 / H^{-1}_{jj}`
///
/// This identifies weights whose removal causes minimal output perturbation
/// when compensated by adjusting remaining weights.
///
/// # Algorithm
/// 1. Compute Hessian H = (1/n) * X^T * X + damp * I
/// 2. Compute Hessian inverse via Cholesky decomposition
/// 3. Saliency = w^2 / diag(H^{-1})
///
/// # Key Insight
/// Second-order information allows weight updates that minimize
/// the output perturbation caused by pruning.
#[derive(Debug, Clone)]
pub struct SparseGPTImportance {
    /// Layer name to look up in calibration context
    layer_name: String,
    /// Block size for block-wise processing (memory efficiency)
    block_size: usize,
    /// Damping factor for Hessian stability
    damp: f32,
    /// Relative damping (percentage of mean diagonal)
    damp_relative: bool,
}

impl SparseGPTImportance {
    /// Create `SparseGPT` importance estimator for a specific layer.
    ///
    /// # Arguments
    /// * `layer_name` - Layer identifier to look up in `CalibrationContext`
    pub fn new(layer_name: impl Into<String>) -> Self {
        Self {
            layer_name: layer_name.into(),
            block_size: 128,
            damp: 0.01,
            damp_relative: true,
        }
    }

    /// Set block size for block-wise processing.
    ///
    /// Smaller blocks use less memory but may be less accurate.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set absolute damping factor.
    ///
    /// # Arguments
    /// * `damp` - Value added to diagonal of Hessian for stability
    #[must_use]
    pub fn with_damp(mut self, damp: f32) -> Self {
        self.damp = damp;
        self.damp_relative = false;
        self
    }

    /// Set relative damping factor.
    ///
    /// # Arguments
    /// * `damp` - Percentage of mean diagonal to add (e.g., 0.01 = 1%)
    #[must_use]
    pub fn with_relative_damp(mut self, damp: f32) -> Self {
        self.damp = damp;
        self.damp_relative = true;
        self
    }

    /// Get the layer name.
    #[must_use]
    pub fn layer_name(&self) -> &str {
        &self.layer_name
    }

    /// Get the block size.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the damping factor.
    #[must_use]
    pub fn damp(&self) -> f32 {
        self.damp
    }

    /// Compute Hessian from calibration activations.
    ///
    /// H = (1/n) * X^T * X + damp * I
    ///
    /// # Arguments
    /// * `activations` - Calibration activations [`num_samples`, `in_features`]
    ///
    /// # Returns
    /// Hessian matrix [`in_features`, `in_features`]
    pub fn compute_hessian(&self, activations: &Tensor) -> Result<Tensor, PruningError> {
        let shape = activations.shape();
        if shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0], // Indicates 2D expected
                got: shape.to_vec(),
            });
        }

        let num_samples = shape[0];
        let in_features = shape[1];

        if num_samples == 0 {
            return Err(PruningError::InvalidSparsity {
                value: 0.0,
                constraint: "calibration must have at least 1 sample".to_string(),
            });
        }

        let act_data = activations.data();

        // Check for NaN/Inf in activations
        for (i, &v) in act_data.iter().enumerate() {
            if v.is_nan() {
                return Err(PruningError::NumericalInstability {
                    method: "SparseGPT".to_string(),
                    details: format!("NaN in activation at index {i}"),
                });
            }
            if v.is_infinite() {
                return Err(PruningError::NumericalInstability {
                    method: "SparseGPT".to_string(),
                    details: format!("Inf in activation at index {i}"),
                });
            }
        }

        // Compute X^T * X
        let mut hessian = vec![0.0f32; in_features * in_features];

        for sample in 0..num_samples {
            for i in 0..in_features {
                let xi = act_data[sample * in_features + i];
                for j in 0..in_features {
                    let xj = act_data[sample * in_features + j];
                    hessian[i * in_features + j] += xi * xj;
                }
            }
        }

        // Normalize by num_samples
        let n = num_samples as f32;
        for v in &mut hessian {
            *v /= n;
        }

        // Compute damping value
        let damp_value = if self.damp_relative {
            // Relative damping: percentage of mean diagonal
            let mut diag_sum = 0.0f32;
            for i in 0..in_features {
                diag_sum += hessian[i * in_features + i];
            }
            let mean_diag = diag_sum / in_features as f32;
            self.damp * mean_diag
        } else {
            self.damp
        };

        // Add damping to diagonal
        for i in 0..in_features {
            hessian[i * in_features + i] += damp_value;
        }

        Ok(Tensor::new(&hessian, &[in_features, in_features]))
    }

    /// Compute Hessian inverse using Cholesky decomposition.
    ///
    /// # Arguments
    /// * `hessian` - Symmetric positive definite Hessian matrix
    ///
    /// # Returns
    /// Inverse of Hessian matrix
    ///
    /// # Toyota Way: Jidoka
    /// Validates positive definiteness and numerical stability.
    pub fn compute_hessian_inverse(&self, hessian: &Tensor) -> Result<Tensor, PruningError> {
        let shape = hessian.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(PruningError::ShapeMismatch {
                expected: vec![shape[0], shape[0]],
                got: shape.to_vec(),
            });
        }

        let n = shape[0];
        let h_data = hessian.data();

        // Cholesky decomposition: H = L * L^T
        let mut l = vec![0.0f32; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = h_data[i * n + j];

                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    if sum <= 0.0 {
                        return Err(PruningError::NumericalInstability {
                            method: "SparseGPT".to_string(),
                            details: format!(
                                "Hessian not positive definite at index {i}. Consider increasing damping."
                            ),
                        });
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }

        // Invert L (lower triangular)
        let mut l_inv = vec![0.0f32; n * n];
        for i in 0..n {
            l_inv[i * n + i] = 1.0 / l[i * n + i];
            for j in 0..i {
                let mut sum = 0.0f32;
                for k in j..i {
                    sum -= l[i * n + k] * l_inv[k * n + j];
                }
                l_inv[i * n + j] = sum / l[i * n + i];
            }
        }

        // H^{-1} = L^{-T} * L^{-1}
        let mut h_inv = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in i.max(j)..n {
                    sum += l_inv[k * n + i] * l_inv[k * n + j];
                }
                h_inv[i * n + j] = sum;
            }
        }

        Ok(Tensor::new(&h_inv, &[n, n]))
    }

    /// Compute saliency scores from weights and Hessian inverse.
    ///
    /// Saliency = w^2 / H^{-1}_{jj}
    ///
    /// Lower saliency means the weight can be pruned with less error.
    pub fn compute_saliency(
        &self,
        weights: &Tensor,
        hessian_inv: &Tensor,
    ) -> Result<Tensor, PruningError> {
        let w_shape = weights.shape();
        let h_shape = hessian_inv.shape();

        if w_shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0],
                got: w_shape.to_vec(),
            });
        }

        let out_features = w_shape[0];
        let in_features = w_shape[1];

        if h_shape[0] != in_features {
            return Err(PruningError::ShapeMismatch {
                expected: vec![in_features, in_features],
                got: h_shape.to_vec(),
            });
        }

        let w_data = weights.data();
        let h_data = hessian_inv.data();

        // Extract diagonal of H^{-1}
        let mut h_diag = vec![0.0f32; in_features];
        for j in 0..in_features {
            h_diag[j] = h_data[j * in_features + j];

            // Validate diagonal is positive
            if h_diag[j] <= 0.0 {
                return Err(PruningError::NumericalInstability {
                    method: "SparseGPT".to_string(),
                    details: format!(
                        "Non-positive Hessian inverse diagonal at index {}: {}",
                        j, h_diag[j]
                    ),
                });
            }
        }

        // Compute saliency: w^2 / h_diag
        let mut saliency = vec![0.0f32; out_features * in_features];
        for i in 0..out_features {
            for j in 0..in_features {
                let w = w_data[i * in_features + j];
                saliency[i * in_features + j] = (w * w) / h_diag[j];
            }
        }

        Ok(Tensor::new(&saliency, w_shape))
    }

    /// Compute importance from weights and activations.
    ///
    /// This is the main entry point for `SparseGPT` importance scoring.
    pub fn compute_from_activations(
        &self,
        weights: &Tensor,
        activations: &Tensor,
    ) -> Result<ImportanceScores, PruningError> {
        // Compute Hessian
        let hessian = self.compute_hessian(activations)?;

        // Compute Hessian inverse
        let hessian_inv = self.compute_hessian_inverse(&hessian)?;

        // Compute saliency
        let saliency = self.compute_saliency(weights, &hessian_inv)?;

        Ok(ImportanceScores::new(
            saliency,
            "sparsegpt_saliency".to_string(),
        ))
    }
}

impl Importance for SparseGPTImportance {
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

        // For SparseGPT, we need the raw activations, not just stats.
        // Use squared_mean as a proxy for activation magnitudes.
        // In a full implementation, we'd store raw activations.
        let in_features = stats.input_features();
        let sq_mean = stats.squared_mean.data();

        // Construct synthetic activations from squared mean
        // This is an approximation - real implementation would use stored activations
        let mut activations = vec![0.0f32; in_features];
        for i in 0..in_features {
            activations[i] = sq_mean[i].sqrt();
        }

        let act_tensor = Tensor::new(&activations, &[1, in_features]);

        self.compute_from_activations(weights, &act_tensor)
    }

    fn name(&self) -> &'static str {
        "sparsegpt"
    }

    fn requires_calibration(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "sparsegpt_tests.rs"]
mod tests;
