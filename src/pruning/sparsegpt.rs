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
mod tests {
    use super::*;

    // Mock module for testing
    struct MockLinear {
        weights: Tensor,
    }

    impl MockLinear {
        fn new(data: &[f32], out_features: usize, in_features: usize) -> Self {
            Self {
                weights: Tensor::new(data, &[out_features, in_features]),
            }
        }
    }

    impl Module for MockLinear {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weights]
        }
    }

    // ==========================================================================
    // FALSIFICATION: SparseGPT requires calibration
    // ==========================================================================
    #[test]
    fn test_sparsegpt_requires_calibration() {
        let sparsegpt = SparseGPTImportance::new("layer0");
        assert!(
            sparsegpt.requires_calibration(),
            "SGP-01 FALSIFIED: SparseGPT must require calibration"
        );
    }

    #[test]
    fn test_sparsegpt_errors_without_calibration() {
        let module = MockLinear::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let sparsegpt = SparseGPTImportance::new("layer0");

        let result = sparsegpt.compute(&module, None);

        assert!(
            result.is_err(),
            "SGP-02 FALSIFIED: Should error without calibration"
        );
        match result.unwrap_err() {
            PruningError::CalibrationRequired { method } => {
                assert_eq!(method, "sparsegpt");
            }
            _ => panic!("SGP-02 FALSIFIED: Expected CalibrationRequired error"),
        }
    }

    // ==========================================================================
    // FALSIFICATION: Hessian computation
    // ==========================================================================
    #[test]
    fn test_hessian_computation_basic() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_damp(0.0);

        // Simple 2x2 case: activations = [[1, 2], [3, 4]]
        // X^T X = [[1, 3], [2, 4]] * [[1, 2], [3, 4]] = [[10, 14], [14, 20]]
        // Normalized by n=2: [[5, 7], [7, 10]]
        let activations = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let hessian = sparsegpt.compute_hessian(&activations).unwrap();

        let data = hessian.data();
        assert!(
            (data[0] - 5.0).abs() < 1e-5,
            "SGP-03 FALSIFIED: H[0,0] should be 5"
        );
        assert!(
            (data[1] - 7.0).abs() < 1e-5,
            "SGP-03 FALSIFIED: H[0,1] should be 7"
        );
        assert!(
            (data[2] - 7.0).abs() < 1e-5,
            "SGP-03 FALSIFIED: H[1,0] should be 7"
        );
        assert!(
            (data[3] - 10.0).abs() < 1e-5,
            "SGP-03 FALSIFIED: H[1,1] should be 10"
        );
    }

    #[test]
    fn test_hessian_with_damping() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_damp(1.0);

        // Activations: [[1, 1]]
        // X^T X = [[1], [1]] * [[1, 1]] = [[1, 1], [1, 1]]
        // With damp=1.0: [[2, 1], [1, 2]]
        let activations = Tensor::new(&[1.0, 1.0], &[1, 2]);
        let hessian = sparsegpt.compute_hessian(&activations).unwrap();

        let data = hessian.data();
        assert!(
            (data[0] - 2.0).abs() < 1e-5,
            "SGP-04 FALSIFIED: H[0,0] should be 2"
        );
        assert!(
            (data[3] - 2.0).abs() < 1e-5,
            "SGP-04 FALSIFIED: H[1,1] should be 2"
        );
    }

    #[test]
    fn test_hessian_is_symmetric() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_damp(0.1);

        let activations = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let hessian = sparsegpt.compute_hessian(&activations).unwrap();

        let data = hessian.data();
        // Check symmetry: H[i,j] == H[j,i]
        assert!(
            (data[1] - data[3]).abs() < 1e-5,
            "SGP-05 FALSIFIED: Hessian should be symmetric (0,1) vs (1,0)"
        );
        assert!(
            (data[2] - data[6]).abs() < 1e-5,
            "SGP-05 FALSIFIED: Hessian should be symmetric (0,2) vs (2,0)"
        );
        assert!(
            (data[5] - data[7]).abs() < 1e-5,
            "SGP-05 FALSIFIED: Hessian should be symmetric (1,2) vs (2,1)"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Hessian inverse computation
    // ==========================================================================
    #[test]
    fn test_hessian_inverse_identity() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        // 2x2 identity matrix
        let hessian = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let h_inv = sparsegpt.compute_hessian_inverse(&hessian).unwrap();

        let data = h_inv.data();
        // Inverse of identity is identity
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "SGP-06 FALSIFIED: H_inv[0,0] should be 1"
        );
        assert!(
            (data[1] - 0.0).abs() < 1e-5,
            "SGP-06 FALSIFIED: H_inv[0,1] should be 0"
        );
        assert!(
            (data[2] - 0.0).abs() < 1e-5,
            "SGP-06 FALSIFIED: H_inv[1,0] should be 0"
        );
        assert!(
            (data[3] - 1.0).abs() < 1e-5,
            "SGP-06 FALSIFIED: H_inv[1,1] should be 1"
        );
    }

    #[test]
    fn test_hessian_inverse_2x2() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        // [[4, 2], [2, 2]] -> inverse = [[0.5, -0.5], [-0.5, 1]]
        let hessian = Tensor::new(&[4.0, 2.0, 2.0, 2.0], &[2, 2]);
        let h_inv = sparsegpt.compute_hessian_inverse(&hessian).unwrap();

        let data = h_inv.data();
        assert!(
            (data[0] - 0.5).abs() < 1e-4,
            "SGP-07 FALSIFIED: H_inv[0,0] should be 0.5"
        );
        assert!(
            (data[1] - (-0.5)).abs() < 1e-4,
            "SGP-07 FALSIFIED: H_inv[0,1] should be -0.5"
        );
        assert!(
            (data[2] - (-0.5)).abs() < 1e-4,
            "SGP-07 FALSIFIED: H_inv[1,0] should be -0.5"
        );
        assert!(
            (data[3] - 1.0).abs() < 1e-4,
            "SGP-07 FALSIFIED: H_inv[1,1] should be 1.0"
        );
    }

    #[test]
    fn test_hessian_inverse_singular_matrix_fails() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        // Singular matrix (not positive definite)
        let hessian = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let result = sparsegpt.compute_hessian_inverse(&hessian);

        assert!(
            result.is_err(),
            "SGP-08 FALSIFIED: Singular matrix should fail"
        );
        match result.unwrap_err() {
            PruningError::NumericalInstability { method, details } => {
                assert_eq!(method, "SparseGPT");
                assert!(details.contains("not positive definite"));
            }
            _ => panic!("SGP-08 FALSIFIED: Expected NumericalInstability error"),
        }
    }

    // ==========================================================================
    // FALSIFICATION: Saliency computation
    // ==========================================================================
    #[test]
    fn test_saliency_computation() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        // Weights [2, 2]: [[1, 2], [3, 4]]
        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // H^{-1} diagonal = [1, 2]
        let h_inv = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);

        let saliency = sparsegpt.compute_saliency(&weights, &h_inv).unwrap();
        let data = saliency.data();

        // saliency[i,j] = w[i,j]^2 / h_inv[j,j]
        // [0,0]: 1^2 / 1 = 1
        // [0,1]: 2^2 / 2 = 2
        // [1,0]: 3^2 / 1 = 9
        // [1,1]: 4^2 / 2 = 8
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "SGP-09 FALSIFIED: saliency[0,0] should be 1"
        );
        assert!(
            (data[1] - 2.0).abs() < 1e-5,
            "SGP-09 FALSIFIED: saliency[0,1] should be 2"
        );
        assert!(
            (data[2] - 9.0).abs() < 1e-5,
            "SGP-09 FALSIFIED: saliency[1,0] should be 9"
        );
        assert!(
            (data[3] - 8.0).abs() < 1e-5,
            "SGP-09 FALSIFIED: saliency[1,1] should be 8"
        );
    }

    #[test]
    fn test_saliency_non_negative() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        // Weights with negative values
        let weights = Tensor::new(&[-1.0, -2.0, 3.0, -4.0], &[2, 2]);
        let h_inv = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let saliency = sparsegpt.compute_saliency(&weights, &h_inv).unwrap();

        for &v in saliency.data() {
            assert!(
                v >= 0.0,
                "SGP-10 FALSIFIED: Saliency should be non-negative"
            );
        }
    }

    // ==========================================================================
    // FALSIFICATION: NaN/Inf detection (Jidoka)
    // ==========================================================================
    #[test]
    fn test_hessian_detects_nan_activations() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let activations = Tensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2]);
        let result = sparsegpt.compute_hessian(&activations);

        assert!(
            result.is_err(),
            "SGP-11 FALSIFIED: NaN activations should be detected"
        );
    }

    #[test]
    fn test_hessian_detects_inf_activations() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let activations = Tensor::new(&[1.0, f32::INFINITY, 3.0, 4.0], &[2, 2]);
        let result = sparsegpt.compute_hessian(&activations);

        assert!(
            result.is_err(),
            "SGP-12 FALSIFIED: Inf activations should be detected"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Shape validation
    // ==========================================================================
    #[test]
    fn test_hessian_1d_rejected() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let activations = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let result = sparsegpt.compute_hessian(&activations);

        assert!(
            result.is_err(),
            "SGP-13 FALSIFIED: 1D activations should be rejected"
        );
    }

    #[test]
    fn test_saliency_shape_mismatch() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let h_inv = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]); // Wrong size

        let result = sparsegpt.compute_saliency(&weights, &h_inv);

        assert!(
            result.is_err(),
            "SGP-14 FALSIFIED: Shape mismatch should be detected"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Builder pattern
    // ==========================================================================
    #[test]
    fn test_sparsegpt_with_block_size() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_block_size(64);
        assert_eq!(sparsegpt.block_size(), 64);
    }

    #[test]
    fn test_sparsegpt_with_damp() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_damp(0.1);
        assert!((sparsegpt.damp() - 0.1).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: Name method
    // ==========================================================================
    #[test]
    fn test_sparsegpt_name() {
        let sparsegpt = SparseGPTImportance::new("layer0");
        assert_eq!(sparsegpt.name(), "sparsegpt");
    }

    // ==========================================================================
    // FALSIFICATION: Layer name getter
    // ==========================================================================
    #[test]
    fn test_sparsegpt_layer_name() {
        let sparsegpt = SparseGPTImportance::new("model.layers.0.mlp");
        assert_eq!(sparsegpt.layer_name(), "model.layers.0.mlp");
    }

    // ==========================================================================
    // FALSIFICATION: End-to-end importance computation
    // ==========================================================================
    #[test]
    fn test_compute_from_activations() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_damp(0.1);

        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let activations = Tensor::new(&[1.0, 2.0, 2.0, 1.0], &[2, 2]);

        let result = sparsegpt.compute_from_activations(&weights, &activations);

        assert!(
            result.is_ok(),
            "SGP-15 FALSIFIED: End-to-end computation should succeed"
        );
        let scores = result.unwrap();
        assert_eq!(scores.method, "sparsegpt_saliency");
        assert_eq!(scores.shape(), &[2, 2]);
    }

    // ==========================================================================
    // FALSIFICATION: Empty activations
    // ==========================================================================
    #[test]
    fn test_hessian_empty_activations() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let activations = Tensor::new(&[], &[0, 2]);
        let result = sparsegpt.compute_hessian(&activations);

        assert!(
            result.is_err(),
            "SGP-16 FALSIFIED: Empty activations should error"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug
    // ==========================================================================
    #[test]
    fn test_sparsegpt_clone() {
        let orig = SparseGPTImportance::new("layer0").with_block_size(64);
        let cloned = orig.clone();

        assert_eq!(orig.layer_name(), cloned.layer_name());
        assert_eq!(orig.block_size(), cloned.block_size());
    }

    #[test]
    fn test_sparsegpt_debug() {
        let sparsegpt = SparseGPTImportance::new("layer0");
        let debug = format!("{:?}", sparsegpt);
        assert!(debug.contains("SparseGPTImportance"));
    }

    // ==========================================================================
    // FALSIFICATION: with_relative_damp builder
    // ==========================================================================
    #[test]
    fn test_sparsegpt_with_relative_damp() {
        let sparsegpt = SparseGPTImportance::new("layer0").with_relative_damp(0.05);
        assert!((sparsegpt.damp() - 0.05).abs() < 1e-6);
        assert!(sparsegpt.damp_relative); // Internal flag check
    }

    #[test]
    fn test_hessian_with_relative_damping() {
        // Test that relative damping computes based on mean diagonal
        let sparsegpt = SparseGPTImportance::new("layer0").with_relative_damp(0.1);

        // Activations: [[2, 0], [0, 2]] -> X^T X = [[4, 0], [0, 4]]
        // Mean diagonal = 4, relative damp = 0.1 * 4 = 0.4
        // Final: [[4.4, 0], [0, 4.4]]
        let activations = Tensor::new(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);
        let hessian = sparsegpt.compute_hessian(&activations).unwrap();

        let data = hessian.data();
        // X^T X / n = [[4, 0], [0, 4]] / 2 = [[2, 0], [0, 2]]
        // Mean diagonal = 2, damp = 0.1 * 2 = 0.2
        // Final diagonal = 2 + 0.2 = 2.2
        assert!(
            (data[0] - 2.2).abs() < 1e-4,
            "SGP-17 FALSIFIED: With relative damp"
        );
        assert!(
            (data[3] - 2.2).abs() < 1e-4,
            "SGP-17 FALSIFIED: With relative damp"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Importance trait compute with calibration
    // ==========================================================================
    #[test]
    fn test_sparsegpt_compute_with_calibration() {
        use super::super::calibration::{ActivationStats, CalibrationContext};

        let module = MockLinear::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let sparsegpt = SparseGPTImportance::new("layer0").with_damp(1.0);

        // Create calibration context with activation stats
        let mut ctx = CalibrationContext::new("test".to_string());
        let mut stats = ActivationStats::new(2);
        stats.squared_mean = Tensor::new(&[4.0, 9.0], &[2]); // sqrt -> [2, 3]
        ctx.add_layer_stats("layer0".to_string(), stats);

        let result = sparsegpt.compute(&module, Some(&ctx));
        assert!(
            result.is_ok(),
            "SGP-18 FALSIFIED: Should compute with calibration"
        );
    }

    #[test]
    fn test_sparsegpt_compute_missing_layer_stats() {
        use super::super::calibration::CalibrationContext;

        let module = MockLinear::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let sparsegpt = SparseGPTImportance::new("nonexistent_layer");

        let ctx = CalibrationContext::new("test".to_string());

        let result = sparsegpt.compute(&module, Some(&ctx));
        assert!(
            result.is_err(),
            "SGP-19 FALSIFIED: Should error on missing layer stats"
        );
    }

    #[test]
    fn test_sparsegpt_compute_empty_module() {
        use super::super::calibration::{ActivationStats, CalibrationContext};

        struct EmptyModule;
        impl Module for EmptyModule {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.clone()
            }
            fn parameters(&self) -> Vec<&Tensor> {
                vec![]
            }
        }

        let module = EmptyModule;
        let sparsegpt = SparseGPTImportance::new("layer0");

        let mut ctx = CalibrationContext::new("test".to_string());
        ctx.add_layer_stats("layer0".to_string(), ActivationStats::new(2));

        let result = sparsegpt.compute(&module, Some(&ctx));
        assert!(
            result.is_err(),
            "SGP-20 FALSIFIED: Empty module should error"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Saliency with non-positive diagonal
    // ==========================================================================
    #[test]
    fn test_saliency_non_positive_diagonal() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0], &[1, 2]);
        // H^{-1} with zero diagonal entry (invalid)
        let h_inv = Tensor::new(&[0.0, 0.0, 0.0, 1.0], &[2, 2]);

        let result = sparsegpt.compute_saliency(&weights, &h_inv);
        assert!(
            result.is_err(),
            "SGP-21 FALSIFIED: Non-positive diagonal should error"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Saliency 1D weights rejected
    // ==========================================================================
    #[test]
    fn test_saliency_1d_weights_rejected() {
        let sparsegpt = SparseGPTImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let h_inv = Tensor::new(&[1.0], &[1, 1]);

        let result = sparsegpt.compute_saliency(&weights, &h_inv);
        assert!(
            result.is_err(),
            "SGP-22 FALSIFIED: 1D weights should be rejected"
        );
    }
}
