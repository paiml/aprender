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
    /// * `weights` - Weight tensor of shape [`out_features`, `in_features`]
    /// * `activation_norms` - L2 norms of input activations [`in_features`]
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
mod tests {
    use super::*;
    use crate::pruning::calibration::ActivationStats;

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
    // FALSIFICATION: Wanda requires calibration (spec item 52)
    // ==========================================================================
    #[test]
    fn test_wanda_requires_calibration() {
        let wanda = WandaImportance::new("layer0");
        assert!(
            wanda.requires_calibration(),
            "WND-01 FALSIFIED: Wanda must require calibration"
        );
    }

    #[test]
    fn test_wanda_errors_without_calibration() {
        let module = MockLinear::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let wanda = WandaImportance::new("layer0");

        // No context provided
        let result = wanda.compute(&module, None);

        assert!(
            result.is_err(),
            "WND-02 FALSIFIED: Should error without calibration"
        );
        match result.unwrap_err() {
            PruningError::CalibrationRequired { method } => {
                assert_eq!(method, "wanda");
            }
            _ => panic!("WND-02 FALSIFIED: Expected CalibrationRequired error"),
        }
    }

    #[test]
    fn test_wanda_errors_missing_layer_stats() {
        let module = MockLinear::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let wanda = WandaImportance::new("nonexistent_layer");

        // Context exists but doesn't have stats for this layer
        let ctx = CalibrationContext::new("test".to_string());
        let result = wanda.compute(&module, Some(&ctx));

        assert!(
            result.is_err(),
            "WND-03 FALSIFIED: Should error on missing layer stats"
        );
        match result.unwrap_err() {
            PruningError::MissingActivationStats { layer } => {
                assert_eq!(layer, "nonexistent_layer");
            }
            _ => panic!("WND-03 FALSIFIED: Expected MissingActivationStats error"),
        }
    }

    // ==========================================================================
    // FALSIFICATION: Zero activations handled (spec item 9)
    // ==========================================================================
    #[test]
    fn test_wanda_zero_activations_handled() {
        let wanda = WandaImportance::new("layer0");

        // Weights [2, 2]
        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // Activation norms with a zero
        let norms = Tensor::new(&[0.0, 1.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);
        assert!(
            result.is_ok(),
            "WND-04 FALSIFIED: Should handle zero activations"
        );

        let importance = result.unwrap();
        for &v in importance.data() {
            assert!(
                !v.is_nan(),
                "WND-04 FALSIFIED: Zero activations should not produce NaN"
            );
            assert!(
                v.is_finite(),
                "WND-04 FALSIFIED: Zero activations should not produce Inf"
            );
        }
    }

    // ==========================================================================
    // FALSIFICATION: Importance is always non-negative (spec item 47)
    // ==========================================================================
    #[test]
    fn test_wanda_importance_non_negative() {
        let wanda = WandaImportance::new("layer0");

        // Mixed positive and negative weights
        let weights = Tensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2]);
        let norms = Tensor::new(&[1.0, 2.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);
        assert!(result.is_ok());

        let importance = result.unwrap();
        for &v in importance.data() {
            assert!(
                v >= 0.0,
                "WND-05 FALSIFIED: Wanda importance should be non-negative, got {}",
                v
            );
        }
    }

    // ==========================================================================
    // FALSIFICATION: Wanda formula correctness
    // ==========================================================================
    #[test]
    fn test_wanda_formula_correctness() {
        let wanda = WandaImportance::new("layer0").with_eps(0.0);

        // Weights [2, 2]: [[1, 2], [3, 4]]
        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // Activation norms: [4.0, 9.0] -> sqrt = [2.0, 3.0]
        let norms = Tensor::new(&[4.0, 9.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms).unwrap();
        let data = result.data();

        // importance[0,0] = |1| * sqrt(4) = 1 * 2 = 2.0
        // importance[0,1] = |2| * sqrt(9) = 2 * 3 = 6.0
        // importance[1,0] = |3| * sqrt(4) = 3 * 2 = 6.0
        // importance[1,1] = |4| * sqrt(9) = 4 * 3 = 12.0
        assert!(
            (data[0] - 2.0).abs() < 1e-6,
            "WND-06 FALSIFIED: importance[0,0] should be 2.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 6.0).abs() < 1e-6,
            "WND-06 FALSIFIED: importance[0,1] should be 6.0, got {}",
            data[1]
        );
        assert!(
            (data[2] - 6.0).abs() < 1e-6,
            "WND-06 FALSIFIED: importance[1,0] should be 6.0, got {}",
            data[2]
        );
        assert!(
            (data[3] - 12.0).abs() < 1e-6,
            "WND-06 FALSIFIED: importance[1,1] should be 12.0, got {}",
            data[3]
        );
    }

    #[test]
    fn test_wanda_with_negative_weights() {
        let wanda = WandaImportance::new("layer0").with_eps(0.0);

        // Negative weights
        let weights = Tensor::new(&[-2.0, -3.0], &[1, 2]);
        let norms = Tensor::new(&[1.0, 4.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms).unwrap();
        let data = result.data();

        // importance[0,0] = |-2| * sqrt(1) = 2 * 1 = 2.0
        // importance[0,1] = |-3| * sqrt(4) = 3 * 2 = 6.0
        assert!(
            (data[0] - 2.0).abs() < 1e-6,
            "WND-07 FALSIFIED: should use absolute weight"
        );
        assert!(
            (data[1] - 6.0).abs() < 1e-6,
            "WND-07 FALSIFIED: should use absolute weight"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Shape validation
    // ==========================================================================
    #[test]
    fn test_wanda_shape_mismatch() {
        let wanda = WandaImportance::new("layer0");

        // Weights [2, 3] but norms [2] (should be [3])
        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let norms = Tensor::new(&[1.0, 2.0], &[2]); // Wrong size!

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(
            result.is_err(),
            "WND-08 FALSIFIED: Should detect shape mismatch"
        );
        match result.unwrap_err() {
            PruningError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, vec![3]); // in_features = 3
                assert_eq!(got, vec![2]);
            }
            _ => panic!("WND-08 FALSIFIED: Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_wanda_1d_weights_rejected() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0, 3.0], &[3]); // 1D, not 2D
        let norms = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(
            result.is_err(),
            "WND-09 FALSIFIED: Should reject 1D weights"
        );
    }

    // ==========================================================================
    // FALSIFICATION: NaN/Inf detection (Jidoka)
    // ==========================================================================
    #[test]
    fn test_wanda_detects_nan_weights() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2]);
        let norms = Tensor::new(&[1.0, 2.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(
            result.is_err(),
            "WND-10 FALSIFIED: Should detect NaN weights"
        );
        match result.unwrap_err() {
            PruningError::NumericalInstability { method, details } => {
                assert_eq!(method, "wanda");
                assert!(details.contains("NaN"));
            }
            _ => panic!("WND-10 FALSIFIED: Expected NumericalInstability error"),
        }
    }

    #[test]
    fn test_wanda_detects_nan_norms() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let norms = Tensor::new(&[1.0, f32::NAN], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(result.is_err(), "WND-11 FALSIFIED: Should detect NaN norms");
    }

    #[test]
    fn test_wanda_detects_inf_weights() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0, f32::INFINITY, 3.0, 4.0], &[2, 2]);
        let norms = Tensor::new(&[1.0, 2.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(
            result.is_err(),
            "WND-12 FALSIFIED: Should detect Inf weights"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Integration with Module
    // ==========================================================================
    #[test]
    fn test_wanda_compute_via_trait() {
        // Create module
        let module = MockLinear::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        // Create calibration context with stats
        let mut ctx = CalibrationContext::new("test".to_string());
        let mut stats = ActivationStats::new(2);
        // Update stats to have some norms
        stats.update(&Tensor::new(&[2.0, 3.0], &[2]));
        ctx.add_layer_stats("layer0".to_string(), stats);

        // Compute via trait
        let wanda = WandaImportance::new("layer0");
        let result = wanda.compute(&module, Some(&ctx));

        assert!(
            result.is_ok(),
            "WND-13 FALSIFIED: Should compute successfully"
        );
        let scores = result.unwrap();
        assert_eq!(scores.method, "wanda");
        assert_eq!(scores.shape(), &[2, 2]);
    }

    // ==========================================================================
    // FALSIFICATION: Name method
    // ==========================================================================
    #[test]
    fn test_wanda_name() {
        let wanda = WandaImportance::new("layer0");
        assert_eq!(wanda.name(), "wanda", "WND-14 FALSIFIED: wrong name");
    }

    // ==========================================================================
    // FALSIFICATION: Getters
    // ==========================================================================
    #[test]
    fn test_wanda_layer_name_getter() {
        let wanda = WandaImportance::new("model.layer.0.mlp");
        assert_eq!(wanda.layer_name(), "model.layer.0.mlp");
    }

    #[test]
    fn test_wanda_pattern_getter() {
        let wanda = WandaImportance::new("layer0");
        assert_eq!(wanda.pattern(), None);

        let wanda = wanda.with_pattern(SparsityPattern::NM { n: 2, m: 4 });
        assert_eq!(wanda.pattern(), Some(SparsityPattern::NM { n: 2, m: 4 }));
    }

    // ==========================================================================
    // FALSIFICATION: Builder pattern
    // ==========================================================================
    #[test]
    fn test_wanda_with_pattern() {
        let wanda = WandaImportance::new("layer0").with_pattern(SparsityPattern::NM { n: 2, m: 4 });

        assert_eq!(wanda.pattern(), Some(SparsityPattern::NM { n: 2, m: 4 }));
    }

    #[test]
    fn test_wanda_with_eps() {
        let wanda = WandaImportance::new("layer0").with_eps(1e-10);

        // Verify it's used by testing with zero activations
        let weights = Tensor::new(&[1.0], &[1, 1]);
        let norms = Tensor::new(&[0.0], &[1]);

        let result = wanda.compute_from_tensors(&weights, &norms).unwrap();
        let data = result.data();

        // With eps=1e-10, sqrt(eps) = 1e-5
        // importance = |1.0| * 1e-5 = 1e-5
        assert!(
            (data[0] - 1e-5).abs() < 1e-8,
            "WND-15 FALSIFIED: custom eps should be used"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug
    // ==========================================================================
    #[test]
    fn test_wanda_clone() {
        let orig = WandaImportance::new("layer0").with_pattern(SparsityPattern::NM { n: 2, m: 4 });
        let cloned = orig.clone();

        assert_eq!(orig.layer_name(), cloned.layer_name());
        assert_eq!(orig.pattern(), cloned.pattern());
    }

    #[test]
    fn test_wanda_debug() {
        let wanda = WandaImportance::new("layer0");
        let debug = format!("{:?}", wanda);
        assert!(debug.contains("WandaImportance"));
        assert!(debug.contains("layer0"));
    }

    // ==========================================================================
    // FALSIFICATION: Empty module
    // ==========================================================================
    #[test]
    fn test_wanda_empty_module() {
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
        let mut ctx = CalibrationContext::new("test".to_string());
        ctx.add_layer_stats("layer0".to_string(), ActivationStats::new(10));

        let wanda = WandaImportance::new("layer0");
        let result = wanda.compute(&module, Some(&ctx));

        assert!(
            result.is_err(),
            "WND-16 FALSIFIED: empty module should error"
        );
        match result.unwrap_err() {
            PruningError::NoParameters { .. } => (),
            _ => panic!("WND-16 FALSIFIED: Expected NoParameters error"),
        }
    }

    // ==========================================================================
    // FALSIFICATION: Shape preserved
    // ==========================================================================
    #[test]
    fn test_wanda_preserves_shape() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0; 12], &[3, 4]);
        let norms = Tensor::new(&[1.0; 4], &[4]);

        let result = wanda.compute_from_tensors(&weights, &norms).unwrap();

        assert_eq!(
            result.shape(),
            &[3, 4],
            "WND-17 FALSIFIED: shape should be preserved"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Inf detection for norms
    // ==========================================================================
    #[test]
    fn test_wanda_detects_inf_norms() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let norms = Tensor::new(&[1.0, f32::INFINITY], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(result.is_err(), "WND-18 FALSIFIED: Should detect Inf norms");
        match result.unwrap_err() {
            PruningError::NumericalInstability { method, details } => {
                assert_eq!(method, "wanda");
                assert!(details.contains("Inf"));
            }
            _ => panic!("WND-18 FALSIFIED: Expected NumericalInstability error"),
        }
    }

    // ==========================================================================
    // FALSIFICATION: Empty norm tensor
    // ==========================================================================
    #[test]
    fn test_wanda_empty_norms() {
        let wanda = WandaImportance::new("layer0");

        let weights = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let norms = Tensor::new(&[], &[0]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(
            result.is_err(),
            "WND-19 FALSIFIED: Should error on empty norms"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Negative activation norms
    // ==========================================================================
    #[test]
    fn test_wanda_negative_norms() {
        let wanda = WandaImportance::new("layer0");

        // Negative norms should be handled gracefully
        let weights = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let norms = Tensor::new(&[-1.0, 2.0], &[2]);

        let result = wanda.compute_from_tensors(&weights, &norms);

        assert!(
            result.is_ok(),
            "WND-20 FALSIFIED: Should handle negative norms"
        );

        let importance = result.unwrap();
        for &v in importance.data() {
            assert!(
                v.is_finite(),
                "WND-20 FALSIFIED: Negative norms should not produce non-finite"
            );
            assert!(
                v >= 0.0,
                "WND-20 FALSIFIED: Importance should be non-negative"
            );
        }
    }
}
