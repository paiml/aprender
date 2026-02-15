
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
