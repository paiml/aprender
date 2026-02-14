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
