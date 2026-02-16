pub(crate) use super::*;

// Mock module for testing
pub(super) struct MockModule {
    weights: Tensor,
}

impl MockModule {
    fn new(data: &[f32], shape: &[usize]) -> Self {
        Self {
            weights: Tensor::new(data, shape),
        }
    }
}

impl Module for MockModule {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights]
    }
}

// Empty module for testing edge case
pub(super) struct EmptyModule;

impl Module for EmptyModule {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ==========================================================================
// FALSIFICATION: L1 importance equals absolute weight (spec item 31)
// ==========================================================================
#[test]
fn test_magnitude_l1_equals_abs_weight() {
    let importance = MagnitudeImportance::l1();
    let weights = Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[4]);

    let scores = importance.compute_for_weights(&weights).unwrap();
    let expected = &[1.0, 2.0, 3.0, 4.0]; // |w|

    assert_eq!(
        scores.data(),
        expected,
        "MAG-01 FALSIFIED: L1 importance should equal |w|"
    );
}

#[test]
fn test_magnitude_l1_zero_weights() {
    let importance = MagnitudeImportance::l1();
    let weights = Tensor::new(&[0.0, 0.0, 0.0], &[3]);

    let scores = importance.compute_for_weights(&weights).unwrap();
    let expected = &[0.0, 0.0, 0.0];

    assert_eq!(
        scores.data(),
        expected,
        "MAG-02 FALSIFIED: L1 of zeros should be zeros"
    );
}

// ==========================================================================
// FALSIFICATION: L2 importance equals weight squared (spec item 32)
// ==========================================================================
#[test]
fn test_magnitude_l2_equals_weight_squared() {
    let importance = MagnitudeImportance::l2();
    let weights = Tensor::new(&[1.0, -2.0, 3.0, -4.0], &[4]);

    let scores = importance.compute_for_weights(&weights).unwrap();
    let expected = &[1.0, 4.0, 9.0, 16.0]; // w²

    assert_eq!(
        scores.data(),
        expected,
        "MAG-03 FALSIFIED: L2 importance should equal w²"
    );
}

#[test]
fn test_magnitude_l2_zero_weights() {
    let importance = MagnitudeImportance::l2();
    let weights = Tensor::new(&[0.0, 0.0, 0.0], &[3]);

    let scores = importance.compute_for_weights(&weights).unwrap();

    for &v in scores.data() {
        assert!(
            v == 0.0,
            "MAG-04 FALSIFIED: L2 of zeros should be zeros, got {}",
            v
        );
    }
}

// ==========================================================================
// FALSIFICATION: Importance is always non-negative (spec item 47)
// ==========================================================================
#[test]
fn test_magnitude_importance_always_non_negative() {
    let l1 = MagnitudeImportance::l1();
    let l2 = MagnitudeImportance::l2();

    // Test with various negative weights
    let weights = Tensor::new(&[-5.0, -0.001, -100.0, -0.0], &[4]);

    let l1_scores = l1.compute_for_weights(&weights).unwrap();
    let l2_scores = l2.compute_for_weights(&weights).unwrap();

    for &v in l1_scores.data() {
        assert!(
            v >= 0.0,
            "MAG-05 FALSIFIED: L1 importance should be non-negative, got {}",
            v
        );
    }

    for &v in l2_scores.data() {
        assert!(
            v >= 0.0,
            "MAG-05 FALSIFIED: L2 importance should be non-negative, got {}",
            v
        );
    }
}

// ==========================================================================
// FALSIFICATION: All zeros produces no NaN (spec item 1)
// ==========================================================================
#[test]
fn test_magnitude_all_zeros_no_nan() {
    let l1 = MagnitudeImportance::l1();
    let l2 = MagnitudeImportance::l2();

    let weights = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);

    let l1_scores = l1.compute_for_weights(&weights).unwrap();
    let l2_scores = l2.compute_for_weights(&weights).unwrap();

    for &v in l1_scores.data() {
        assert!(
            !v.is_nan(),
            "MAG-06 FALSIFIED: L1 should not produce NaN on zeros"
        );
    }

    for &v in l2_scores.data() {
        assert!(
            !v.is_nan(),
            "MAG-06 FALSIFIED: L2 should not produce NaN on zeros"
        );
    }
}

// ==========================================================================
// FALSIFICATION: NaN weights detected (Jidoka)
// ==========================================================================
#[test]
fn test_magnitude_detects_nan_weights() {
    let importance = MagnitudeImportance::l2();
    let weights = Tensor::new(&[1.0, f32::NAN, 3.0], &[3]);

    let result = importance.compute_for_weights(&weights);

    assert!(
        result.is_err(),
        "MAG-07 FALSIFIED: NaN weights should be detected"
    );
    match result.unwrap_err() {
        PruningError::NumericalInstability { method, details } => {
            assert_eq!(method, "magnitude_l2");
            assert!(details.contains("NaN"));
        }
        _ => panic!("MAG-07 FALSIFIED: Expected NumericalInstability error"),
    }
}

#[test]
fn test_magnitude_detects_inf_weights() {
    let importance = MagnitudeImportance::l1();
    let weights = Tensor::new(&[1.0, f32::INFINITY, 3.0], &[3]);

    let result = importance.compute_for_weights(&weights);

    assert!(
        result.is_err(),
        "MAG-08 FALSIFIED: Inf weights should be detected"
    );
    match result.unwrap_err() {
        PruningError::NumericalInstability { method, details } => {
            assert_eq!(method, "magnitude_l1");
            assert!(details.contains("Inf"));
        }
        _ => panic!("MAG-08 FALSIFIED: Expected NumericalInstability error"),
    }
}

#[test]
fn test_magnitude_detects_neg_inf_weights() {
    let importance = MagnitudeImportance::l2();
    let weights = Tensor::new(&[1.0, f32::NEG_INFINITY, 3.0], &[3]);

    let result = importance.compute_for_weights(&weights);

    assert!(
        result.is_err(),
        "MAG-09 FALSIFIED: -Inf weights should be detected"
    );
}

// ==========================================================================
// FALSIFICATION: Does not require calibration
// ==========================================================================
#[test]
fn test_magnitude_does_not_require_calibration() {
    let l1 = MagnitudeImportance::l1();
    let l2 = MagnitudeImportance::l2();

    assert!(
        !l1.requires_calibration(),
        "MAG-10 FALSIFIED: L1 should not require calibration"
    );
    assert!(
        !l2.requires_calibration(),
        "MAG-10 FALSIFIED: L2 should not require calibration"
    );
}

// ==========================================================================
// FALSIFICATION: Module integration via Importance trait
// ==========================================================================
#[test]
fn test_magnitude_compute_via_trait() {
    let module = MockModule::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2]);
    let importance = MagnitudeImportance::l2();

    let scores = importance.compute(&module, None).unwrap();

    assert_eq!(scores.method, "magnitude_l2");
    assert_eq!(scores.shape(), &[2, 2]);
    assert_eq!(scores.values.data(), &[1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_magnitude_empty_module_error() {
    let module = EmptyModule;
    let importance = MagnitudeImportance::l1();

    let result = importance.compute(&module, None);

    assert!(
        result.is_err(),
        "MAG-11 FALSIFIED: empty module should error"
    );
    match result.unwrap_err() {
        PruningError::NoParameters { .. } => (),
        _ => panic!("MAG-11 FALSIFIED: Expected NoParameters error"),
    }
}

// ==========================================================================
// FALSIFICATION: Name method
// ==========================================================================
#[test]
fn test_magnitude_name() {
    let l1 = MagnitudeImportance::l1();
    let l2 = MagnitudeImportance::l2();

    assert_eq!(l1.name(), "magnitude_l1", "MAG-12 FALSIFIED: wrong L1 name");
    assert_eq!(l2.name(), "magnitude_l2", "MAG-12 FALSIFIED: wrong L2 name");
}

// ==========================================================================
// FALSIFICATION: Norm getter
// ==========================================================================
#[test]
fn test_magnitude_norm_getter() {
    let l1 = MagnitudeImportance::l1();
    let l2 = MagnitudeImportance::l2();

    assert_eq!(l1.norm(), NormType::L1);
    assert_eq!(l2.norm(), NormType::L2);
}

// ==========================================================================
// FALSIFICATION: with_norm constructor
// ==========================================================================
#[test]
fn test_magnitude_with_norm() {
    let l1 = MagnitudeImportance::with_norm(NormType::L1);
    let l2 = MagnitudeImportance::with_norm(NormType::L2);

    assert_eq!(l1.norm(), NormType::L1);
    assert_eq!(l2.norm(), NormType::L2);
}

// ==========================================================================
// FALSIFICATION: Shape preserved
// ==========================================================================
#[test]
fn test_magnitude_preserves_shape() {
    let importance = MagnitudeImportance::l2();

    // 2D
    let weights_2d = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let scores_2d = importance.compute_for_weights(&weights_2d).unwrap();
    assert_eq!(
        scores_2d.shape(),
        &[2, 3],
        "MAG-13 FALSIFIED: 2D shape should be preserved"
    );

    // 1D
    let weights_1d = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let scores_1d = importance.compute_for_weights(&weights_1d).unwrap();
    assert_eq!(
        scores_1d.shape(),
        &[3],
        "MAG-13 FALSIFIED: 1D shape should be preserved"
    );
}

// ==========================================================================
// FALSIFICATION: Very small weights handled correctly
// ==========================================================================
#[test]
fn test_magnitude_small_weights() {
    let importance = MagnitudeImportance::l2();
    let weights = Tensor::new(&[1e-10, 1e-20, 1e-30], &[3]);

    let scores = importance.compute_for_weights(&weights).unwrap();

    // L2 of very small numbers should still be >= 0 and not NaN
    for &v in scores.data() {
        assert!(
            v >= 0.0,
            "MAG-14 FALSIFIED: small weight score should be >= 0"
        );
        assert!(
            !v.is_nan(),
            "MAG-14 FALSIFIED: small weight score should not be NaN"
        );
    }
}

// ==========================================================================
// FALSIFICATION: Large weights handled correctly
// ==========================================================================
#[test]
fn test_magnitude_large_weights() {
    let importance = MagnitudeImportance::l1();
    let weights = Tensor::new(&[1e10, 1e20, 1e30], &[3]);

    let scores = importance.compute_for_weights(&weights).unwrap();

    assert!(
        (scores.data()[0] - 1e10).abs() < 1e5,
        "MAG-15 FALSIFIED: large weight L1 should be preserved"
    );
}

// ==========================================================================
// FALSIFICATION: Clone and Debug
// ==========================================================================
#[test]
fn test_magnitude_clone() {
    let orig = MagnitudeImportance::l2();
    let cloned = orig.clone();

    assert_eq!(orig.norm(), cloned.norm());
}

#[test]
fn test_magnitude_debug() {
    let importance = MagnitudeImportance::l1();
    let debug = format!("{:?}", importance);
    assert!(debug.contains("MagnitudeImportance"));
}

#[test]
fn test_norm_type_eq() {
    assert_eq!(NormType::L1, NormType::L1);
    assert_eq!(NormType::L2, NormType::L2);
    assert_ne!(NormType::L1, NormType::L2);
}

// ==========================================================================
// FALSIFICATION: NormType Copy and Clone
// ==========================================================================
#[test]
fn test_norm_type_copy() {
    let norm = NormType::L1;
    let copied = norm;
    assert_eq!(norm, copied);
}

#[test]
fn test_norm_type_clone() {
    let norm = NormType::L2;
    let cloned = norm.clone();
    assert_eq!(norm, cloned);
}

#[test]
fn test_norm_type_debug() {
    let l1 = NormType::L1;
    let l2 = NormType::L2;
    let debug_l1 = format!("{:?}", l1);
    let debug_l2 = format!("{:?}", l2);
    assert!(debug_l1.contains("L1"));
    assert!(debug_l2.contains("L2"));
}

// ==========================================================================
// FALSIFICATION: L1 formula with very precise values
// ==========================================================================
#[test]
fn test_magnitude_l1_precise() {
    let importance = MagnitudeImportance::l1();
    let weights = Tensor::new(&[-0.5, 0.5], &[2]);

    let scores = importance.compute_for_weights(&weights).unwrap();

    assert!(
        (scores.data()[0] - 0.5).abs() < 1e-6,
        "MAG-16 FALSIFIED: L1 of -0.5 should be 0.5"
    );
    assert!(
        (scores.data()[1] - 0.5).abs() < 1e-6,
        "MAG-16 FALSIFIED: L1 of 0.5 should be 0.5"
    );
}

// ==========================================================================
// FALSIFICATION: L2 formula with very precise values
// ==========================================================================
#[test]
fn test_magnitude_l2_precise() {
    let importance = MagnitudeImportance::l2();
    let weights = Tensor::new(&[0.5, -0.5, 2.0, -2.0], &[4]);

    let scores = importance.compute_for_weights(&weights).unwrap();

    assert!(
        (scores.data()[0] - 0.25).abs() < 1e-6,
        "MAG-17 FALSIFIED: L2 of 0.5 should be 0.25"
    );
    assert!(
        (scores.data()[1] - 0.25).abs() < 1e-6,
        "MAG-17 FALSIFIED: L2 of -0.5 should be 0.25"
    );
    assert!(
        (scores.data()[2] - 4.0).abs() < 1e-6,
        "MAG-17 FALSIFIED: L2 of 2.0 should be 4.0"
    );
    assert!(
        (scores.data()[3] - 4.0).abs() < 1e-6,
        "MAG-17 FALSIFIED: L2 of -2.0 should be 4.0"
    );
}
