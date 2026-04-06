//! Backend consistency and equivalence integration tests.

use proptest::prelude::*;
use trueno::{Backend, Vector};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))] // Fewer cases for backend tests

    /// Integration test: Backend consistency
    #[test]
    fn integration_backend_consistency(
        data in prop::collection::vec(-50.0f32..50.0, 50..100)
    ) {
        // Test that Auto backend produces valid results
        let v_auto = Vector::from_slice(&data);

        // Scalar backend as reference
        let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);

        // Test a few operations for consistency
        let sum_auto = v_auto.sum()?;
        let sum_scalar = v_scalar.sum()?;
        prop_assert!((sum_auto - sum_scalar).abs() < 1e-3 * sum_scalar.abs().max(1.0));

        let max_auto = v_auto.max()?;
        let max_scalar = v_scalar.max()?;
        prop_assert_eq!(max_auto, max_scalar);
    }

    /// Integration test: Backend equivalence for activation functions
    #[test]
    fn integration_backend_equivalence_activations(
        data in prop::collection::vec(-10.0f32..10.0, 50..100)
    ) {
        // Scalar backend as reference
        let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
        let v_auto = Vector::from_slice(&data);

        // ReLU - should produce identical results across backends
        let relu_scalar = v_scalar.relu()?;
        let relu_auto = v_auto.relu()?;
        for (s, a) in relu_scalar.as_slice().iter().zip(relu_auto.as_slice().iter()) {
            prop_assert!((s - a).abs() < 1e-5, "ReLU mismatch: scalar={}, auto={}", s, a);
        }

        // Sigmoid - tolerance for numerical differences
        let sigmoid_scalar = v_scalar.sigmoid()?;
        let sigmoid_auto = v_auto.sigmoid()?;
        for (s, a) in sigmoid_scalar.as_slice().iter().zip(sigmoid_auto.as_slice().iter()) {
            prop_assert!((s - a).abs() < 1e-5, "Sigmoid mismatch: scalar={}, auto={}", s, a);
        }

        // GELU - tolerance for numerical differences
        let gelu_scalar = v_scalar.gelu()?;
        let gelu_auto = v_auto.gelu()?;
        for (s, a) in gelu_scalar.as_slice().iter().zip(gelu_auto.as_slice().iter()) {
            prop_assert!((s - a).abs() < 1e-4, "GELU mismatch: scalar={}, auto={}", s, a);
        }

        // Swish - tolerance for numerical differences
        let swish_scalar = v_scalar.swish()?;
        let swish_auto = v_auto.swish()?;
        for (s, a) in swish_scalar.as_slice().iter().zip(swish_auto.as_slice().iter()) {
            prop_assert!((s - a).abs() < 1e-5, "Swish mismatch: scalar={}, auto={}", s, a);
        }
    }

    /// Integration test: Backend equivalence for norms and reductions
    #[test]
    fn integration_backend_equivalence_norms(
        data in prop::collection::vec(-50.0f32..50.0, 50..100)
    ) {
        let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
        let v_auto = Vector::from_slice(&data);

        // L1 norm
        let l1_scalar = v_scalar.norm_l1()?;
        let l1_auto = v_auto.norm_l1()?;
        prop_assert!((l1_scalar - l1_auto).abs() < 1e-3 * l1_scalar.abs().max(1.0),
            "L1 norm mismatch: scalar={}, auto={}", l1_scalar, l1_auto);

        // L2 norm
        let l2_scalar = v_scalar.norm_l2()?;
        let l2_auto = v_auto.norm_l2()?;
        prop_assert!((l2_scalar - l2_auto).abs() < 1e-3 * l2_scalar.abs().max(1.0),
            "L2 norm mismatch: scalar={}, auto={}", l2_scalar, l2_auto);

        // Dot product
        let dot_scalar = v_scalar.dot(&v_scalar)?;
        let dot_auto = v_auto.dot(&v_auto)?;
        prop_assert!((dot_scalar - dot_auto).abs() < 1e-3 * dot_scalar.abs().max(1.0),
            "Dot product mismatch: scalar={}, auto={}", dot_scalar, dot_auto);
    }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[test]
fn integration_error_handling() {
    use trueno::Matrix;

    // Size mismatch errors
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[1.0, 2.0]);
    assert!(a.add(&b).is_err());
    assert!(a.dot(&b).is_err());

    // Empty vector handling (most operations return 0 or NaN for empty)
    let empty = Vector::from_slice(&[]);
    // sum() returns 0.0 for empty vectors
    assert_eq!(empty.sum().unwrap(), 0.0);
    // mean() returns error for empty vectors
    assert!(empty.mean().is_err());

    // Single element vectors (variance is 0)
    let single = Vector::from_slice(&[1.0]);
    assert_eq!(single.variance().unwrap(), 0.0); // Single element has zero variance

    // Matrix dimension mismatch
    let m1 = Matrix::zeros(2, 3);
    let m2 = Matrix::zeros(2, 2);
    assert!(m1.matmul(&m2).is_err());

    // Matrix from_vec with wrong size
    let data = vec![1.0, 2.0, 3.0];
    assert!(Matrix::from_vec(2, 2, data).is_err());
}

// ============================================================================
// BACKEND SELECTION
// ============================================================================

#[test]
fn integration_backend_selection() {
    use trueno::select_best_available_backend;

    let backend = select_best_available_backend();

    // Verify backend is one of the valid options
    assert!(matches!(
        backend,
        Backend::Scalar
            | Backend::SSE2
            | Backend::AVX
            | Backend::AVX2
            | Backend::AVX512
            | Backend::NEON
            | Backend::WasmSIMD
    ));

    // Verify Backend::select_best() works
    let backend2 = Backend::select_best();
    assert_eq!(backend, backend2);

    // Test creating vectors with explicit backends
    let data = vec![1.0, 2.0, 3.0];
    let _v_auto = Vector::from_slice(&data);
    let _v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
}
