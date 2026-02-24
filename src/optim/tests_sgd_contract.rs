// =========================================================================
// FALSIFY-SGD: optimization-v1.yaml contract (aprender SGD)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-SGD-* tests
//   Why 2: SGD tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from optimization-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: SGD was "obviously correct" (x = x - lr * grad)
//
// References:
//   - provable-contracts/contracts/optimization-v1.yaml
//   - Robbins & Monro (1951) "A Stochastic Approximation Method"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-SGD-001: SGD step decreases objective on gradient direction
#[test]
fn falsify_sgd_001_step_reduces_param() {
    let mut sgd = SGD::new(0.1);
    let mut params = Vector::from_vec(vec![5.0, -3.0]);
    let gradients = Vector::from_vec(vec![10.0, -6.0]); // gradient at this point

    sgd.step(&mut params, &gradients);

    // params should have moved in negative gradient direction
    // new_x = 5.0 - 0.1 * 10.0 = 4.0
    // new_y = -3.0 - 0.1 * (-6.0) = -2.4
    assert!(
        (params[0] - 4.0).abs() < 1e-5,
        "FALSIFIED SGD-001: x={}, expected 4.0",
        params[0]
    );
    assert!(
        (params[1] - (-2.4)).abs() < 1e-5,
        "FALSIFIED SGD-001: y={}, expected -2.4",
        params[1]
    );
}

/// FALSIFY-SGD-002: SGD result is finite
#[test]
fn falsify_sgd_002_finite_result() {
    let mut sgd = SGD::new(0.01);
    let mut params = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = Vector::from_vec(vec![0.5, -0.3, 0.8]);

    sgd.step(&mut params, &gradients);

    for i in 0..3 {
        assert!(
            params[i].is_finite(),
            "FALSIFIED SGD-002: params[{i}] is not finite"
        );
    }
}

/// FALSIFY-SGD-003: SGD with zero gradient leaves params unchanged
#[test]
fn falsify_sgd_003_zero_gradient_identity() {
    let mut sgd = SGD::new(0.1);
    let mut params = Vector::from_vec(vec![2.0, 3.0]);
    let zero_grad = Vector::from_vec(vec![0.0, 0.0]);

    sgd.step(&mut params, &zero_grad);

    assert!(
        (params[0] - 2.0).abs() < 1e-6,
        "FALSIFIED SGD-003: x={}, expected 2.0 (unchanged)",
        params[0]
    );
    assert!(
        (params[1] - 3.0).abs() < 1e-6,
        "FALSIFIED SGD-003: y={}, expected 3.0 (unchanged)",
        params[1]
    );
}
