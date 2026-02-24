// =========================================================================
// FALSIFY-LBFGS: lbfgs-kernel-v1.yaml contract (aprender LBFGS)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LBFGS-* tests
//   Why 2: LBFGS tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from lbfgs-kernel-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: L-BFGS was "obviously correct" (quasi-Newton, well-studied)
//
// References:
//   - provable-contracts/contracts/lbfgs-kernel-v1.yaml
//   - Nocedal (1980) "Updating Quasi-Newton Matrices with Limited Storage"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-LBFGS-001: Converges on convex quadratic f(x) = x²
#[test]
fn falsify_lbfgs_001_quadratic_convergence() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let x0 = Vector::from_vec(vec![5.0]);
    let result = lbfgs.minimize(objective, gradient, x0);

    assert!(
        result.solution[0].abs() < 0.01,
        "FALSIFIED LBFGS-001: minimizer x={}, expected ≈ 0",
        result.solution[0]
    );
}

/// FALSIFY-LBFGS-002: Result objective value decreases from initial
#[test]
fn falsify_lbfgs_002_objective_decreases() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] + x[1] * x[1] };
    let gradient =
        |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    let x0 = Vector::from_vec(vec![3.0, 4.0]);
    let initial_obj = objective(&x0);

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let result = lbfgs.minimize(objective, gradient, x0);

    assert!(
        result.objective_value < initial_obj,
        "FALSIFIED LBFGS-002: final obj {} >= initial obj {}",
        result.objective_value,
        initial_obj
    );
}

/// FALSIFY-LBFGS-003: Result has finite values
#[test]
fn falsify_lbfgs_003_finite_result() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LBFGS::new(50, 1e-6, 5);
    let x0 = Vector::from_vec(vec![10.0]);
    let result = lbfgs.minimize(objective, gradient, x0);

    assert!(
        result.solution[0].is_finite(),
        "FALSIFIED LBFGS-003: result x is not finite"
    );
    assert!(
        result.objective_value.is_finite(),
        "FALSIFIED LBFGS-003: objective value is not finite"
    );
}

mod lbfgs_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-LBFGS-001-prop: L-BFGS converges on quadratic from random starts
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_lbfgs_001_prop_quadratic_convergence(
            x0_val in -50.0f32..50.0,
        ) {
            let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
            let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

            let mut lbfgs = LBFGS::new(100, 1e-6, 10);
            let x0 = Vector::from_vec(vec![x0_val]);
            let result = lbfgs.minimize(objective, gradient, x0);

            prop_assert!(
                result.solution[0].abs() < 1.0,
                "FALSIFIED LBFGS-001-prop: x={} for start={}",
                result.solution[0], x0_val
            );
        }
    }

    /// FALSIFY-LBFGS-002-prop: L-BFGS objective decreases from random starts
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_lbfgs_002_prop_objective_decreases(
            x0_val in -20.0f32..20.0,
            y0_val in -20.0f32..20.0,
        ) {
            let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] + x[1] * x[1] };
            let gradient = |x: &Vector<f32>| -> Vector<f32> {
                Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
            };

            let x0 = Vector::from_vec(vec![x0_val, y0_val]);
            let initial_obj = objective(&x0);

            let mut lbfgs = LBFGS::new(100, 1e-6, 10);
            let result = lbfgs.minimize(objective, gradient, x0);

            if initial_obj > 1e-10 {
                prop_assert!(
                    result.objective_value < initial_obj,
                    "FALSIFIED LBFGS-002-prop: final {} >= initial {} for start=({}, {})",
                    result.objective_value, initial_obj, x0_val, y0_val
                );
            }
        }
    }
}
