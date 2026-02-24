// =========================================================================
// FALSIFY-CG: optimization-v1.yaml contract (aprender Conjugate Gradient)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-CG-* tests
//   Why 2: CG tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from optimization-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: CG was "obviously correct" (well-studied Fletcher-Reeves)
//
// References:
//   - provable-contracts/contracts/optimization-v1.yaml
//   - Fletcher & Reeves (1964) "Function Minimization by Conjugate Gradients"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-CG-001: CG converges on convex quadratic f(x) = x²
#[test]
fn falsify_cg_001_quadratic_convergence() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] + x[1] * x[1] };
    let gradient =
        |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    let mut cg = ConjugateGradient::new(100, 1e-6, CGBetaFormula::FletcherReeves);
    let x0 = Vector::from_vec(vec![5.0, -3.0]);
    let result = cg.minimize(objective, gradient, x0);

    assert!(
        result.objective_value < 0.01,
        "FALSIFIED CG-001: objective {} >= 0.01 after minimization",
        result.objective_value
    );
}

/// FALSIFY-CG-002: CG result has finite values
#[test]
fn falsify_cg_002_finite_result() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut cg = ConjugateGradient::new(50, 1e-6, CGBetaFormula::PolakRibiere);
    let x0 = Vector::from_vec(vec![10.0]);
    let result = cg.minimize(objective, gradient, x0);

    assert!(
        result.solution[0].is_finite(),
        "FALSIFIED CG-002: solution is not finite"
    );
    assert!(
        result.objective_value.is_finite(),
        "FALSIFIED CG-002: objective value is not finite"
    );
}

/// FALSIFY-CG-003: CG objective decreases from initial value
#[test]
fn falsify_cg_003_objective_decreases() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] + x[1] * x[1] };
    let gradient =
        |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    let x0 = Vector::from_vec(vec![3.0, 4.0]);
    let initial_obj = objective(&x0);

    let mut cg = ConjugateGradient::new(100, 1e-6, CGBetaFormula::FletcherReeves);
    let result = cg.minimize(objective, gradient, x0);

    assert!(
        result.objective_value < initial_obj,
        "FALSIFIED CG-003: final obj {} >= initial obj {}",
        result.objective_value,
        initial_obj
    );
}

mod cg_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-CG-001-prop: CG converges on shifted quadratic for random starts
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cg_001_prop_quadratic_convergence(
            x0_val in -50.0f32..50.0,
            y0_val in -50.0f32..50.0,
        ) {
            let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] + x[1] * x[1] };
            let gradient = |x: &Vector<f32>| -> Vector<f32> {
                Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]])
            };

            let mut cg = ConjugateGradient::new(200, 1e-6, CGBetaFormula::FletcherReeves);
            let x0 = Vector::from_vec(vec![x0_val, y0_val]);
            let result = cg.minimize(objective, gradient, x0);

            prop_assert!(
                result.objective_value < 1.0,
                "FALSIFIED CG-001-prop: obj={} for start=({}, {})",
                result.objective_value, x0_val, y0_val
            );
        }
    }

    /// FALSIFY-CG-003-prop: CG objective decreases from random starts
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cg_003_prop_objective_decreases(
            x0_val in -20.0f32..20.0,
        ) {
            let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
            let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

            let x0 = Vector::from_vec(vec![x0_val]);
            let initial_obj = objective(&x0);

            let mut cg = ConjugateGradient::new(100, 1e-6, CGBetaFormula::FletcherReeves);
            let result = cg.minimize(objective, gradient, x0);

            // For x0 != 0, objective should decrease (or stay at 0 if already at minimum)
            if initial_obj > 1e-10 {
                prop_assert!(
                    result.objective_value < initial_obj,
                    "FALSIFIED CG-003-prop: final {} >= initial {} for x0={}",
                    result.objective_value, initial_obj, x0_val
                );
            }
        }
    }
}
