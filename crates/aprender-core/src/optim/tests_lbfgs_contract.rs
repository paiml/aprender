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

// =========================================================================
// FALSIFY-LBFGS-003 NON-FINITE MATRIX (contract equation `nonfinite_input_status`)
//
// Four input CHANNELS can carry a non-finite value into the solver, and each
// reaches a different guard:
//
//   (i)   x0 itself                      -> entry check on the start point
//   (ii)  x0 itself, +inf rather than NaN-> entry check on the start point
//   (iii) the objective's return value   -> entry check on f(x0)
//   (iv)  the gradient's return value    -> entry check on grad(x0)
//   (iv-b) a value that is finite at x0 and non-finite only at a LINE-SEARCH
//          TRIAL POINT -> the Wolfe loop's OWN guard, which the entry checks
//          cannot see
//
// Every case asserts the STATUS DISCRIMINANT (not a message) and must not
// panic. T-3-02 in the plan's threat register claimed non-finite handling
// that no test exercised; this block is that test.
// =========================================================================

/// Channel (i), f32: `x0` contains NaN.
#[test]
fn falsify_lbfgs_003_nonfinite_x0_nan_f32() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let result = lbfgs.minimize(objective, gradient, Vector::from_vec(vec![f32::NAN]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: NaN in x0 (f32) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (ii), f32: `x0` contains +inf.
#[test]
fn falsify_lbfgs_003_nonfinite_x0_inf_f32() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let result = lbfgs.minimize(objective, gradient, Vector::from_vec(vec![f32::INFINITY]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: +inf in x0 (f32) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (iii), f32: the objective returns NaN at the start point while the
/// gradient stays finite. The objective is finite elsewhere, so ONLY the entry
/// check on `f(x0)` can catch this — every later evaluation looks healthy.
#[test]
fn falsify_lbfgs_003_nonfinite_objective_nan_f32() {
    let objective = |x: &Vector<f32>| -> f32 {
        if x[0] == 1.0 {
            f32::NAN
        } else {
            x[0] * x[0]
        }
    };
    let gradient = |x: &Vector<f32>| -> Vector<f32> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let result = lbfgs.minimize(objective, gradient, Vector::from_vec(vec![1.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: NaN objective at x0 (f32) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (iv), f32: the gradient returns a vector containing +inf at the
/// start point while the objective stays finite.
#[test]
fn falsify_lbfgs_003_nonfinite_gradient_inf_f32() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> {
        if x[0] == 1.0 {
            Vector::from_vec(vec![f32::INFINITY])
        } else {
            Vector::from_vec(vec![2.0 * x[0]])
        }
    };

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let result = lbfgs.minimize(objective, gradient, Vector::from_vec(vec![1.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: +inf gradient at x0 (f32) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (iv-b), f32: everything is FINITE at `x0` and the gradient goes
/// non-finite only at line-search trial points. The entry checks cannot see
/// this; the Wolfe loop's own guard must.
#[test]
fn falsify_lbfgs_003_nonfinite_linesearch_trial_f32() {
    let objective = |x: &Vector<f32>| -> f32 { x[0] * x[0] };
    let gradient = |x: &Vector<f32>| -> Vector<f32> {
        if x[0] == 1.0 {
            Vector::from_vec(vec![2.0])
        } else {
            Vector::from_vec(vec![f32::INFINITY])
        }
    };

    let mut lbfgs = LBFGS::new(100, 1e-6, 10);
    let result = lbfgs.minimize(objective, gradient, Vector::from_vec(vec![1.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: +inf gradient at a line-search trial point (f32) reported {:?}, \
         expected NumericalError — a poisoned search must not be reported as a benign tiny step",
        result.status
    );
}

// =========================================================================
// FALSIFY-LBFGS f64 TWINS
//
// The point of the f64 entry point is that it reaches tolerances the f32 path
// CANNOT. That claim is easy to write and easy to make vacuous, so it is
// pinned by three tests rather than one:
//
//   * `falsify_lbfgs_001_f64`                       — the contract quadratic
//   * `..._tolerance_1e6_cannot_distinguish_widths` — the VACUITY witness: at
//     1e-6 both widths converge, so a 1e-6 assertion proves nothing about width
//   * `..._1e10_width_is_real`                      — the DISCRIMINATING case:
//     at 1e-10 f64 converges and f32 provably cannot
//
// Measured on this host (see 03-01-SUMMARY.md): on the contract quadratic BOTH
// widths reach gradient norm exactly 0 in one iteration, because its optimum is
// exactly representable in f32. A tolerance test on that problem therefore
// cannot demonstrate the width no matter how tight it is — hence the dense
// least-squares problem below, whose optimum is representable in neither width.
// =========================================================================

/// 6x4 design matrix. Entries are deliberately "untidy" so the least-squares
/// optimum is not a value either width can represent exactly.
const LSQ_A: [[f64; 4]; 6] = [
    [0.13, -0.27, 0.41, 0.08],
    [0.22, 0.19, -0.33, 0.47],
    [-0.31, 0.44, 0.12, -0.26],
    [0.38, -0.11, 0.29, 0.35],
    [0.07, 0.33, -0.48, 0.21],
    [-0.24, 0.16, 0.37, -0.09],
];
const LSQ_B: [f64; 6] = [0.051, -0.037, 0.083, 0.019, -0.062, 0.044];

/// Least-squares objective, f32: f(x) = ||A x - b||².
fn lsq_obj_f32(x: &Vector<f32>) -> f32 {
    let mut total = 0.0f32;
    for j in 0..6 {
        let mut r = 0.0f32;
        for i in 0..4 {
            r += LSQ_A[j][i] as f32 * x[i];
        }
        r -= LSQ_B[j] as f32;
        total += r * r;
    }
    total
}

/// Least-squares gradient, f32: ∇f(x) = 2 Aᵀ(A x - b).
fn lsq_grad_f32(x: &Vector<f32>) -> Vector<f32> {
    let mut r = [0.0f32; 6];
    for j in 0..6 {
        let mut acc = 0.0f32;
        for i in 0..4 {
            acc += LSQ_A[j][i] as f32 * x[i];
        }
        r[j] = acc - LSQ_B[j] as f32;
    }
    let mut g = vec![0.0f32; 4];
    for (i, gi) in g.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (j, rj) in r.iter().enumerate() {
            acc += LSQ_A[j][i] as f32 * rj;
        }
        *gi = 2.0 * acc;
    }
    Vector::from_vec(g)
}

/// Least-squares objective, f64.
fn lsq_obj_f64(x: &Vector<f64>) -> f64 {
    let mut total = 0.0f64;
    for j in 0..6 {
        let mut r = 0.0f64;
        for i in 0..4 {
            r += LSQ_A[j][i] * x[i];
        }
        r -= LSQ_B[j];
        total += r * r;
    }
    total
}

/// Least-squares gradient, f64.
fn lsq_grad_f64(x: &Vector<f64>) -> Vector<f64> {
    let mut r = [0.0f64; 6];
    for j in 0..6 {
        let mut acc = 0.0f64;
        for i in 0..4 {
            acc += LSQ_A[j][i] * x[i];
        }
        r[j] = acc - LSQ_B[j];
    }
    let mut g = vec![0.0f64; 4];
    for (i, gi) in g.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (j, rj) in r.iter().enumerate() {
            acc += LSQ_A[j][i] * rj;
        }
        *gi = 2.0 * acc;
    }
    Vector::from_vec(g)
}

/// FALSIFY-LBFGS-001-f64: `LbfgsF64` converges on the contract quadratic with
/// gradient norm below 1e-10.
#[test]
fn falsify_lbfgs_001_f64() {
    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] };
    let gradient = |x: &Vector<f64>| -> Vector<f64> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LbfgsF64::new(200, 1e-10, 10);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![5.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::Converged,
        "FALSIFIED LBFGS-001-f64: status {:?}, expected Converged",
        result.status
    );
    assert!(
        result.gradient_norm < 1e-10,
        "FALSIFIED LBFGS-001-f64: gradient_norm {} not below 1e-10",
        result.gradient_norm
    );
    assert!(
        result.solution[0].abs() < 1e-10,
        "FALSIFIED LBFGS-001-f64: minimizer x={}, expected ≈ 0",
        result.solution[0]
    );
}

/// VACUITY WITNESS. At an f32-reachable tolerance (1e-6) BOTH widths converge
/// on the least-squares problem, so a 1e-6 assertion says nothing about
/// precision. Recorded as a test so the discriminating test below cannot
/// silently be loosened back into vacuity.
#[test]
fn falsify_lbfgs_001_f64_tolerance_1e6_cannot_distinguish_widths() {
    let mut o32 = LBFGS::new(500, 1e-6, 10);
    let r32 = o32.minimize(lsq_obj_f32, lsq_grad_f32, Vector::from_vec(vec![0.0f32; 4]));

    let mut o64 = LbfgsF64::new(500, 1e-6, 10);
    let r64 = o64.minimize(
        lsq_obj_f64,
        lsq_grad_f64,
        &Vector::from_vec(vec![0.0f64; 4]),
    );

    assert_eq!(
        r32.status,
        ConvergenceStatus::Converged,
        "VACUITY WITNESS broken: f32 no longer reaches 1e-6 (status {:?}, grad_norm {})",
        r32.status,
        r32.gradient_norm
    );
    assert_eq!(
        r64.status,
        ConvergenceStatus::Converged,
        "VACUITY WITNESS broken: f64 no longer reaches 1e-6 (status {:?}, grad_norm {})",
        r64.status,
        r64.gradient_norm
    );
}

/// FALSIFY-LBFGS-001-f64 (DISCRIMINATING): at 1e-10 the f64 path converges and
/// the f32 path provably cannot. This is what proves the width is real rather
/// than a cast — if `LbfgsF64` were f32 arithmetic behind an f64 signature,
/// this test would fail.
#[test]
fn falsify_lbfgs_001_f64_1e10_width_is_real() {
    let mut o32 = LBFGS::new(500, 1e-10, 10);
    let r32 = o32.minimize(lsq_obj_f32, lsq_grad_f32, Vector::from_vec(vec![0.0f32; 4]));

    let mut o64 = LbfgsF64::new(500, 1e-10, 10);
    let r64 = o64.minimize(
        lsq_obj_f64,
        lsq_grad_f64,
        &Vector::from_vec(vec![0.0f64; 4]),
    );

    assert!(
        f64::from(r32.gradient_norm) > 1e-10,
        "FALSIFIED LBFGS-001-f64: the f32 path reached gradient_norm {} <= 1e-10, so this test \
         no longer demonstrates that the f64 width buys anything",
        r32.gradient_norm
    );
    assert_ne!(
        r32.status,
        ConvergenceStatus::Converged,
        "FALSIFIED LBFGS-001-f64: f32 reported Converged at tol 1e-10"
    );

    assert_eq!(
        r64.status,
        ConvergenceStatus::Converged,
        "FALSIFIED LBFGS-001-f64: f64 status {:?} at tol 1e-10, expected Converged",
        r64.status
    );
    assert!(
        r64.gradient_norm < 1e-10,
        "FALSIFIED LBFGS-001-f64: f64 gradient_norm {} not below 1e-10",
        r64.gradient_norm
    );
}

/// FALSIFY-LBFGS-002-f64: the objective strictly decreases across accepted
/// steps, and the reported `objective_value` really is the objective AT the
/// reported solution.
#[test]
fn falsify_lbfgs_002_f64() {
    use std::cell::RefCell;

    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
    let gradient =
        |x: &Vector<f64>| -> Vector<f64> { Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    let x0 = Vector::from_vec(vec![3.0, 4.0]);
    let initial_obj = objective(&x0);

    let calls: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    let recorded = |x: &Vector<f64>| {
        let value = objective(x);
        calls.borrow_mut().push(value);
        value
    };

    let mut lbfgs = LbfgsF64::new(200, 1e-10, 10);
    let result = lbfgs.minimize(&recorded, gradient, &x0);
    let calls = calls.into_inner();

    // Accepted steps = the strictly-decreasing running minimum of every
    // evaluation the solver made.
    let mut accepted = Vec::new();
    let mut best = f64::INFINITY;
    for value in &calls {
        if *value < best {
            best = *value;
            accepted.push(*value);
        }
    }

    assert!(
        accepted.len() >= 2,
        "FALSIFIED LBFGS-002-f64: only {} accepted step(s); the solver made no progress",
        accepted.len()
    );
    for pair in accepted.windows(2) {
        assert!(
            pair[1] < pair[0],
            "FALSIFIED LBFGS-002-f64: accepted objective did not strictly decrease: {} -> {}",
            pair[0],
            pair[1]
        );
    }
    assert!(
        result.objective_value < initial_obj,
        "FALSIFIED LBFGS-002-f64: final obj {} >= initial obj {}",
        result.objective_value,
        initial_obj
    );
    // The reported pair (solution, objective_value) must be self-consistent.
    let recomputed = objective(&result.solution);
    assert!(
        (recomputed - result.objective_value).abs() <= f64::EPSILON * recomputed.abs().max(1.0),
        "FALSIFIED LBFGS-002-f64: reported objective_value {} does not match f(solution) {}",
        result.objective_value,
        recomputed
    );
}

/// FALSIFY-LBFGS-003-f64: solution and gradient norm stay finite on a
/// pathological-but-FINITE input — a diagonal Hessian with condition number
/// 1e10, which drives a near-degenerate curvature pair.
#[test]
fn falsify_lbfgs_003_f64() {
    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] + 1e10 * x[1] * x[1] };
    let gradient =
        |x: &Vector<f64>| -> Vector<f64> { Vector::from_vec(vec![2.0 * x[0], 2e10 * x[1]]) };

    let mut lbfgs = LbfgsF64::new(100, 1e-10, 5);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![1.0, 1.0]));

    assert!(
        result.solution[0].is_finite() && result.solution[1].is_finite(),
        "FALSIFIED LBFGS-003-f64: solution is not finite: [{}, {}]",
        result.solution[0],
        result.solution[1]
    );
    assert!(
        result.objective_value.is_finite(),
        "FALSIFIED LBFGS-003-f64: objective value is not finite"
    );
    assert!(
        result.gradient_norm.is_finite(),
        "FALSIFIED LBFGS-003-f64: gradient norm is not finite"
    );
    assert_ne!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003-f64: a FINITE pathological input was reported as NumericalError"
    );
}

/// Channel (i), f64: `x0` contains NaN.
#[test]
fn falsify_lbfgs_003_nonfinite_x0_nan_f64() {
    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] };
    let gradient = |x: &Vector<f64>| -> Vector<f64> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LbfgsF64::new(100, 1e-10, 10);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![f64::NAN]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: NaN in x0 (f64) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (ii), f64: `x0` contains +inf.
#[test]
fn falsify_lbfgs_003_nonfinite_x0_inf_f64() {
    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] };
    let gradient = |x: &Vector<f64>| -> Vector<f64> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LbfgsF64::new(100, 1e-10, 10);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![f64::INFINITY]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: +inf in x0 (f64) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (iii), f64: the objective returns NaN at the start point while the
/// gradient stays finite.
#[test]
fn falsify_lbfgs_003_nonfinite_objective_nan_f64() {
    let objective = |x: &Vector<f64>| -> f64 {
        if x[0] == 1.0 {
            f64::NAN
        } else {
            x[0] * x[0]
        }
    };
    let gradient = |x: &Vector<f64>| -> Vector<f64> { Vector::from_vec(vec![2.0 * x[0]]) };

    let mut lbfgs = LbfgsF64::new(100, 1e-10, 10);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![1.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: NaN objective at x0 (f64) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (iv), f64: the gradient returns a vector containing +inf at the
/// start point while the objective stays finite.
#[test]
fn falsify_lbfgs_003_nonfinite_gradient_inf_f64() {
    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] };
    let gradient = |x: &Vector<f64>| -> Vector<f64> {
        if x[0] == 1.0 {
            Vector::from_vec(vec![f64::INFINITY])
        } else {
            Vector::from_vec(vec![2.0 * x[0]])
        }
    };

    let mut lbfgs = LbfgsF64::new(100, 1e-10, 10);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![1.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: +inf gradient at x0 (f64) reported {:?}, expected NumericalError",
        result.status
    );
}

/// Channel (iv-b), f64: finite at `x0`, non-finite only at line-search trial
/// points — the Wolfe loop's own guard.
#[test]
fn falsify_lbfgs_003_nonfinite_linesearch_trial_f64() {
    let objective = |x: &Vector<f64>| -> f64 { x[0] * x[0] };
    let gradient = |x: &Vector<f64>| -> Vector<f64> {
        if x[0] == 1.0 {
            Vector::from_vec(vec![2.0])
        } else {
            Vector::from_vec(vec![f64::INFINITY])
        }
    };

    let mut lbfgs = LbfgsF64::new(100, 1e-10, 10);
    let result = lbfgs.minimize(objective, gradient, &Vector::from_vec(vec![1.0]));

    assert_eq!(
        result.status,
        ConvergenceStatus::NumericalError,
        "FALSIFIED LBFGS-003: +inf gradient at a line-search trial point (f64) reported {:?}, \
         expected NumericalError — a poisoned search must not be reported as a benign tiny step",
        result.status
    );
}

mod lbfgs_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // FALSIFY-LBFGS-001-prop: L-BFGS converges on quadratic from random starts
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

    // FALSIFY-LBFGS-002-prop: L-BFGS objective decreases from random starts
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

    // FALSIFY-LBFGS-001-f64-prop: random PSD quadratics converge in f64 at a
    // tolerance (1e-10) the f32 path cannot generally reach.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_lbfgs_001_f64_prop_psd_quadratic_convergence(
            a in 0.5f64..5.0,
            b in 0.5f64..5.0,
            x0_val in -10.0f64..10.0,
            y0_val in -10.0f64..10.0,
        ) {
            let objective = |x: &Vector<f64>| -> f64 { a * x[0] * x[0] + b * x[1] * x[1] };
            let gradient = |x: &Vector<f64>| -> Vector<f64> {
                Vector::from_vec(vec![2.0 * a * x[0], 2.0 * b * x[1]])
            };

            let mut lbfgs = LbfgsF64::new(500, 1e-10, 10);
            let result = lbfgs.minimize(
                objective,
                gradient,
                &Vector::from_vec(vec![x0_val, y0_val]),
            );

            prop_assert_eq!(
                result.status,
                ConvergenceStatus::Converged,
                "FALSIFIED LBFGS-001-f64-prop: status {:?} (grad_norm {}) for a={}, b={}, start=({}, {})",
                result.status, result.gradient_norm, a, b, x0_val, y0_val
            );
            prop_assert!(
                result.gradient_norm < 1e-10,
                "FALSIFIED LBFGS-001-f64-prop: gradient_norm {} not below 1e-10",
                result.gradient_norm
            );
        }
    }
}
