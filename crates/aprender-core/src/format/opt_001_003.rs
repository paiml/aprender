// SHIP-TWO-001 — `optimization-v1` algorithm-level PARTIAL discharge
// for FALSIFY-OPT-001..003.
//
// Contract: `contracts/optimization-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Three Conjugate Gradient gates from Nocedal & Wright (2006) Ch. 5:
//
// - OPT-001 (monotone decrease): f(x_{k+1}) ≤ f(x_k) per iteration.
// - OPT-002 (finite iterates): all x_k finite (no NaN, no Inf).
// - OPT-003 (gradient norm convergence): ||g_final|| < ||g_initial||.
//
// All three are pure properties of the iteration trace (function-value
// sequence, iterate sequence, gradient-norm pair) — no actual line-
// search / matrix algebra wired at this layer.

/// Tolerance for monotone-decrease check: f_{k+1} ≤ f_k + EPS allows
/// small numerical wobble. Standard FP32 trust region.
pub const AC_OPT_001_MONOTONE_EPS: f32 = 1e-5;

/// Required gradient-norm decrease ratio: ||g_final|| < ||g_initial||
/// strictly (no slack — we only assert the contract's own inequality).
pub const AC_OPT_003_DECREASE_REQUIRED: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference quadratic optimizer (steepest descent for clarity).
// -----------------------------------------------------------------------------

/// Compute quadratic f(x) = 0.5 x^T A x + b^T x for a diagonal A.
#[must_use]
pub fn quadratic(a_diag: &[f32], b: &[f32], x: &[f32]) -> f32 {
    let mut f = 0.0_f32;
    for i in 0..x.len() {
        f += 0.5 * a_diag[i] * x[i] * x[i] + b[i] * x[i];
    }
    f
}

/// Gradient of f at x: g_i = a_i * x_i + b_i.
#[must_use]
pub fn quadratic_grad(a_diag: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    (0..x.len()).map(|i| a_diag[i] * x[i] + b[i]).collect()
}

/// Run steepest-descent on a strictly-convex diagonal quadratic for
/// `n_iters` iterations. Returns trace of (f_k, x_k, ||g_k||).
#[must_use]
pub fn steepest_descent_quadratic(
    a_diag: &[f32],
    b: &[f32],
    x0: &[f32],
    learning_rate: f32,
    n_iters: usize,
) -> Vec<(f32, Vec<f32>, f32)> {
    let mut x = x0.to_vec();
    let mut trace = Vec::with_capacity(n_iters + 1);
    for _ in 0..=n_iters {
        let f = quadratic(a_diag, b, &x);
        let g = quadratic_grad(a_diag, b, &x);
        let g_norm = g.iter().map(|gi| gi * gi).sum::<f32>().sqrt();
        trace.push((f, x.clone(), g_norm));
        // x_{k+1} = x_k - lr * g_k
        for i in 0..x.len() {
            x[i] -= learning_rate * g[i];
        }
    }
    trace
}

// -----------------------------------------------------------------------------
// Verdict 1: OPT-001 — monotone decrease.
// -----------------------------------------------------------------------------

/// Pass iff f_values is non-empty AND every consecutive pair satisfies
/// `f_{k+1} ≤ f_k + AC_OPT_001_MONOTONE_EPS`.
#[must_use]
pub fn verdict_from_monotone_decrease(f_values: &[f32]) -> OptVerdict {
    if f_values.is_empty() {
        return OptVerdict::Fail;
    }
    for w in f_values.windows(2) {
        let f_k = w[0];
        let f_kp1 = w[1];
        if !f_k.is_finite() || !f_kp1.is_finite() {
            return OptVerdict::Fail;
        }
        if f_kp1 > f_k + AC_OPT_001_MONOTONE_EPS {
            return OptVerdict::Fail;
        }
    }
    OptVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: OPT-002 — finite iterates.
// -----------------------------------------------------------------------------

/// Pass iff every iterate is finite (no NaN, no Inf).
#[must_use]
pub fn verdict_from_finite_iterates(iterates: &[Vec<f32>]) -> OptVerdict {
    if iterates.is_empty() {
        return OptVerdict::Fail;
    }
    for x_k in iterates {
        for &xi in x_k {
            if !xi.is_finite() {
                return OptVerdict::Fail;
            }
        }
    }
    OptVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: OPT-003 — gradient norm decrease.
// -----------------------------------------------------------------------------

/// Pass iff `g_final_norm < g_initial_norm` (strict; both finite).
#[must_use]
pub fn verdict_from_gradient_decrease(g_initial_norm: f32, g_final_norm: f32) -> OptVerdict {
    if !g_initial_norm.is_finite() || !g_final_norm.is_finite() {
        return OptVerdict::Fail;
    }
    if g_initial_norm < 0.0 || g_final_norm < 0.0 {
        return OptVerdict::Fail;
    }
    if g_final_norm < g_initial_norm {
        OptVerdict::Pass
    } else {
        OptVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_monotone_eps_1e_5() {
        assert_eq!(AC_OPT_001_MONOTONE_EPS, 1e-5);
    }

    #[test]
    fn provenance_decrease_required() {
        assert!(AC_OPT_003_DECREASE_REQUIRED);
    }

    // -------------------------------------------------------------------------
    // Section 2: OPT-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn opt001_pass_strictly_decreasing() {
        let f = vec![10.0_f32, 8.0, 5.0, 3.0, 1.0];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Pass);
    }

    #[test]
    fn opt001_pass_constant() {
        let f = vec![5.0_f32, 5.0, 5.0];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Pass);
    }

    #[test]
    fn opt001_pass_after_convergence() {
        let f = vec![100.0_f32, 50.0, 25.0, 12.5, 12.5, 12.5];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Pass);
    }

    #[test]
    fn opt001_pass_within_eps_wobble() {
        let f = vec![5.0_f32, 5.000005]; // 5e-6 < 1e-5
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: OPT-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn opt001_fail_increasing() {
        let f = vec![1.0_f32, 5.0, 3.0];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Fail);
    }

    #[test]
    fn opt001_fail_one_jump() {
        let f = vec![10.0_f32, 8.0, 9.0, 5.0]; // 8 → 9 is ascent
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Fail);
    }

    #[test]
    fn opt001_fail_empty() {
        let f: Vec<f32> = vec![];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Fail);
    }

    #[test]
    fn opt001_fail_nan() {
        let f = vec![5.0_f32, f32::NAN];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Fail);
    }

    #[test]
    fn opt001_fail_inf() {
        let f = vec![5.0_f32, f32::INFINITY];
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: OPT-002 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn opt002_pass_finite_2d() {
        let iters = vec![vec![1.0_f32, 2.0], vec![0.5, 1.0], vec![0.25, 0.5]];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Pass);
    }

    #[test]
    fn opt002_pass_zeros() {
        let iters = vec![vec![0.0_f32, 0.0]];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Pass);
    }

    #[test]
    fn opt002_pass_negative_values() {
        let iters = vec![vec![-100.0_f32, -50.0], vec![-25.0, -10.0]];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: OPT-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn opt002_fail_nan_iterate() {
        let iters = vec![vec![1.0_f32], vec![f32::NAN]];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Fail);
    }

    #[test]
    fn opt002_fail_inf_iterate() {
        let iters = vec![vec![1.0_f32, f32::INFINITY]];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Fail);
    }

    #[test]
    fn opt002_fail_neg_inf() {
        let iters = vec![vec![1.0_f32], vec![f32::NEG_INFINITY]];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Fail);
    }

    #[test]
    fn opt002_fail_empty_iterates() {
        let iters: Vec<Vec<f32>> = vec![];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: OPT-003 — gradient decrease.
    // -------------------------------------------------------------------------
    #[test]
    fn opt003_pass_strict_decrease() {
        assert_eq!(verdict_from_gradient_decrease(10.0, 1.0), OptVerdict::Pass);
    }

    #[test]
    fn opt003_pass_near_zero_final() {
        assert_eq!(
            verdict_from_gradient_decrease(5.0, 1e-10),
            OptVerdict::Pass
        );
    }

    #[test]
    fn opt003_fail_no_change() {
        // Strict <, no equality.
        assert_eq!(verdict_from_gradient_decrease(5.0, 5.0), OptVerdict::Fail);
    }

    #[test]
    fn opt003_fail_increase() {
        assert_eq!(verdict_from_gradient_decrease(1.0, 10.0), OptVerdict::Fail);
    }

    #[test]
    fn opt003_fail_negative() {
        // Norms must be non-negative.
        assert_eq!(
            verdict_from_gradient_decrease(-1.0, 0.5),
            OptVerdict::Fail
        );
        assert_eq!(
            verdict_from_gradient_decrease(1.0, -0.5),
            OptVerdict::Fail
        );
    }

    #[test]
    fn opt003_fail_nan() {
        assert_eq!(
            verdict_from_gradient_decrease(f32::NAN, 1.0),
            OptVerdict::Fail
        );
        assert_eq!(
            verdict_from_gradient_decrease(1.0, f32::NAN),
            OptVerdict::Fail
        );
    }

    #[test]
    fn opt003_fail_inf() {
        assert_eq!(
            verdict_from_gradient_decrease(f32::INFINITY, 1.0),
            OptVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — reference steepest descent.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_steepest_descent_converges_on_quadratic() {
        // f(x) = 0.5*(x_0^2 + 2*x_1^2) + (-x_0 - 2*x_1)
        // Minimum at x* = [1, 1] with f* = -1.5.
        let a = vec![1.0_f32, 2.0];
        let b = vec![-1.0_f32, -2.0];
        let x0 = vec![5.0_f32, 5.0];
        let trace = steepest_descent_quadratic(&a, &b, &x0, 0.1, 50);

        let f_values: Vec<f32> = trace.iter().map(|(f, _, _)| *f).collect();
        let iterates: Vec<Vec<f32>> = trace.iter().map(|(_, x, _)| x.clone()).collect();
        let g_initial = trace.first().unwrap().2;
        let g_final = trace.last().unwrap().2;

        assert_eq!(verdict_from_monotone_decrease(&f_values), OptVerdict::Pass);
        assert_eq!(verdict_from_finite_iterates(&iterates), OptVerdict::Pass);
        assert_eq!(
            verdict_from_gradient_decrease(g_initial, g_final),
            OptVerdict::Pass
        );

        // Final point should be near (1, 1). At lr=0.1, 50 iters, the
        // x0 dimension (eigenvalue 1) converges as 0.9^50 ≈ 0.005 of
        // initial residual; the x1 dimension (eigenvalue 2) is faster.
        // Allow 0.05 slack — pure correctness check is monotone+gradient,
        // not the absolute distance.
        let x_final = trace.last().unwrap().1.clone();
        assert!((x_final[0] - 1.0).abs() < 0.05, "x0={}", x_final[0]);
        assert!((x_final[1] - 1.0).abs() < 0.05, "x1={}", x_final[1]);
    }

    #[test]
    fn domain_quadratic_value_at_minimum() {
        // f(x) = 0.5*x^2 - x; min at x=1, f*=-0.5.
        let a = vec![1.0_f32];
        let b = vec![-1.0_f32];
        let x = vec![1.0_f32];
        let f = quadratic(&a, &b, &x);
        assert!((f - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn domain_quadratic_gradient_zero_at_minimum() {
        let a = vec![2.0_f32];
        let b = vec![-4.0_f32]; // min at x = -b/a = 2
        let x = vec![2.0_f32];
        let g = quadratic_grad(&a, &b, &x);
        assert!(g[0].abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Section 8: Sweep — initial points.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_steepest_descent_various_starts() {
        let a = vec![1.0_f32];
        let b = vec![-2.0_f32]; // min at x=2, f*=-2
        for x0_val in [-10.0_f32, -1.0, 0.0, 5.0, 100.0] {
            let trace = steepest_descent_quadratic(&a, &b, &[x0_val], 0.1, 100);
            let f_values: Vec<f32> = trace.iter().map(|(f, _, _)| *f).collect();
            let iterates: Vec<Vec<f32>> = trace.iter().map(|(_, x, _)| x.clone()).collect();
            assert_eq!(
                verdict_from_monotone_decrease(&f_values),
                OptVerdict::Pass,
                "x0={x0_val}"
            );
            assert_eq!(
                verdict_from_finite_iterates(&iterates),
                OptVerdict::Pass,
                "x0={x0_val}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 9: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_armijo_violation_caught() {
        // OPT-001 if_fails: "Line search not satisfying Armijo condition".
        // Simulate by injecting a non-monotone f sequence.
        let f = vec![5.0_f32, 3.0, 7.0, 2.0]; // 3→7 is ascent
        assert_eq!(verdict_from_monotone_decrease(&f), OptVerdict::Fail);
    }

    #[test]
    fn realistic_gradient_explosion_caught() {
        // OPT-002 if_fails: "Step size too large or gradient explosion".
        // Simulate runaway iteration producing Inf.
        let iters = vec![
            vec![1.0_f32, 2.0],
            vec![100.0, 200.0],
            vec![10000.0, 20000.0],
            vec![f32::INFINITY, f32::INFINITY],
        ];
        assert_eq!(verdict_from_finite_iterates(&iters), OptVerdict::Fail);
    }

    #[test]
    fn realistic_beta_overflow_ascent() {
        // OPT-003 if_fails: "Conjugate direction computation error or
        // beta overflow". Final gradient norm exceeds initial.
        assert_eq!(
            verdict_from_gradient_decrease(2.0, 5.0),
            OptVerdict::Fail
        );
    }

    #[test]
    fn realistic_lr_too_large_diverges() {
        // Steepest descent with lr too large diverges; iterates blow up.
        let a = vec![1.0_f32];
        let b = vec![0.0_f32];
        let x0 = vec![1.0_f32];
        let trace = steepest_descent_quadratic(&a, &b, &x0, 5.0, 20); // lr=5 too large

        let iterates: Vec<Vec<f32>> = trace.iter().map(|(_, x, _)| x.clone()).collect();
        // x grows without bound — eventually overflows to Inf.
        // First check that values became extreme.
        let last_x = iterates.last().unwrap()[0].abs();
        assert!(
            last_x > 1e6 || !last_x.is_finite(),
            "lr=5 on x_0=1 must diverge; |x_final|={last_x}"
        );
    }

    #[test]
    fn realistic_full_optimization_pipeline() {
        // f(x) = 0.5*(x_0^2 + x_1^2 + x_2^2); minimum at origin, f*=0.
        let a = vec![1.0_f32; 3];
        let b = vec![0.0_f32; 3];
        let x0 = vec![3.0_f32, -2.0, 1.5];
        let trace = steepest_descent_quadratic(&a, &b, &x0, 0.5, 100);

        let f_values: Vec<f32> = trace.iter().map(|(f, _, _)| *f).collect();
        let iterates: Vec<Vec<f32>> = trace.iter().map(|(_, x, _)| x.clone()).collect();

        assert_eq!(verdict_from_monotone_decrease(&f_values), OptVerdict::Pass);
        assert_eq!(verdict_from_finite_iterates(&iterates), OptVerdict::Pass);
        let (_, _, g_init) = trace.first().unwrap();
        let (f_final, x_final, g_final) = trace.last().unwrap();
        assert_eq!(
            verdict_from_gradient_decrease(*g_init, *g_final),
            OptVerdict::Pass
        );

        // Should be close to origin.
        for &xi in x_final {
            assert!(xi.abs() < 1e-5, "x_final={x_final:?}");
        }
        assert!(f_final.abs() < 1e-9);
    }
}
