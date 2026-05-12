// SHIP-TWO-001 — `adamw-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-AW-001..011 (closes 11/11 sweep).
//
// Contract: `contracts/adamw-kernel-v1.yaml`.
//
// Bundles 11 verdict fns + a stand-alone scalar AdamW reference
// implementation so the algorithm-level rule is testable offline
// against drift in the SIMD kernel.

// ===========================================================================
// Reference scalar AdamW step (Loshchilov & Hutter 2019).
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub struct AdamWHyperparams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AdamWState {
    pub theta: f32,
    pub m: f32,
    pub v: f32,
    pub t: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum AdamWError {
    InvalidBeta1,
    InvalidBeta2,
    InvalidEps,
    InvalidLr,
}

/// Reference scalar AdamW step. Returns `Err(...)` for boundary
/// hyperparameter values (FALSIFY-AW-007).
pub fn adamw_step(
    state: AdamWState,
    grad: f32,
    h: AdamWHyperparams,
) -> Result<AdamWState, AdamWError> {
    if !(0.0 < h.beta1 && h.beta1 < 1.0) { return Err(AdamWError::InvalidBeta1); }
    if !(0.0 < h.beta2 && h.beta2 < 1.0) { return Err(AdamWError::InvalidBeta2); }
    if h.eps <= 0.0 { return Err(AdamWError::InvalidEps); }
    if !h.lr.is_finite() { return Err(AdamWError::InvalidLr); }

    let t = state.t.saturating_add(1);
    let m = h.beta1 * state.m + (1.0 - h.beta1) * grad;
    let v = h.beta2 * state.v + (1.0 - h.beta2) * grad * grad;
    let m_hat = m / (1.0 - h.beta1.powi(t as i32));
    let v_hat = v / (1.0 - h.beta2.powi(t as i32));
    let theta = state.theta - h.lr * (m_hat / (v_hat.sqrt() + h.eps) + h.weight_decay * state.theta);
    Ok(AdamWState { theta, m, v, t })
}

// ===========================================================================
// AW-001 — Decoupled weight decay: AdamW != L2-regularized Adam
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw001Verdict { Pass, Fail }

/// Pass iff the AdamW theta differs from a coupled-L2 Adam theta
/// when `weight_decay > 0` AND `theta != 0`. With wd == 0 OR theta == 0
/// the two formulas coincide; the contract rule predicts inequality
/// only for `lambda > 0`.
#[must_use]
pub fn verdict_from_decoupled_weight_decay(
    adamw_theta: f32,
    coupled_l2_adam_theta: f32,
    weight_decay: f32,
    initial_theta: f32,
) -> Aw001Verdict {
    if weight_decay <= 0.0 || initial_theta == 0.0 {
        // Vacuous Pass — both formulas agree, but the contract rule
        // is over the lambda > 0 regime.
        return Aw001Verdict::Pass;
    }
    if !adamw_theta.is_finite() || !coupled_l2_adam_theta.is_finite() {
        return Aw001Verdict::Fail;
    }
    let diff = (adamw_theta - coupled_l2_adam_theta).abs();
    if diff > 1e-7 { Aw001Verdict::Pass } else { Aw001Verdict::Fail }
}

// ===========================================================================
// AW-002 — Second moment v_t >= 0 after a single update
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_second_moment_nonnegative(v_t: f32) -> Aw002Verdict {
    if !v_t.is_finite() { return Aw002Verdict::Fail; }
    if v_t >= 0.0 { Aw002Verdict::Pass } else { Aw002Verdict::Fail }
}

// ===========================================================================
// AW-003 — Bias correction: 1/(1 - beta^t) > 1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw003Verdict { Pass, Fail }

/// Pass iff `1 / (1 - beta^t) > 1` for `beta in (0, 1)` and `t >= 1`.
#[must_use]
pub fn verdict_from_bias_correction(beta: f32, t: u32) -> Aw003Verdict {
    if !(0.0 < beta && beta < 1.0) { return Aw003Verdict::Fail; }
    if t == 0 { return Aw003Verdict::Fail; }
    let denom = 1.0 - beta.powi(t as i32);
    if denom <= 0.0 || !denom.is_finite() { return Aw003Verdict::Fail; }
    let correction = 1.0 / denom;
    if correction > 1.0 && correction.is_finite() { Aw003Verdict::Pass } else { Aw003Verdict::Fail }
}

// ===========================================================================
// AW-004 — Update finiteness with finite gradient and eps > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_update_finiteness(
    g: f32,
    eps: f32,
    final_theta: f32,
) -> Aw004Verdict {
    if !g.is_finite() || eps <= 0.0 { return Aw004Verdict::Fail; }
    if final_theta.is_finite() { Aw004Verdict::Pass } else { Aw004Verdict::Fail }
}

// ===========================================================================
// AW-005 — SIMD vs scalar: |simd - scalar| < 8 ULP
// ===========================================================================

pub const AC_AW_005_MAX_ULP: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw005Verdict { Pass, Fail }

/// Compute the absolute ULP distance between two f32 values, treating
/// NaN as a Fail signal upstream. For values with the same sign, ULPs
/// are the absolute difference of their bit-monotonic mappings.
fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() { return None; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    if (ai < 0) != (bi < 0) {
        // Different signs: cross zero — count both magnitudes.
        let abs_diff = ai.unsigned_abs() + bi.unsigned_abs();
        return Some(abs_diff);
    }
    Some((ai - bi).unsigned_abs())
}

/// Pass iff `ulp_distance(simd, scalar) < 8`.
#[must_use]
pub fn verdict_from_simd_equivalence(simd: f32, scalar: f32) -> Aw005Verdict {
    match ulp_distance(simd, scalar) {
        Some(d) if d < AC_AW_005_MAX_ULP => Aw005Verdict::Pass,
        _ => Aw005Verdict::Fail,
    }
}

// ===========================================================================
// AW-006 — Zero-gradient boundary: only weight decay modifies theta
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw006Verdict { Pass, Fail }

/// Pass iff with `g == 0.0` (and `wd > 0`, `lr > 0`), the new theta
/// reflects pure weight-decay shrinkage:
///   theta_new ≈ theta_old * (1 - lr * wd)
/// within ULP-scale tolerance.
#[must_use]
pub fn verdict_from_zero_gradient_only_decay(
    theta_old: f32,
    theta_new: f32,
    lr: f32,
    weight_decay: f32,
) -> Aw006Verdict {
    if !theta_old.is_finite() || !theta_new.is_finite() { return Aw006Verdict::Fail; }
    if lr <= 0.0 || weight_decay <= 0.0 { return Aw006Verdict::Fail; }
    let expected = theta_old * (1.0 - lr * weight_decay);
    let tol = expected.abs().mul_add(1e-5, 1e-7);
    if (theta_new - expected).abs() <= tol { Aw006Verdict::Pass } else { Aw006Verdict::Fail }
}

// ===========================================================================
// AW-007 — Hyperparameter validation: invalid combos return Err
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw007Verdict { Pass, Fail }

/// Pass iff invalid hyperparameter combinations return `Err`. Exercises
/// the reference impl directly against the (β1=0, β2=1, ε=0) boundaries.
#[must_use]
pub fn verdict_from_hyperparam_validation() -> Aw007Verdict {
    let state = AdamWState { theta: 1.0, m: 0.0, v: 0.0, t: 0 };
    let bad: [AdamWHyperparams; 4] = [
        AdamWHyperparams { lr: 1e-3, beta1: 0.0, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 },
        AdamWHyperparams { lr: 1e-3, beta1: 0.9, beta2: 1.0,   eps: 1e-8, weight_decay: 0.0 },
        AdamWHyperparams { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 0.0,  weight_decay: 0.0 },
        AdamWHyperparams { lr: 1e-3, beta1: 1.0, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 },
    ];
    for h in &bad {
        if adamw_step(state, 0.5, *h).is_ok() {
            return Aw007Verdict::Fail;
        }
    }
    Aw007Verdict::Pass
}

// ===========================================================================
// AW-008 — Frame condition: gradient buffer unchanged after step
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw008Verdict { Pass, Fail }

/// Pass iff the post-step gradient buffer is byte-equal to the
/// pre-step snapshot.
#[must_use]
pub fn verdict_from_grad_unchanged(grad_before: &[f32], grad_after: &[f32]) -> Aw008Verdict {
    if grad_before.len() != grad_after.len() { return Aw008Verdict::Fail; }
    for (a, b) in grad_before.iter().zip(grad_after) {
        if a.to_bits() != b.to_bits() { return Aw008Verdict::Fail; }
    }
    Aw008Verdict::Pass
}

// ===========================================================================
// AW-009 — Loop invariant: v_t >= 0 across N steps
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw009Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_v_history_nonnegative(v_history: &[f32]) -> Aw009Verdict {
    if v_history.is_empty() { return Aw009Verdict::Fail; }
    for v in v_history {
        if !v.is_finite() || *v < 0.0 { return Aw009Verdict::Fail; }
    }
    Aw009Verdict::Pass
}

// ===========================================================================
// AW-010 — First moment formula: m = β1·old_m + (1-β1)·g
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw010Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_first_moment_formula(
    old_m: f32,
    g: f32,
    beta1: f32,
    new_m: f32,
) -> Aw010Verdict {
    if !old_m.is_finite() || !g.is_finite() || !new_m.is_finite() { return Aw010Verdict::Fail; }
    if !(0.0 < beta1 && beta1 < 1.0) { return Aw010Verdict::Fail; }
    let expected = beta1 * old_m + (1.0 - beta1) * g;
    let tol = expected.abs().mul_add(1e-5, 1e-7);
    if (new_m - expected).abs() <= tol { Aw010Verdict::Pass } else { Aw010Verdict::Fail }
}

// ===========================================================================
// AW-011 — Loop variant: training terminates at max_steps
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aw011Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_loop_terminates(executed_steps: u32, max_steps: u32) -> Aw011Verdict {
    if max_steps == 0 { return Aw011Verdict::Fail; }
    if executed_steps == max_steps { Aw011Verdict::Pass } else { Aw011Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_h() -> AdamWHyperparams {
        AdamWHyperparams {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    // ----- Reference impl spot checks ----------------------------------------

    #[test]
    fn reference_step_runs_without_panic() {
        let state = AdamWState { theta: 0.5, m: 0.0, v: 0.0, t: 0 };
        let new = adamw_step(state, 0.1, default_h()).unwrap();
        assert!(new.theta.is_finite());
        assert!(new.m.is_finite());
        assert!(new.v.is_finite());
        assert_eq!(new.t, 1);
    }

    // ----- AW-001 ------------------------------------------------------------

    #[test]
    fn aw001_pass_diverging_thetas() {
        // Trivial divergent values — emulates AdamW vs coupled-Adam.
        assert_eq!(
            verdict_from_decoupled_weight_decay(0.5, 0.6, 0.01, 0.5),
            Aw001Verdict::Pass
        );
    }

    #[test]
    fn aw001_fail_when_identical_under_lambda_pos() {
        // Same theta despite wd > 0 — implies coupled L2 path.
        assert_eq!(
            verdict_from_decoupled_weight_decay(0.5, 0.5, 0.01, 0.5),
            Aw001Verdict::Fail
        );
    }

    #[test]
    fn aw001_pass_vacuous_zero_lambda() {
        // wd == 0: contract rule is vacuously true.
        assert_eq!(
            verdict_from_decoupled_weight_decay(0.5, 0.5, 0.0, 0.5),
            Aw001Verdict::Pass
        );
    }

    // ----- AW-002 ------------------------------------------------------------

    #[test] fn aw002_pass_zero() { assert_eq!(verdict_from_second_moment_nonnegative(0.0), Aw002Verdict::Pass); }
    #[test] fn aw002_pass_positive() { assert_eq!(verdict_from_second_moment_nonnegative(1.5), Aw002Verdict::Pass); }
    #[test] fn aw002_fail_negative() { assert_eq!(verdict_from_second_moment_nonnegative(-0.001), Aw002Verdict::Fail); }
    #[test] fn aw002_fail_nan() { assert_eq!(verdict_from_second_moment_nonnegative(f32::NAN), Aw002Verdict::Fail); }

    // ----- AW-003 ------------------------------------------------------------

    #[test]
    fn aw003_pass_canonical() {
        // Pass band: any beta in (0, 1) and t in [1, ~200] for beta=0.9
        // (beyond ~700 steps with beta=0.9, f32 underflows beta^t to 0.0
        //  and the correction collapses to 1.0 — that's a pure FP artifact,
        //  not a contract violation, so we test inside the safe range).
        assert_eq!(verdict_from_bias_correction(0.9, 1), Aw003Verdict::Pass);
        assert_eq!(verdict_from_bias_correction(0.9, 100), Aw003Verdict::Pass);
        assert_eq!(verdict_from_bias_correction(0.999, 100), Aw003Verdict::Pass);
        assert_eq!(verdict_from_bias_correction(0.999, 5000), Aw003Verdict::Pass);
    }

    #[test]
    fn aw003_fail_zero_t() {
        assert_eq!(verdict_from_bias_correction(0.9, 0), Aw003Verdict::Fail);
    }

    #[test]
    fn aw003_fail_beta_out_of_range() {
        assert_eq!(verdict_from_bias_correction(0.0, 5), Aw003Verdict::Fail);
        assert_eq!(verdict_from_bias_correction(1.0, 5), Aw003Verdict::Fail);
    }

    // ----- AW-004 ------------------------------------------------------------

    #[test] fn aw004_pass_finite_update() { assert_eq!(verdict_from_update_finiteness(1.0, 1e-8, 0.5), Aw004Verdict::Pass); }
    #[test] fn aw004_fail_zero_eps() { assert_eq!(verdict_from_update_finiteness(1.0, 0.0, 0.5), Aw004Verdict::Fail); }
    #[test] fn aw004_fail_inf_grad() { assert_eq!(verdict_from_update_finiteness(f32::INFINITY, 1e-8, 0.5), Aw004Verdict::Fail); }
    #[test] fn aw004_fail_nan_theta() { assert_eq!(verdict_from_update_finiteness(1.0, 1e-8, f32::NAN), Aw004Verdict::Fail); }

    // ----- AW-005 ------------------------------------------------------------

    #[test]
    fn aw005_pass_exact() {
        assert_eq!(verdict_from_simd_equivalence(0.5, 0.5), Aw005Verdict::Pass);
    }

    #[test]
    fn aw005_pass_within_8_ulp() {
        let scalar = 1.0_f32;
        // Increment 5 ULPs.
        let simd = f32::from_bits(scalar.to_bits() + 5);
        assert_eq!(verdict_from_simd_equivalence(simd, scalar), Aw005Verdict::Pass);
    }

    #[test]
    fn aw005_fail_far_apart() {
        let simd = f32::from_bits(1.0_f32.to_bits() + 100);
        assert_eq!(verdict_from_simd_equivalence(simd, 1.0), Aw005Verdict::Fail);
    }

    #[test]
    fn aw005_fail_nan() {
        assert_eq!(verdict_from_simd_equivalence(f32::NAN, 1.0), Aw005Verdict::Fail);
    }

    // ----- AW-006 ------------------------------------------------------------

    #[test]
    fn aw006_pass_pure_decay() {
        // theta_new = 1.0 * (1 - 1e-3 * 0.01) = 0.99999
        let theta_new = 1.0 * (1.0 - 1e-3 * 0.01);
        assert_eq!(
            verdict_from_zero_gradient_only_decay(1.0, theta_new, 1e-3, 0.01),
            Aw006Verdict::Pass
        );
    }

    #[test]
    fn aw006_fail_extra_step() {
        // theta_new diverges from pure-decay expectation.
        assert_eq!(
            verdict_from_zero_gradient_only_decay(1.0, 0.5, 1e-3, 0.01),
            Aw006Verdict::Fail
        );
    }

    #[test]
    fn aw006_fail_zero_lr() {
        assert_eq!(
            verdict_from_zero_gradient_only_decay(1.0, 1.0, 0.0, 0.01),
            Aw006Verdict::Fail
        );
    }

    // ----- AW-007 ------------------------------------------------------------

    #[test]
    fn aw007_pass_validation() {
        assert_eq!(verdict_from_hyperparam_validation(), Aw007Verdict::Pass);
    }

    // ----- AW-008 ------------------------------------------------------------

    #[test]
    fn aw008_pass_unchanged() {
        let g = vec![0.1, 0.2, 0.3];
        assert_eq!(verdict_from_grad_unchanged(&g, &g), Aw008Verdict::Pass);
    }

    #[test]
    fn aw008_fail_modified() {
        let before = [0.1, 0.2, 0.3];
        let after = [0.1, 0.2, 0.5];
        assert_eq!(verdict_from_grad_unchanged(&before, &after), Aw008Verdict::Fail);
    }

    #[test]
    fn aw008_fail_resized() {
        let before = [0.1, 0.2];
        let after = [0.1, 0.2, 0.3];
        assert_eq!(verdict_from_grad_unchanged(&before, &after), Aw008Verdict::Fail);
    }

    // ----- AW-009 ------------------------------------------------------------

    #[test]
    fn aw009_pass_all_nonneg() {
        let history = vec![0.0, 0.1, 0.5, 1.0, 0.99];
        assert_eq!(verdict_from_v_history_nonnegative(&history), Aw009Verdict::Pass);
    }

    #[test]
    fn aw009_fail_one_negative() {
        let history = vec![0.0, 0.1, -1e-10, 1.0];
        assert_eq!(verdict_from_v_history_nonnegative(&history), Aw009Verdict::Fail);
    }

    #[test]
    fn aw009_fail_empty() {
        assert_eq!(verdict_from_v_history_nonnegative(&[]), Aw009Verdict::Fail);
    }

    // ----- AW-010 ------------------------------------------------------------

    #[test]
    fn aw010_pass_correct_formula() {
        let new_m = 0.9 * 0.5 + 0.1 * 1.0; // = 0.55
        assert_eq!(
            verdict_from_first_moment_formula(0.5, 1.0, 0.9, new_m),
            Aw010Verdict::Pass
        );
    }

    #[test]
    fn aw010_fail_swapped_coeffs() {
        // (1-β1)·old_m + β1·g — wrong direction.
        let bad = 0.1 * 0.5 + 0.9 * 1.0; // = 0.95
        assert_eq!(
            verdict_from_first_moment_formula(0.5, 1.0, 0.9, bad),
            Aw010Verdict::Fail
        );
    }

    // ----- AW-011 ------------------------------------------------------------

    #[test] fn aw011_pass_exact() { assert_eq!(verdict_from_loop_terminates(100, 100), Aw011Verdict::Pass); }
    #[test] fn aw011_fail_short() { assert_eq!(verdict_from_loop_terminates(99, 100), Aw011Verdict::Fail); }
    #[test] fn aw011_fail_long() { assert_eq!(verdict_from_loop_terminates(101, 100), Aw011Verdict::Fail); }
    #[test] fn aw011_fail_zero_max() { assert_eq!(verdict_from_loop_terminates(0, 0), Aw011Verdict::Fail); }

    // Provenance pin
    #[test] fn provenance_max_ulp() { assert_eq!(AC_AW_005_MAX_ULP, 8); }
}
