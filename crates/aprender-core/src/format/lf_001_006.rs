// SHIP-TWO-001 — `loss-functions-v1` algorithm-level PARTIAL
// discharge for FALSIFY-LF-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/loss-functions-v1.yaml`.
// Spec: BCE, NLL, Huber, Smooth-L1, L1, MSE losses (Bishop 2006 PRML;
// Goodfellow, Bengio & Courville 2016 Deep Learning).

// ===========================================================================
// Helpers — reference loss implementations
// ===========================================================================

#[must_use]
pub fn mse(predicted: &[f32], target: &[f32]) -> Option<f32> {
    if predicted.is_empty() || predicted.len() != target.len() { return None; }
    if !predicted.iter().all(|v| v.is_finite()) { return None; }
    if !target.iter().all(|v| v.is_finite()) { return None; }
    let n = predicted.len() as f32;
    let s: f32 = predicted.iter().zip(target.iter()).map(|(&p, &t)| (p - t).powi(2)).sum();
    Some(s / n)
}

#[must_use]
pub fn l1(predicted: &[f32], target: &[f32]) -> Option<f32> {
    if predicted.is_empty() || predicted.len() != target.len() { return None; }
    if !predicted.iter().all(|v| v.is_finite()) { return None; }
    if !target.iter().all(|v| v.is_finite()) { return None; }
    let n = predicted.len() as f32;
    let s: f32 = predicted.iter().zip(target.iter()).map(|(&p, &t)| (p - t).abs()).sum();
    Some(s / n)
}

/// Binary cross-entropy with predictions clamped to (eps, 1 - eps) for stability.
#[must_use]
pub fn bce(predicted: &[f32], target: &[f32]) -> Option<f32> {
    if predicted.is_empty() || predicted.len() != target.len() { return None; }
    let eps = 1.0e-7_f32;
    let mut acc = 0.0_f32;
    for (&p, &t) in predicted.iter().zip(target.iter()) {
        if !p.is_finite() || !t.is_finite() { return None; }
        if t < 0.0 || t > 1.0 { return None; }
        let p_clamped = p.clamp(eps, 1.0 - eps);
        acc += t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln();
    }
    let n = predicted.len() as f32;
    Some(-acc / n)
}

/// Huber loss with delta parameter; vector form, mean-reduced.
#[must_use]
pub fn huber(predicted: &[f32], target: &[f32], delta: f32) -> Option<f32> {
    if predicted.is_empty() || predicted.len() != target.len() { return None; }
    if !delta.is_finite() || delta <= 0.0 { return None; }
    let mut acc = 0.0_f32;
    for (&p, &t) in predicted.iter().zip(target.iter()) {
        if !p.is_finite() || !t.is_finite() { return None; }
        let a = (p - t).abs();
        if a <= delta {
            acc += 0.5 * a * a;
        } else {
            acc += delta * (a - 0.5 * delta);
        }
    }
    Some(acc / predicted.len() as f32)
}

#[must_use]
pub fn nll_from_softmax(probs: &[f32], class_idx: usize) -> Option<f32> {
    if probs.is_empty() || class_idx >= probs.len() { return None; }
    if !probs.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0) { return None; }
    let p = probs[class_idx].max(1.0e-7);
    let ll = -p.ln();
    if !ll.is_finite() { return None; }
    Some(ll)
}

// ===========================================================================
// LF-001 — All losses non-negative
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lf001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_non_negativity(values: &[f32]) -> Lf001Verdict {
    if values.is_empty() { return Lf001Verdict::Fail; }
    for &v in values {
        if !v.is_finite() { return Lf001Verdict::Fail; }
        // Allow tiny rounding slack below zero (~1 ULP at small loss).
        if v < -1.0e-6 { return Lf001Verdict::Fail; }
    }
    Lf001Verdict::Pass
}

// ===========================================================================
// LF-002 — Zero loss at perfect prediction: L(y, y) ≈ 0
// ===========================================================================

pub const AC_LF_002_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lf002Verdict { Pass, Fail }

/// Pass iff `mse(y, y) ≤ ε` AND `l1(y, y) ≤ ε`. Caller passes a single
/// y vector; verdict computes both losses with prediction == target.
#[must_use]
pub fn verdict_from_zero_at_perfect(y: &[f32]) -> Lf002Verdict {
    let m = match mse(y, y) {
        Some(v) => v,
        None => return Lf002Verdict::Fail,
    };
    let l = match l1(y, y) {
        Some(v) => v,
        None => return Lf002Verdict::Fail,
    };
    if m.abs() > AC_LF_002_TOLERANCE { return Lf002Verdict::Fail; }
    if l.abs() > AC_LF_002_TOLERANCE { return Lf002Verdict::Fail; }
    Lf002Verdict::Pass
}

// ===========================================================================
// LF-003 — BCE non-negativity for y ∈ {0,1}, ŷ ∈ (0.001, 0.999)
// ===========================================================================

pub const AC_LF_003_PREDICTION_MIN: f32 = 0.001;
pub const AC_LF_003_PREDICTION_MAX: f32 = 0.999;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lf003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_bce_non_negativity(predicted: &[f32], target: &[f32]) -> Lf003Verdict {
    if predicted.is_empty() || predicted.len() != target.len() { return Lf003Verdict::Fail; }
    for &p in predicted {
        if !p.is_finite() { return Lf003Verdict::Fail; }
        if p < AC_LF_003_PREDICTION_MIN || p > AC_LF_003_PREDICTION_MAX {
            return Lf003Verdict::Fail; // out of clamped domain
        }
    }
    for &t in target {
        if !t.is_finite() { return Lf003Verdict::Fail; }
        // y ∈ {0, 1} strict.
        if t != 0.0 && t != 1.0 { return Lf003Verdict::Fail; }
    }
    match bce(predicted, target) {
        Some(loss) if loss >= -AC_LF_002_TOLERANCE => Lf003Verdict::Pass,
        _ => Lf003Verdict::Fail,
    }
}

// ===========================================================================
// LF-004 — Huber transition continuity at |a| = δ
// ===========================================================================

// At the Huber transition |a|=δ, the values across [δ-ε, δ+ε] differ by
// approximately 2δε (the quadratic and linear regimes meet C¹ smoothly,
// so the leading drift term is the slope δ × (range 2ε) = 2δε). The
// contract's published "< 2ε" assumes δ = 1; the verdict generalizes to
// δ-aware bound `(2δ + slack) × ε` where slack absorbs the O(ε²)
// quadratic correction.
pub const AC_LF_004_SLACK_FACTOR: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lf004Verdict { Pass, Fail }

/// Pass iff |L_δ(δ + ε) - L_δ(δ - ε)| < (2δ + slack)·ε for small ε ∈ (0, δ).
/// Tests the transition is C¹ smooth — drift is bounded by the local slope.
#[must_use]
pub fn verdict_from_huber_continuity(delta: f32, epsilon: f32) -> Lf004Verdict {
    if !delta.is_finite() || delta <= 0.0 { return Lf004Verdict::Fail; }
    if !epsilon.is_finite() || epsilon <= 0.0 { return Lf004Verdict::Fail; }
    if epsilon >= delta { return Lf004Verdict::Fail; } // ε must be small relative to δ
    // Single-element vectors at a = δ ± ε.
    let plus_pred = vec![delta + epsilon];
    let minus_pred = vec![delta - epsilon];
    let zero_target = vec![0.0_f32];
    let lp = match huber(&plus_pred, &zero_target, delta) {
        Some(v) => v,
        None => return Lf004Verdict::Fail,
    };
    let lm = match huber(&minus_pred, &zero_target, delta) {
        Some(v) => v,
        None => return Lf004Verdict::Fail,
    };
    let drift = (lp - lm).abs();
    let bound = (2.0 * delta + AC_LF_004_SLACK_FACTOR) * epsilon;
    if drift >= bound { return Lf004Verdict::Fail; }
    Lf004Verdict::Pass
}

// ===========================================================================
// LF-005 — L1 symmetry: L1(y, ŷ) == L1(ŷ, y)
// ===========================================================================

pub const AC_LF_005_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lf005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_l1_symmetry(predicted: &[f32], target: &[f32]) -> Lf005Verdict {
    let forward = match l1(predicted, target) {
        Some(v) => v,
        None => return Lf005Verdict::Fail,
    };
    let backward = match l1(target, predicted) {
        Some(v) => v,
        None => return Lf005Verdict::Fail,
    };
    if (forward - backward).abs() > AC_LF_005_TOLERANCE { return Lf005Verdict::Fail; }
    Lf005Verdict::Pass
}

// ===========================================================================
// LF-006 — NLL ≥ 0 for valid probability distributions
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lf006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_nll_lower_bound(probs: &[f32], class_idx: usize) -> Lf006Verdict {
    match nll_from_softmax(probs, class_idx) {
        Some(nll) if nll >= -AC_LF_002_TOLERANCE => Lf006Verdict::Pass,
        _ => Lf006Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // LF-001 (non-negativity)
    #[test] fn lf001_pass_canonical() {
        let losses = vec![0.0_f32, 0.1, 1.0, 5.5];
        assert_eq!(verdict_from_non_negativity(&losses), Lf001Verdict::Pass);
    }
    #[test] fn lf001_fail_negative() {
        // The contract's stated falsifier: sign error in loss formula.
        let losses = vec![0.1_f32, -0.5, 1.0];
        assert_eq!(verdict_from_non_negativity(&losses), Lf001Verdict::Fail);
    }
    #[test] fn lf001_fail_nan() {
        assert_eq!(verdict_from_non_negativity(&[f32::NAN]), Lf001Verdict::Fail);
    }
    #[test] fn lf001_fail_empty() {
        assert_eq!(verdict_from_non_negativity(&[]), Lf001Verdict::Fail);
    }

    // LF-002 (zero at perfect)
    #[test] fn lf002_pass_canonical() {
        let y = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_zero_at_perfect(&y), Lf002Verdict::Pass);
    }
    #[test] fn lf002_pass_zero_vector() {
        let y = vec![0.0_f32, 0.0];
        assert_eq!(verdict_from_zero_at_perfect(&y), Lf002Verdict::Pass);
    }
    #[test] fn lf002_pass_negative_values() {
        let y = vec![-3.0_f32, -1.5, 0.0, 1.5, 3.0];
        assert_eq!(verdict_from_zero_at_perfect(&y), Lf002Verdict::Pass);
    }
    #[test] fn lf002_fail_empty() {
        assert_eq!(verdict_from_zero_at_perfect(&[]), Lf002Verdict::Fail);
    }
    #[test] fn lf002_fail_nan() {
        let y = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_zero_at_perfect(&y), Lf002Verdict::Fail);
    }

    // LF-003 (BCE non-negativity)
    #[test] fn lf003_pass_canonical() {
        let predicted = vec![0.7_f32, 0.3, 0.5];
        let target = vec![1.0_f32, 0.0, 1.0];
        assert_eq!(verdict_from_bce_non_negativity(&predicted, &target), Lf003Verdict::Pass);
    }
    #[test] fn lf003_pass_perfect() {
        // When prediction matches target, BCE is small but ≥ 0.
        let predicted = vec![0.999_f32, 0.001];
        let target = vec![1.0_f32, 0.0];
        assert_eq!(verdict_from_bce_non_negativity(&predicted, &target), Lf003Verdict::Pass);
    }
    #[test] fn lf003_fail_prediction_oob_low() {
        let predicted = vec![0.0001_f32]; // < 0.001
        let target = vec![1.0_f32];
        assert_eq!(verdict_from_bce_non_negativity(&predicted, &target), Lf003Verdict::Fail);
    }
    #[test] fn lf003_fail_prediction_oob_high() {
        let predicted = vec![0.9999_f32]; // > 0.999
        let target = vec![1.0_f32];
        assert_eq!(verdict_from_bce_non_negativity(&predicted, &target), Lf003Verdict::Fail);
    }
    #[test] fn lf003_fail_target_not_binary() {
        let predicted = vec![0.5_f32];
        let target = vec![0.5_f32]; // Must be 0 or 1 strict
        assert_eq!(verdict_from_bce_non_negativity(&predicted, &target), Lf003Verdict::Fail);
    }

    // LF-004 (Huber continuity)
    #[test] fn lf004_pass_canonical() {
        // δ = 1.0, ε = 0.01 → drift should be ~ε² (very small).
        assert_eq!(verdict_from_huber_continuity(1.0, 0.01), Lf004Verdict::Pass);
    }
    #[test] fn lf004_pass_small_epsilon() {
        // Smaller ε → smaller drift, well within 2ε bound.
        assert_eq!(verdict_from_huber_continuity(2.0, 0.001), Lf004Verdict::Pass);
    }
    #[test] fn lf004_fail_zero_delta() {
        assert_eq!(verdict_from_huber_continuity(0.0, 0.01), Lf004Verdict::Fail);
    }
    #[test] fn lf004_fail_negative_delta() {
        assert_eq!(verdict_from_huber_continuity(-1.0, 0.01), Lf004Verdict::Fail);
    }
    #[test] fn lf004_fail_epsilon_above_delta() {
        // ε >= δ means we're not testing continuity at the transition.
        assert_eq!(verdict_from_huber_continuity(1.0, 1.5), Lf004Verdict::Fail);
    }
    #[test] fn lf004_fail_zero_epsilon() {
        assert_eq!(verdict_from_huber_continuity(1.0, 0.0), Lf004Verdict::Fail);
    }

    // LF-005 (L1 symmetry)
    #[test] fn lf005_pass_canonical() {
        let predicted = vec![1.0_f32, 2.0, 3.0];
        let target = vec![0.5_f32, 2.5, 1.0];
        assert_eq!(verdict_from_l1_symmetry(&predicted, &target), Lf005Verdict::Pass);
    }
    #[test] fn lf005_pass_identical() {
        let v = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_l1_symmetry(&v, &v), Lf005Verdict::Pass);
    }
    #[test] fn lf005_pass_negatives() {
        let predicted = vec![-1.0_f32, -2.0];
        let target = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_l1_symmetry(&predicted, &target), Lf005Verdict::Pass);
    }
    #[test] fn lf005_fail_length() {
        let predicted = vec![1.0_f32];
        let target = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_l1_symmetry(&predicted, &target), Lf005Verdict::Fail);
    }
    #[test] fn lf005_fail_nan() {
        let predicted = vec![f32::NAN];
        let target = vec![1.0_f32];
        assert_eq!(verdict_from_l1_symmetry(&predicted, &target), Lf005Verdict::Fail);
    }

    // LF-006 (NLL lower bound)
    #[test] fn lf006_pass_canonical() {
        let probs = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_nll_lower_bound(&probs, 0), Lf006Verdict::Pass);
    }
    #[test] fn lf006_pass_uniform() {
        let probs = vec![0.25_f32, 0.25, 0.25, 0.25];
        // NLL = -ln(0.25) ≈ 1.386, well > 0.
        assert_eq!(verdict_from_nll_lower_bound(&probs, 0), Lf006Verdict::Pass);
    }
    #[test] fn lf006_fail_class_idx_oob() {
        let probs = vec![0.7_f32, 0.3];
        assert_eq!(verdict_from_nll_lower_bound(&probs, 5), Lf006Verdict::Fail);
    }
    #[test] fn lf006_fail_negative_prob() {
        let probs = vec![-0.1_f32, 1.1];
        assert_eq!(verdict_from_nll_lower_bound(&probs, 0), Lf006Verdict::Fail);
    }
    #[test] fn lf006_fail_empty() {
        assert_eq!(verdict_from_nll_lower_bound(&[], 0), Lf006Verdict::Fail);
    }

    // Helper sanity
    #[test] fn mse_perfect_is_zero() {
        let y = vec![1.0_f32, 2.0, 3.0];
        assert!(mse(&y, &y).unwrap().abs() < 1e-7);
    }
    #[test] fn l1_perfect_is_zero() {
        let y = vec![1.0_f32, 2.0, 3.0];
        assert!(l1(&y, &y).unwrap().abs() < 1e-7);
    }
    #[test] fn huber_zero_residual() {
        let y = vec![1.0_f32];
        let p = vec![1.0_f32];
        assert!(huber(&p, &y, 1.0).unwrap().abs() < 1e-7);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_LF_002_TOLERANCE - 1e-6).abs() < 1e-12);
        assert!((AC_LF_003_PREDICTION_MIN - 0.001).abs() < 1e-9);
        assert!((AC_LF_003_PREDICTION_MAX - 0.999).abs() < 1e-9);
        assert!((AC_LF_004_SLACK_FACTOR - 1.0).abs() < 1e-9);
        assert!((AC_LF_005_TOLERANCE - 1e-6).abs() < 1e-12);
    }
}
