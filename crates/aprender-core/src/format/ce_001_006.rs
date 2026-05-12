// SHIP-TWO-001 — `cross-entropy-kernel-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CE-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/cross-entropy-kernel-v1.yaml`.
// Spec: Cross-entropy kernel — log-sum-exp stable cross-entropy loss
// (Shannon 1948 + Milakov & Gimelshein 2018 online normalizer).

// ===========================================================================
// Helpers — log_softmax + cross_entropy reference impls
// ===========================================================================

#[must_use]
pub fn log_softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() { return vec![]; }
    if !x.iter().all(|v| v.is_finite()) { return vec![]; }
    let m = x.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    if !m.is_finite() { return vec![]; }
    let exps: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s == 0.0 || !s.is_finite() { return vec![]; }
    let log_s = s.ln();
    if !log_s.is_finite() { return vec![]; }
    x.iter().map(|&v| v - m - log_s).collect()
}

/// CE(targets, logits) = -Σ targets_i · log_softmax(logits)_i
/// where targets is a probability vector summing to 1.
#[must_use]
pub fn cross_entropy(targets: &[f32], logits: &[f32]) -> Option<f32> {
    if targets.is_empty() || logits.is_empty() { return None; }
    if targets.len() != logits.len() { return None; }
    if !targets.iter().all(|v| v.is_finite() && *v >= 0.0) { return None; }
    if !logits.iter().all(|v| v.is_finite()) { return None; }
    let ls = log_softmax(logits);
    if ls.is_empty() { return None; }
    let mut acc = 0.0_f32;
    for (&t, &l) in targets.iter().zip(ls.iter()) {
        acc -= t * l;
    }
    if !acc.is_finite() { return None; }
    Some(acc)
}

// ===========================================================================
// CE-001 — Non-negativity: CE(targets, logits) ≥ 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ce001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_non_negativity(targets: &[f32], logits: &[f32]) -> Ce001Verdict {
    match cross_entropy(targets, logits) {
        Some(ce) if ce >= -1.0e-6 => Ce001Verdict::Pass, // tiny rounding slack
        _ => Ce001Verdict::Fail,
    }
}

// ===========================================================================
// CE-002 — Log-softmax bounded above by zero: log_softmax(x)_i ≤ 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ce002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_log_softmax_upper_bound(logits: &[f32]) -> Ce002Verdict {
    let ls = log_softmax(logits);
    if ls.is_empty() { return Ce002Verdict::Fail; }
    for &v in &ls {
        if !v.is_finite() { return Ce002Verdict::Fail; }
        // Allow tiny f32 rounding above zero (~1 ULP at log scale).
        if v > 1.0e-6 { return Ce002Verdict::Fail; }
    }
    Ce002Verdict::Pass
}

// ===========================================================================
// CE-003 — Numerical stability: no NaN/Inf for finite logits
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ce003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_numerical_stability(targets: &[f32], logits: &[f32]) -> Ce003Verdict {
    if targets.is_empty() || logits.is_empty() { return Ce003Verdict::Fail; }
    if !logits.iter().all(|v| v.is_finite()) { return Ce003Verdict::Fail; }
    if !targets.iter().all(|v| v.is_finite() && *v >= 0.0) { return Ce003Verdict::Fail; }
    match cross_entropy(targets, logits) {
        Some(ce) if ce.is_finite() => Ce003Verdict::Pass,
        _ => Ce003Verdict::Fail,
    }
}

// ===========================================================================
// CE-004 — Decomposition: |CE(t, x) - NLL(t, log_softmax(x))| ≤ 1e-6
// ===========================================================================

pub const AC_CE_004_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ce004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_decomposition(targets: &[f32], logits: &[f32]) -> Ce004Verdict {
    let fused = match cross_entropy(targets, logits) {
        Some(ce) => ce,
        None => return Ce004Verdict::Fail,
    };
    let ls = log_softmax(logits);
    if ls.is_empty() { return Ce004Verdict::Fail; }
    let nll: f32 = -targets.iter().zip(ls.iter()).map(|(&t, &l)| t * l).sum::<f32>();
    if !nll.is_finite() { return Ce004Verdict::Fail; }
    if (fused - nll).abs() > AC_CE_004_TOLERANCE { return Ce004Verdict::Fail; }
    Ce004Verdict::Pass
}

// ===========================================================================
// CE-005 — SIMD parity within 8 ULP
// ===========================================================================

pub const AC_CE_005_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ce005Verdict { Pass, Fail }

#[must_use]
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    if !a.is_finite() || !b.is_finite() { return u32::MAX; }
    if a == b { return 0; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    let ord_a = if ai < 0 { i32::MIN.wrapping_sub(ai).wrapping_add(1) } else { ai };
    let ord_b = if bi < 0 { i32::MIN.wrapping_sub(bi).wrapping_add(1) } else { bi };
    ord_a.wrapping_sub(ord_b).unsigned_abs()
}

#[must_use]
pub fn verdict_from_simd_parity(scalar: f32, simd: f32) -> Ce005Verdict {
    if !scalar.is_finite() || !simd.is_finite() { return Ce005Verdict::Fail; }
    if ulp_distance(scalar, simd) > AC_CE_005_ULP_TOLERANCE { return Ce005Verdict::Fail; }
    Ce005Verdict::Pass
}

// ===========================================================================
// CE-006 — Perfect prediction boundary: CE → 0 as dominant_logit → ∞
// ===========================================================================

pub const AC_CE_006_TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ce006Verdict { Pass, Fail }

/// Pass iff CE(one_hot(k), logits) is nearly 0 when logits[k] >> logits[j].
/// Caller passes a logit vector with logits[target_idx] much larger than
/// the rest; the verdict computes CE and verifies it's small.
#[must_use]
pub fn verdict_from_perfect_prediction(target_idx: usize, logits: &[f32]) -> Ce006Verdict {
    if logits.is_empty() || target_idx >= logits.len() { return Ce006Verdict::Fail; }
    if !logits.iter().all(|v| v.is_finite()) { return Ce006Verdict::Fail; }
    let mut targets = vec![0.0_f32; logits.len()];
    targets[target_idx] = 1.0;
    match cross_entropy(&targets, logits) {
        Some(ce) if ce.is_finite() && ce < AC_CE_006_TOLERANCE => Ce006Verdict::Pass,
        _ => Ce006Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CE-001 (non-negativity)
    #[test] fn ce001_pass_uniform_target() {
        let targets = vec![1.0_f32, 0.0, 0.0];
        let logits = vec![0.5_f32, 0.3, 0.2];
        assert_eq!(verdict_from_non_negativity(&targets, &logits), Ce001Verdict::Pass);
    }
    #[test] fn ce001_pass_perfect_match() {
        // When logits match target perfectly, CE is small but ≥ 0.
        let targets = vec![1.0_f32, 0.0];
        let logits = vec![100.0_f32, -100.0];
        assert_eq!(verdict_from_non_negativity(&targets, &logits), Ce001Verdict::Pass);
    }
    #[test] fn ce001_fail_empty() {
        assert_eq!(verdict_from_non_negativity(&[], &[]), Ce001Verdict::Fail);
    }
    #[test] fn ce001_fail_negative_target() {
        // Targets must be ≥ 0; verdict rejects.
        let targets = vec![-0.1_f32, 1.1];
        let logits = vec![0.5_f32, 0.5];
        assert_eq!(verdict_from_non_negativity(&targets, &logits), Ce001Verdict::Fail);
    }

    // CE-002 (log-softmax upper bound)
    #[test] fn ce002_pass_canonical() {
        let logits = vec![0.5_f32, 1.0, 1.5, 2.0];
        assert_eq!(verdict_from_log_softmax_upper_bound(&logits), Ce002Verdict::Pass);
    }
    #[test] fn ce002_pass_extreme_logits() {
        // Numerically stable log-softmax handles wide range.
        let logits = vec![100.0_f32, -100.0, 50.0];
        assert_eq!(verdict_from_log_softmax_upper_bound(&logits), Ce002Verdict::Pass);
    }
    #[test] fn ce002_fail_empty() {
        assert_eq!(verdict_from_log_softmax_upper_bound(&[]), Ce002Verdict::Fail);
    }
    #[test] fn ce002_fail_nan() {
        assert_eq!(verdict_from_log_softmax_upper_bound(&[f32::NAN]), Ce002Verdict::Fail);
    }

    // CE-003 (numerical stability)
    #[test] fn ce003_pass_canonical() {
        let targets = vec![1.0_f32, 0.0];
        let logits = vec![0.3_f32, 0.7];
        assert_eq!(verdict_from_numerical_stability(&targets, &logits), Ce003Verdict::Pass);
    }
    #[test] fn ce003_pass_extreme_logits() {
        // The contract's stated falsifier: "Remove max subtraction from
        // log-sum-exp" would produce NaN here — log-sum-exp trick prevents.
        let targets = vec![1.0_f32, 0.0];
        let logits = vec![1000.0_f32, -1000.0];
        assert_eq!(verdict_from_numerical_stability(&targets, &logits), Ce003Verdict::Pass);
    }
    #[test] fn ce003_fail_inf_logit() {
        let targets = vec![1.0_f32];
        let logits = vec![f32::INFINITY];
        assert_eq!(verdict_from_numerical_stability(&targets, &logits), Ce003Verdict::Fail);
    }
    #[test] fn ce003_fail_nan_target() {
        let targets = vec![f32::NAN];
        let logits = vec![1.0_f32];
        assert_eq!(verdict_from_numerical_stability(&targets, &logits), Ce003Verdict::Fail);
    }

    // CE-004 (decomposition)
    #[test] fn ce004_pass_canonical() {
        let targets = vec![1.0_f32, 0.0, 0.0];
        let logits = vec![0.5_f32, 0.3, 0.2];
        assert_eq!(verdict_from_decomposition(&targets, &logits), Ce004Verdict::Pass);
    }
    #[test] fn ce004_pass_uniform_distribution() {
        let targets = vec![0.25_f32, 0.25, 0.25, 0.25];
        let logits = vec![1.0_f32, 1.0, 1.0, 1.0];
        assert_eq!(verdict_from_decomposition(&targets, &logits), Ce004Verdict::Pass);
    }
    #[test] fn ce004_fail_empty() {
        assert_eq!(verdict_from_decomposition(&[], &[]), Ce004Verdict::Fail);
    }

    // CE-005 (SIMD parity)
    #[test] fn ce005_pass_identical() {
        assert_eq!(verdict_from_simd_parity(0.5, 0.5), Ce005Verdict::Pass);
    }
    #[test] fn ce005_pass_within_8_ulp() {
        let a = 0.5_f32;
        let b = f32::from_bits(a.to_bits() + 4);
        assert_eq!(verdict_from_simd_parity(a, b), Ce005Verdict::Pass);
    }
    #[test] fn ce005_fail_above_8_ulp() {
        let a = 0.5_f32;
        let b = f32::from_bits(a.to_bits() + 100);
        assert_eq!(verdict_from_simd_parity(a, b), Ce005Verdict::Fail);
    }
    #[test] fn ce005_fail_nan() {
        assert_eq!(verdict_from_simd_parity(0.5, f32::NAN), Ce005Verdict::Fail);
    }

    // CE-006 (perfect prediction)
    #[test] fn ce006_pass_dominant_logit() {
        // logits[1] >> others → CE(one_hot(1), logits) ≈ 0.
        let logits = vec![-50.0_f32, 50.0, -50.0];
        assert_eq!(verdict_from_perfect_prediction(1, &logits), Ce006Verdict::Pass);
    }
    #[test] fn ce006_fail_uniform_logits() {
        // Uniform logits → CE = log(V) ≈ 1.099 for V=3, far from 0.
        let logits = vec![1.0_f32, 1.0, 1.0];
        assert_eq!(verdict_from_perfect_prediction(0, &logits), Ce006Verdict::Fail);
    }
    #[test] fn ce006_fail_target_idx_oob() {
        let logits = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_perfect_prediction(5, &logits), Ce006Verdict::Fail);
    }
    #[test] fn ce006_fail_empty() {
        assert_eq!(verdict_from_perfect_prediction(0, &[]), Ce006Verdict::Fail);
    }
    #[test] fn ce006_fail_nan() {
        let logits = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_perfect_prediction(0, &logits), Ce006Verdict::Fail);
    }

    // Helper sanity
    #[test] fn log_softmax_uniform() {
        let ls = log_softmax(&[1.0_f32, 1.0, 1.0]);
        for &v in &ls {
            assert!((v - (-3.0_f32.ln())).abs() < 1e-5);
        }
    }
    #[test] fn cross_entropy_one_hot() {
        // Perfect prediction with dominant logit → CE near 0.
        let targets = vec![0.0_f32, 1.0];
        let logits = vec![-100.0_f32, 100.0];
        let ce = cross_entropy(&targets, &logits).unwrap();
        assert!(ce < 1e-3);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_CE_004_TOLERANCE - 1e-6).abs() < 1e-12);
        assert_eq!(AC_CE_005_ULP_TOLERANCE, 8);
        assert!((AC_CE_006_TOLERANCE - 1e-3).abs() < 1e-9);
    }
}
