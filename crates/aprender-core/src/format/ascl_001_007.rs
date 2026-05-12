// SHIP-TWO-001 — `attention-scaling-v1` algorithm-level PARTIAL
// discharge for FALSIFY-ASCL-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/attention-scaling-v1.yaml`.

// ===========================================================================
// Reference scaled-attention scoring (Q @ K^T / sqrt(d_k))
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionError { ShapeMismatch, EmptyInput, NonFiniteInput }

#[must_use]
pub fn matmul_qk_t(q: &[Vec<f32>], k: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    if q.is_empty() || k.is_empty() { return None; }
    let d_k = q[0].len();
    if d_k == 0 { return None; }
    if q.iter().any(|r| r.len() != d_k) || k.iter().any(|r| r.len() != d_k) { return None; }
    let n = q.len();
    let m = k.len();
    let mut s = vec![vec![0.0_f32; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut acc = 0.0_f32;
            for kk in 0..d_k {
                acc += q[i][kk] * k[j][kk];
            }
            s[i][j] = acc;
        }
    }
    Some(s)
}

#[must_use]
pub fn scaled_scores(q: &[Vec<f32>], k: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    let d_k = q.first().map(|r| r.len()).unwrap_or(0);
    if d_k == 0 { return None; }
    let scale = (d_k as f32).sqrt();
    let mut s = matmul_qk_t(q, k)?;
    for row in &mut s {
        for v in row.iter_mut() { *v /= scale; }
    }
    Some(s)
}

#[must_use]
pub fn softmax_row(x: &[f32]) -> Option<Vec<f64>> {
    if x.is_empty() { return None; }
    if x.iter().any(|v| !v.is_finite()) { return None; }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    let exps: Vec<f64> = x.iter().map(|v| {
        let e = ((*v - max) as f64).exp();
        sum += e;
        e
    }).collect();
    if sum == 0.0 { return None; }
    Some(exps.into_iter().map(|e| e / sum).collect())
}

/// Shannon entropy in nats: H = -Σ p_i * ln(p_i).
#[must_use]
pub fn entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0_f64;
    for p in probs {
        if *p > 0.0 { h -= p * p.ln(); }
    }
    h
}

// ===========================================================================
// ASCL-001 — Scaling factor: variance ≈ 1 (with iid unit-variance Q,K)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl001Verdict { Pass, Fail }

/// Pass iff observed_variance is in (0.5, 2.0). Tight bound for d_k
/// large, but for small d_k samples drift; the contract's prediction
/// is "≈ 1" — accept a 2× wedge.
#[must_use]
pub fn verdict_from_score_variance(observed_variance: f64) -> Ascl001Verdict {
    if !observed_variance.is_finite() || observed_variance <= 0.0 {
        return Ascl001Verdict::Fail;
    }
    if observed_variance > 0.5 && observed_variance < 2.0 {
        Ascl001Verdict::Pass
    } else {
        Ascl001Verdict::Fail
    }
}

// ===========================================================================
// ASCL-002 — Score bound: |score_ij| <= sqrt(d_k) under unit-norm Q, K
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_score_bound(scores: &[Vec<f32>], d_k: usize) -> Ascl002Verdict {
    if scores.is_empty() || d_k == 0 { return Ascl002Verdict::Fail; }
    let bound = (d_k as f32).sqrt();
    for row in scores {
        for v in row {
            if !v.is_finite() || v.abs() > bound + 1e-5 { return Ascl002Verdict::Fail; }
        }
    }
    Ascl002Verdict::Pass
}

// ===========================================================================
// ASCL-003 — Entropy non-negative
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_entropy_nonnegative(probs_rows: &[Vec<f64>]) -> Ascl003Verdict {
    if probs_rows.is_empty() { return Ascl003Verdict::Fail; }
    for row in probs_rows {
        let h = entropy(row);
        if !h.is_finite() || h < -1e-12 { return Ascl003Verdict::Fail; }
    }
    Ascl003Verdict::Pass
}

// ===========================================================================
// ASCL-004 — Max-subtraction equivalence: softmax(x - max) == softmax(x)
// ===========================================================================

pub const AC_ASCL_004_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_max_subtraction(x: &[f32]) -> Ascl004Verdict {
    if x.is_empty() { return Ascl004Verdict::Fail; }
    if x.iter().any(|v| !v.is_finite()) { return Ascl004Verdict::Fail; }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = x.iter().map(|v| v - max).collect();
    let a = match softmax_row(x) { Some(v) => v, None => return Ascl004Verdict::Fail };
    let b = match softmax_row(&shifted) { Some(v) => v, None => return Ascl004Verdict::Fail };
    for (p, q) in a.iter().zip(b.iter()) {
        if (p - q).abs() > AC_ASCL_004_TOLERANCE { return Ascl004Verdict::Fail; }
    }
    Ascl004Verdict::Pass
}

// ===========================================================================
// ASCL-005 — Saturation prevention: scaled entropy >= unscaled entropy
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_saturation_prevention(scaled_entropy: f64, unscaled_entropy: f64) -> Ascl005Verdict {
    if !scaled_entropy.is_finite() || !unscaled_entropy.is_finite() { return Ascl005Verdict::Fail; }
    if scaled_entropy >= unscaled_entropy { Ascl005Verdict::Pass } else { Ascl005Verdict::Fail }
}

// ===========================================================================
// ASCL-006 — Shape correctness: scores.len() == n, scores[i].len() == m
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_score_shape(scores: &[Vec<f32>], expected_n: usize, expected_m: usize) -> Ascl006Verdict {
    if scores.len() != expected_n { return Ascl006Verdict::Fail; }
    for row in scores {
        if row.len() != expected_m { return Ascl006Verdict::Fail; }
    }
    Ascl006Verdict::Pass
}

// ===========================================================================
// ASCL-007 — Entropy upper bound: H(attn_i) <= ln(m)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ascl007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_entropy_upper_bound(probs_rows: &[Vec<f64>]) -> Ascl007Verdict {
    if probs_rows.is_empty() { return Ascl007Verdict::Fail; }
    for row in probs_rows {
        if row.is_empty() { return Ascl007Verdict::Fail; }
        let m = row.len() as f64;
        let max_entropy = m.ln();
        let h = entropy(row);
        if !h.is_finite() || h > max_entropy + 1e-9 { return Ascl007Verdict::Fail; }
    }
    Ascl007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_q_k(d_k: usize, n: usize, m: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let q: Vec<Vec<f32>> = (0..n).map(|i| {
            let v: Vec<f32> = (0..d_k).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { v.iter().map(|x| x / norm).collect() } else { vec![1.0 / (d_k as f32).sqrt(); d_k] }
        }).collect();
        let k: Vec<Vec<f32>> = (0..m).map(|i| {
            let v: Vec<f32> = (0..d_k).map(|j| ((i * 2 + j) as f32 * 0.1).cos()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { v.iter().map(|x| x / norm).collect() } else { vec![1.0 / (d_k as f32).sqrt(); d_k] }
        }).collect();
        (q, k)
    }

    // Reference impl spot checks
    #[test] fn ref_dot_self_unit_norm_is_one() {
        let q = vec![vec![1.0_f32 / 2.0_f32.sqrt(), 1.0_f32 / 2.0_f32.sqrt()]];
        let s = matmul_qk_t(&q, &q).unwrap();
        assert!((s[0][0] - 1.0).abs() < 1e-5);
    }

    // ASCL-001
    #[test] fn ascl001_pass_in_band() {
        assert_eq!(verdict_from_score_variance(1.0), Ascl001Verdict::Pass);
        assert_eq!(verdict_from_score_variance(0.7), Ascl001Verdict::Pass);
        assert_eq!(verdict_from_score_variance(1.5), Ascl001Verdict::Pass);
    }
    #[test] fn ascl001_fail_too_low() {
        assert_eq!(verdict_from_score_variance(0.4), Ascl001Verdict::Fail);
    }
    #[test] fn ascl001_fail_too_high() {
        assert_eq!(verdict_from_score_variance(2.5), Ascl001Verdict::Fail);
    }
    #[test] fn ascl001_fail_zero() {
        assert_eq!(verdict_from_score_variance(0.0), Ascl001Verdict::Fail);
    }

    // ASCL-002
    #[test] fn ascl002_pass_unit_norm() {
        let d_k = 64;
        let (q, k) = unit_q_k(d_k, 4, 4);
        let scores = matmul_qk_t(&q, &k).unwrap();
        assert_eq!(verdict_from_score_bound(&scores, d_k), Ascl002Verdict::Pass);
    }
    #[test] fn ascl002_fail_above_bound() {
        let d_k = 4;
        // Invent a score above sqrt(4) = 2.0.
        let scores = vec![vec![3.0_f32, 1.0], vec![1.0, 0.5]];
        assert_eq!(verdict_from_score_bound(&scores, d_k), Ascl002Verdict::Fail);
    }
    #[test] fn ascl002_fail_zero_d_k() {
        assert_eq!(verdict_from_score_bound(&[vec![0.0]], 0), Ascl002Verdict::Fail);
    }

    // ASCL-003
    #[test] fn ascl003_pass_uniform() {
        let probs = vec![vec![0.25_f64; 4], vec![0.5, 0.5]];
        assert_eq!(verdict_from_entropy_nonnegative(&probs), Ascl003Verdict::Pass);
    }
    #[test] fn ascl003_pass_one_hot() {
        let probs = vec![vec![1.0_f64, 0.0, 0.0, 0.0]];
        assert_eq!(verdict_from_entropy_nonnegative(&probs), Ascl003Verdict::Pass);
    }
    #[test] fn ascl003_fail_empty() {
        assert_eq!(verdict_from_entropy_nonnegative(&[]), Ascl003Verdict::Fail);
    }

    // ASCL-004
    #[test] fn ascl004_pass_normal() {
        let x = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_max_subtraction(&x), Ascl004Verdict::Pass);
    }
    #[test] fn ascl004_pass_extreme() {
        let x = vec![100.0_f32, 1.0, -100.0];
        assert_eq!(verdict_from_max_subtraction(&x), Ascl004Verdict::Pass);
    }
    #[test] fn ascl004_fail_empty() {
        assert_eq!(verdict_from_max_subtraction(&[]), Ascl004Verdict::Fail);
    }
    #[test] fn ascl004_fail_nan() {
        assert_eq!(verdict_from_max_subtraction(&[f32::NAN, 1.0]), Ascl004Verdict::Fail);
    }

    // ASCL-005
    #[test] fn ascl005_pass_scaled_higher() {
        // Scaled (smaller logits) → softmax closer to uniform → higher entropy.
        let scaled = entropy(&[0.25_f64, 0.30, 0.25, 0.20]);
        let unscaled = entropy(&[0.99_f64, 0.01, 0.0, 0.0]); // saturated
        assert_eq!(verdict_from_saturation_prevention(scaled, unscaled), Ascl005Verdict::Pass);
    }
    #[test] fn ascl005_pass_equal() {
        assert_eq!(verdict_from_saturation_prevention(1.5, 1.5), Ascl005Verdict::Pass);
    }
    #[test] fn ascl005_fail_unscaled_higher() {
        assert_eq!(verdict_from_saturation_prevention(0.5, 1.5), Ascl005Verdict::Fail);
    }
    #[test] fn ascl005_fail_nan() {
        assert_eq!(verdict_from_saturation_prevention(f64::NAN, 1.0), Ascl005Verdict::Fail);
    }

    // ASCL-006
    #[test] fn ascl006_pass_canonical() {
        let scores = vec![vec![0.0_f32; 5]; 4];
        assert_eq!(verdict_from_score_shape(&scores, 4, 5), Ascl006Verdict::Pass);
    }
    #[test] fn ascl006_fail_n_drift() {
        let scores = vec![vec![0.0_f32; 5]; 4];
        assert_eq!(verdict_from_score_shape(&scores, 5, 5), Ascl006Verdict::Fail);
    }
    #[test] fn ascl006_fail_m_drift() {
        let scores = vec![vec![0.0_f32; 5]; 4];
        assert_eq!(verdict_from_score_shape(&scores, 4, 6), Ascl006Verdict::Fail);
    }
    #[test] fn ascl006_fail_jagged() {
        let scores = vec![vec![0.0_f32; 5], vec![0.0_f32; 4]];
        assert_eq!(verdict_from_score_shape(&scores, 2, 5), Ascl006Verdict::Fail);
    }

    // ASCL-007
    #[test] fn ascl007_pass_uniform() {
        let probs = vec![vec![0.25_f64; 4]];
        // ln(4) ≈ 1.386; uniform entropy = ln(4); within 1e-9 tol.
        assert_eq!(verdict_from_entropy_upper_bound(&probs), Ascl007Verdict::Pass);
    }
    #[test] fn ascl007_pass_concentrated() {
        let probs = vec![vec![0.97_f64, 0.01, 0.01, 0.01]];
        assert_eq!(verdict_from_entropy_upper_bound(&probs), Ascl007Verdict::Pass);
    }
    #[test] fn ascl007_fail_empty_row() {
        let probs = vec![vec![]];
        assert_eq!(verdict_from_entropy_upper_bound(&probs), Ascl007Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_tolerance() {
        assert!((AC_ASCL_004_TOLERANCE - 1e-9).abs() < 1e-15);
    }
}
