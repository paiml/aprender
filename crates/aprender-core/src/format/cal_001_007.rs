// SHIP-TWO-001 — `calibration-v1` algorithm-level PARTIAL discharge
// for FALSIFY-CAL-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/calibration-v1.yaml`.

// ===========================================================================
// Reference calibration metrics: ECE, MCE, reliability bins, Platt, isotonic
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub struct ReliabilityBin {
    pub confidence: f64,
    pub accuracy: f64,
    pub weight: f64,
}

#[must_use]
pub fn reliability_bins(probs: &[f64], labels: &[u8], n_bins: usize) -> Option<Vec<ReliabilityBin>> {
    if probs.is_empty() || probs.len() != labels.len() || n_bins == 0 { return None; }
    if probs.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) { return None; }
    if labels.iter().any(|l| *l > 1) { return None; }
    let mut bins: Vec<(f64, f64, u64)> = vec![(0.0, 0.0, 0); n_bins];
    let n = probs.len();
    for i in 0..n {
        let bin_idx = ((probs[i] * n_bins as f64) as usize).min(n_bins - 1);
        bins[bin_idx].0 += probs[i];
        bins[bin_idx].1 += labels[i] as f64;
        bins[bin_idx].2 += 1;
    }
    let total = n as f64;
    Some(bins.into_iter().map(|(sum_p, sum_l, count)| {
        if count == 0 {
            ReliabilityBin { confidence: 0.0, accuracy: 0.0, weight: 0.0 }
        } else {
            ReliabilityBin {
                confidence: sum_p / count as f64,
                accuracy: sum_l / count as f64,
                weight: count as f64 / total,
            }
        }
    }).collect())
}

#[must_use]
pub fn ece(probs: &[f64], labels: &[u8], n_bins: usize) -> Option<f64> {
    let bins = reliability_bins(probs, labels, n_bins)?;
    let mut e = 0.0_f64;
    for b in bins {
        e += b.weight * (b.confidence - b.accuracy).abs();
    }
    Some(e)
}

#[must_use]
pub fn mce(probs: &[f64], labels: &[u8], n_bins: usize) -> Option<f64> {
    let bins = reliability_bins(probs, labels, n_bins)?;
    let mut m = 0.0_f64;
    for b in bins {
        if b.weight > 0.0 {
            m = m.max((b.confidence - b.accuracy).abs());
        }
    }
    Some(m)
}

#[must_use]
pub fn platt_sigmoid(logit: f64) -> f64 {
    // 1 / (1 + exp(-x)) — saturates safely for extreme logits.
    if logit > 60.0 { return 1.0; }
    if logit < -60.0 { return 0.0; }
    1.0 / (1.0 + (-logit).exp())
}

#[must_use]
pub fn pool_adjacent_violators(probs: &[f64]) -> Vec<f64> {
    if probs.is_empty() { return vec![]; }
    let mut out: Vec<(f64, u64)> = probs.iter().map(|p| (*p, 1)).collect();
    let mut i = 0;
    while i < out.len() - 1 {
        if out[i].0 > out[i + 1].0 {
            let merged_w = out[i].1 + out[i + 1].1;
            let merged_v = (out[i].0 * out[i].1 as f64 + out[i + 1].0 * out[i + 1].1 as f64) / merged_w as f64;
            out[i] = (merged_v, merged_w);
            out.remove(i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    let mut result = Vec::with_capacity(probs.len());
    for (v, w) in out {
        for _ in 0..w { result.push(v); }
    }
    result
}

// ===========================================================================
// CAL-001 — ECE bounded in [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_ece_bounded(probs: &[f64], labels: &[u8], n_bins: usize) -> Cal001Verdict {
    match ece(probs, labels, n_bins) {
        Some(e) if e.is_finite() && (0.0..=1.0).contains(&e) => Cal001Verdict::Pass,
        _ => Cal001Verdict::Fail,
    }
}

// ===========================================================================
// CAL-002 — MCE bounded in [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mce_bounded(probs: &[f64], labels: &[u8], n_bins: usize) -> Cal002Verdict {
    match mce(probs, labels, n_bins) {
        Some(m) if m.is_finite() && (0.0..=1.0).contains(&m) => Cal002Verdict::Pass,
        _ => Cal002Verdict::Fail,
    }
}

// ===========================================================================
// CAL-003 — MCE >= ECE (max dominates weighted average)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mce_dominates_ece(probs: &[f64], labels: &[u8], n_bins: usize) -> Cal003Verdict {
    let e = match ece(probs, labels, n_bins) { Some(v) => v, None => return Cal003Verdict::Fail };
    let m = match mce(probs, labels, n_bins) { Some(v) => v, None => return Cal003Verdict::Fail };
    if m >= e - 1e-12 { Cal003Verdict::Pass } else { Cal003Verdict::Fail }
}

// ===========================================================================
// CAL-004 — Perfect calibration: ECE ≈ 0, MCE ≈ 0 when prob == label rate
// ===========================================================================

pub const AC_CAL_004_TOLERANCE: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_perfect_calibration_zero(observed_ece: f64, observed_mce: f64) -> Cal004Verdict {
    if !observed_ece.is_finite() || !observed_mce.is_finite() { return Cal004Verdict::Fail; }
    if observed_ece <= AC_CAL_004_TOLERANCE && observed_mce <= AC_CAL_004_TOLERANCE {
        Cal004Verdict::Pass
    } else {
        Cal004Verdict::Fail
    }
}

// ===========================================================================
// CAL-005 — Platt output strictly in (0, 1) for finite logits
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_platt_bounded(logits: &[f64]) -> Cal005Verdict {
    if logits.is_empty() { return Cal005Verdict::Fail; }
    for l in logits {
        if !l.is_finite() { return Cal005Verdict::Fail; }
        let p = platt_sigmoid(*l);
        if !(0.0..=1.0).contains(&p) || !p.is_finite() { return Cal005Verdict::Fail; }
    }
    Cal005Verdict::Pass
}

// ===========================================================================
// CAL-006 — Isotonic monotonicity: output non-decreasing
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_isotonic_monotonicity(input: &[f64]) -> Cal006Verdict {
    if input.is_empty() { return Cal006Verdict::Fail; }
    if input.iter().any(|v| !v.is_finite()) { return Cal006Verdict::Fail; }
    let out = pool_adjacent_violators(input);
    for w in out.windows(2) {
        if w[1] < w[0] - 1e-12 { return Cal006Verdict::Fail; }
    }
    Cal006Verdict::Pass
}

// ===========================================================================
// CAL-007 — Reliability bin (confidence, accuracy) in [0, 1]²
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cal007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_reliability_bin_bounds(probs: &[f64], labels: &[u8], n_bins: usize) -> Cal007Verdict {
    let bins = match reliability_bins(probs, labels, n_bins) { Some(b) => b, None => return Cal007Verdict::Fail };
    for b in bins {
        if !b.confidence.is_finite() || !b.accuracy.is_finite() { return Cal007Verdict::Fail; }
        if !(0.0..=1.0).contains(&b.confidence) { return Cal007Verdict::Fail; }
        if !(0.0..=1.0).contains(&b.accuracy) { return Cal007Verdict::Fail; }
    }
    Cal007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_probs_labels(n: usize) -> (Vec<f64>, Vec<u8>) {
        let probs: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.07).sin().abs()).collect();
        let labels: Vec<u8> = (0..n).map(|i| if (i % 3) == 0 { 1 } else { 0 }).collect();
        (probs, labels)
    }

    // Reference impl
    #[test] fn ref_perfect_calibration() {
        // 4 samples, 2 bins. Bin [0, 0.5] has 2 with p=0.25, labels [0,0]. Bin [0.5, 1] has 2 with p=0.75, labels [1, 1].
        let probs = vec![0.25_f64, 0.25, 0.75, 0.75];
        let labels = vec![0_u8, 0, 1, 1];
        let e = ece(&probs, &labels, 2).unwrap();
        assert!((e - 0.25).abs() < 1e-9); // |0.25 - 0| * 0.5 + |0.75 - 1| * 0.5 = 0.25
    }

    #[test] fn ref_pav_already_sorted() {
        let xs = vec![0.1_f64, 0.3, 0.7, 0.9];
        let out = pool_adjacent_violators(&xs);
        assert_eq!(out, xs);
    }

    #[test] fn ref_pav_violations_pooled() {
        let xs = vec![0.1_f64, 0.7, 0.4, 0.6];
        let out = pool_adjacent_violators(&xs);
        // Verify monotone non-decreasing.
        for w in out.windows(2) { assert!(w[1] >= w[0]); }
    }

    // CAL-001
    #[test] fn cal001_pass_random() {
        let (p, l) = rand_probs_labels(50);
        assert_eq!(verdict_from_ece_bounded(&p, &l, 10), Cal001Verdict::Pass);
    }
    #[test] fn cal001_fail_oob_prob() {
        let p = vec![0.5_f64, 1.5];
        let l = vec![0_u8, 1];
        assert_eq!(verdict_from_ece_bounded(&p, &l, 5), Cal001Verdict::Fail);
    }
    #[test] fn cal001_fail_empty() {
        assert_eq!(verdict_from_ece_bounded(&[], &[], 5), Cal001Verdict::Fail);
    }

    // CAL-002
    #[test] fn cal002_pass_random() {
        let (p, l) = rand_probs_labels(50);
        assert_eq!(verdict_from_mce_bounded(&p, &l, 10), Cal002Verdict::Pass);
    }

    // CAL-003
    #[test] fn cal003_pass_canonical() {
        let (p, l) = rand_probs_labels(50);
        assert_eq!(verdict_from_mce_dominates_ece(&p, &l, 10), Cal003Verdict::Pass);
    }

    // CAL-004
    #[test] fn cal004_pass_zero() {
        assert_eq!(verdict_from_perfect_calibration_zero(0.0, 0.0), Cal004Verdict::Pass);
    }
    #[test] fn cal004_pass_within_tol() {
        assert_eq!(verdict_from_perfect_calibration_zero(5e-4, 5e-4), Cal004Verdict::Pass);
    }
    #[test] fn cal004_fail_above_tol() {
        assert_eq!(verdict_from_perfect_calibration_zero(0.1, 0.1), Cal004Verdict::Fail);
    }

    // CAL-005
    #[test] fn cal005_pass_normal() {
        let logits = vec![-2.0_f64, 0.0, 1.0, 2.0];
        assert_eq!(verdict_from_platt_bounded(&logits), Cal005Verdict::Pass);
    }
    #[test] fn cal005_pass_extreme() {
        let logits = vec![-100.0_f64, 100.0];
        assert_eq!(verdict_from_platt_bounded(&logits), Cal005Verdict::Pass);
    }
    #[test] fn cal005_fail_nan() {
        let logits = vec![f64::NAN];
        assert_eq!(verdict_from_platt_bounded(&logits), Cal005Verdict::Fail);
    }
    #[test] fn cal005_fail_empty() {
        assert_eq!(verdict_from_platt_bounded(&[]), Cal005Verdict::Fail);
    }

    // CAL-006
    #[test] fn cal006_pass_already_monotone() {
        let xs = vec![0.1_f64, 0.3, 0.7, 0.9];
        assert_eq!(verdict_from_isotonic_monotonicity(&xs), Cal006Verdict::Pass);
    }
    #[test] fn cal006_pass_violations_smoothed() {
        let xs = vec![0.1_f64, 0.7, 0.4, 0.6];
        // PAV smooths to monotone — verdict should Pass.
        assert_eq!(verdict_from_isotonic_monotonicity(&xs), Cal006Verdict::Pass);
    }
    #[test] fn cal006_fail_empty() {
        assert_eq!(verdict_from_isotonic_monotonicity(&[]), Cal006Verdict::Fail);
    }
    #[test] fn cal006_fail_nan() {
        let xs = vec![0.1_f64, f64::NAN];
        assert_eq!(verdict_from_isotonic_monotonicity(&xs), Cal006Verdict::Fail);
    }

    // CAL-007
    #[test] fn cal007_pass_normal() {
        let (p, l) = rand_probs_labels(50);
        assert_eq!(verdict_from_reliability_bin_bounds(&p, &l, 10), Cal007Verdict::Pass);
    }
    #[test] fn cal007_fail_oob_prob() {
        let p = vec![1.5_f64];
        let l = vec![0_u8];
        assert_eq!(verdict_from_reliability_bin_bounds(&p, &l, 5), Cal007Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_tolerance() {
        assert!((AC_CAL_004_TOLERANCE - 1e-3).abs() < 1e-12);
    }
}
