// SHIP-TWO-001 — `online-softmax-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OSM-001..010 (closes 10/10 sweep).
//
// Contract: `contracts/online-softmax-v1.yaml`.
//
// Bundles 10 verdict fns + a stand-alone reference online-softmax
// scan. The scan tracks `(m_i, d_i)` where `m_i = max(x_1..x_i)` and
// `d_i = Σ_{j=1}^{i} exp(x_j - m_i)`. After the scan,
// `softmax(x)_j = exp(x_j - m_n) / d_n`.

// ===========================================================================
// Reference scalar online softmax
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineState { pub m: f32, pub d: f32 }

#[must_use]
pub const fn empty_state() -> OnlineState {
    OnlineState { m: f32::NEG_INFINITY, d: 0.0 }
}

/// Update `(m, d)` after observing `x_i`. Mathematically:
///   m_new = max(m_old, x_i)
///   d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
#[must_use]
pub fn online_update(prev: OnlineState, x: f32) -> OnlineState {
    if !x.is_finite() { return prev; }
    let m_new = prev.m.max(x);
    let scale = if prev.m == f32::NEG_INFINITY { 0.0 } else { (prev.m - m_new).exp() };
    let d_new = prev.d * scale + (x - m_new).exp();
    OnlineState { m: m_new, d: d_new }
}

/// Full scan returning the per-step state vector (length = n+1, including
/// initial empty state).
#[must_use]
pub fn online_scan(xs: &[f32]) -> Vec<OnlineState> {
    let mut out = Vec::with_capacity(xs.len() + 1);
    let mut s = empty_state();
    out.push(s);
    for x in xs {
        s = online_update(s, *x);
        out.push(s);
    }
    out
}

/// Compute softmax via online scan: `y_i = exp(x_i - m_n) / d_n`.
#[must_use]
pub fn online_softmax(xs: &[f32]) -> Vec<f32> {
    if xs.is_empty() { return vec![]; }
    let mut s = empty_state();
    for x in xs { s = online_update(s, *x); }
    xs.iter().map(|x| (x - s.m).exp() / s.d).collect()
}

/// Standard (two-pass max-subtraction) softmax for comparison.
#[must_use]
pub fn standard_softmax(xs: &[f32]) -> Vec<f32> {
    if xs.is_empty() { return vec![]; }
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    let exps: Vec<f32> = xs.iter().map(|x| {
        let e = (x - m).exp();
        sum += e;
        e
    }).collect();
    exps.into_iter().map(|e| e / sum).collect()
}

// ===========================================================================
// OSM-001 — Online matches standard softmax element-wise
// ===========================================================================

pub const AC_OSM_001_TOLERANCE: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_online_vs_standard(xs: &[f32]) -> Osm001Verdict {
    if xs.is_empty() { return Osm001Verdict::Fail; }
    if xs.iter().any(|v| !v.is_finite()) { return Osm001Verdict::Fail; }
    let online = online_softmax(xs);
    let std_ref = standard_softmax(xs);
    for (a, b) in online.iter().zip(std_ref.iter()) {
        if (a - b).abs() > AC_OSM_001_TOLERANCE { return Osm001Verdict::Fail; }
    }
    Osm001Verdict::Pass
}

// ===========================================================================
// OSM-002 — Sum-to-one
// ===========================================================================

pub const AC_OSM_002_SUM_TOLERANCE: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_sum_to_one(xs: &[f32]) -> Osm002Verdict {
    if xs.is_empty() { return Osm002Verdict::Fail; }
    if xs.iter().any(|v| !v.is_finite()) { return Osm002Verdict::Fail; }
    let y = online_softmax(xs);
    let sum: f32 = y.iter().sum();
    if (sum - 1.0).abs() < AC_OSM_002_SUM_TOLERANCE { Osm002Verdict::Pass } else { Osm002Verdict::Fail }
}

// ===========================================================================
// OSM-003 — Positivity: every output > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_positivity(xs: &[f32]) -> Osm003Verdict {
    if xs.is_empty() { return Osm003Verdict::Fail; }
    let y = online_softmax(xs);
    for v in y {
        // Underflow may produce 0.0; we require strictly positive.
        if v <= 0.0 || !v.is_finite() { return Osm003Verdict::Fail; }
    }
    Osm003Verdict::Pass
}

// ===========================================================================
// OSM-004 — Shift invariance: softmax(x + c) ≈ softmax(x)
// ===========================================================================

pub const AC_OSM_004_TOLERANCE: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_shift_invariance(xs: &[f32], shift: f32) -> Osm004Verdict {
    if xs.is_empty() || !shift.is_finite() { return Osm004Verdict::Fail; }
    if xs.iter().any(|v| !v.is_finite()) { return Osm004Verdict::Fail; }
    let shifted: Vec<f32> = xs.iter().map(|x| x + shift).collect();
    if shifted.iter().any(|v| !v.is_finite()) { return Osm004Verdict::Fail; }
    let a = online_softmax(xs);
    let b = online_softmax(&shifted);
    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > AC_OSM_004_TOLERANCE { return Osm004Verdict::Fail; }
    }
    Osm004Verdict::Pass
}

// ===========================================================================
// OSM-005 — Decoder attention dimensions: works at canonical kv_lens
// ===========================================================================

pub const AC_OSM_005_KV_LENS: &[usize] = &[1, 6, 64, 448, 1500];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_decoder_kv_lens() -> Osm005Verdict {
    for &n in AC_OSM_005_KV_LENS {
        // Reproducible deterministic input.
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let y = online_softmax(&xs);
        if y.len() != n { return Osm005Verdict::Fail; }
        let sum: f32 = y.iter().sum();
        // FP summation error grows ~sqrt(n) for naive sum; scale the
        // tolerance by sqrt(n) over the n=1 baseline.
        let tol = AC_OSM_002_SUM_TOLERANCE * (n as f32).sqrt().max(1.0);
        if (sum - 1.0).abs() > tol { return Osm005Verdict::Fail; }
        if y.iter().any(|v| *v <= 0.0 || !v.is_finite()) { return Osm005Verdict::Fail; }
    }
    Osm005Verdict::Pass
}

// ===========================================================================
// OSM-006 — Single-element softmax([x]) == [1.0]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_single_element(x: f32) -> Osm006Verdict {
    if !x.is_finite() { return Osm006Verdict::Fail; }
    let y = online_softmax(&[x]);
    if y.len() != 1 { return Osm006Verdict::Fail; }
    if (y[0] - 1.0).abs() < 1e-6 { Osm006Verdict::Pass } else { Osm006Verdict::Fail }
}

// ===========================================================================
// OSM-007 — Loop invariant: m_i = max(x_1..x_i)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_running_max_invariant(xs: &[f32]) -> Osm007Verdict {
    if xs.is_empty() { return Osm007Verdict::Fail; }
    if xs.iter().any(|v| !v.is_finite()) { return Osm007Verdict::Fail; }
    let states = online_scan(xs);
    for i in 1..=xs.len() {
        let expected = xs[..i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if states[i].m != expected { return Osm007Verdict::Fail; }
    }
    Osm007Verdict::Pass
}

// ===========================================================================
// OSM-008 — Loop variant: counter advances by 1 per step, terminates at n
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_loop_termination(executed_steps: usize, expected_n: usize) -> Osm008Verdict {
    if expected_n == 0 { return Osm008Verdict::Fail; }
    if executed_steps == expected_n { Osm008Verdict::Pass } else { Osm008Verdict::Fail }
}

// ===========================================================================
// OSM-009 — Old state: d_i computed from old d_{i-1} matches recompute
// ===========================================================================

pub const AC_OSM_009_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm009Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_normalizer_recompute(xs: &[f32]) -> Osm009Verdict {
    if xs.is_empty() { return Osm009Verdict::Fail; }
    if xs.iter().any(|v| !v.is_finite()) { return Osm009Verdict::Fail; }
    let states = online_scan(xs);
    for i in 1..=xs.len() {
        let m_i = states[i].m;
        let recompute: f32 = xs[..i].iter().map(|x| (x - m_i).exp()).sum();
        let online_d = states[i].d;
        let denom = recompute.abs().max(1.0);
        if (online_d - recompute).abs() / denom > AC_OSM_009_TOLERANCE {
            return Osm009Verdict::Fail;
        }
    }
    Osm009Verdict::Pass
}

// ===========================================================================
// OSM-010 — Loop invariant: d_i == Σ exp(x_j - m_i) for j=1..i
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Osm010Verdict { Pass, Fail }

/// Same as OSM-009 in this stand-alone reference (the contract draws
/// the distinction between an in-loop instrument vs. a post-step
/// snapshot; for the algorithm-level rule both reduce to the same
/// equality check).
#[must_use]
pub fn verdict_from_partial_sum_invariant(xs: &[f32]) -> Osm010Verdict {
    match verdict_from_normalizer_recompute(xs) {
        Osm009Verdict::Pass => Osm010Verdict::Pass,
        Osm009Verdict::Fail => Osm010Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() <= eps }

    // ----- Reference impl spot checks ----------------------------------------

    #[test]
    fn ref_softmax_uniform() {
        let y = online_softmax(&[1.0, 1.0, 1.0]);
        for v in y { assert!(approx_eq(v, 1.0 / 3.0, 1e-6)); }
    }

    #[test]
    fn ref_softmax_one_hot() {
        // Large gap: result is essentially [1, 0, 0].
        let y = online_softmax(&[100.0, 0.0, 0.0]);
        assert!(y[0] > 0.999);
        assert!(y[1] < 1e-6);
    }

    // ----- OSM-001 ------------------------------------------------------------

    #[test] fn osm001_pass_uniform() { assert_eq!(verdict_from_online_vs_standard(&[1.0; 32]), Osm001Verdict::Pass); }
    #[test] fn osm001_pass_random_like() {
        let xs: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7 - 5.3).sin()).collect();
        assert_eq!(verdict_from_online_vs_standard(&xs), Osm001Verdict::Pass);
    }
    #[test] fn osm001_pass_extreme_range() {
        assert_eq!(verdict_from_online_vs_standard(&[100.0, -100.0, 50.0]), Osm001Verdict::Pass);
    }
    #[test] fn osm001_fail_empty() { assert_eq!(verdict_from_online_vs_standard(&[]), Osm001Verdict::Fail); }
    #[test] fn osm001_fail_nan() { assert_eq!(verdict_from_online_vs_standard(&[f32::NAN, 1.0]), Osm001Verdict::Fail); }

    // ----- OSM-002 ------------------------------------------------------------

    #[test] fn osm002_pass_uniform() { assert_eq!(verdict_from_sum_to_one(&[1.0; 4096]), Osm002Verdict::Pass); }
    #[test] fn osm002_pass_extreme() { assert_eq!(verdict_from_sum_to_one(&[1000.0, -1000.0, 0.0]), Osm002Verdict::Pass); }
    #[test] fn osm002_fail_empty() { assert_eq!(verdict_from_sum_to_one(&[]), Osm002Verdict::Fail); }
    #[test] fn osm002_fail_inf() { assert_eq!(verdict_from_sum_to_one(&[f32::INFINITY, 1.0]), Osm002Verdict::Fail); }

    // ----- OSM-003 ------------------------------------------------------------

    #[test] fn osm003_pass_normal() { assert_eq!(verdict_from_positivity(&[1.0, 2.0, 3.0]), Osm003Verdict::Pass); }
    #[test] fn osm003_fail_underflow_to_zero() {
        // Extreme negative → underflow to exactly 0.
        let result = online_softmax(&[1000.0, -1000.0]);
        assert!(result[1] == 0.0);  // Confirms underflow.
        assert_eq!(verdict_from_positivity(&[1000.0, -1000.0]), Osm003Verdict::Fail);
    }
    #[test] fn osm003_fail_empty() { assert_eq!(verdict_from_positivity(&[]), Osm003Verdict::Fail); }

    // ----- OSM-004 ------------------------------------------------------------

    #[test] fn osm004_pass_zero_shift() {
        assert_eq!(verdict_from_shift_invariance(&[1.0, 2.0, 3.0], 0.0), Osm004Verdict::Pass);
    }
    #[test] fn osm004_pass_positive_shift() {
        assert_eq!(verdict_from_shift_invariance(&[1.0, 2.0, 3.0], 100.0), Osm004Verdict::Pass);
    }
    #[test] fn osm004_pass_negative_shift() {
        assert_eq!(verdict_from_shift_invariance(&[1.0, 2.0, 3.0], -100.0), Osm004Verdict::Pass);
    }
    #[test] fn osm004_fail_empty() { assert_eq!(verdict_from_shift_invariance(&[], 1.0), Osm004Verdict::Fail); }

    // ----- OSM-005 ------------------------------------------------------------

    #[test] fn osm005_pass_decoder_dims() { assert_eq!(verdict_from_decoder_kv_lens(), Osm005Verdict::Pass); }

    // ----- OSM-006 ------------------------------------------------------------

    #[test] fn osm006_pass_zero() { assert_eq!(verdict_from_single_element(0.0), Osm006Verdict::Pass); }
    #[test] fn osm006_pass_negative() { assert_eq!(verdict_from_single_element(-100.0), Osm006Verdict::Pass); }
    #[test] fn osm006_pass_large() { assert_eq!(verdict_from_single_element(1000.0), Osm006Verdict::Pass); }
    #[test] fn osm006_fail_nan() { assert_eq!(verdict_from_single_element(f32::NAN), Osm006Verdict::Fail); }

    // ----- OSM-007 ------------------------------------------------------------

    #[test] fn osm007_pass_increasing() {
        assert_eq!(verdict_from_running_max_invariant(&[1.0, 2.0, 3.0]), Osm007Verdict::Pass);
    }
    #[test] fn osm007_pass_zigzag() {
        assert_eq!(verdict_from_running_max_invariant(&[5.0, 1.0, 9.0, 2.0]), Osm007Verdict::Pass);
    }
    #[test] fn osm007_pass_constant() {
        assert_eq!(verdict_from_running_max_invariant(&[7.0; 10]), Osm007Verdict::Pass);
    }
    #[test] fn osm007_fail_empty() { assert_eq!(verdict_from_running_max_invariant(&[]), Osm007Verdict::Fail); }

    // ----- OSM-008 ------------------------------------------------------------

    #[test] fn osm008_pass() { assert_eq!(verdict_from_loop_termination(10, 10), Osm008Verdict::Pass); }
    #[test] fn osm008_fail_short() { assert_eq!(verdict_from_loop_termination(9, 10), Osm008Verdict::Fail); }
    #[test] fn osm008_fail_long() { assert_eq!(verdict_from_loop_termination(11, 10), Osm008Verdict::Fail); }
    #[test] fn osm008_fail_zero() { assert_eq!(verdict_from_loop_termination(0, 0), Osm008Verdict::Fail); }

    // ----- OSM-009 ------------------------------------------------------------

    #[test] fn osm009_pass_normal() {
        assert_eq!(verdict_from_normalizer_recompute(&[1.0, 2.0, 3.0]), Osm009Verdict::Pass);
    }
    #[test] fn osm009_pass_extreme_range() {
        assert_eq!(verdict_from_normalizer_recompute(&[100.0, -100.0, 50.0]), Osm009Verdict::Pass);
    }
    #[test] fn osm009_fail_empty() {
        assert_eq!(verdict_from_normalizer_recompute(&[]), Osm009Verdict::Fail);
    }

    // ----- OSM-010 ------------------------------------------------------------

    #[test] fn osm010_pass_normal() {
        assert_eq!(verdict_from_partial_sum_invariant(&[1.0, 2.0, 3.0]), Osm010Verdict::Pass);
    }
    #[test] fn osm010_fail_empty() {
        assert_eq!(verdict_from_partial_sum_invariant(&[]), Osm010Verdict::Fail);
    }

    // ----- Provenance --------------------------------------------------------

    #[test] fn provenance_decoder_kv_lens() {
        assert_eq!(AC_OSM_005_KV_LENS, &[1, 6, 64, 448, 1500][..]);
    }

    #[test] fn provenance_tolerances() {
        assert!((AC_OSM_001_TOLERANCE - 1e-5).abs() < f32::EPSILON);
        assert!((AC_OSM_002_SUM_TOLERANCE - 1e-6).abs() < f32::EPSILON);
        assert!((AC_OSM_004_TOLERANCE - 1e-5).abs() < f32::EPSILON);
    }
}
