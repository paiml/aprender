// SHIP-TWO-001 — `rope-extrapolation-v1` algorithm-level PARTIAL
// discharge for FALSIFY-REXT-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/rope-extrapolation-v1.yaml`.
//
// Bundles 8 verdict fns + RoPE base-frequency / NTK-scaling /
// linear-interpolation / YaRN-ramp / 2D-rotation reference impls.


// ===========================================================================
// Reference: base RoPE freq vector
//   freq_i = theta^(-2*i / d) for i in [0, d/2)
// ===========================================================================

#[must_use]
pub fn base_freqs(theta: f64, d: u64) -> Vec<f64> {
    if d == 0 || !d.is_multiple_of(2) || theta <= 1.0 { return vec![]; }
    let half = d / 2;
    (0..half).map(|i| {
        let p = -2.0_f64 * (i as f64) / (d as f64);
        theta.powf(p)
    }).collect()
}

// ===========================================================================
// REXT-001 — Base frequency: freq_0 == 1.0 AND strictly decreasing
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_base_freq_shape(theta: f64, d: u64) -> Rext001Verdict {
    let freqs = base_freqs(theta, d);
    if freqs.len() < 2 { return Rext001Verdict::Fail; }
    if (freqs[0] - 1.0).abs() > 1e-12 { return Rext001Verdict::Fail; }
    for w in freqs.windows(2) {
        if w[0] <= w[1] { return Rext001Verdict::Fail; }
    }
    Rext001Verdict::Pass
}

// ===========================================================================
// REXT-002 — NTK identity: theta unchanged when L_new == L_orig
// ===========================================================================
//
// NTK-scaled base: theta' = theta * (L_new / L_orig)^(d / (d - 2))
// When L_new == L_orig, the ratio is 1 and theta' == theta.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext002Verdict { Pass, Fail }

#[must_use]
pub fn ntk_scaled_base(theta: f64, d: u64, l_orig: u64, l_new: u64) -> Option<f64> {
    if d < 2 || l_orig == 0 || l_new == 0 || theta <= 0.0 { return None; }
    let ratio = (l_new as f64) / (l_orig as f64);
    let exponent = (d as f64) / ((d - 2) as f64);
    Some(theta * ratio.powf(exponent))
}

#[must_use]
pub fn verdict_from_ntk_identity(theta: f64, d: u64, l: u64) -> Rext002Verdict {
    let theta_new = match ntk_scaled_base(theta, d, l, l) { Some(v) => v, None => return Rext002Verdict::Fail };
    if (theta_new - theta).abs() <= theta.abs() * 1e-12 { Rext002Verdict::Pass } else { Rext002Verdict::Fail }
}

// ===========================================================================
// REXT-003 — Linear interpolation preserves freq ratios
// ===========================================================================
//
// Linear interp scales freqs uniformly: freq'_i = freq_i / s.
// Therefore freq'_i / freq'_j = freq_i / freq_j.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_linear_interp_ratio(theta: f64, d: u64, scale: f64) -> Rext003Verdict {
    if scale <= 0.0 || !scale.is_finite() { return Rext003Verdict::Fail; }
    let base = base_freqs(theta, d);
    if base.len() < 2 { return Rext003Verdict::Fail; }
    let scaled: Vec<f64> = base.iter().map(|f| f / scale).collect();
    // Pick i=0, j=1 as canonical pair.
    let r_orig = base[0] / base[1];
    let r_new = scaled[0] / scaled[1];
    if (r_orig - r_new).abs() <= r_orig.abs() * 1e-12 { Rext003Verdict::Pass } else { Rext003Verdict::Fail }
}

// ===========================================================================
// REXT-004 — YaRN ramp: clamp output to [0, 1]
// ===========================================================================
//
// ramp(r) = clamp((r - low) / (high - low), 0, 1)

#[must_use]
pub fn yarn_ramp(r: f64, low: f64, high: f64) -> f64 {
    if high <= low { return 0.0; }
    let raw = (r - low) / (high - low);
    raw.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_yarn_ramp_clamp(r: f64, low: f64, high: f64) -> Rext004Verdict {
    if !r.is_finite() || !low.is_finite() || !high.is_finite() { return Rext004Verdict::Fail; }
    let v = yarn_ramp(r, low, high);
    if (0.0..=1.0).contains(&v) { Rext004Verdict::Pass } else { Rext004Verdict::Fail }
}

// ===========================================================================
// REXT-005 — 2D rotation orthogonality: R^T R = I within 1e-12
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext005Verdict { Pass, Fail }

/// `R = [[cos, -sin], [sin, cos]]`. Orthogonal iff R^T R == I.
#[must_use]
pub fn verdict_from_rotation_orthogonality(angle: f64) -> Rext005Verdict {
    if !angle.is_finite() { return Rext005Verdict::Fail; }
    let c = angle.cos();
    let s = angle.sin();
    // R^T R produces:
    //   [[c²+s², 0], [0, c²+s²]]
    let diag = c * c + s * s;
    if (diag - 1.0).abs() > 1e-12 { return Rext005Verdict::Fail; }
    Rext005Verdict::Pass
}

// ===========================================================================
// REXT-006 — Position 0 identity: cos(0) = 1, sin(0) = 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_position_zero_identity() -> Rext006Verdict {
    let c = 0.0_f64.cos();
    let s = 0.0_f64.sin();
    if (c - 1.0).abs() > 1e-15 { return Rext006Verdict::Fail; }
    if s.abs() > 1e-15 { return Rext006Verdict::Fail; }
    Rext006Verdict::Pass
}

// ===========================================================================
// REXT-007 — NTK base monotonicity: L1 < L2 ⇒ base'(L1) <= base'(L2)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_ntk_monotonicity(theta: f64, d: u64, l_orig: u64, l1: u64, l2: u64) -> Rext007Verdict {
    if l1 >= l2 { return Rext007Verdict::Fail; }
    let b1 = match ntk_scaled_base(theta, d, l_orig, l1) { Some(v) => v, None => return Rext007Verdict::Fail };
    let b2 = match ntk_scaled_base(theta, d, l_orig, l2) { Some(v) => v, None => return Rext007Verdict::Fail };
    if b1 <= b2 { Rext007Verdict::Pass } else { Rext007Verdict::Fail }
}

// ===========================================================================
// REXT-008 — YaRN ramp non-decreasing: r1 < r2 ⇒ ramp(r1) <= ramp(r2)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rext008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_yarn_ramp_monotonic(r1: f64, r2: f64, low: f64, high: f64) -> Rext008Verdict {
    if r1 >= r2 { return Rext008Verdict::Fail; }
    if !r1.is_finite() || !r2.is_finite() || !low.is_finite() || !high.is_finite() { return Rext008Verdict::Fail; }
    let y1 = yarn_ramp(r1, low, high);
    let y2 = yarn_ramp(r2, low, high);
    if y1 <= y2 { Rext008Verdict::Pass } else { Rext008Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Reference impl spot checks
    #[test] fn ref_base_freqs_canonical() {
        let f = base_freqs(10000.0, 64);
        assert_eq!(f.len(), 32);
        assert!((f[0] - 1.0).abs() < 1e-12);
        assert!(f[31] < f[0]);
    }

    // REXT-001
    #[test] fn rext_001_pass_canonical() {
        assert_eq!(verdict_from_base_freq_shape(10000.0, 64), Rext001Verdict::Pass);
    }
    #[test] fn rext_001_pass_qwen2() {
        assert_eq!(verdict_from_base_freq_shape(1_000_000.0, 128), Rext001Verdict::Pass);
    }
    #[test] fn rext_001_fail_zero_d() {
        assert_eq!(verdict_from_base_freq_shape(10000.0, 0), Rext001Verdict::Fail);
    }
    #[test] fn rext_001_fail_odd_d() {
        assert_eq!(verdict_from_base_freq_shape(10000.0, 65), Rext001Verdict::Fail);
    }
    #[test] fn rext_001_fail_theta_one() {
        assert_eq!(verdict_from_base_freq_shape(1.0, 64), Rext001Verdict::Fail);
    }

    // REXT-002
    #[test] fn rext_002_pass() {
        assert_eq!(verdict_from_ntk_identity(10000.0, 128, 4096), Rext002Verdict::Pass);
    }
    #[test] fn rext_002_pass_large_theta() {
        assert_eq!(verdict_from_ntk_identity(1_000_000.0, 128, 32768), Rext002Verdict::Pass);
    }

    // REXT-003
    #[test] fn rext_003_pass_scale_1() {
        assert_eq!(verdict_from_linear_interp_ratio(10000.0, 64, 1.0), Rext003Verdict::Pass);
    }
    #[test] fn rext_003_pass_scale_2() {
        assert_eq!(verdict_from_linear_interp_ratio(10000.0, 64, 2.0), Rext003Verdict::Pass);
    }
    #[test] fn rext_003_pass_scale_8() {
        assert_eq!(verdict_from_linear_interp_ratio(10000.0, 64, 8.0), Rext003Verdict::Pass);
    }
    #[test] fn rext_003_fail_zero_scale() {
        assert_eq!(verdict_from_linear_interp_ratio(10000.0, 64, 0.0), Rext003Verdict::Fail);
    }

    // REXT-004
    #[test] fn rext_004_pass_in_range() {
        assert_eq!(verdict_from_yarn_ramp_clamp(0.5, 0.0, 1.0), Rext004Verdict::Pass);
    }
    #[test] fn rext_004_pass_below_low() {
        assert_eq!(verdict_from_yarn_ramp_clamp(-100.0, 0.0, 1.0), Rext004Verdict::Pass);
    }
    #[test] fn rext_004_pass_above_high() {
        assert_eq!(verdict_from_yarn_ramp_clamp(100.0, 0.0, 1.0), Rext004Verdict::Pass);
    }
    #[test] fn rext_004_fail_nan() {
        assert_eq!(verdict_from_yarn_ramp_clamp(f64::NAN, 0.0, 1.0), Rext004Verdict::Fail);
    }

    // REXT-005
    #[test] fn rext_005_pass_zero() {
        assert_eq!(verdict_from_rotation_orthogonality(0.0), Rext005Verdict::Pass);
    }
    #[test] fn rext_005_pass_pi_2() {
        assert_eq!(verdict_from_rotation_orthogonality(PI / 2.0), Rext005Verdict::Pass);
    }
    #[test] fn rext_005_pass_random_angle() {
        assert_eq!(verdict_from_rotation_orthogonality(1.234), Rext005Verdict::Pass);
    }
    #[test] fn rext_005_fail_nan() {
        assert_eq!(verdict_from_rotation_orthogonality(f64::NAN), Rext005Verdict::Fail);
    }

    // REXT-006
    #[test] fn rext_006_pass() {
        assert_eq!(verdict_from_position_zero_identity(), Rext006Verdict::Pass);
    }

    // REXT-007
    #[test] fn rext_007_pass_canonical() {
        assert_eq!(
            verdict_from_ntk_monotonicity(10000.0, 128, 4096, 8192, 16384),
            Rext007Verdict::Pass
        );
    }
    #[test] fn rext_007_fail_l1_eq_l2() {
        assert_eq!(
            verdict_from_ntk_monotonicity(10000.0, 128, 4096, 8192, 8192),
            Rext007Verdict::Fail
        );
    }

    // REXT-008
    #[test] fn rext_008_pass_increasing() {
        assert_eq!(verdict_from_yarn_ramp_monotonic(0.2, 0.8, 0.0, 1.0), Rext008Verdict::Pass);
    }
    #[test] fn rext_008_pass_below_band() {
        assert_eq!(verdict_from_yarn_ramp_monotonic(-1.0, -0.5, 0.0, 1.0), Rext008Verdict::Pass);
    }
    #[test] fn rext_008_pass_above_band() {
        assert_eq!(verdict_from_yarn_ramp_monotonic(2.0, 3.0, 0.0, 1.0), Rext008Verdict::Pass);
    }
    #[test] fn rext_008_fail_r1_ge_r2() {
        assert_eq!(verdict_from_yarn_ramp_monotonic(0.5, 0.5, 0.0, 1.0), Rext008Verdict::Fail);
    }
}
