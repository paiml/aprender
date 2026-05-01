// SHIP-TWO-001 — `gated-delta-net-v1` algorithm-level PARTIAL
// discharge for FALSIFY-GDN-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/gated-delta-net-v1.yaml`.
// Spec: Qwen3.5 linear attention with decay, delta rule, causal conv1d
// (Yang et al. 2024 — Gated Delta Networks: Improving Mamba2 with Delta Rule).

// ===========================================================================
// GDN-001 — Decay bound: sigmoid(x) ∈ (0, 1) for all finite x
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn001Verdict { Pass, Fail }

/// Pure scalar sigmoid; matches the algorithm-level decay equation
/// `α_t = sigmoid(A_log.exp() * dt + dt_bias)`.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[must_use]
pub fn verdict_from_decay_bound(probes: &[f32]) -> Gdn001Verdict {
    if probes.is_empty() { return Gdn001Verdict::Fail; }
    for &x in probes {
        if !x.is_finite() { return Gdn001Verdict::Fail; }
        let s = sigmoid(x);
        if !s.is_finite() || s <= 0.0 || s >= 1.0 {
            return Gdn001Verdict::Fail;
        }
    }
    Gdn001Verdict::Pass
}

// ===========================================================================
// GDN-002 — State shape preserved across recurrence updates
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn002Verdict { Pass, Fail }

/// Pass iff every state_t in the recurrence trace has shape [k_dim, v_dim].
#[must_use]
pub fn verdict_from_state_shape(states: &[(u64, u64)], k_dim: u64, v_dim: u64) -> Gdn002Verdict {
    if states.is_empty() || k_dim == 0 || v_dim == 0 { return Gdn002Verdict::Fail; }
    for &(k, v) in states {
        if k != k_dim || v != v_dim { return Gdn002Verdict::Fail; }
    }
    Gdn002Verdict::Pass
}

// ===========================================================================
// GDN-003 — Causal conv1d: modifying future input does not change current output
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn003Verdict { Pass, Fail }

/// Pass iff `output_a[..=t] == output_b[..=t]` whenever `input_a[..=t] == input_b[..=t]`
/// (regardless of input differences past t).
#[must_use]
pub fn verdict_from_causal_conv(
    input_a: &[f32],
    output_a: &[f32],
    input_b: &[f32],
    output_b: &[f32],
    t: usize,
) -> Gdn003Verdict {
    if input_a.is_empty() || input_b.is_empty() || output_a.is_empty() || output_b.is_empty() {
        return Gdn003Verdict::Fail;
    }
    if input_a.len() != input_b.len() || output_a.len() != output_b.len() {
        return Gdn003Verdict::Fail;
    }
    if t >= output_a.len() || t >= input_a.len() { return Gdn003Verdict::Fail; }
    // Precondition: prefix [0..=t] of inputs is identical
    for i in 0..=t {
        if input_a[i] != input_b[i] { return Gdn003Verdict::Fail; }
    }
    // Postcondition: prefix [0..=t] of outputs is identical
    for i in 0..=t {
        if output_a[i] != output_b[i] { return Gdn003Verdict::Fail; }
    }
    Gdn003Verdict::Pass
}

// ===========================================================================
// GDN-004 — L2 norm preserves direction: cos(L2(q), q) ≈ 1
// ===========================================================================

pub const AC_GDN_004_COSINE_TOLERANCE: f32 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn004Verdict { Pass, Fail }

#[must_use]
pub fn l2_norm(v: &[f32]) -> f32 {
    let s: f32 = v.iter().map(|&x| x * x).sum();
    s.sqrt()
}

#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na == 0.0 || nb == 0.0 { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot / (na * nb)
}

/// Pass iff cos(normalized, original) ≈ 1.0 within tolerance.
#[must_use]
pub fn verdict_from_l2_direction(input: &[f32]) -> Gdn004Verdict {
    if input.is_empty() { return Gdn004Verdict::Fail; }
    if !input.iter().all(|v| v.is_finite()) { return Gdn004Verdict::Fail; }
    let n = l2_norm(input);
    if n == 0.0 || !n.is_finite() { return Gdn004Verdict::Fail; }
    let normalized: Vec<f32> = input.iter().map(|&x| x / n).collect();
    let cos = cosine_similarity(&normalized, input);
    if !cos.is_finite() { return Gdn004Verdict::Fail; }
    if (cos - 1.0).abs() <= AC_GDN_004_COSINE_TOLERANCE { Gdn004Verdict::Pass } else { Gdn004Verdict::Fail }
}

// ===========================================================================
// GDN-005 — SIMD matches scalar within 8 ULP (per contract tolerance)
// ===========================================================================

pub const AC_GDN_005_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdn005Verdict { Pass, Fail }

/// ULP distance between two f32 values (treats -0.0 == 0.0).
#[must_use]
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    if !a.is_finite() || !b.is_finite() { return u32::MAX; }
    if a == b { return 0; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    // Sign-flipping handling
    let ord_a = if ai < 0 { i32::MIN.wrapping_sub(ai).wrapping_add(1) } else { ai };
    let ord_b = if bi < 0 { i32::MIN.wrapping_sub(bi).wrapping_add(1) } else { bi };
    ord_a.wrapping_sub(ord_b).unsigned_abs()
}

/// Pass iff every element pair (scalar[i], simd[i]) is within `tol_ulp`.
#[must_use]
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Gdn005Verdict {
    if scalar.is_empty() || simd.is_empty() { return Gdn005Verdict::Fail; }
    if scalar.len() != simd.len() { return Gdn005Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Gdn005Verdict::Fail; }
        if ulp_distance(s, v) > AC_GDN_005_ULP_TOLERANCE { return Gdn005Verdict::Fail; }
    }
    Gdn005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // GDN-001 (decay bound)
    #[test] fn gdn001_pass_zero() {
        assert_eq!(verdict_from_decay_bound(&[0.0, 1.0, -1.0, 5.0, -5.0]), Gdn001Verdict::Pass);
    }
    #[test] fn gdn001_pass_non_saturating() {
        // f32 sigmoid saturates at |x| ≳ 16 (returns exactly 1.0 / 0.0).
        // The strict-(0,1) verdict only holds in the non-saturating regime;
        // saturation is an OOB precision regime for the algorithm-level rule.
        assert_eq!(
            verdict_from_decay_bound(&[10.0, -10.0, 5.0, -5.0, 15.0, -15.0]),
            Gdn001Verdict::Pass
        );
    }
    #[test] fn gdn001_fail_at_saturation() {
        // sigmoid(50.0) saturates to 1.0 in f32 → violates strict-(0,1).
        // This documents the verdict's behavior at the precision cliff:
        // saturated outputs Fail the strict-bound check.
        assert_eq!(verdict_from_decay_bound(&[50.0]), Gdn001Verdict::Fail);
    }
    #[test] fn gdn001_fail_nan() {
        assert_eq!(verdict_from_decay_bound(&[f32::NAN]), Gdn001Verdict::Fail);
    }
    #[test] fn gdn001_fail_inf() {
        assert_eq!(verdict_from_decay_bound(&[f32::INFINITY]), Gdn001Verdict::Fail);
    }
    #[test] fn sigmoid_strict_in_unit_interval_below_saturation() {
        // Within the f32 non-saturating regime |x| ≤ 15, sigmoid output
        // is strictly in (0, 1).
        for &x in &[-15.0_f32, -10.0, -1.0, 0.0, 1.0, 10.0, 15.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({}) = {}", x, s);
        }
    }
    #[test] fn gdn001_fail_empty() {
        assert_eq!(verdict_from_decay_bound(&[]), Gdn001Verdict::Fail);
    }

    // GDN-002 (state shape)
    #[test] fn gdn002_pass_constant_shape() {
        let states = [(64_u64, 128), (64, 128), (64, 128)];
        assert_eq!(verdict_from_state_shape(&states, 64, 128), Gdn002Verdict::Pass);
    }
    #[test] fn gdn002_fail_drift() {
        let states = [(64_u64, 128), (64, 128), (65, 128)];
        assert_eq!(verdict_from_state_shape(&states, 64, 128), Gdn002Verdict::Fail);
    }
    #[test] fn gdn002_fail_zero_dim() {
        let states = [(0_u64, 128)];
        assert_eq!(verdict_from_state_shape(&states, 0, 128), Gdn002Verdict::Fail);
    }
    #[test] fn gdn002_fail_empty() {
        assert_eq!(verdict_from_state_shape(&[], 64, 128), Gdn002Verdict::Fail);
    }

    // GDN-003 (causal conv)
    #[test] fn gdn003_pass_identical_prefix() {
        // Future-only difference: prefix [0..=2] outputs identical.
        let in_a = [0.1, 0.2, 0.3, 0.4, 0.5];
        let in_b = [0.1, 0.2, 0.3, 99.0, 99.0];
        let out_a = [0.1, 0.3, 0.6, 1.0, 1.5];
        let out_b = [0.1, 0.3, 0.6, 99.0, 99.0]; // future drifts but past doesn't
        assert_eq!(verdict_from_causal_conv(&in_a, &out_a, &in_b, &out_b, 2), Gdn003Verdict::Pass);
    }
    #[test] fn gdn003_fail_acausal_leak() {
        // Future input change DID alter past output — non-causal regression.
        let in_a = [0.1, 0.2, 0.3, 0.4, 0.5];
        let in_b = [0.1, 0.2, 0.3, 99.0, 99.0];
        let out_a = [0.1, 0.3, 0.6, 1.0, 1.5];
        let out_b = [0.1, 0.3, 999.0, 99.0, 99.0]; // past output @ t=2 leaked
        assert_eq!(verdict_from_causal_conv(&in_a, &out_a, &in_b, &out_b, 2), Gdn003Verdict::Fail);
    }
    #[test] fn gdn003_fail_prefix_input_mismatch() {
        // Caller violated precondition.
        let in_a = [0.1, 0.2, 0.3];
        let in_b = [0.1, 0.99, 0.3];
        let out_a = [0.1, 0.3, 0.6];
        let out_b = [0.1, 0.3, 0.6];
        assert_eq!(verdict_from_causal_conv(&in_a, &out_a, &in_b, &out_b, 2), Gdn003Verdict::Fail);
    }

    // GDN-004 (L2 direction)
    #[test] fn gdn004_pass_typical() {
        let v = [1.0_f32, 2.0, -3.0, 4.0];
        assert_eq!(verdict_from_l2_direction(&v), Gdn004Verdict::Pass);
    }
    #[test] fn gdn004_pass_unit_vector() {
        let v = [1.0_f32, 0.0, 0.0];
        assert_eq!(verdict_from_l2_direction(&v), Gdn004Verdict::Pass);
    }
    #[test] fn gdn004_fail_zero_vector() {
        // Cannot normalize the zero vector → Fail (degenerate input).
        let v = [0.0_f32, 0.0, 0.0];
        assert_eq!(verdict_from_l2_direction(&v), Gdn004Verdict::Fail);
    }
    #[test] fn gdn004_fail_nan() {
        let v = [1.0_f32, f32::NAN];
        assert_eq!(verdict_from_l2_direction(&v), Gdn004Verdict::Fail);
    }

    // GDN-005 (SIMD parity)
    #[test] fn gdn005_pass_identical() {
        let a = vec![1.0_f32; 64];
        assert_eq!(verdict_from_simd_parity(&a, &a), Gdn005Verdict::Pass);
    }
    #[test] fn gdn005_pass_within_ulp() {
        // 1-ULP perturbation is well within 8.
        let a = [1.0_f32, 2.0, 3.0];
        let b = [
            f32::from_bits(1.0_f32.to_bits() + 1),
            2.0,
            f32::from_bits(3.0_f32.to_bits() + 2),
        ];
        assert_eq!(verdict_from_simd_parity(&a, &b), Gdn005Verdict::Pass);
    }
    #[test] fn gdn005_fail_above_8_ulp() {
        let a = [1.0_f32];
        let b = [f32::from_bits(1.0_f32.to_bits() + 100)]; // 100 ULP > 8
        assert_eq!(verdict_from_simd_parity(&a, &b), Gdn005Verdict::Fail);
    }
    #[test] fn gdn005_fail_length_mismatch() {
        let a = [1.0_f32];
        let b = [1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Gdn005Verdict::Fail);
    }
    #[test] fn gdn005_fail_nan() {
        let a = [1.0_f32];
        let b = [f32::NAN];
        assert_eq!(verdict_from_simd_parity(&a, &b), Gdn005Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_GDN_005_ULP_TOLERANCE, 8);
        assert!((AC_GDN_004_COSINE_TOLERANCE - 1.0e-5).abs() < 1e-12);
    }

    // Sigmoid sanity (helper)
    #[test] fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
    }
    #[test] fn sigmoid_monotonic() {
        let a = sigmoid(-1.0);
        let b = sigmoid(0.0);
        let c = sigmoid(1.0);
        assert!(a < b && b < c);
    }
}
