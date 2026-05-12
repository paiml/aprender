// SHIP-TWO-001 — `rmsnorm-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-RN-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/rmsnorm-kernel-v1.yaml`.
//
// Reference scalar RMSNorm: y_i = x_i * γ_i / sqrt(mean(x_j^2) + ε).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormError { LengthMismatch, NonFiniteInput, EpsNonPositive, EmptyInput }

pub fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>, RmsNormError> {
    if x.is_empty() { return Err(RmsNormError::EmptyInput); }
    if x.len() != gamma.len() { return Err(RmsNormError::LengthMismatch); }
    if eps <= 0.0 || !eps.is_finite() { return Err(RmsNormError::EpsNonPositive); }
    if x.iter().chain(gamma.iter()).any(|v| !v.is_finite()) {
        return Err(RmsNormError::NonFiniteInput);
    }
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n + eps).sqrt();
    Ok(x.iter().zip(gamma).map(|(a, g)| a * g / rms).collect())
}

// ===========================================================================
// RN-001 — Finiteness with eps > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_finiteness(x: &[f32], gamma: &[f32], eps: f32) -> Rn001Verdict {
    let y = match rmsnorm(x, gamma, eps) { Ok(y) => y, Err(_) => return Rn001Verdict::Fail };
    if y.iter().all(|v| v.is_finite()) { Rn001Verdict::Pass } else { Rn001Verdict::Fail }
}

// ===========================================================================
// RN-002 — Scale invariance: RMSNorm(α·x) ≈ sign(α)·RMSNorm(x)
// ===========================================================================

pub const AC_RN_002_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_scale_invariance(x: &[f32], gamma: &[f32], alpha: f32, eps: f32) -> Rn002Verdict {
    if alpha == 0.0 || !alpha.is_finite() { return Rn002Verdict::Fail; }
    let scaled: Vec<f32> = x.iter().map(|v| v * alpha).collect();
    let y_orig = match rmsnorm(x, gamma, eps) { Ok(y) => y, Err(_) => return Rn002Verdict::Fail };
    let y_scaled = match rmsnorm(&scaled, gamma, eps) { Ok(y) => y, Err(_) => return Rn002Verdict::Fail };
    let sign = if alpha > 0.0 { 1.0 } else { -1.0 };
    for (a, b) in y_orig.iter().zip(y_scaled.iter()) {
        if (sign * a - b).abs() > AC_RN_002_TOLERANCE { return Rn002Verdict::Fail; }
    }
    Rn002Verdict::Pass
}

// ===========================================================================
// RN-003 — SIMD vs scalar: |simd - scalar| < 4 ULPs
// ===========================================================================

pub const AC_RN_003_MAX_ULP: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn003Verdict { Pass, Fail }

fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() { return None; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    if (ai < 0) != (bi < 0) {
        return Some(ai.unsigned_abs() + bi.unsigned_abs());
    }
    Some((ai - bi).unsigned_abs())
}

#[must_use]
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> Rn003Verdict {
    if simd.len() != scalar.len() || simd.is_empty() { return Rn003Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_RN_003_MAX_ULP => {}
            _ => return Rn003Verdict::Fail,
        }
    }
    Rn003Verdict::Pass
}

// ===========================================================================
// RN-004 — Zero vector → zero output
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_zero_input(n: usize, eps: f32) -> Rn004Verdict {
    if n == 0 { return Rn004Verdict::Fail; }
    let x = vec![0.0_f32; n];
    let gamma = vec![1.0_f32; n];
    let y = match rmsnorm(&x, &gamma, eps) { Ok(y) => y, Err(_) => return Rn004Verdict::Fail };
    for v in y {
        if v != 0.0 { return Rn004Verdict::Fail; }
    }
    Rn004Verdict::Pass
}

// ===========================================================================
// RN-005 — Unit gamma: RMS(RMSNorm(x)) ≈ 1
// ===========================================================================

pub const AC_RN_005_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_unit_gamma_rms(x: &[f32], eps: f32) -> Rn005Verdict {
    if x.is_empty() { return Rn005Verdict::Fail; }
    let gamma = vec![1.0_f32; x.len()];
    let y = match rmsnorm(x, &gamma, eps) { Ok(y) => y, Err(_) => return Rn005Verdict::Fail };
    let n = y.len() as f32;
    let rms_y = (y.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
    // For non-trivial x, rms_y should be ≈ 1 (when eps is small relative to mean(x^2)).
    if (rms_y - 1.0).abs() < AC_RN_005_TOLERANCE { Rn005Verdict::Pass } else { Rn005Verdict::Fail }
}

// ===========================================================================
// RN-006 — Length mismatch → Err
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_length_validation() -> Rn006Verdict {
    let x = vec![1.0_f32; 4];
    let gamma_short = vec![1.0_f32; 3];
    let gamma_long = vec![1.0_f32; 5];
    if !matches!(rmsnorm(&x, &gamma_short, 1e-6), Err(RmsNormError::LengthMismatch)) {
        return Rn006Verdict::Fail;
    }
    if !matches!(rmsnorm(&x, &gamma_long, 1e-6), Err(RmsNormError::LengthMismatch)) {
        return Rn006Verdict::Fail;
    }
    Rn006Verdict::Pass
}

// ===========================================================================
// RN-007 — Frame condition: input buffers byte-identical
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_input_immutable(
    x_before: &[f32], x_after: &[f32],
    gamma_before: &[f32], gamma_after: &[f32],
) -> Rn007Verdict {
    if x_before.len() != x_after.len() || gamma_before.len() != gamma_after.len() {
        return Rn007Verdict::Fail;
    }
    for (a, b) in x_before.iter().zip(x_after) {
        if a.to_bits() != b.to_bits() { return Rn007Verdict::Fail; }
    }
    for (a, b) in gamma_before.iter().zip(gamma_after) {
        if a.to_bits() != b.to_bits() { return Rn007Verdict::Fail; }
    }
    Rn007Verdict::Pass
}

// ===========================================================================
// RN-008 — Postcondition: len(out) == len(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rn008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_length_preserved(x: &[f32], gamma: &[f32], eps: f32) -> Rn008Verdict {
    let y = match rmsnorm(x, gamma, eps) { Ok(y) => y, Err(_) => return Rn008Verdict::Fail };
    if y.len() == x.len() { Rn008Verdict::Pass } else { Rn008Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() <= eps }

    // Reference impl spot checks
    #[test] fn ref_unit_input() {
        let y = rmsnorm(&[1.0; 4], &[1.0; 4], 1e-6).unwrap();
        for v in y { assert!(approx(v, 1.0, 1e-3)); }
    }

    #[test] fn ref_gamma_amplification() {
        let y = rmsnorm(&[1.0; 4], &[2.0; 4], 1e-6).unwrap();
        for v in y { assert!(approx(v, 2.0, 1e-3)); }
    }

    // RN-001
    #[test] fn rn001_pass_normal() {
        assert_eq!(verdict_from_finiteness(&[1.0, 2.0], &[1.0, 1.0], 1e-6), Rn001Verdict::Pass);
    }
    #[test] fn rn001_pass_near_zero() {
        assert_eq!(verdict_from_finiteness(&[1e-10; 8], &[1.0; 8], 1e-6), Rn001Verdict::Pass);
    }
    #[test] fn rn001_fail_eps_zero() {
        assert_eq!(verdict_from_finiteness(&[1.0], &[1.0], 0.0), Rn001Verdict::Fail);
    }

    // RN-002
    #[test] fn rn002_pass_positive_alpha() {
        assert_eq!(
            verdict_from_scale_invariance(&[1.0, 2.0, 3.0], &[1.0; 3], 5.0, 1e-6),
            Rn002Verdict::Pass
        );
    }
    #[test] fn rn002_pass_negative_alpha() {
        assert_eq!(
            verdict_from_scale_invariance(&[1.0, 2.0, 3.0], &[1.0; 3], -2.0, 1e-6),
            Rn002Verdict::Pass
        );
    }
    #[test] fn rn002_fail_zero_alpha() {
        assert_eq!(
            verdict_from_scale_invariance(&[1.0, 2.0], &[1.0; 2], 0.0, 1e-6),
            Rn002Verdict::Fail
        );
    }

    // RN-003
    #[test] fn rn003_pass_exact() {
        let scalar = vec![1.0, 2.0, 3.0];
        assert_eq!(verdict_from_simd_equivalence(&scalar, &scalar), Rn003Verdict::Pass);
    }
    #[test] fn rn003_pass_within_3_ulp() {
        let scalar = [1.0_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 2)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Rn003Verdict::Pass);
    }
    #[test] fn rn003_fail_above_4_ulp() {
        let scalar = [1.0_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 10)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Rn003Verdict::Fail);
    }

    // RN-004
    #[test] fn rn004_pass_n4() { assert_eq!(verdict_from_zero_input(4, 1e-6), Rn004Verdict::Pass); }
    #[test] fn rn004_pass_n128() { assert_eq!(verdict_from_zero_input(128, 1e-6), Rn004Verdict::Pass); }
    #[test] fn rn004_fail_zero_n() { assert_eq!(verdict_from_zero_input(0, 1e-6), Rn004Verdict::Fail); }

    // RN-005
    #[test] fn rn005_pass_uniform() {
        assert_eq!(verdict_from_unit_gamma_rms(&[5.0; 16], 1e-6), Rn005Verdict::Pass);
    }
    #[test] fn rn005_pass_random_like() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 5.0).collect();
        assert_eq!(verdict_from_unit_gamma_rms(&x, 1e-6), Rn005Verdict::Pass);
    }

    // RN-006
    #[test] fn rn006_pass() { assert_eq!(verdict_from_length_validation(), Rn006Verdict::Pass); }

    // RN-007
    #[test] fn rn007_pass_unchanged() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let g = vec![1.0_f32; 3];
        assert_eq!(verdict_from_input_immutable(&x, &x, &g, &g), Rn007Verdict::Pass);
    }
    #[test] fn rn007_fail_x_modified() {
        let xb = [1.0_f32, 2.0];
        let xa = [1.0_f32, 5.0];
        let g = [1.0_f32, 1.0];
        assert_eq!(verdict_from_input_immutable(&xb, &xa, &g, &g), Rn007Verdict::Fail);
    }
    #[test] fn rn007_fail_gamma_modified() {
        let x = [1.0_f32, 2.0];
        let gb = [1.0_f32, 1.0];
        let ga = [1.0_f32, 2.0];
        assert_eq!(verdict_from_input_immutable(&x, &x, &gb, &ga), Rn007Verdict::Fail);
    }

    // RN-008
    #[test] fn rn008_pass_n4() {
        assert_eq!(verdict_from_length_preserved(&[1.0; 4], &[1.0; 4], 1e-6), Rn008Verdict::Pass);
    }
    #[test] fn rn008_fail_length_mismatch() {
        assert_eq!(verdict_from_length_preserved(&[1.0; 4], &[1.0; 3], 1e-6), Rn008Verdict::Fail);
    }
    #[test] fn rn008_fail_empty() {
        assert_eq!(verdict_from_length_preserved(&[], &[], 1e-6), Rn008Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_max_ulp() { assert_eq!(AC_RN_003_MAX_ULP, 4); }
    #[test] fn provenance_tolerance_002() { assert!((AC_RN_002_TOLERANCE - 1e-3).abs() < f32::EPSILON); }
}
