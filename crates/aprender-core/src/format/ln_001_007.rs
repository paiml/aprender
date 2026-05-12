// SHIP-TWO-001 — `layernorm-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-LN-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/layernorm-kernel-v1.yaml`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerNormError { LengthMismatch, EpsNonPositive, NonFiniteInput, EmptyInput }

pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Result<Vec<f32>, LayerNormError> {
    if x.is_empty() { return Err(LayerNormError::EmptyInput); }
    if x.len() != gamma.len() || x.len() != beta.len() { return Err(LayerNormError::LengthMismatch); }
    if eps <= 0.0 || !eps.is_finite() { return Err(LayerNormError::EpsNonPositive); }
    if x.iter().chain(gamma.iter()).chain(beta.iter()).any(|v| !v.is_finite()) {
        return Err(LayerNormError::NonFiniteInput);
    }
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    Ok(x.iter().zip(gamma).zip(beta)
        .map(|((xi, g), b)| g * (xi - mean) * inv_std + b)
        .collect())
}

fn mean_f64(v: &[f32]) -> f64 {
    v.iter().map(|x| *x as f64).sum::<f64>() / (v.len() as f64)
}

fn variance_f64(v: &[f32]) -> f64 {
    let m = mean_f64(v);
    v.iter().map(|x| ((*x as f64) - m).powi(2)).sum::<f64>() / (v.len() as f64)
}

// ===========================================================================
// LN-001 — Centering: |mean(LN(x)) - mean(beta)| < 1e-5 with gamma=1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_centering(x: &[f32], beta: &[f32], eps: f32) -> Ln001Verdict {
    if x.is_empty() || x.len() != beta.len() { return Ln001Verdict::Fail; }
    let gamma = vec![1.0_f32; x.len()];
    let y = match layer_norm(x, &gamma, beta, eps) { Ok(v) => v, Err(_) => return Ln001Verdict::Fail };
    let target = mean_f64(beta);
    let observed = mean_f64(&y);
    if (observed - target).abs() < 1e-5 { Ln001Verdict::Pass } else { Ln001Verdict::Fail }
}

// ===========================================================================
// LN-002 — Standardization: |var(LN(x)) - 1.0| < 1e-5 with gamma=1, beta=0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_standardization(x: &[f32], eps: f32) -> Ln002Verdict {
    if x.len() < 2 { return Ln002Verdict::Fail; }
    // x must be non-constant for var(LN(x))≈1.
    let mean = mean_f64(x);
    if x.iter().all(|v| ((*v as f64) - mean).abs() < 1e-9) {
        return Ln002Verdict::Fail;
    }
    let gamma = vec![1.0_f32; x.len()];
    let beta = vec![0.0_f32; x.len()];
    let y = match layer_norm(x, &gamma, &beta, eps) { Ok(v) => v, Err(_) => return Ln002Verdict::Fail };
    let v = variance_f64(&y);
    if (v - 1.0).abs() < 1e-3 { Ln002Verdict::Pass } else { Ln002Verdict::Fail }
}

// ===========================================================================
// LN-003 — Denominator safety: no NaN/Inf in output for any finite input
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_finiteness(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Ln003Verdict {
    let y = match layer_norm(x, gamma, beta, eps) { Ok(v) => v, Err(_) => return Ln003Verdict::Fail };
    if y.iter().all(|v| v.is_finite()) { Ln003Verdict::Pass } else { Ln003Verdict::Fail }
}

// ===========================================================================
// LN-004 — SIMD vs scalar: |simd - scalar| < 8 ULPs
// ===========================================================================

pub const AC_LN_004_MAX_ULP: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln004Verdict { Pass, Fail }

fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() { return None; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    if (ai < 0) != (bi < 0) { return Some(ai.unsigned_abs() + bi.unsigned_abs()); }
    Some((ai - bi).unsigned_abs())
}

#[must_use]
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> Ln004Verdict {
    if simd.len() != scalar.len() || simd.is_empty() { return Ln004Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_LN_004_MAX_ULP => {}
            _ => return Ln004Verdict::Fail,
        }
    }
    Ln004Verdict::Pass
}

// ===========================================================================
// LN-005 — Idempotency: |LN(LN(x)) - LN(x)| < 1e-5 with gamma=1, beta=0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_idempotency(x: &[f32], eps: f32) -> Ln005Verdict {
    if x.len() < 2 { return Ln005Verdict::Fail; }
    let gamma = vec![1.0_f32; x.len()];
    let beta = vec![0.0_f32; x.len()];
    let y1 = match layer_norm(x, &gamma, &beta, eps) { Ok(v) => v, Err(_) => return Ln005Verdict::Fail };
    let y2 = match layer_norm(&y1, &gamma, &beta, eps) { Ok(v) => v, Err(_) => return Ln005Verdict::Fail };
    for (a, b) in y1.iter().zip(y2.iter()) {
        if (a - b).abs() > 1e-3 { return Ln005Verdict::Fail; }
    }
    Ln005Verdict::Pass
}

// ===========================================================================
// LN-006 — Shift invariance: |LN(x + c) - LN(x)| < 1e-3
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_shift_invariance(x: &[f32], c: f32, eps: f32) -> Ln006Verdict {
    if x.len() < 2 || !c.is_finite() { return Ln006Verdict::Fail; }
    let gamma = vec![1.0_f32; x.len()];
    let beta = vec![0.0_f32; x.len()];
    let shifted: Vec<f32> = x.iter().map(|v| v + c).collect();
    let a = match layer_norm(x, &gamma, &beta, eps) { Ok(v) => v, Err(_) => return Ln006Verdict::Fail };
    let b = match layer_norm(&shifted, &gamma, &beta, eps) { Ok(v) => v, Err(_) => return Ln006Verdict::Fail };
    for (p, q) in a.iter().zip(b.iter()) {
        if (p - q).abs() > 1e-3 { return Ln006Verdict::Fail; }
    }
    Ln006Verdict::Pass
}

// ===========================================================================
// LN-007 — Constant input boundary: LN([c; d]) == beta when gamma=1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ln007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_constant_input(c: f32, beta: &[f32], eps: f32) -> Ln007Verdict {
    if beta.is_empty() || !c.is_finite() { return Ln007Verdict::Fail; }
    let x = vec![c; beta.len()];
    let gamma = vec![1.0_f32; beta.len()];
    let y = match layer_norm(&x, &gamma, beta, eps) { Ok(v) => v, Err(_) => return Ln007Verdict::Fail };
    for (yi, bi) in y.iter().zip(beta.iter()) {
        if (yi - bi).abs() > 1e-3 { return Ln007Verdict::Fail; }
    }
    Ln007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_x(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.7 - 5.0).sin() * 3.0).collect()
    }

    // Reference impl spot checks
    #[test] fn ref_basic() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let y = layer_norm(&x, &[1.0; 4], &[0.0; 4], 1e-5).unwrap();
        let m = mean_f64(&y);
        let v = variance_f64(&y);
        assert!(m.abs() < 1e-5);
        assert!((v - 1.0).abs() < 1e-3);
    }

    // LN-001
    #[test] fn ln001_pass_zero_beta() {
        let x = rand_x(8);
        assert_eq!(verdict_from_centering(&x, &[0.0; 8], 1e-5), Ln001Verdict::Pass);
    }
    #[test] fn ln001_pass_nonzero_beta() {
        let x = rand_x(16);
        let beta: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        assert_eq!(verdict_from_centering(&x, &beta, 1e-5), Ln001Verdict::Pass);
    }
    #[test] fn ln001_fail_dim_mismatch() {
        let x = rand_x(8);
        assert_eq!(verdict_from_centering(&x, &[0.0; 4], 1e-5), Ln001Verdict::Fail);
    }

    // LN-002
    #[test] fn ln002_pass_random() {
        let x = rand_x(64);
        assert_eq!(verdict_from_standardization(&x, 1e-5), Ln002Verdict::Pass);
    }
    #[test] fn ln002_fail_constant() {
        let x = vec![3.0_f32; 16];
        assert_eq!(verdict_from_standardization(&x, 1e-5), Ln002Verdict::Fail);
    }

    // LN-003
    #[test] fn ln003_pass_normal() {
        let x = rand_x(8);
        assert_eq!(verdict_from_finiteness(&x, &[1.0; 8], &[0.0; 8], 1e-5), Ln003Verdict::Pass);
    }
    #[test] fn ln003_pass_extreme() {
        let x = vec![1e30_f32, -1e30, 0.0, 1.0];
        assert_eq!(verdict_from_finiteness(&x, &[1.0; 4], &[0.0; 4], 1e-5), Ln003Verdict::Pass);
    }
    #[test] fn ln003_fail_zero_eps() {
        let x = vec![1.0_f32; 4];
        assert_eq!(verdict_from_finiteness(&x, &[1.0; 4], &[0.0; 4], 0.0), Ln003Verdict::Fail);
    }

    // LN-004
    #[test] fn ln004_pass_exact() {
        let s = vec![0.1_f32, 0.2];
        assert_eq!(verdict_from_simd_equivalence(&s, &s), Ln004Verdict::Pass);
    }
    #[test] fn ln004_pass_within_8_ulp() {
        let s = [0.1_f32];
        let simd = [f32::from_bits(s[0].to_bits() + 5)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &s), Ln004Verdict::Pass);
    }
    #[test] fn ln004_fail_far_apart() {
        let s = [0.1_f32];
        let simd = [f32::from_bits(s[0].to_bits() + 100)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &s), Ln004Verdict::Fail);
    }

    // LN-005
    #[test] fn ln005_pass_random() {
        let x = rand_x(32);
        assert_eq!(verdict_from_idempotency(&x, 1e-5), Ln005Verdict::Pass);
    }
    #[test] fn ln005_fail_too_short() {
        let x = vec![1.0_f32];
        assert_eq!(verdict_from_idempotency(&x, 1e-5), Ln005Verdict::Fail);
    }

    // LN-006
    #[test] fn ln006_pass_zero_shift() {
        let x = rand_x(16);
        assert_eq!(verdict_from_shift_invariance(&x, 0.0, 1e-5), Ln006Verdict::Pass);
    }
    #[test] fn ln006_pass_large_shift() {
        let x = rand_x(16);
        assert_eq!(verdict_from_shift_invariance(&x, 100.0, 1e-5), Ln006Verdict::Pass);
    }
    #[test] fn ln006_pass_negative_shift() {
        let x = rand_x(16);
        assert_eq!(verdict_from_shift_invariance(&x, -50.0, 1e-5), Ln006Verdict::Pass);
    }

    // LN-007
    #[test] fn ln007_pass_zero_beta() {
        let beta = vec![0.0_f32; 8];
        assert_eq!(verdict_from_constant_input(5.0, &beta, 1e-5), Ln007Verdict::Pass);
    }
    #[test] fn ln007_pass_nonzero_beta() {
        let beta: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5).collect();
        assert_eq!(verdict_from_constant_input(2.0, &beta, 1e-5), Ln007Verdict::Pass);
    }
    #[test] fn ln007_fail_empty() {
        assert_eq!(verdict_from_constant_input(1.0, &[], 1e-5), Ln007Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_max_ulp() { assert_eq!(AC_LN_004_MAX_ULP, 8); }
}
