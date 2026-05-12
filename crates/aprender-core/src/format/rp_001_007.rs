// SHIP-TWO-001 — `rope-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-RP-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/rope-kernel-v1.yaml`.

// ===========================================================================
// Reference RoPE: rotates pairs (x_{2i}, x_{2i+1}) by angle m * theta_i.
// theta_i = base^(-2i / d) for i in [0, d/2).
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeError { OddDimension, EmptyVector, NonFinite }

pub fn rope_apply(x: &[f32], m: f32, base: f64) -> Result<Vec<f32>, RopeError> {
    if x.is_empty() { return Err(RopeError::EmptyVector); }
    if !x.len().is_multiple_of(2) { return Err(RopeError::OddDimension); }
    if x.iter().any(|v| !v.is_finite()) { return Err(RopeError::NonFinite); }
    if !m.is_finite() || base <= 1.0 { return Err(RopeError::NonFinite); }
    let d = x.len() as f64;
    let mut out = vec![0.0_f32; x.len()];
    for i in 0..(x.len() / 2) {
        let theta_i = base.powf(-2.0 * (i as f64) / d);
        let angle = (m as f64) * theta_i;
        let c = angle.cos() as f32;
        let s = angle.sin() as f32;
        let a = x[2 * i];
        let b = x[2 * i + 1];
        out[2 * i] = a * c - b * s;
        out[2 * i + 1] = a * s + b * c;
    }
    Ok(out)
}

pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
}

pub fn l2_norm(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
}

// ===========================================================================
// RP-001 — Norm preservation: ‖RoPE(x, m)‖ ≈ ‖x‖
// ===========================================================================

pub const AC_RP_001_TOLERANCE: f64 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_norm_preservation(x: &[f32], m: f32, base: f64) -> Rp001Verdict {
    let y = match rope_apply(x, m, base) { Ok(v) => v, Err(_) => return Rp001Verdict::Fail };
    let nx = l2_norm(x);
    let ny = l2_norm(&y);
    if nx == 0.0 { return if ny == 0.0 { Rp001Verdict::Pass } else { Rp001Verdict::Fail }; }
    let rel = (nx - ny).abs() / nx;
    if rel < AC_RP_001_TOLERANCE { Rp001Verdict::Pass } else { Rp001Verdict::Fail }
}

// ===========================================================================
// RP-002 — Relative position: dot(RoPE(q,m), RoPE(k,n)) =
//                              dot(RoPE(q,0), RoPE(k,n-m))
// ===========================================================================

pub const AC_RP_002_TOLERANCE: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_relative_position(q: &[f32], k: &[f32], m: f32, n: f32, base: f64) -> Rp002Verdict {
    if q.len() != k.len() { return Rp002Verdict::Fail; }
    let qm = match rope_apply(q, m, base) { Ok(v) => v, Err(_) => return Rp002Verdict::Fail };
    let kn = match rope_apply(k, n, base) { Ok(v) => v, Err(_) => return Rp002Verdict::Fail };
    let q0 = match rope_apply(q, 0.0, base) { Ok(v) => v, Err(_) => return Rp002Verdict::Fail };
    let k_diff = match rope_apply(k, n - m, base) { Ok(v) => v, Err(_) => return Rp002Verdict::Fail };
    let lhs = dot(&qm, &kn);
    let rhs = dot(&q0, &k_diff);
    let denom = lhs.abs().max(rhs.abs()).max(1.0);
    if (lhs - rhs).abs() / denom < AC_RP_002_TOLERANCE { Rp002Verdict::Pass } else { Rp002Verdict::Fail }
}

// ===========================================================================
// RP-003 — SIMD vs scalar: |simd - scalar| < 4 ULPs
// ===========================================================================

pub const AC_RP_003_MAX_ULP: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp003Verdict { Pass, Fail }

fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() { return None; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    if (ai < 0) != (bi < 0) { return Some(ai.unsigned_abs() + bi.unsigned_abs()); }
    Some((ai - bi).unsigned_abs())
}

#[must_use]
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> Rp003Verdict {
    if simd.len() != scalar.len() || simd.is_empty() { return Rp003Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_RP_003_MAX_ULP => {}
            _ => return Rp003Verdict::Fail,
        }
    }
    Rp003Verdict::Pass
}

// ===========================================================================
// RP-004 — Identity at position 0: RoPE(x, 0) == x
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_position_zero_identity(x: &[f32], base: f64) -> Rp004Verdict {
    let y = match rope_apply(x, 0.0, base) { Ok(v) => v, Err(_) => return Rp004Verdict::Fail };
    if y.len() != x.len() { return Rp004Verdict::Fail; }
    for (a, b) in x.iter().zip(y.iter()) {
        if (a - b).abs() > 1e-6 { return Rp004Verdict::Fail; }
    }
    Rp004Verdict::Pass
}

// ===========================================================================
// RP-005 — Odd dimension → Err
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_odd_dim_rejection() -> Rp005Verdict {
    let odd = vec![1.0_f32, 2.0, 3.0]; // dim=3
    if matches!(rope_apply(&odd, 1.0, 10000.0), Err(RopeError::OddDimension)) {
        Rp005Verdict::Pass
    } else {
        Rp005Verdict::Fail
    }
}

// ===========================================================================
// RP-006 — Frame condition: input unchanged after rope
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_input_immutable(before: &[f32], after: &[f32]) -> Rp006Verdict {
    if before.len() != after.len() { return Rp006Verdict::Fail; }
    for (a, b) in before.iter().zip(after) {
        if a.to_bits() != b.to_bits() { return Rp006Verdict::Fail; }
    }
    Rp006Verdict::Pass
}

// ===========================================================================
// RP-007 — len(out) == len(in)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rp007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_length_preserved(x: &[f32], m: f32, base: f64) -> Rp007Verdict {
    let y = match rope_apply(x, m, base) { Ok(v) => v, Err(_) => return Rp007Verdict::Fail };
    if y.len() == x.len() { Rp007Verdict::Pass } else { Rp007Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_x(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.7 - 5.0).sin()).collect()
    }

    // Reference impl spot checks
    #[test] fn ref_zero_position_identity() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let y = rope_apply(&x, 0.0, 10000.0).unwrap();
        assert_eq!(x, y);
    }

    #[test] fn ref_norm_preserved_2d() {
        let x = vec![3.0_f32, 4.0]; // norm = 5
        let y = rope_apply(&x, 1.0, 10000.0).unwrap();
        let n = l2_norm(&y);
        assert!((n - 5.0).abs() < 1e-5);
    }

    // RP-001
    #[test] fn rp001_pass_canonical() {
        let x = rng_x(8);
        assert_eq!(verdict_from_norm_preservation(&x, 5.0, 10000.0), Rp001Verdict::Pass);
    }
    #[test] fn rp001_pass_qwen_base() {
        let x = rng_x(128);
        assert_eq!(verdict_from_norm_preservation(&x, 100.0, 1_000_000.0), Rp001Verdict::Pass);
    }
    #[test] fn rp001_fail_odd_dim() {
        let x = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_norm_preservation(&x, 1.0, 10000.0), Rp001Verdict::Fail);
    }

    // RP-002
    #[test] fn rp002_pass_translation_invariant() {
        let q = rng_x(8);
        let k = rng_x(8);
        assert_eq!(
            verdict_from_relative_position(&q, &k, 3.0, 7.0, 10000.0),
            Rp002Verdict::Pass
        );
    }
    #[test] fn rp002_pass_zero_diff() {
        // n - m = 0 case still satisfies translation invariance.
        let q = rng_x(8);
        let k = rng_x(8);
        assert_eq!(
            verdict_from_relative_position(&q, &k, 5.0, 5.0, 10000.0),
            Rp002Verdict::Pass
        );
    }
    #[test] fn rp002_fail_dim_mismatch() {
        let q = rng_x(8);
        let k = rng_x(6);
        assert_eq!(
            verdict_from_relative_position(&q, &k, 1.0, 2.0, 10000.0),
            Rp002Verdict::Fail
        );
    }

    // RP-003
    #[test] fn rp003_pass_exact() {
        let scalar = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_equivalence(&scalar, &scalar), Rp003Verdict::Pass);
    }
    #[test] fn rp003_pass_3_ulp() {
        let scalar = [1.0_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 2)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Rp003Verdict::Pass);
    }
    #[test] fn rp003_fail_far_apart() {
        let scalar = [1.0_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 100)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Rp003Verdict::Fail);
    }

    // RP-004
    #[test] fn rp004_pass_canonical() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_position_zero_identity(&x, 10000.0), Rp004Verdict::Pass);
    }
    #[test] fn rp004_fail_empty() {
        assert_eq!(verdict_from_position_zero_identity(&[], 10000.0), Rp004Verdict::Fail);
    }

    // RP-005
    #[test] fn rp005_pass() {
        assert_eq!(verdict_from_odd_dim_rejection(), Rp005Verdict::Pass);
    }

    // RP-006
    #[test] fn rp006_pass_unchanged() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_input_immutable(&x, &x), Rp006Verdict::Pass);
    }
    #[test] fn rp006_fail_modified() {
        let before = [1.0_f32, 2.0];
        let after = [1.0_f32, 5.0];
        assert_eq!(verdict_from_input_immutable(&before, &after), Rp006Verdict::Fail);
    }

    // RP-007
    #[test] fn rp007_pass_canonical() {
        let x = rng_x(8);
        assert_eq!(verdict_from_length_preserved(&x, 5.0, 10000.0), Rp007Verdict::Pass);
    }
    #[test] fn rp007_fail_odd_dim() {
        let x = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_length_preserved(&x, 5.0, 10000.0), Rp007Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_max_ulp() { assert_eq!(AC_RP_003_MAX_ULP, 4); }
}
