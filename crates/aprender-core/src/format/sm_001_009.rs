// SHIP-TWO-001 — `softmax-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-SM-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/softmax-kernel-v1.yaml`.
//
// Bundles 9 verdict fns + a stand-alone scalar reference softmax
// (max-subtraction trick). Returns `None` for empty inputs (the
// canonical input-validation contract per SM-007).

// ===========================================================================
// Reference scalar softmax (max-subtraction)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxError { EmptyInput, NonFiniteInput }

pub fn softmax(xs: &[f32]) -> Result<Vec<f32>, SoftmaxError> {
    if xs.is_empty() { return Err(SoftmaxError::EmptyInput); }
    if xs.iter().any(|v| !v.is_finite()) { return Err(SoftmaxError::NonFiniteInput); }
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    let exps: Vec<f32> = xs.iter().map(|x| {
        let e = (x - m).exp();
        sum += e;
        e
    }).collect();
    Ok(exps.into_iter().map(|e| e / sum).collect())
}

// ===========================================================================
// SM-001 — Normalization: sum(softmax(x)) ≈ 1.0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_normalization(xs: &[f32]) -> Sm001Verdict {
    let y = match softmax(xs) {
        Ok(y) => y,
        Err(_) => return Sm001Verdict::Fail,
    };
    let sum: f32 = y.iter().sum();
    let n = y.len() as f32;
    let tol = 1e-6_f32 * n.sqrt().max(1.0);
    if (sum - 1.0).abs() < tol { Sm001Verdict::Pass } else { Sm001Verdict::Fail }
}

// ===========================================================================
// SM-002 — Positivity: every output > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_positivity(xs: &[f32]) -> Sm002Verdict {
    let y = match softmax(xs) { Ok(y) => y, Err(_) => return Sm002Verdict::Fail };
    for v in y {
        if v <= 0.0 || !v.is_finite() { return Sm002Verdict::Fail; }
    }
    Sm002Verdict::Pass
}

// ===========================================================================
// SM-003 — Order preservation: argmax(softmax(x)) == argmax(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm003Verdict { Pass, Fail }

fn argmax(xs: &[f32]) -> Option<usize> {
    if xs.is_empty() { return None; }
    let mut best_i = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best_i] { best_i = i; }
    }
    Some(best_i)
}

#[must_use]
pub fn verdict_from_order_preservation(xs: &[f32]) -> Sm003Verdict {
    let y = match softmax(xs) { Ok(y) => y, Err(_) => return Sm003Verdict::Fail };
    let arg_x = match argmax(xs) { Some(i) => i, None => return Sm003Verdict::Fail };
    let arg_y = match argmax(&y) { Some(i) => i, None => return Sm003Verdict::Fail };
    if arg_x == arg_y { Sm003Verdict::Pass } else { Sm003Verdict::Fail }
}

// ===========================================================================
// SM-004 — SIMD vs scalar: |simd - scalar| < 8 ULPs
// ===========================================================================

pub const AC_SM_004_MAX_ULP: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm004Verdict { Pass, Fail }

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
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> Sm004Verdict {
    if simd.len() != scalar.len() { return Sm004Verdict::Fail; }
    if simd.is_empty() { return Sm004Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_SM_004_MAX_ULP => {}
            _ => return Sm004Verdict::Fail,
        }
    }
    Sm004Verdict::Pass
}

// ===========================================================================
// SM-005 — Single-element softmax([x]) == [1.0]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_single_element(x: f32) -> Sm005Verdict {
    if !x.is_finite() { return Sm005Verdict::Fail; }
    let y = match softmax(&[x]) { Ok(y) => y, Err(_) => return Sm005Verdict::Fail };
    if y.len() != 1 { return Sm005Verdict::Fail; }
    if (y[0] - 1.0).abs() < 1e-6 { Sm005Verdict::Pass } else { Sm005Verdict::Fail }
}

// ===========================================================================
// SM-006 — Constant input → uniform output [1/n; n]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_uniform_on_constant(c: f32, n: usize) -> Sm006Verdict {
    if n == 0 || !c.is_finite() { return Sm006Verdict::Fail; }
    let xs = vec![c; n];
    let y = match softmax(&xs) { Ok(y) => y, Err(_) => return Sm006Verdict::Fail };
    let expected = 1.0_f32 / n as f32;
    let tol = 1e-6_f32 * (n as f32).sqrt().max(1.0);
    for v in y {
        if (v - expected).abs() > tol { return Sm006Verdict::Fail; }
    }
    Sm006Verdict::Pass
}

// ===========================================================================
// SM-007 — Precondition: empty input → Err; NaN/Inf → Err
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_input_validation() -> Sm007Verdict {
    if softmax(&[]).is_ok() { return Sm007Verdict::Fail; }
    if softmax(&[f32::NAN, 1.0]).is_ok() { return Sm007Verdict::Fail; }
    if softmax(&[f32::INFINITY, 1.0]).is_ok() { return Sm007Verdict::Fail; }
    Sm007Verdict::Pass
}

// ===========================================================================
// SM-008 — Postcondition: len(softmax(x)) == len(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_length_preserved(xs: &[f32]) -> Sm008Verdict {
    let y = match softmax(xs) { Ok(y) => y, Err(_) => return Sm008Verdict::Fail };
    if y.len() == xs.len() { Sm008Verdict::Pass } else { Sm008Verdict::Fail }
}

// ===========================================================================
// SM-009 — Frame condition: input buffer byte-identical after call
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sm009Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_input_immutable(input_before: &[f32], input_after: &[f32]) -> Sm009Verdict {
    if input_before.len() != input_after.len() { return Sm009Verdict::Fail; }
    for (a, b) in input_before.iter().zip(input_after) {
        if a.to_bits() != b.to_bits() { return Sm009Verdict::Fail; }
    }
    Sm009Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference impl spot checks
    #[test] fn ref_uniform() {
        let y = softmax(&[1.0, 1.0, 1.0]).unwrap();
        for v in y { assert!((v - 1.0 / 3.0).abs() < 1e-6); }
    }

    #[test] fn ref_max_subtraction_extreme() {
        // Without max-subtraction this would overflow.
        let y = softmax(&[1000.0, 1000.0, 1000.0]).unwrap();
        for v in y { assert!((v - 1.0 / 3.0).abs() < 1e-5); }
    }

    // SM-001
    #[test] fn sm001_pass_uniform() { assert_eq!(verdict_from_normalization(&[1.0; 128]), Sm001Verdict::Pass); }
    #[test] fn sm001_pass_extreme() { assert_eq!(verdict_from_normalization(&[1000.0, -1000.0, 0.0]), Sm001Verdict::Pass); }
    #[test] fn sm001_fail_empty() { assert_eq!(verdict_from_normalization(&[]), Sm001Verdict::Fail); }

    // SM-002
    #[test] fn sm002_pass_normal() { assert_eq!(verdict_from_positivity(&[1.0, 2.0, 3.0]), Sm002Verdict::Pass); }
    #[test] fn sm002_fail_underflow() { assert_eq!(verdict_from_positivity(&[1000.0, -1000.0]), Sm002Verdict::Fail); }
    #[test] fn sm002_fail_empty() { assert_eq!(verdict_from_positivity(&[]), Sm002Verdict::Fail); }

    // SM-003
    #[test] fn sm003_pass_canonical() { assert_eq!(verdict_from_order_preservation(&[1.0, 5.0, 2.0]), Sm003Verdict::Pass); }
    #[test] fn sm003_pass_decreasing() { assert_eq!(verdict_from_order_preservation(&[5.0, 4.0, 3.0]), Sm003Verdict::Pass); }
    #[test] fn sm003_fail_empty() { assert_eq!(verdict_from_order_preservation(&[]), Sm003Verdict::Fail); }

    // SM-004
    #[test] fn sm004_pass_exact() {
        let scalar = vec![0.1, 0.2, 0.7];
        assert_eq!(verdict_from_simd_equivalence(&scalar, &scalar), Sm004Verdict::Pass);
    }

    #[test] fn sm004_pass_within_tolerance() {
        let scalar = [0.1_f32, 0.2, 0.7];
        let simd: Vec<f32> = scalar.iter().map(|v| f32::from_bits(v.to_bits() + 3)).collect();
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Sm004Verdict::Pass);
    }

    #[test] fn sm004_fail_far_apart() {
        let scalar = [0.1_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 100)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Sm004Verdict::Fail);
    }

    #[test] fn sm004_fail_length_drift() {
        let scalar = [0.1_f32, 0.2];
        let simd = [0.1_f32];
        assert_eq!(verdict_from_simd_equivalence(&simd, &scalar), Sm004Verdict::Fail);
    }

    // SM-005
    #[test] fn sm005_pass_zero() { assert_eq!(verdict_from_single_element(0.0), Sm005Verdict::Pass); }
    #[test] fn sm005_pass_large() { assert_eq!(verdict_from_single_element(1000.0), Sm005Verdict::Pass); }
    #[test] fn sm005_fail_nan() { assert_eq!(verdict_from_single_element(f32::NAN), Sm005Verdict::Fail); }

    // SM-006
    #[test] fn sm006_pass_n3() { assert_eq!(verdict_from_uniform_on_constant(7.5, 3), Sm006Verdict::Pass); }
    #[test] fn sm006_pass_n128() { assert_eq!(verdict_from_uniform_on_constant(0.0, 128), Sm006Verdict::Pass); }
    #[test] fn sm006_fail_zero_n() { assert_eq!(verdict_from_uniform_on_constant(0.0, 0), Sm006Verdict::Fail); }

    // SM-007
    #[test] fn sm007_pass() { assert_eq!(verdict_from_input_validation(), Sm007Verdict::Pass); }

    // SM-008
    #[test] fn sm008_pass_n5() { assert_eq!(verdict_from_length_preserved(&[1.0; 5]), Sm008Verdict::Pass); }
    #[test] fn sm008_fail_empty() { assert_eq!(verdict_from_length_preserved(&[]), Sm008Verdict::Fail); }

    // SM-009
    #[test] fn sm009_pass_unchanged() {
        let v = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_input_immutable(&v, &v), Sm009Verdict::Pass);
    }

    #[test] fn sm009_fail_modified() {
        let before = [0.1_f32, 0.2];
        let after = [0.1_f32, 0.5];
        assert_eq!(verdict_from_input_immutable(&before, &after), Sm009Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_max_ulp() { assert_eq!(AC_SM_004_MAX_ULP, 8); }
}
