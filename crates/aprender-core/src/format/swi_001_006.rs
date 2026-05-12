// SHIP-TWO-001 — `swiglu-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-SG-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/swiglu-kernel-v1.yaml`.
// Spec: SwiGLU kernel — SwiGLU(x, W, V, b, c) = SiLU(xW + b) * (xV + c)
// (Shazeer 2020 GLU Variants; SiLU from Ramachandran et al. 2017).

// ===========================================================================
// Helpers — SiLU and reference SwiGLU (in-module, no external deps)
// ===========================================================================

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
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Linear projection: out[i] = Σ_j x[j] * w[j*h + i] + b[i].
/// `w` shape: (d × h) row-major; `x` shape: d; `b` shape: h.
#[must_use]
pub fn linear(x: &[f32], w: &[f32], b: &[f32], d: usize, h: usize) -> Vec<f32> {
    if x.len() != d || w.len() != d * h || b.len() != h { return vec![]; }
    let mut out = vec![0.0_f32; h];
    for i in 0..h {
        let mut acc = b[i];
        for j in 0..d {
            acc += x[j] * w[j * h + i];
        }
        out[i] = acc;
    }
    out
}

/// SwiGLU reference (component-wise): silu(xW + b) * (xV + c)
#[must_use]
pub fn swiglu_unfused(
    x: &[f32],
    w_gate: &[f32],
    b_gate: &[f32],
    w_value: &[f32],
    b_value: &[f32],
    d: usize,
    h: usize,
) -> Vec<f32> {
    let gate = linear(x, w_gate, b_gate, d, h);
    let value = linear(x, w_value, b_value, d, h);
    if gate.len() != h || value.len() != h { return vec![]; }
    gate.iter().zip(value.iter()).map(|(&g, &v)| silu(g) * v).collect()
}

// ===========================================================================
// SG-001 — Zero preservation: SwiGLU(0, W, V, 0, 0) = 0
// ===========================================================================

pub const AC_SG_001_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sg001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_zero_preservation(
    w_gate: &[f32],
    w_value: &[f32],
    d: usize,
    h: usize,
) -> Sg001Verdict {
    if d == 0 || h == 0 { return Sg001Verdict::Fail; }
    if w_gate.len() != d * h || w_value.len() != d * h { return Sg001Verdict::Fail; }
    if !w_gate.iter().all(|v| v.is_finite()) || !w_value.iter().all(|v| v.is_finite()) {
        return Sg001Verdict::Fail;
    }
    let x = vec![0.0_f32; d];
    let b = vec![0.0_f32; h];
    let out = swiglu_unfused(&x, w_gate, &b, w_value, &b, d, h);
    if out.len() != h { return Sg001Verdict::Fail; }
    for &v in &out {
        if !v.is_finite() || v.abs() > AC_SG_001_TOLERANCE { return Sg001Verdict::Fail; }
    }
    Sg001Verdict::Pass
}

// ===========================================================================
// SG-002 — Fused equivalence: |fused - unfused| < 1e-6
// ===========================================================================

pub const AC_SG_002_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sg002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_fused_equivalence(fused: &[f32], unfused: &[f32]) -> Sg002Verdict {
    if fused.is_empty() || unfused.is_empty() { return Sg002Verdict::Fail; }
    if fused.len() != unfused.len() { return Sg002Verdict::Fail; }
    for (&a, &b) in fused.iter().zip(unfused.iter()) {
        if !a.is_finite() || !b.is_finite() { return Sg002Verdict::Fail; }
        if (a - b).abs() > AC_SG_002_TOLERANCE { return Sg002Verdict::Fail; }
    }
    Sg002Verdict::Pass
}

// ===========================================================================
// SG-003 — SiLU lower bound: SiLU(z) > -0.279 for finite z
// ===========================================================================

pub const AC_SG_003_LOWER_BOUND: f32 = -0.279;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sg003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_silu_lower_bound(probes: &[f32]) -> Sg003Verdict {
    if probes.is_empty() { return Sg003Verdict::Fail; }
    for &z in probes {
        if !z.is_finite() { return Sg003Verdict::Fail; }
        let s = silu(z);
        if !s.is_finite() { return Sg003Verdict::Fail; }
        if s <= AC_SG_003_LOWER_BOUND { return Sg003Verdict::Fail; }
    }
    Sg003Verdict::Pass
}

// ===========================================================================
// SG-004 — SIMD parity within 8 ULP
// ===========================================================================

pub const AC_SG_004_ULP_TOLERANCE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sg004Verdict { Pass, Fail }

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
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Sg004Verdict {
    if scalar.is_empty() || simd.is_empty() { return Sg004Verdict::Fail; }
    if scalar.len() != simd.len() { return Sg004Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Sg004Verdict::Fail; }
        if ulp_distance(s, v) > AC_SG_004_ULP_TOLERANCE { return Sg004Verdict::Fail; }
    }
    Sg004Verdict::Pass
}

// ===========================================================================
// SG-005 — Boundary: empty input → empty output
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sg005Verdict { Pass, Fail }

/// Pass iff `output.is_empty()` matches `x.is_empty()` (length contract).
/// Caller passes the actual computed output of their kernel.
#[must_use]
pub fn verdict_from_empty_boundary(x: &[f32], output: &[f32]) -> Sg005Verdict {
    if x.is_empty() {
        if output.is_empty() { Sg005Verdict::Pass } else { Sg005Verdict::Fail }
    } else if output.is_empty() {
        Sg005Verdict::Fail
    } else {
        Sg005Verdict::Pass
    }
}

// ===========================================================================
// SG-006 — Gate monotonicity: SiLU is monotonic on x > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sg006Verdict { Pass, Fail }

/// Pass iff for sorted `gate_inputs` (all > 0), `silu(gate)` produces a
/// non-decreasing sequence.
#[must_use]
pub fn verdict_from_gate_monotonicity(gate_inputs: &[f32]) -> Sg006Verdict {
    if gate_inputs.is_empty() { return Sg006Verdict::Fail; }
    let mut prev = f32::NEG_INFINITY;
    let mut last_silu = f32::NEG_INFINITY;
    for &z in gate_inputs {
        if !z.is_finite() || z <= 0.0 { return Sg006Verdict::Fail; }
        if z < prev { return Sg006Verdict::Fail; } // require sorted ascending
        let s = silu(z);
        if !s.is_finite() { return Sg006Verdict::Fail; }
        if s < last_silu { return Sg006Verdict::Fail; }
        prev = z;
        last_silu = s;
    }
    Sg006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // SG-001 (zero preservation)
    #[test] fn sg001_pass_random_w() {
        // d=2, h=3
        let w_gate = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w_value = vec![-0.1_f32, 0.05, -0.2, 0.3, -0.15, 0.25];
        assert_eq!(verdict_from_zero_preservation(&w_gate, &w_value, 2, 3), Sg001Verdict::Pass);
    }
    #[test] fn sg001_fail_dim_zero() {
        let w_gate = vec![1.0_f32];
        let w_value = vec![1.0_f32];
        assert_eq!(verdict_from_zero_preservation(&w_gate, &w_value, 0, 0), Sg001Verdict::Fail);
    }
    #[test] fn sg001_fail_nan_w() {
        let w_gate = vec![f32::NAN, 0.2, 0.3, 0.4];
        let w_value = vec![0.1_f32, 0.2, 0.3, 0.4];
        assert_eq!(verdict_from_zero_preservation(&w_gate, &w_value, 2, 2), Sg001Verdict::Fail);
    }

    // SG-002 (fused equivalence)
    #[test] fn sg002_pass_identical() {
        let a = vec![0.1_f32, 0.5, -0.3];
        assert_eq!(verdict_from_fused_equivalence(&a, &a), Sg002Verdict::Pass);
    }
    #[test] fn sg002_pass_within_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32 + 1e-7]; // < 1e-6
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Sg002Verdict::Pass);
    }
    #[test] fn sg002_fail_above_tolerance() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32 + 1e-3]; // > 1e-6
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Sg002Verdict::Fail);
    }
    #[test] fn sg002_fail_length() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Sg002Verdict::Fail);
    }
    #[test] fn sg002_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![1.0_f32];
        assert_eq!(verdict_from_fused_equivalence(&a, &b), Sg002Verdict::Fail);
    }

    // SG-003 (SiLU lower bound)
    #[test] fn sg003_pass_canonical_range() {
        // SiLU global minimum is at z ≈ -1.278, value ≈ -0.2785.
        // Probe across a wide range; all values must be > -0.279.
        let probes: Vec<f32> = (-1000..1000)
            .step_by(50)
            .map(|i| i as f32 / 10.0)
            .collect();
        assert_eq!(verdict_from_silu_lower_bound(&probes), Sg003Verdict::Pass);
    }
    #[test] fn sg003_pass_at_global_minimum() {
        // Exactly at the global min: silu(-1.278) ≈ -0.2784645 > -0.279.
        let probes = [-1.2784_f32, -1.0, -1.5];
        assert_eq!(verdict_from_silu_lower_bound(&probes), Sg003Verdict::Pass);
    }
    #[test] fn sg003_fail_nan() {
        assert_eq!(verdict_from_silu_lower_bound(&[f32::NAN]), Sg003Verdict::Fail);
    }
    #[test] fn sg003_fail_inf() {
        assert_eq!(verdict_from_silu_lower_bound(&[f32::INFINITY]), Sg003Verdict::Fail);
    }
    #[test] fn sg003_fail_empty() {
        assert_eq!(verdict_from_silu_lower_bound(&[]), Sg003Verdict::Fail);
    }
    #[test] fn silu_zero_is_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
    }

    // SG-004 (SIMD parity)
    #[test] fn sg004_pass_identical() {
        let a = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &a), Sg004Verdict::Pass);
    }
    #[test] fn sg004_pass_within_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 4)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Sg004Verdict::Pass);
    }
    #[test] fn sg004_fail_above_8_ulp() {
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 100)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Sg004Verdict::Fail);
    }

    // SG-005 (empty boundary)
    #[test] fn sg005_pass_empty_to_empty() {
        assert_eq!(verdict_from_empty_boundary(&[], &[]), Sg005Verdict::Pass);
    }
    #[test] fn sg005_pass_nonempty_to_nonempty() {
        let x = [1.0_f32];
        let out = [1.0_f32];
        assert_eq!(verdict_from_empty_boundary(&x, &out), Sg005Verdict::Pass);
    }
    #[test] fn sg005_fail_empty_to_nonempty() {
        let out = [1.0_f32];
        assert_eq!(verdict_from_empty_boundary(&[], &out), Sg005Verdict::Fail);
    }
    #[test] fn sg005_fail_nonempty_to_empty() {
        let x = [1.0_f32];
        assert_eq!(verdict_from_empty_boundary(&x, &[]), Sg005Verdict::Fail);
    }

    // SG-006 (gate monotonicity)
    #[test] fn sg006_pass_sorted_positive() {
        let z = [0.1_f32, 0.5, 1.0, 2.0, 5.0, 10.0];
        assert_eq!(verdict_from_gate_monotonicity(&z), Sg006Verdict::Pass);
    }
    #[test] fn sg006_fail_negative_input() {
        // SiLU is NOT monotonic on negatives (has a min near -1.28).
        // The verdict explicitly requires positive domain.
        let z = [-1.0_f32, -0.5, 0.5];
        assert_eq!(verdict_from_gate_monotonicity(&z), Sg006Verdict::Fail);
    }
    #[test] fn sg006_fail_unsorted() {
        let z = [1.0_f32, 0.5, 2.0]; // not ascending
        assert_eq!(verdict_from_gate_monotonicity(&z), Sg006Verdict::Fail);
    }
    #[test] fn sg006_fail_zero() {
        let z = [0.0_f32, 1.0];
        assert_eq!(verdict_from_gate_monotonicity(&z), Sg006Verdict::Fail);
    }
    #[test] fn sg006_fail_empty() {
        assert_eq!(verdict_from_gate_monotonicity(&[]), Sg006Verdict::Fail);
    }

    // SwiGLU helper sanity
    #[test] fn swiglu_zero_input_zero_bias() {
        let w_gate = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w_value = vec![5.0_f32, 6.0, 7.0, 8.0];
        let x = vec![0.0_f32, 0.0];
        let b = vec![0.0_f32, 0.0];
        let out = swiglu_unfused(&x, &w_gate, &b, &w_value, &b, 2, 2);
        assert_eq!(out, vec![0.0_f32, 0.0]);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_SG_001_TOLERANCE - 1e-6).abs() < 1e-12);
        assert!((AC_SG_002_TOLERANCE - 1e-6).abs() < 1e-12);
        assert!((AC_SG_003_LOWER_BOUND - (-0.279)).abs() < 1e-9);
        assert_eq!(AC_SG_004_ULP_TOLERANCE, 8);
    }
}
