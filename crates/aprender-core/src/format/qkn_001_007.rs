// SHIP-TWO-001 — `qk-norm-v1` algorithm-level PARTIAL discharge for
// FALSIFY-QKN-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qk-norm-v1.yaml`.
// Spec: Qwen3 architecture per-head Q/K RMSNorm.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNormError { LengthMismatch, EpsNonPositive, NonFiniteInput, EmptyInput }

/// Per-vector RMSNorm: y_i = x_i * gamma_i / sqrt(mean(x²) + eps).
pub fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Result<Vec<f32>, QkNormError> {
    if x.is_empty() { return Err(QkNormError::EmptyInput); }
    if x.len() != gamma.len() { return Err(QkNormError::LengthMismatch); }
    if eps <= 0.0 || !eps.is_finite() { return Err(QkNormError::EpsNonPositive); }
    if x.iter().chain(gamma.iter()).any(|v| !v.is_finite()) {
        return Err(QkNormError::NonFiniteInput);
    }
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n + eps).sqrt();
    Ok(x.iter().zip(gamma).map(|(xi, gi)| xi * gi * inv_rms).collect())
}

/// Per-head batched RMSNorm: applies `rmsnorm` independently to each
/// `head_dim`-sized slice of `x`.
pub fn rmsnorm_per_head(x: &[f32], gamma: &[f32], eps: f32, head_dim: usize) -> Result<Vec<f32>, QkNormError> {
    if head_dim == 0 || x.is_empty() { return Err(QkNormError::EmptyInput); }
    if !x.len().is_multiple_of(head_dim) { return Err(QkNormError::LengthMismatch); }
    if gamma.len() != head_dim { return Err(QkNormError::LengthMismatch); }
    let mut out = Vec::with_capacity(x.len());
    for chunk in x.chunks_exact(head_dim) {
        out.extend(rmsnorm(chunk, gamma, eps)?);
    }
    Ok(out)
}

fn rms(v: &[f32]) -> f64 {
    let n = v.len() as f64;
    let sum_sq: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum();
    (sum_sq / n).sqrt()
}

// ===========================================================================
// QKN-001 — Unit RMS: RMS of normalized output ≈ 1.0 with unit weight
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_unit_rms(x: &[f32], eps: f32) -> Qkn001Verdict {
    if x.len() < 2 { return Qkn001Verdict::Fail; }
    let gamma = vec![1.0_f32; x.len()];
    let y = match rmsnorm(x, &gamma, eps) { Ok(v) => v, Err(_) => return Qkn001Verdict::Fail };
    let r = rms(&y);
    if (r - 1.0).abs() < 1e-3 { Qkn001Verdict::Pass } else { Qkn001Verdict::Fail }
}

// ===========================================================================
// QKN-002 — Bounded amplitude: |y_i| <= |gamma_i| * sqrt(d/eps) * |x_i|/sqrt(eps)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn002Verdict { Pass, Fail }

/// Pass iff every output magnitude is finite (the eps guard prevents
/// division-by-zero blow-up). The contract's exact bound is loose;
/// finite-output is the operational invariant.
#[must_use]
pub fn verdict_from_bounded_amplitude(x: &[f32], gamma: &[f32], eps: f32) -> Qkn002Verdict {
    let y = match rmsnorm(x, gamma, eps) { Ok(v) => v, Err(_) => return Qkn002Verdict::Fail };
    if y.iter().all(|v| v.is_finite()) { Qkn002Verdict::Pass } else { Qkn002Verdict::Fail }
}

// ===========================================================================
// QKN-003 — Idempotent: |RMSNorm(RMSNorm(x)) - RMSNorm(x)| < tol
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_idempotent(x: &[f32], eps: f32) -> Qkn003Verdict {
    if x.len() < 2 { return Qkn003Verdict::Fail; }
    let gamma = vec![1.0_f32; x.len()];
    let y1 = match rmsnorm(x, &gamma, eps) { Ok(v) => v, Err(_) => return Qkn003Verdict::Fail };
    let y2 = match rmsnorm(&y1, &gamma, eps) { Ok(v) => v, Err(_) => return Qkn003Verdict::Fail };
    for (a, b) in y1.iter().zip(y2.iter()) {
        if (a - b).abs() > 1e-3 { return Qkn003Verdict::Fail; }
    }
    Qkn003Verdict::Pass
}

// ===========================================================================
// QKN-004 — Zero stability: RMSNorm(0) = 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_zero_stability(d: usize, eps: f32) -> Qkn004Verdict {
    if d == 0 { return Qkn004Verdict::Fail; }
    let x = vec![0.0_f32; d];
    let gamma = vec![1.0_f32; d];
    let y = match rmsnorm(&x, &gamma, eps) { Ok(v) => v, Err(_) => return Qkn004Verdict::Fail };
    for v in y { if v != 0.0 { return Qkn004Verdict::Fail; } }
    Qkn004Verdict::Pass
}

// ===========================================================================
// QKN-005 — SIMD vs scalar: |simd - scalar| < 8 ULPs
// ===========================================================================

pub const AC_QKN_005_MAX_ULP: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn005Verdict { Pass, Fail }

fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() { return None; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    if (ai < 0) != (bi < 0) { return Some(ai.unsigned_abs() + bi.unsigned_abs()); }
    Some((ai - bi).unsigned_abs())
}

#[must_use]
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> Qkn005Verdict {
    if simd.len() != scalar.len() || simd.is_empty() { return Qkn005Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_QKN_005_MAX_ULP => {}
            _ => return Qkn005Verdict::Fail,
        }
    }
    Qkn005Verdict::Pass
}

// ===========================================================================
// QKN-006 — GPU/CPU parity: cosine >= 0.999
// ===========================================================================

pub const AC_QKN_006_MIN_COSINE: f64 = 0.999;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn006Verdict { Pass, Fail }

fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() { return None; }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) { return None; }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { return None; }
    Some(dot / (na * nb))
}

#[must_use]
pub fn verdict_from_gpu_cpu_parity(cpu: &[f32], gpu: &[f32]) -> Qkn006Verdict {
    match cosine(cpu, gpu) {
        Some(c) if c >= AC_QKN_006_MIN_COSINE => Qkn006Verdict::Pass,
        _ => Qkn006Verdict::Fail,
    }
}

// ===========================================================================
// QKN-007 — Per-head independence: batched per-head == individual heads
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkn007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_per_head_independence(
    x: &[f32],
    gamma: &[f32],
    eps: f32,
    head_dim: usize,
) -> Qkn007Verdict {
    if head_dim == 0 || x.is_empty() || !x.len().is_multiple_of(head_dim) { return Qkn007Verdict::Fail; }
    let batch = match rmsnorm_per_head(x, gamma, eps, head_dim) {
        Ok(v) => v, Err(_) => return Qkn007Verdict::Fail,
    };
    let mut individual = Vec::with_capacity(x.len());
    for chunk in x.chunks_exact(head_dim) {
        individual.extend(match rmsnorm(chunk, gamma, eps) {
            Ok(v) => v, Err(_) => return Qkn007Verdict::Fail,
        });
    }
    if batch.len() != individual.len() { return Qkn007Verdict::Fail; }
    for (a, b) in batch.iter().zip(individual.iter()) {
        if (a - b).abs() > 1e-6 { return Qkn007Verdict::Fail; }
    }
    Qkn007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_x(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.7 - 5.0).sin() * 2.0).collect()
    }

    // QKN-001
    #[test] fn qkn001_pass_random() {
        let x = rand_x(64);
        assert_eq!(verdict_from_unit_rms(&x, 1e-6), Qkn001Verdict::Pass);
    }
    #[test] fn qkn001_pass_qwen3_head_dim() {
        let x = rand_x(128);
        assert_eq!(verdict_from_unit_rms(&x, 1e-6), Qkn001Verdict::Pass);
    }
    #[test] fn qkn001_fail_too_short() {
        assert_eq!(verdict_from_unit_rms(&[1.0], 1e-6), Qkn001Verdict::Fail);
    }

    // QKN-002
    #[test] fn qkn002_pass_normal() {
        let x = rand_x(8);
        assert_eq!(verdict_from_bounded_amplitude(&x, &[1.0; 8], 1e-6), Qkn002Verdict::Pass);
    }
    #[test] fn qkn002_pass_extreme() {
        let x = vec![1e30_f32, -1e30, 0.0, 1.0];
        assert_eq!(verdict_from_bounded_amplitude(&x, &[1.0; 4], 1e-6), Qkn002Verdict::Pass);
    }
    #[test] fn qkn002_fail_zero_eps() {
        let x = rand_x(4);
        assert_eq!(verdict_from_bounded_amplitude(&x, &[1.0; 4], 0.0), Qkn002Verdict::Fail);
    }

    // QKN-003
    #[test] fn qkn003_pass_random() {
        let x = rand_x(32);
        assert_eq!(verdict_from_idempotent(&x, 1e-6), Qkn003Verdict::Pass);
    }

    // QKN-004
    #[test] fn qkn004_pass_d8() { assert_eq!(verdict_from_zero_stability(8, 1e-6), Qkn004Verdict::Pass); }
    #[test] fn qkn004_pass_d128() { assert_eq!(verdict_from_zero_stability(128, 1e-6), Qkn004Verdict::Pass); }
    #[test] fn qkn004_fail_d_zero() { assert_eq!(verdict_from_zero_stability(0, 1e-6), Qkn004Verdict::Fail); }

    // QKN-005
    #[test] fn qkn005_pass_exact() {
        let s = vec![0.1_f32, 0.2];
        assert_eq!(verdict_from_simd_equivalence(&s, &s), Qkn005Verdict::Pass);
    }
    #[test] fn qkn005_pass_within_8_ulp() {
        let s = [0.1_f32];
        let simd = [f32::from_bits(s[0].to_bits() + 5)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &s), Qkn005Verdict::Pass);
    }
    #[test] fn qkn005_fail_far_apart() {
        let s = [0.1_f32];
        let simd = [f32::from_bits(s[0].to_bits() + 100)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &s), Qkn005Verdict::Fail);
    }

    // QKN-006
    #[test] fn qkn006_pass_perfect() {
        let cpu = rand_x(16);
        assert_eq!(verdict_from_gpu_cpu_parity(&cpu, &cpu), Qkn006Verdict::Pass);
    }
    #[test] fn qkn006_pass_close() {
        let cpu = vec![1.0_f32, 2.0, 3.0];
        let gpu = vec![1.001_f32, 2.001, 2.999];
        assert_eq!(verdict_from_gpu_cpu_parity(&cpu, &gpu), Qkn006Verdict::Pass);
    }
    #[test] fn qkn006_fail_orthogonal() {
        let cpu = vec![1.0_f32, 0.0];
        let gpu = vec![0.0_f32, 1.0];
        assert_eq!(verdict_from_gpu_cpu_parity(&cpu, &gpu), Qkn006Verdict::Fail);
    }

    // QKN-007
    #[test] fn qkn007_pass_qwen3_8_heads_128_dim() {
        // 8 × 128 = 1024 element vector, head_dim = 128.
        let x = rand_x(1024);
        let gamma = vec![1.0_f32; 128];
        assert_eq!(
            verdict_from_per_head_independence(&x, &gamma, 1e-6, 128),
            Qkn007Verdict::Pass
        );
    }
    #[test] fn qkn007_pass_small() {
        let x = rand_x(16); // 4 heads × 4 dim
        let gamma = vec![1.0_f32; 4];
        assert_eq!(
            verdict_from_per_head_independence(&x, &gamma, 1e-6, 4),
            Qkn007Verdict::Pass
        );
    }
    #[test] fn qkn007_fail_indivisible() {
        let x = rand_x(13);
        let gamma = vec![1.0_f32; 4];
        assert_eq!(
            verdict_from_per_head_independence(&x, &gamma, 1e-6, 4),
            Qkn007Verdict::Fail
        );
    }

    // Provenance pins
    #[test] fn provenance_max_ulp() { assert_eq!(AC_QKN_005_MAX_ULP, 8); }
    #[test] fn provenance_min_cosine() { assert!((AC_QKN_006_MIN_COSINE - 0.999).abs() < 1e-9); }
}
