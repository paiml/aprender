// SHIP-TWO-001 — `gqa-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-GQ-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/gqa-kernel-v1.yaml`.

// ===========================================================================
// GQ-001 — Attention weights normalize to 1.0 per query position
// ===========================================================================

pub const AC_GQ_001_TOLERANCE: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq001Verdict { Pass, Fail }

/// Pass iff every row of `attn_weights` (shape [num_queries][seq_len])
/// sums to 1.0 within tolerance.
#[must_use]
pub fn verdict_from_attn_weight_normalization(weights: &[Vec<f32>]) -> Gq001Verdict {
    if weights.is_empty() { return Gq001Verdict::Fail; }
    for row in weights {
        if row.is_empty() { return Gq001Verdict::Fail; }
        if row.iter().any(|v| !v.is_finite()) { return Gq001Verdict::Fail; }
        let sum: f32 = row.iter().sum();
        if (sum - 1.0).abs() > AC_GQ_001_TOLERANCE { return Gq001Verdict::Fail; }
    }
    Gq001Verdict::Pass
}

// ===========================================================================
// GQ-002 — MHA degeneration: GQA(kv=h) ≈ MHA when kv_heads == num_heads
// ===========================================================================

pub const AC_GQ_002_TOLERANCE: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mha_degeneration(
    gqa_output: &[f32],
    mha_output: &[f32],
    num_heads: u32,
    num_kv_heads: u32,
) -> Gq002Verdict {
    if num_heads != num_kv_heads || num_heads == 0 { return Gq002Verdict::Fail; }
    if gqa_output.len() != mha_output.len() || gqa_output.is_empty() { return Gq002Verdict::Fail; }
    for (a, b) in gqa_output.iter().zip(mha_output.iter()) {
        if (a - b).abs() > AC_GQ_002_TOLERANCE { return Gq002Verdict::Fail; }
    }
    Gq002Verdict::Pass
}

// ===========================================================================
// GQ-003 — Convex combination: min(V) <= output_i <= max(V) per head
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_convex_combination(output: &[f32], v_values: &[f32]) -> Gq003Verdict {
    if output.is_empty() || v_values.is_empty() { return Gq003Verdict::Fail; }
    if output.iter().chain(v_values.iter()).any(|v| !v.is_finite()) { return Gq003Verdict::Fail; }
    let v_min = v_values.iter().copied().fold(f32::INFINITY, f32::min);
    let v_max = v_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Allow a small FP tolerance band.
    let tol = (v_max - v_min).abs().mul_add(1e-5, 1e-7);
    for o in output {
        if *o < v_min - tol || *o > v_max + tol { return Gq003Verdict::Fail; }
    }
    Gq003Verdict::Pass
}

// ===========================================================================
// GQ-004 — Head divisibility: num_heads.is_multiple_of(num_kv_heads)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_head_divisibility(num_heads: u32, num_kv_heads: u32) -> Gq004Verdict {
    if num_heads == 0 || num_kv_heads == 0 { return Gq004Verdict::Fail; }
    if num_heads.is_multiple_of(num_kv_heads) { Gq004Verdict::Pass } else { Gq004Verdict::Fail }
}

// ===========================================================================
// GQ-005 — SIMD vs scalar: |simd - scalar| < 8 ULPs
// ===========================================================================

pub const AC_GQ_005_MAX_ULP: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq005Verdict { Pass, Fail }

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
pub fn verdict_from_simd_equivalence(simd: &[f32], scalar: &[f32]) -> Gq005Verdict {
    if simd.len() != scalar.len() || simd.is_empty() { return Gq005Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_GQ_005_MAX_ULP => {}
            _ => return Gq005Verdict::Fail,
        }
    }
    Gq005Verdict::Pass
}

// ===========================================================================
// GQ-006 — MQA boundary: kv_heads=1 broadcasts to all q heads
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq006Verdict { Pass, Fail }

/// Pass iff for every query head index `q`, the KV head index used
/// is 0 (i.e., the single KV head is broadcast to all queries) AND
/// `num_kv_heads == 1`.
#[must_use]
pub fn verdict_from_mqa_broadcast(num_heads: u32, num_kv_heads: u32, kv_indices: &[u32]) -> Gq006Verdict {
    if num_kv_heads != 1 { return Gq006Verdict::Fail; }
    if kv_indices.len() != num_heads as usize || num_heads == 0 { return Gq006Verdict::Fail; }
    for idx in kv_indices {
        if *idx != 0 { return Gq006Verdict::Fail; }
    }
    Gq006Verdict::Pass
}

// ===========================================================================
// GQ-007 — GPU/CPU parity: cosine >= 0.98 for non-power-of-2 ratio
// ===========================================================================

pub const AC_GQ_007_MIN_COSINE: f32 = 0.98;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq007Verdict { Pass, Fail }

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() { return None; }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) { return None; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { return None; }
    Some(dot / (na * nb))
}

#[must_use]
pub fn verdict_from_gpu_cpu_parity(cpu: &[f32], gpu: &[f32]) -> Gq007Verdict {
    match cosine_similarity(cpu, gpu) {
        Some(c) if c >= AC_GQ_007_MIN_COSINE => Gq007Verdict::Pass,
        _ => Gq007Verdict::Fail,
    }
}

// ===========================================================================
// GQ-008 — GPU/CPU parity: cosine >= 0.98 for power-of-2 ratio
// ===========================================================================
//
// Same decision rule as GQ-007; configuration differs (heads=32, kv=8).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gpu_cpu_parity_pow2(cpu: &[f32], gpu: &[f32]) -> Gq008Verdict {
    match verdict_from_gpu_cpu_parity(cpu, gpu) {
        Gq007Verdict::Pass => Gq008Verdict::Pass,
        Gq007Verdict::Fail => Gq008Verdict::Fail,
    }
}

// ===========================================================================
// GQ-009 — Head mapping: kv_head_idx(q) == q * num_kv_heads / num_heads
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gq009Verdict { Pass, Fail }

#[must_use]
pub const fn expected_kv_head_idx(q: u32, num_heads: u32, num_kv_heads: u32) -> Option<u32> {
    if num_heads == 0 || num_kv_heads == 0 { return None; }
    if q >= num_heads { return None; }
    Some(q * num_kv_heads / num_heads)
}

#[must_use]
pub fn verdict_from_head_mapping(
    num_heads: u32,
    num_kv_heads: u32,
    observed_indices: &[u32],
) -> Gq009Verdict {
    if num_heads == 0 || num_kv_heads == 0 { return Gq009Verdict::Fail; }
    if !num_heads.is_multiple_of(num_kv_heads) { return Gq009Verdict::Fail; }
    if observed_indices.len() != num_heads as usize { return Gq009Verdict::Fail; }
    for q in 0..num_heads {
        let expected = match expected_kv_head_idx(q, num_heads, num_kv_heads) {
            Some(e) => e, None => return Gq009Verdict::Fail,
        };
        if observed_indices[q as usize] != expected { return Gq009Verdict::Fail; }
    }
    Gq009Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- GQ-001 ----------------------------------------------------------

    #[test] fn gq001_pass_uniform() {
        let w = vec![vec![0.25_f32; 4], vec![0.25; 4]];
        assert_eq!(verdict_from_attn_weight_normalization(&w), Gq001Verdict::Pass);
    }
    #[test] fn gq001_fail_unnormalized() {
        let w = vec![vec![0.5_f32, 0.4]];
        assert_eq!(verdict_from_attn_weight_normalization(&w), Gq001Verdict::Fail);
    }
    #[test] fn gq001_fail_empty() {
        assert_eq!(verdict_from_attn_weight_normalization(&[]), Gq001Verdict::Fail);
    }

    // ----- GQ-002 ----------------------------------------------------------

    #[test] fn gq002_pass_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_mha_degeneration(&v, &v, 8, 8), Gq002Verdict::Pass);
    }
    #[test] fn gq002_fail_different() {
        assert_eq!(
            verdict_from_mha_degeneration(&[1.0, 2.0], &[1.0, 3.0], 4, 4),
            Gq002Verdict::Fail
        );
    }
    #[test] fn gq002_fail_unequal_heads() {
        let v = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_mha_degeneration(&v, &v, 8, 4), Gq002Verdict::Fail);
    }

    // ----- GQ-003 ----------------------------------------------------------

    #[test] fn gq003_pass_in_range() {
        assert_eq!(
            verdict_from_convex_combination(&[1.5, 2.5, 1.0], &[0.0, 1.0, 2.0, 3.0]),
            Gq003Verdict::Pass
        );
    }
    #[test] fn gq003_fail_below_min() {
        assert_eq!(
            verdict_from_convex_combination(&[-1.0], &[0.0, 1.0, 2.0]),
            Gq003Verdict::Fail
        );
    }
    #[test] fn gq003_fail_above_max() {
        assert_eq!(
            verdict_from_convex_combination(&[5.0], &[0.0, 1.0, 2.0]),
            Gq003Verdict::Fail
        );
    }

    // ----- GQ-004 ----------------------------------------------------------

    #[test] fn gq004_pass_qwen2_5_7b() { assert_eq!(verdict_from_head_divisibility(28, 4), Gq004Verdict::Pass); }
    #[test] fn gq004_pass_pow2() { assert_eq!(verdict_from_head_divisibility(32, 8), Gq004Verdict::Pass); }
    #[test] fn gq004_pass_mqa() { assert_eq!(verdict_from_head_divisibility(8, 1), Gq004Verdict::Pass); }
    #[test] fn gq004_pass_mha() { assert_eq!(verdict_from_head_divisibility(8, 8), Gq004Verdict::Pass); }
    #[test] fn gq004_fail_indivisible() { assert_eq!(verdict_from_head_divisibility(7, 3), Gq004Verdict::Fail); }
    #[test] fn gq004_fail_zero() { assert_eq!(verdict_from_head_divisibility(0, 4), Gq004Verdict::Fail); }

    // ----- GQ-005 ----------------------------------------------------------

    #[test] fn gq005_pass_exact() {
        let s = vec![0.1_f32, 0.2];
        assert_eq!(verdict_from_simd_equivalence(&s, &s), Gq005Verdict::Pass);
    }
    #[test] fn gq005_pass_within_8_ulp() {
        let s = [0.1_f32];
        let simd = [f32::from_bits(s[0].to_bits() + 5)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &s), Gq005Verdict::Pass);
    }
    #[test] fn gq005_fail_far_apart() {
        let s = [0.1_f32];
        let simd = [f32::from_bits(s[0].to_bits() + 100)];
        assert_eq!(verdict_from_simd_equivalence(&simd, &s), Gq005Verdict::Fail);
    }

    // ----- GQ-006 ----------------------------------------------------------

    #[test] fn gq006_pass_mqa_8_heads() {
        let kv = vec![0_u32; 8];
        assert_eq!(verdict_from_mqa_broadcast(8, 1, &kv), Gq006Verdict::Pass);
    }
    #[test] fn gq006_fail_non_mqa() {
        let kv = vec![0_u32, 0, 1, 1, 2, 2, 3, 3];
        assert_eq!(verdict_from_mqa_broadcast(8, 4, &kv), Gq006Verdict::Fail);
    }
    #[test] fn gq006_fail_wrong_index() {
        let kv = vec![0_u32, 0, 1, 0]; // contains a non-zero
        assert_eq!(verdict_from_mqa_broadcast(4, 1, &kv), Gq006Verdict::Fail);
    }

    // ----- GQ-007 / GQ-008 -------------------------------------------------

    #[test] fn gq007_pass_perfect() {
        let cpu = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_gpu_cpu_parity(&cpu, &cpu), Gq007Verdict::Pass);
    }
    #[test] fn gq007_pass_close() {
        let cpu = vec![1.0_f32, 2.0, 3.0];
        let gpu = vec![1.01_f32, 2.01, 2.99];
        assert_eq!(verdict_from_gpu_cpu_parity(&cpu, &gpu), Gq007Verdict::Pass);
    }
    #[test] fn gq007_fail_orthogonal() {
        let cpu = vec![1.0_f32, 0.0];
        let gpu = vec![0.0_f32, 1.0];
        assert_eq!(verdict_from_gpu_cpu_parity(&cpu, &gpu), Gq007Verdict::Fail);
    }
    #[test] fn gq008_delegates_to_gq007() {
        let cpu = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_gpu_cpu_parity_pow2(&cpu, &cpu), Gq008Verdict::Pass);
    }

    // ----- GQ-009 ----------------------------------------------------------

    #[test] fn gq009_pass_qwen2_5_7b_mapping() {
        // 28 heads, 4 kv → ratio 7. Each kv gets 7 q heads.
        let mut kv = Vec::with_capacity(28);
        for q in 0_u32..28 { kv.push(q * 4 / 28); }
        assert_eq!(verdict_from_head_mapping(28, 4, &kv), Gq009Verdict::Pass);
    }

    #[test] fn gq009_pass_pow2_mapping() {
        // 32 heads, 8 kv → ratio 4. Each kv gets 4 q heads.
        let mut kv = Vec::with_capacity(32);
        for q in 0_u32..32 { kv.push(q * 8 / 32); }
        assert_eq!(verdict_from_head_mapping(32, 8, &kv), Gq009Verdict::Pass);
    }

    #[test] fn gq009_fail_off_by_one_mapping() {
        let mut kv = Vec::with_capacity(28);
        for q in 0_u32..28 { kv.push(q * 4 / 28); }
        kv[15] += 1; // Bump one entry.
        assert_eq!(verdict_from_head_mapping(28, 4, &kv), Gq009Verdict::Fail);
    }

    #[test] fn gq009_fail_indivisible() {
        assert_eq!(verdict_from_head_mapping(7, 3, &[0, 0, 0, 1, 1, 2, 2]), Gq009Verdict::Fail);
    }

    // ----- Provenance pins ---------------------------------------------------

    #[test] fn provenance_max_ulp() { assert_eq!(AC_GQ_005_MAX_ULP, 8); }
    #[test] fn provenance_min_cosine() { assert!((AC_GQ_007_MIN_COSINE - 0.98).abs() < 1e-6); }
}
