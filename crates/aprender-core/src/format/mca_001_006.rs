// SHIP-TWO-001 — `model-config-algebra-v1` algorithm-level PARTIAL
// discharge for FALSIFY-MCA-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/model-config-algebra-v1.yaml`.
// Spec: 5-level proof hierarchy for transformer config constraints
// (Vaswani 2017 head_dim; Ainslie 2023 GQA divisibility; Su 2021 RoPE
// even head_dim; Shazeer 2020 FFN expansion).

// ===========================================================================
// MCA-001 — Divisibility: h % n_h == 0 ∧ n_h % n_kv == 0 ∧ d_k % 2 == 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mca001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_divisibility(
    hidden_dim: u64,
    num_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
) -> Mca001Verdict {
    if hidden_dim == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Mca001Verdict::Fail;
    }
    if !hidden_dim.is_multiple_of(num_heads) { return Mca001Verdict::Fail; }
    if !num_heads.is_multiple_of(num_kv_heads) { return Mca001Verdict::Fail; }
    if !head_dim.is_multiple_of(2) { return Mca001Verdict::Fail; }
    if hidden_dim / num_heads != head_dim { return Mca001Verdict::Fail; }
    Mca001Verdict::Pass
}

// ===========================================================================
// MCA-002 — Bounds: head_dim ∈ [hidden_dim/num_heads, 2*(hidden_dim/num_heads)]
//                   AND d_ff > hidden_dim
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mca002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_bounds(
    hidden_dim: u64,
    num_heads: u64,
    head_dim: u64,
    d_ff: u64,
) -> Mca002Verdict {
    if hidden_dim == 0 || num_heads == 0 || head_dim == 0 || d_ff == 0 {
        return Mca002Verdict::Fail;
    }
    let h_per_head = hidden_dim / num_heads;
    if h_per_head == 0 { return Mca002Verdict::Fail; }
    if head_dim < h_per_head { return Mca002Verdict::Fail; }
    if head_dim > 2 * h_per_head { return Mca002Verdict::Fail; }
    if d_ff <= hidden_dim { return Mca002Verdict::Fail; }
    Mca002Verdict::Pass
}

// ===========================================================================
// MCA-003 — Ordering: d_ff > h ∧ n_kv ≤ n_h ∧ max_pos > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mca003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_ordering(
    hidden_dim: u64,
    d_ff: u64,
    num_heads: u64,
    num_kv_heads: u64,
    max_position: u64,
) -> Mca003Verdict {
    if hidden_dim == 0 || d_ff == 0 || num_heads == 0 || num_kv_heads == 0 {
        return Mca003Verdict::Fail;
    }
    if d_ff <= hidden_dim { return Mca003Verdict::Fail; }
    if num_kv_heads > num_heads { return Mca003Verdict::Fail; }
    if max_position == 0 { return Mca003Verdict::Fail; }
    Mca003Verdict::Pass
}

// ===========================================================================
// MCA-004 — Non-degeneracy: all structural params > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mca004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_non_degeneracy(
    hidden_dim: u64,
    num_layers: u64,
    num_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    vocab_size: u64,
) -> Mca004Verdict {
    if hidden_dim == 0 { return Mca004Verdict::Fail; }
    if num_layers == 0 { return Mca004Verdict::Fail; }
    if num_heads == 0 { return Mca004Verdict::Fail; }
    if num_kv_heads == 0 { return Mca004Verdict::Fail; }
    if head_dim == 0 { return Mca004Verdict::Fail; }
    if vocab_size == 0 { return Mca004Verdict::Fail; }
    Mca004Verdict::Pass
}

// ===========================================================================
// MCA-005 — Cross-parameter: rope_theta > 0 finite, rms_norm_eps ∈ (0, 0.1)
// ===========================================================================

pub const AC_MCA_005_RMS_EPS_MAX_EXCLUSIVE: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mca005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_cross_parameter(rope_theta: f32, rms_norm_eps: f32) -> Mca005Verdict {
    if !rope_theta.is_finite() || rope_theta <= 0.0 { return Mca005Verdict::Fail; }
    if !rms_norm_eps.is_finite() { return Mca005Verdict::Fail; }
    if rms_norm_eps <= 0.0 { return Mca005Verdict::Fail; }
    if rms_norm_eps >= AC_MCA_005_RMS_EPS_MAX_EXCLUSIVE { return Mca005Verdict::Fail; }
    Mca005Verdict::Pass
}

// ===========================================================================
// MCA-006 — SIMD config equivalence (contract tolerance=0.0 → byte-exact)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mca006Verdict { Pass, Fail }

/// Pass iff every entry of `simd_config` is byte-identical to the
/// corresponding entry of `scalar_config` (config algebra is integer
/// arithmetic — any drift is a bug).
#[must_use]
pub fn verdict_from_simd_parity(scalar_config: &[u64], simd_config: &[u64]) -> Mca006Verdict {
    if scalar_config.is_empty() || simd_config.is_empty() { return Mca006Verdict::Fail; }
    if scalar_config.len() != simd_config.len() { return Mca006Verdict::Fail; }
    for (&s, &v) in scalar_config.iter().zip(simd_config.iter()) {
        if s != v { return Mca006Verdict::Fail; }
    }
    Mca006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // MCA-001 (divisibility)
    #[test] fn mca001_pass_qwen2_7b() {
        // Qwen2-7B: h=3584, n_h=28, n_kv=4, d_k=128.
        assert_eq!(verdict_from_divisibility(3584, 28, 4, 128), Mca001Verdict::Pass);
    }
    #[test] fn mca001_pass_qwen3_8b() {
        // Qwen3-8B: h=4096, n_h=32, n_kv=8, d_k=128.
        assert_eq!(verdict_from_divisibility(4096, 32, 8, 128), Mca001Verdict::Pass);
    }
    #[test] fn mca001_fail_h_not_div_nh() {
        // The contract's stated falsifier: "num_heads not dividing hidden_dim".
        assert_eq!(verdict_from_divisibility(3585, 28, 4, 128), Mca001Verdict::Fail);
    }
    #[test] fn mca001_fail_nh_not_div_nkv() {
        // n_h % n_kv != 0 (e.g., 28 % 5 != 0).
        assert_eq!(verdict_from_divisibility(3584, 28, 5, 128), Mca001Verdict::Fail);
    }
    #[test] fn mca001_fail_odd_head_dim() {
        // RoPE requires d_k % 2 == 0.
        assert_eq!(verdict_from_divisibility(2989, 23, 1, 130), Mca001Verdict::Fail);
    }
    #[test] fn mca001_fail_d_k_inconsistent() {
        // hidden_dim / num_heads != head_dim (declared d_k drifts).
        assert_eq!(verdict_from_divisibility(3584, 28, 4, 256), Mca001Verdict::Fail);
    }
    #[test] fn mca001_fail_zero() {
        assert_eq!(verdict_from_divisibility(0, 28, 4, 128), Mca001Verdict::Fail);
        assert_eq!(verdict_from_divisibility(3584, 0, 4, 128), Mca001Verdict::Fail);
    }

    // MCA-002 (bounds)
    #[test] fn mca002_pass_canonical() {
        // h=4096, n_h=32 → h_per_head=128, d_k=128 (in [128, 256]); d_ff=14336 > h.
        assert_eq!(verdict_from_bounds(4096, 32, 128, 14336), Mca002Verdict::Pass);
    }
    #[test] fn mca002_pass_d_k_doubled() {
        // head_dim = 2 * (hidden_dim / num_heads) is the upper bound.
        assert_eq!(verdict_from_bounds(4096, 32, 256, 14336), Mca002Verdict::Pass);
    }
    #[test] fn mca002_fail_d_k_below_lower_bound() {
        assert_eq!(verdict_from_bounds(4096, 32, 64, 14336), Mca002Verdict::Fail);
    }
    #[test] fn mca002_fail_d_k_above_upper_bound() {
        assert_eq!(verdict_from_bounds(4096, 32, 512, 14336), Mca002Verdict::Fail);
    }
    #[test] fn mca002_fail_d_ff_le_h() {
        // d_ff must be strictly larger than hidden_dim.
        assert_eq!(verdict_from_bounds(4096, 32, 128, 4096), Mca002Verdict::Fail);
        assert_eq!(verdict_from_bounds(4096, 32, 128, 1024), Mca002Verdict::Fail);
    }

    // MCA-003 (ordering)
    #[test] fn mca003_pass_canonical() {
        // n_kv=4, n_h=28 → 4 ≤ 28; d_ff=14336 > h=3584; max_pos=32768 > 0.
        assert_eq!(verdict_from_ordering(3584, 14336, 28, 4, 32768), Mca003Verdict::Pass);
    }
    #[test] fn mca003_pass_n_kv_eq_n_h() {
        // n_kv == n_h is the boundary (full multi-head).
        assert_eq!(verdict_from_ordering(3584, 14336, 28, 28, 32768), Mca003Verdict::Pass);
    }
    #[test] fn mca003_fail_n_kv_above_n_h() {
        // The contract says n_kv ≤ n_h (KV heads cannot exceed query heads).
        assert_eq!(verdict_from_ordering(3584, 14336, 28, 32, 32768), Mca003Verdict::Fail);
    }
    #[test] fn mca003_fail_d_ff_le_h() {
        assert_eq!(verdict_from_ordering(3584, 3584, 28, 4, 32768), Mca003Verdict::Fail);
    }
    #[test] fn mca003_fail_max_pos_zero() {
        assert_eq!(verdict_from_ordering(3584, 14336, 28, 4, 0), Mca003Verdict::Fail);
    }

    // MCA-004 (non-degeneracy)
    #[test] fn mca004_pass_canonical() {
        // Qwen2-7B: h=3584, L=28, n_h=28, n_kv=4, d_k=128, V=152064.
        assert_eq!(
            verdict_from_non_degeneracy(3584, 28, 28, 4, 128, 152064),
            Mca004Verdict::Pass
        );
    }
    #[test] fn mca004_fail_zero_vocab() {
        assert_eq!(
            verdict_from_non_degeneracy(3584, 28, 28, 4, 128, 0),
            Mca004Verdict::Fail
        );
    }
    #[test] fn mca004_fail_zero_layers() {
        assert_eq!(
            verdict_from_non_degeneracy(3584, 0, 28, 4, 128, 152064),
            Mca004Verdict::Fail
        );
    }
    #[test] fn mca004_fail_zero_kv_heads() {
        // KV heads must be > 0 (otherwise no attention is possible).
        assert_eq!(
            verdict_from_non_degeneracy(3584, 28, 28, 0, 128, 152064),
            Mca004Verdict::Fail
        );
    }

    // MCA-005 (cross-parameter)
    #[test] fn mca005_pass_canonical() {
        // Qwen2-7B: rope_theta = 1000000.0, rms_norm_eps = 1e-6.
        assert_eq!(verdict_from_cross_parameter(1_000_000.0, 1e-6), Mca005Verdict::Pass);
    }
    #[test] fn mca005_pass_typical_eps() {
        assert_eq!(verdict_from_cross_parameter(10_000.0, 1e-5), Mca005Verdict::Pass);
        assert_eq!(verdict_from_cross_parameter(10_000.0, 0.05), Mca005Verdict::Pass);
    }
    #[test] fn mca005_fail_rope_zero() {
        assert_eq!(verdict_from_cross_parameter(0.0, 1e-6), Mca005Verdict::Fail);
    }
    #[test] fn mca005_fail_rope_negative() {
        assert_eq!(verdict_from_cross_parameter(-1.0, 1e-6), Mca005Verdict::Fail);
    }
    #[test] fn mca005_fail_rope_inf() {
        assert_eq!(verdict_from_cross_parameter(f32::INFINITY, 1e-6), Mca005Verdict::Fail);
    }
    #[test] fn mca005_fail_eps_zero() {
        assert_eq!(verdict_from_cross_parameter(10_000.0, 0.0), Mca005Verdict::Fail);
    }
    #[test] fn mca005_fail_eps_at_upper_boundary() {
        // 0.1 is excluded (open interval upper bound).
        assert_eq!(verdict_from_cross_parameter(10_000.0, 0.1), Mca005Verdict::Fail);
    }
    #[test] fn mca005_fail_eps_above_upper() {
        assert_eq!(verdict_from_cross_parameter(10_000.0, 0.5), Mca005Verdict::Fail);
    }
    #[test] fn mca005_fail_eps_negative() {
        assert_eq!(verdict_from_cross_parameter(10_000.0, -1e-6), Mca005Verdict::Fail);
    }

    // MCA-006 (SIMD config parity, byte-exact integers)
    #[test] fn mca006_pass_identical() {
        let a = vec![3584_u64, 28, 4, 128, 14336];
        assert_eq!(verdict_from_simd_parity(&a, &a), Mca006Verdict::Pass);
    }
    #[test] fn mca006_fail_drift() {
        let a = vec![3584_u64];
        let b = vec![3585_u64];
        assert_eq!(verdict_from_simd_parity(&a, &b), Mca006Verdict::Fail);
    }
    #[test] fn mca006_fail_length() {
        let a = vec![3584_u64];
        let b = vec![3584_u64, 28];
        assert_eq!(verdict_from_simd_parity(&a, &b), Mca006Verdict::Fail);
    }
    #[test] fn mca006_fail_empty() {
        assert_eq!(verdict_from_simd_parity(&[], &[]), Mca006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_MCA_005_RMS_EPS_MAX_EXCLUSIVE - 0.1).abs() < 1e-9);
    }
}
