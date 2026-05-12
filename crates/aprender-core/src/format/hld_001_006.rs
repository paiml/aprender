// SHIP-TWO-001 — `hybrid-layer-dispatch-v1` algorithm-level PARTIAL
// discharge for FALSIFY-HL-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/hybrid-layer-dispatch-v1.yaml`.
// Spec: Qwen3.5 hybrid attention layer dispatch and linear attention
// invariants (head grouping, causal conv1d, SIMD parity).

// ===========================================================================
// HL-001 — Exhaustive partition: every layer index has exactly one type
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlLayerType { Attention, Linear }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hl001Verdict { Pass, Fail }

/// Pass iff `len(layer_types) == num_hidden_layers`. The enum makes
/// "duplicate or missing assignment" structurally impossible at the
/// algorithm level — the only failure modes are wrong length or zero L.
#[must_use]
pub fn verdict_from_partition(layer_types: &[HlLayerType], num_hidden_layers: u64) -> Hl001Verdict {
    if num_hidden_layers == 0 { return Hl001Verdict::Fail; }
    if layer_types.len() as u64 != num_hidden_layers { return Hl001Verdict::Fail; }
    Hl001Verdict::Pass
}

// ===========================================================================
// HL-002 — Matrix associativity: (A @ B) @ C == A @ (B @ C) within tolerance
// ===========================================================================

pub const AC_HL_002_TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hl002Verdict { Pass, Fail }

/// Pass iff every element of the two computed groupings agrees within
/// `AC_HL_002_TOLERANCE`. Caller computes both forms; verdict checks
/// element-wise drift.
#[must_use]
pub fn verdict_from_associativity(left_grouping: &[f32], right_grouping: &[f32]) -> Hl002Verdict {
    if left_grouping.is_empty() || right_grouping.is_empty() { return Hl002Verdict::Fail; }
    if left_grouping.len() != right_grouping.len() { return Hl002Verdict::Fail; }
    for (&a, &b) in left_grouping.iter().zip(right_grouping.iter()) {
        if !a.is_finite() || !b.is_finite() { return Hl002Verdict::Fail; }
        if (a - b).abs() > AC_HL_002_TOLERANCE { return Hl002Verdict::Fail; }
    }
    Hl002Verdict::Pass
}

// ===========================================================================
// HL-003 — Head grouping exact: n_v % n_k == 0 AND n_v >= n_k
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hl003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_head_grouping(n_v: u64, n_k: u64) -> Hl003Verdict {
    if n_k == 0 || n_v == 0 { return Hl003Verdict::Fail; }
    if n_v < n_k { return Hl003Verdict::Fail; }
    if !n_v.is_multiple_of(n_k) { return Hl003Verdict::Fail; }
    Hl003Verdict::Pass
}

// ===========================================================================
// HL-004 — Residual shape preservation: O_proj output dim == hidden_dim
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hl004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_o_proj_shape(o_out_dim: u64, hidden_dim: u64) -> Hl004Verdict {
    if hidden_dim == 0 || o_out_dim == 0 { return Hl004Verdict::Fail; }
    if o_out_dim == hidden_dim { Hl004Verdict::Pass } else { Hl004Verdict::Fail }
}

// ===========================================================================
// HL-005 — Causal conv1d: output_len == input_len with padding = k - 1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hl005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_conv1d_causal(
    input_len: u64,
    output_len: u64,
    kernel_size: u64,
    padding: u64,
) -> Hl005Verdict {
    if input_len == 0 || kernel_size == 0 { return Hl005Verdict::Fail; }
    // Causal conv1d requires padding == k - 1 AND output_len == input_len.
    if padding + 1 != kernel_size { return Hl005Verdict::Fail; }
    if output_len != input_len { return Hl005Verdict::Fail; }
    Hl005Verdict::Pass
}

// ===========================================================================
// HL-006 — SIMD linear attention equivalence (tolerance 0.0 → byte-equal)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hl006Verdict { Pass, Fail }

/// Per contract: `tolerance: 0.0` for SIMD linear attention equivalence.
/// Verdict requires byte-exact match (every element bit-identical).
#[must_use]
pub fn verdict_from_simd_parity(scalar: &[f32], simd: &[f32]) -> Hl006Verdict {
    if scalar.is_empty() || simd.is_empty() { return Hl006Verdict::Fail; }
    if scalar.len() != simd.len() { return Hl006Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if !s.is_finite() || !v.is_finite() { return Hl006Verdict::Fail; }
        if s.to_bits() != v.to_bits() { return Hl006Verdict::Fail; }
    }
    Hl006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // HL-001 (exhaustive partition)
    #[test] fn hl001_pass_full_attention() {
        let lt = vec![HlLayerType::Attention; 32];
        assert_eq!(verdict_from_partition(&lt, 32), Hl001Verdict::Pass);
    }
    #[test] fn hl001_pass_hybrid_alternating() {
        let mut lt = Vec::with_capacity(32);
        for i in 0..32 {
            lt.push(if i % 4 == 0 { HlLayerType::Attention } else { HlLayerType::Linear });
        }
        assert_eq!(verdict_from_partition(&lt, 32), Hl001Verdict::Pass);
    }
    #[test] fn hl001_fail_length_mismatch() {
        let lt = vec![HlLayerType::Linear; 30];
        assert_eq!(verdict_from_partition(&lt, 32), Hl001Verdict::Fail);
    }
    #[test] fn hl001_fail_zero_layers() {
        assert_eq!(verdict_from_partition(&[], 0), Hl001Verdict::Fail);
    }

    // HL-002 (matrix associativity)
    #[test] fn hl002_pass_identical_groupings() {
        // Two precomputed groupings that agree to machine precision.
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_associativity(&a, &b), Hl002Verdict::Pass);
    }
    #[test] fn hl002_pass_within_tolerance() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32 + 1e-5, 2.0 - 1e-5, 3.0]; // well within 1e-3
        assert_eq!(verdict_from_associativity(&a, &b), Hl002Verdict::Pass);
    }
    #[test] fn hl002_fail_above_tolerance() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0 + 0.5, 3.0]; // 0.5 > 1e-3
        assert_eq!(verdict_from_associativity(&a, &b), Hl002Verdict::Fail);
    }
    #[test] fn hl002_fail_length_mismatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_associativity(&a, &b), Hl002Verdict::Fail);
    }
    #[test] fn hl002_fail_nan() {
        let a = vec![1.0_f32, f32::NAN];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_associativity(&a, &b), Hl002Verdict::Fail);
    }

    // HL-003 (head grouping)
    #[test] fn hl003_pass_qwen35_canonical() {
        // Qwen3.5: n_v=4, n_k=4 (1:1 GQA-like) is valid.
        assert_eq!(verdict_from_head_grouping(4, 4), Hl003Verdict::Pass);
    }
    #[test] fn hl003_pass_2to1() {
        assert_eq!(verdict_from_head_grouping(8, 4), Hl003Verdict::Pass);
    }
    #[test] fn hl003_pass_8to1() {
        // Qwen2 typical: 28 query heads to 4 KV heads (7:1) — but this
        // contract is HL-003, n_v >= n_k semantics (V vs K head counts,
        // not Q vs K). Demonstrate 32 V heads / 4 K heads.
        assert_eq!(verdict_from_head_grouping(32, 4), Hl003Verdict::Pass);
    }
    #[test] fn hl003_fail_indivisible() {
        // The contract's stated falsifier: "Set n_v not divisible by n_k".
        assert_eq!(verdict_from_head_grouping(6, 4), Hl003Verdict::Fail);
    }
    #[test] fn hl003_fail_n_v_below_n_k() {
        assert_eq!(verdict_from_head_grouping(2, 4), Hl003Verdict::Fail);
    }
    #[test] fn hl003_fail_zero() {
        assert_eq!(verdict_from_head_grouping(0, 4), Hl003Verdict::Fail);
        assert_eq!(verdict_from_head_grouping(4, 0), Hl003Verdict::Fail);
    }

    // HL-004 (O_proj shape)
    #[test] fn hl004_pass_match() {
        assert_eq!(verdict_from_o_proj_shape(4096, 4096), Hl004Verdict::Pass);
    }
    #[test] fn hl004_fail_drift() {
        assert_eq!(verdict_from_o_proj_shape(3584, 4096), Hl004Verdict::Fail);
    }
    #[test] fn hl004_fail_zero() {
        assert_eq!(verdict_from_o_proj_shape(0, 4096), Hl004Verdict::Fail);
    }

    // HL-005 (conv1d causal)
    #[test] fn hl005_pass_kernel_4() {
        // padding = k - 1 = 3, output_len = input_len.
        assert_eq!(verdict_from_conv1d_causal(128, 128, 4, 3), Hl005Verdict::Pass);
    }
    #[test] fn hl005_pass_kernel_1() {
        // padding = 0, output_len = input_len.
        assert_eq!(verdict_from_conv1d_causal(64, 64, 1, 0), Hl005Verdict::Pass);
    }
    #[test] fn hl005_fail_wrong_padding() {
        // k=4 but padding=2 → not causal (would lose 1 element).
        assert_eq!(verdict_from_conv1d_causal(128, 128, 4, 2), Hl005Verdict::Fail);
    }
    #[test] fn hl005_fail_length_drift() {
        // Padding correct but output dropped one element.
        assert_eq!(verdict_from_conv1d_causal(128, 127, 4, 3), Hl005Verdict::Fail);
    }
    #[test] fn hl005_fail_zero_input() {
        assert_eq!(verdict_from_conv1d_causal(0, 0, 4, 3), Hl005Verdict::Fail);
    }

    // HL-006 (SIMD parity, byte-exact per contract tolerance=0.0)
    #[test] fn hl006_pass_identical() {
        let a = vec![1.0_f32; 64];
        assert_eq!(verdict_from_simd_parity(&a, &a), Hl006Verdict::Pass);
    }
    #[test] fn hl006_fail_one_ulp_off() {
        // Contract says tolerance=0.0 — even 1 ULP fails.
        let a = vec![1.0_f32];
        let b = vec![f32::from_bits(1.0_f32.to_bits() + 1)];
        assert_eq!(verdict_from_simd_parity(&a, &b), Hl006Verdict::Fail);
    }
    #[test] fn hl006_fail_length_mismatch() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_parity(&a, &b), Hl006Verdict::Fail);
    }
    #[test] fn hl006_fail_nan() {
        let a = vec![f32::NAN];
        let b = vec![f32::NAN];
        // NaN != NaN at the contract level; also caller may have produced
        // NaN in only one path. Verdict Fails on any non-finite.
        assert_eq!(verdict_from_simd_parity(&a, &b), Hl006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_HL_002_TOLERANCE - 1.0e-3).abs() < 1e-9);
    }
}
