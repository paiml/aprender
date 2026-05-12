// `attention-backward-v1` algorithm-level PARTIAL discharge for the 2
// FlashAttention-style backward-pass falsifiers (gradient correctness,
// causal mask preservation in backward).
//
// Contract: `contracts/attention-backward-v1.yaml`.
// Refs: Dao et al. (2022). FlashAttention: Fast and Memory-Efficient
// Exact Attention.
//
// ## Disambiguation
//
// `attention-kernel-v1.yaml` (task #308) covers FORWARD attention
// (5/5 ATT-* gates). This contract covers BACKWARD only. Module suffix
// `attnbwd_` disambiguates from any `att_*` forward modules.

/// Tolerance for tiled-vs-naive backward gradient comparison
/// (FlashAttention reordering FMA introduces FP roundoff).
pub const AC_ATTNBWD_GRAD_TOLERANCE: f32 = 1.0e-3;

// =============================================================================
// FALSIFY-ATTENTION_BACKWARD_V1_001 — gradient correctness
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnBwdGradVerdict {
    /// max |tiled_dQ[i] - naive_dQ[i]| < ε across all heads.
    Pass,
    /// At least one element exceeds tolerance.
    Fail,
}

/// Pure verdict for FALSIFY-ATTENTION_BACKWARD_V1_001.
///
/// Inputs are (head, element)-flattened gradient tensors from the
/// tiled and naive backward implementations.
#[must_use]
pub fn verdict_from_attn_bwd_grad(tiled_dq: &[f32], naive_dq: &[f32]) -> AttnBwdGradVerdict {
    if tiled_dq.len() != naive_dq.len() {
        return AttnBwdGradVerdict::Fail;
    }
    if tiled_dq.is_empty() {
        return AttnBwdGradVerdict::Fail;
    }
    for (a, b) in tiled_dq.iter().zip(naive_dq.iter()) {
        if (a - b).abs() >= AC_ATTNBWD_GRAD_TOLERANCE {
            return AttnBwdGradVerdict::Fail;
        }
    }
    AttnBwdGradVerdict::Pass
}

// =============================================================================
// FALSIFY-ATTENTION_BACKWARD_V1_002 — causal mask preservation
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalMaskVerdict {
    /// ∀ i < j: attention_weight[i, j] == 0.0 (lower-triangular pass-through).
    Pass,
    /// At least one upper-triangular position has non-zero weight —
    /// causal mask leaked information across the time barrier.
    Fail,
}

/// Pure verdict for FALSIFY-ATTENTION_BACKWARD_V1_002.
///
/// `attention_weights` is a row-major (seq_len × seq_len) matrix.
#[must_use]
pub fn verdict_from_causal_mask(seq_len: usize, attention_weights: &[f32]) -> CausalMaskVerdict {
    if seq_len == 0 {
        return CausalMaskVerdict::Fail;
    }
    if attention_weights.len() != seq_len * seq_len {
        return CausalMaskVerdict::Fail;
    }
    for i in 0..seq_len {
        for j in 0..seq_len {
            if i < j {
                let w = attention_weights[i * seq_len + j];
                if w != 0.0 {
                    return CausalMaskVerdict::Fail;
                }
            }
        }
    }
    CausalMaskVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_grad_tolerance_1e_neg3() {
        assert!((AC_ATTNBWD_GRAD_TOLERANCE - 1.0e-3).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: ABV1-001 gradient correctness.
    // -------------------------------------------------------------------------
    #[test]
    fn fa001_pass_exact_match() {
        let dq = vec![1.0; 64];
        assert_eq!(
            verdict_from_attn_bwd_grad(&dq, &dq),
            AttnBwdGradVerdict::Pass
        );
    }

    #[test]
    fn fa001_pass_within_tolerance() {
        let tiled = vec![1.0001, 2.0001, 3.0001];
        let naive = vec![1.0, 2.0, 3.0];
        assert_eq!(
            verdict_from_attn_bwd_grad(&tiled, &naive),
            AttnBwdGradVerdict::Pass
        );
    }

    #[test]
    fn fa001_fail_above_tolerance() {
        let tiled = vec![1.5];
        let naive = vec![1.0];
        assert_eq!(
            verdict_from_attn_bwd_grad(&tiled, &naive),
            AttnBwdGradVerdict::Fail
        );
    }

    #[test]
    fn fa001_fail_at_threshold() {
        // Strict less-than: 1e-3 exactly fails.
        let tiled = vec![1.001];
        let naive = vec![1.0];
        assert_eq!(
            verdict_from_attn_bwd_grad(&tiled, &naive),
            AttnBwdGradVerdict::Fail
        );
    }

    #[test]
    fn fa001_fail_length_mismatch() {
        let tiled = vec![1.0, 2.0];
        let naive = vec![1.0];
        assert_eq!(
            verdict_from_attn_bwd_grad(&tiled, &naive),
            AttnBwdGradVerdict::Fail
        );
    }

    #[test]
    fn fa001_fail_empty() {
        assert_eq!(
            verdict_from_attn_bwd_grad(&[], &[]),
            AttnBwdGradVerdict::Fail
        );
    }

    #[test]
    fn fa001_pass_multi_head_4x16() {
        // 4 heads * 16 elements = 64 grad entries.
        let dq: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        assert_eq!(
            verdict_from_attn_bwd_grad(&dq, &dq),
            AttnBwdGradVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: ABV1-002 causal mask.
    // -------------------------------------------------------------------------
    #[test]
    fn fa002_pass_lower_triangular_only() {
        // 3x3 lower-triangular: rows 0,1,2 have non-zero only at j ≤ i.
        let w = vec![
            0.5, 0.0, 0.0, // i=0
            0.3, 0.7, 0.0, // i=1
            0.1, 0.4, 0.5, // i=2
        ];
        assert_eq!(verdict_from_causal_mask(3, &w), CausalMaskVerdict::Pass);
    }

    #[test]
    fn fa002_pass_diagonal_only() {
        let w = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        assert_eq!(verdict_from_causal_mask(3, &w), CausalMaskVerdict::Pass);
    }

    #[test]
    fn fa002_pass_seq_len_1() {
        let w = vec![1.0];
        assert_eq!(verdict_from_causal_mask(1, &w), CausalMaskVerdict::Pass);
    }

    #[test]
    fn fa002_fail_upper_triangular_leak() {
        // i=0, j=1 has weight 0.5 — future attended in past position.
        let w = vec![
            0.5, 0.5, 0.0,
            0.3, 0.7, 0.0,
            0.1, 0.4, 0.5,
        ];
        assert_eq!(verdict_from_causal_mask(3, &w), CausalMaskVerdict::Fail);
    }

    #[test]
    fn fa002_fail_corner_leak() {
        // i=0, j=2 (top-right corner).
        let w = vec![
            1.0, 0.0, 0.0001,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        assert_eq!(verdict_from_causal_mask(3, &w), CausalMaskVerdict::Fail);
    }

    #[test]
    fn fa002_fail_zero_seq_len() {
        assert_eq!(verdict_from_causal_mask(0, &[]), CausalMaskVerdict::Fail);
    }

    #[test]
    fn fa002_fail_size_mismatch() {
        let w = vec![1.0, 2.0]; // 2 elements but seq_len=3 → expect 9
        assert_eq!(verdict_from_causal_mask(3, &w), CausalMaskVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Realistic — full healthy backward pass.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_backward_passes_both() {
        // 4 heads × 8 elements gradient, tiled matches naive within 5e-4.
        let naive: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let tiled: Vec<f32> = naive.iter().map(|&x| x + 5e-4).collect();
        assert_eq!(
            verdict_from_attn_bwd_grad(&tiled, &naive),
            AttnBwdGradVerdict::Pass
        );

        // 4x4 strict lower-triangular attention weights.
        let w = vec![
            0.2, 0.0, 0.0, 0.0,
            0.1, 0.3, 0.0, 0.0,
            0.05, 0.1, 0.4, 0.0,
            0.05, 0.1, 0.2, 0.5,
        ];
        assert_eq!(verdict_from_causal_mask(4, &w), CausalMaskVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_both_failures() {
        // 001: tile-boundary FMA reordering bug — gradient diverges by 0.5.
        let naive = vec![1.0; 32];
        let mut tiled = naive.clone();
        tiled[16] += 0.5;
        assert_eq!(
            verdict_from_attn_bwd_grad(&tiled, &naive),
            AttnBwdGradVerdict::Fail
        );

        // 002: backward pass forgot to apply causal mask, all weights non-zero.
        let w = vec![
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
        ];
        assert_eq!(verdict_from_causal_mask(4, &w), CausalMaskVerdict::Fail);
    }
}
