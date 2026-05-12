// `apr-architecture-schema-v1` algorithm-level PARTIAL discharge for the
// 8 transformer-architecture-schema falsifiers (head divisibility, GQA
// group size, attn shapes, FFN transpose, norm count, embedding shape,
// rope_type, total tensor count tolerance).
//
// Contract: `contracts/apr-architecture-schema-v1.yaml`.
// Refs: Vaswani et al. (2017), Shazeer (2020) GLU Variants, Su et al.
// (2021) RoFormer, Ainslie et al. (2023) GQA.
//
// ## Disambiguation
//
// `architecture-requirements-v1.yaml` (task #260) is a sibling contract
// covering 12 different ARCH-* gates (build/install requirements, not
// transformer architecture). Despite both using the FALSIFY-ARCH-* gate
// prefix, the two contracts cover orthogonal invariants. Module suffix
// `archschema_` disambiguates from any `arch_` module bound to
// architecture-requirements-v1.

/// Tolerance bound on `abs(actual - expected_tensor_count)` per
/// FALSIFY-ARCH-008.
pub const AC_ARCHSCHEMA_TENSOR_COUNT_TOLERANCE: i32 = 5;

/// Valid RoPE type values per FALSIFY-ARCH-007 (spec CORRECTNESS-011).
pub const AC_ARCHSCHEMA_VALID_ROPE_TYPES: [u32; 2] = [0, 2];

// =============================================================================
// FALSIFY-ARCH-001 — head_dim divides hidden_size evenly
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadDimVerdict {
    /// hidden_size.is_multiple_of(num_heads).
    Pass,
    /// Hidden-size not divisible by num_heads — head_dim would be non-integer.
    Fail,
}

#[must_use]
pub fn verdict_from_head_dim(hidden_size: u32, num_heads: u32) -> HeadDimVerdict {
    if num_heads == 0 {
        return HeadDimVerdict::Fail;
    }
    if hidden_size.is_multiple_of(num_heads) {
        HeadDimVerdict::Pass
    } else {
        HeadDimVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-ARCH-002 — GQA group size divides num_heads evenly
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqaGroupVerdict {
    /// num_heads.is_multiple_of(num_kv_heads).
    Pass,
    /// num_heads not divisible by num_kv_heads — GQA group size non-integer.
    Fail,
}

#[must_use]
pub fn verdict_from_gqa_group(num_heads: u32, num_kv_heads: u32) -> GqaGroupVerdict {
    if num_kv_heads == 0 {
        return GqaGroupVerdict::Fail;
    }
    if num_heads.is_multiple_of(num_kv_heads) {
        GqaGroupVerdict::Pass
    } else {
        GqaGroupVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-ARCH-003 — attention Q/K/V/O shapes match config
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnShapeVerdict {
    /// Q=[h, n_h*d], K=V=[h, n_kv*d], O=[n_h*d, h] all match config.
    Pass,
    /// Any projection's shape mismatches.
    Fail,
}

#[must_use]
pub fn verdict_from_attn_shape(
    hidden_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    q_shape: (u32, u32),
    k_shape: (u32, u32),
    v_shape: (u32, u32),
    o_shape: (u32, u32),
) -> AttnShapeVerdict {
    let q_inner = num_heads * head_dim;
    let kv_inner = num_kv_heads * head_dim;
    if q_shape != (hidden_size, q_inner) {
        return AttnShapeVerdict::Fail;
    }
    if k_shape != (hidden_size, kv_inner) {
        return AttnShapeVerdict::Fail;
    }
    if v_shape != (hidden_size, kv_inner) {
        return AttnShapeVerdict::Fail;
    }
    if o_shape != (q_inner, hidden_size) {
        return AttnShapeVerdict::Fail;
    }
    AttnShapeVerdict::Pass
}

// =============================================================================
// FALSIFY-ARCH-004 — FFN gate/up/down transpose consistency
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnShapeVerdict {
    /// gate.shape == up.shape AND down.shape == transpose(gate.shape).
    Pass,
    /// Any inconsistency.
    Fail,
}

#[must_use]
pub fn verdict_from_ffn_shape(
    gate_shape: (u32, u32),
    up_shape: (u32, u32),
    down_shape: (u32, u32),
) -> FfnShapeVerdict {
    if gate_shape != up_shape {
        return FfnShapeVerdict::Fail;
    }
    let (a, b) = gate_shape;
    if down_shape != (b, a) {
        return FfnShapeVerdict::Fail;
    }
    FfnShapeVerdict::Pass
}

// =============================================================================
// FALSIFY-ARCH-005 — exactly 2 norm tensors per layer + 1 final norm
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormCountVerdict {
    /// total_norms == 2 * num_layers + 1 (attn_norm + ffn_norm per layer + final).
    Pass,
    /// Norm count off — missing or extra norm tensor.
    Fail,
}

#[must_use]
pub fn verdict_from_norm_count(num_layers: u32, total_norms: u32) -> NormCountVerdict {
    let expected = 2 * num_layers + 1;
    if total_norms == expected {
        NormCountVerdict::Pass
    } else {
        NormCountVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-ARCH-006 — embedding shape matches vocab/hidden
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingShapeVerdict {
    /// Embedding shape == [vocab_size, hidden_size].
    Pass,
    /// Off-by-one or other mismatch.
    Fail,
}

#[must_use]
pub fn verdict_from_embedding_shape(
    vocab_size: u32,
    hidden_size: u32,
    embedding_shape: (u32, u32),
) -> EmbeddingShapeVerdict {
    if embedding_shape == (vocab_size, hidden_size) {
        EmbeddingShapeVerdict::Pass
    } else {
        EmbeddingShapeVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-ARCH-007 — RoPE type ∈ {0, 2}
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeTypeVerdict {
    /// rope_type ∈ {0 = NORM, 2 = NEOX}.
    Pass,
    /// Unknown rope_type — silent default to 0 is the regression class.
    Fail,
}

#[must_use]
pub fn verdict_from_rope_type(rope_type: u32) -> RopeTypeVerdict {
    if AC_ARCHSCHEMA_VALID_ROPE_TYPES.contains(&rope_type) {
        RopeTypeVerdict::Pass
    } else {
        RopeTypeVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-ARCH-008 — total tensor count within ±5 tolerance
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCountVerdict {
    /// abs(actual - expected) <= 5.
    Pass,
    /// Out of tolerance — model has too many or too few tensors.
    Fail,
}

#[must_use]
pub fn verdict_from_tensor_count(actual: u32, expected: u32) -> TensorCountVerdict {
    let diff = (actual as i64 - expected as i64).abs();
    if diff <= AC_ARCHSCHEMA_TENSOR_COUNT_TOLERANCE as i64 {
        TensorCountVerdict::Pass
    } else {
        TensorCountVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_tensor_count_tolerance_5() {
        assert_eq!(AC_ARCHSCHEMA_TENSOR_COUNT_TOLERANCE, 5);
    }

    #[test]
    fn provenance_valid_rope_types_count_2() {
        assert_eq!(AC_ARCHSCHEMA_VALID_ROPE_TYPES.len(), 2);
        assert!(AC_ARCHSCHEMA_VALID_ROPE_TYPES.contains(&0));
        assert!(AC_ARCHSCHEMA_VALID_ROPE_TYPES.contains(&2));
    }

    // -------------------------------------------------------------------------
    // Section 2: ARCH-001 head dim.
    // -------------------------------------------------------------------------
    #[test]
    fn fa001_pass_qwen2_7b() {
        // Qwen2.5-Coder-7B: hidden=3584, heads=28 → head_dim=128.
        assert_eq!(verdict_from_head_dim(3584, 28), HeadDimVerdict::Pass);
    }

    #[test]
    fn fa001_pass_llama_70b() {
        assert_eq!(verdict_from_head_dim(8192, 64), HeadDimVerdict::Pass);
    }

    #[test]
    fn fa001_fail_prime_hidden() {
        // 769 / 12 not integer.
        assert_eq!(verdict_from_head_dim(769, 12), HeadDimVerdict::Fail);
    }

    #[test]
    fn fa001_fail_zero_heads() {
        assert_eq!(verdict_from_head_dim(512, 0), HeadDimVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: ARCH-002 GQA group size.
    // -------------------------------------------------------------------------
    #[test]
    fn fa002_pass_qwen2_7b_4_kv() {
        // 28 heads, 4 kv heads → 7:1 GQA.
        assert_eq!(verdict_from_gqa_group(28, 4), GqaGroupVerdict::Pass);
    }

    #[test]
    fn fa002_pass_mha_no_gqa() {
        // num_heads == num_kv_heads (full MHA).
        assert_eq!(verdict_from_gqa_group(12, 12), GqaGroupVerdict::Pass);
    }

    #[test]
    fn fa002_fail_non_divisible() {
        assert_eq!(verdict_from_gqa_group(12, 5), GqaGroupVerdict::Fail);
    }

    #[test]
    fn fa002_fail_zero_kv_heads() {
        assert_eq!(verdict_from_gqa_group(12, 0), GqaGroupVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: ARCH-003 attention shape.
    // -------------------------------------------------------------------------
    #[test]
    fn fa003_pass_qwen2_5_coder_7b_layer_shapes() {
        // hidden=3584, n_h=28, n_kv=4, d=128.
        let q = (3584, 28 * 128);
        let k = (3584, 4 * 128);
        let v = (3584, 4 * 128);
        let o = (28 * 128, 3584);
        assert_eq!(
            verdict_from_attn_shape(3584, 28, 4, 128, q, k, v, o),
            AttnShapeVerdict::Pass
        );
    }

    #[test]
    fn fa003_fail_q_off_by_one() {
        let q = (3584, 28 * 128 + 1); // bad
        let k = (3584, 4 * 128);
        let v = (3584, 4 * 128);
        let o = (28 * 128, 3584);
        assert_eq!(
            verdict_from_attn_shape(3584, 28, 4, 128, q, k, v, o),
            AttnShapeVerdict::Fail
        );
    }

    #[test]
    fn fa003_fail_o_wrong_shape() {
        // Note: for Qwen2 hidden_size == num_heads * head_dim, so Q is
        // square and transposing Q gives the same shape. Use a different
        // wrong O shape to expose the regression.
        let q = (3584, 28 * 128);
        let k = (3584, 4 * 128);
        let v = (3584, 4 * 128);
        let o = (1, 1); // obviously wrong
        assert_eq!(
            verdict_from_attn_shape(3584, 28, 4, 128, q, k, v, o),
            AttnShapeVerdict::Fail
        );
    }

    #[test]
    fn fa003_fail_kv_uses_q_inner_not_kv() {
        let q = (3584, 28 * 128);
        let k = (3584, 28 * 128); // wrong: used q dim instead of kv
        let v = (3584, 4 * 128);
        let o = (28 * 128, 3584);
        assert_eq!(
            verdict_from_attn_shape(3584, 28, 4, 128, q, k, v, o),
            AttnShapeVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: ARCH-004 FFN transpose.
    // -------------------------------------------------------------------------
    #[test]
    fn fa004_pass_qwen2_7b_ffn() {
        // hidden=3584, intermediate=18944.
        let gate = (3584, 18944);
        let up = (3584, 18944);
        let down = (18944, 3584);
        assert_eq!(verdict_from_ffn_shape(gate, up, down), FfnShapeVerdict::Pass);
    }

    #[test]
    fn fa004_fail_gate_up_mismatch() {
        let gate = (3584, 18944);
        let up = (3584, 18943); // off by one
        let down = (18944, 3584);
        assert_eq!(verdict_from_ffn_shape(gate, up, down), FfnShapeVerdict::Fail);
    }

    #[test]
    fn fa004_fail_down_not_transposed() {
        let gate = (3584, 18944);
        let up = (3584, 18944);
        let down = (3584, 18944); // not transposed
        assert_eq!(verdict_from_ffn_shape(gate, up, down), FfnShapeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: ARCH-005 norm count.
    // -------------------------------------------------------------------------
    #[test]
    fn fa005_pass_qwen2_7b_28_layers() {
        // 28 layers * 2 norms + 1 final = 57.
        assert_eq!(verdict_from_norm_count(28, 57), NormCountVerdict::Pass);
    }

    #[test]
    fn fa005_pass_minimal_1_layer() {
        // 1 layer * 2 + 1 final = 3.
        assert_eq!(verdict_from_norm_count(1, 3), NormCountVerdict::Pass);
    }

    #[test]
    fn fa005_fail_missing_one_norm() {
        // 28 layers expects 57; 56 = missing one ffn_norm.
        assert_eq!(verdict_from_norm_count(28, 56), NormCountVerdict::Fail);
    }

    #[test]
    fn fa005_fail_extra_norm() {
        assert_eq!(verdict_from_norm_count(28, 58), NormCountVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: ARCH-006 embedding shape.
    // -------------------------------------------------------------------------
    #[test]
    fn fa006_pass_qwen2_7b_embedding() {
        let e = (152064, 3584);
        assert_eq!(verdict_from_embedding_shape(152064, 3584, e), EmbeddingShapeVerdict::Pass);
    }

    #[test]
    fn fa006_fail_off_by_one_vocab() {
        let e = (152063, 3584);
        assert_eq!(verdict_from_embedding_shape(152064, 3584, e), EmbeddingShapeVerdict::Fail);
    }

    #[test]
    fn fa006_fail_swapped_axes() {
        let e = (3584, 152064);
        assert_eq!(verdict_from_embedding_shape(152064, 3584, e), EmbeddingShapeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: ARCH-007 rope type.
    // -------------------------------------------------------------------------
    #[test]
    fn fa007_pass_norm_type_0() {
        assert_eq!(verdict_from_rope_type(0), RopeTypeVerdict::Pass);
    }

    #[test]
    fn fa007_pass_neox_type_2() {
        assert_eq!(verdict_from_rope_type(2), RopeTypeVerdict::Pass);
    }

    #[test]
    fn fa007_fail_invalid_type_1() {
        assert_eq!(verdict_from_rope_type(1), RopeTypeVerdict::Fail);
    }

    #[test]
    fn fa007_fail_invalid_type_3() {
        assert_eq!(verdict_from_rope_type(3), RopeTypeVerdict::Fail);
    }

    #[test]
    fn fa007_fail_invalid_large() {
        assert_eq!(verdict_from_rope_type(99), RopeTypeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 9: ARCH-008 total tensor count.
    // -------------------------------------------------------------------------
    #[test]
    fn fa008_pass_exact_match() {
        assert_eq!(verdict_from_tensor_count(339, 339), TensorCountVerdict::Pass);
    }

    #[test]
    fn fa008_pass_within_tolerance_plus5() {
        assert_eq!(verdict_from_tensor_count(344, 339), TensorCountVerdict::Pass);
    }

    #[test]
    fn fa008_pass_within_tolerance_minus5() {
        assert_eq!(verdict_from_tensor_count(334, 339), TensorCountVerdict::Pass);
    }

    #[test]
    fn fa008_fail_minus_10() {
        assert_eq!(verdict_from_tensor_count(329, 339), TensorCountVerdict::Fail);
    }

    #[test]
    fn fa008_fail_plus_50() {
        assert_eq!(verdict_from_tensor_count(389, 339), TensorCountVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — full Qwen2.5-Coder-7B passes all 8.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_qwen2_7b_passes_all_8() {
        // Qwen2.5-Coder-7B: hidden=3584, layers=28, heads=28, kv_heads=4,
        // head_dim=128, intermediate=18944, vocab=152064, rope_type=0, 339 tensors.
        assert_eq!(verdict_from_head_dim(3584, 28), HeadDimVerdict::Pass);
        assert_eq!(verdict_from_gqa_group(28, 4), GqaGroupVerdict::Pass);
        assert_eq!(
            verdict_from_attn_shape(3584, 28, 4, 128,
                (3584, 3584), (3584, 512), (3584, 512), (3584, 3584)),
            AttnShapeVerdict::Pass
        );
        assert_eq!(
            verdict_from_ffn_shape((3584, 18944), (3584, 18944), (18944, 3584)),
            FfnShapeVerdict::Pass
        );
        assert_eq!(verdict_from_norm_count(28, 57), NormCountVerdict::Pass);
        assert_eq!(
            verdict_from_embedding_shape(152064, 3584, (152064, 3584)),
            EmbeddingShapeVerdict::Pass
        );
        assert_eq!(verdict_from_rope_type(0), RopeTypeVerdict::Pass);
        assert_eq!(verdict_from_tensor_count(339, 339), TensorCountVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_8_failures() {
        // Each gate's regression class.
        assert_eq!(verdict_from_head_dim(769, 12), HeadDimVerdict::Fail);
        assert_eq!(verdict_from_gqa_group(12, 5), GqaGroupVerdict::Fail);
        assert_eq!(
            verdict_from_attn_shape(3584, 28, 4, 128,
                (3584, 3585), (3584, 512), (3584, 512), (3584, 3584)),
            AttnShapeVerdict::Fail
        );
        assert_eq!(
            verdict_from_ffn_shape((3584, 18944), (3584, 18944), (3584, 18944)),
            FfnShapeVerdict::Fail
        );
        assert_eq!(verdict_from_norm_count(28, 56), NormCountVerdict::Fail);
        assert_eq!(
            verdict_from_embedding_shape(152064, 3584, (152063, 3584)),
            EmbeddingShapeVerdict::Fail
        );
        assert_eq!(verdict_from_rope_type(3), RopeTypeVerdict::Fail);
        assert_eq!(verdict_from_tensor_count(329, 339), TensorCountVerdict::Fail);
    }
}
