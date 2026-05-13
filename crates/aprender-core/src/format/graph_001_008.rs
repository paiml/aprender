// SHIP-TWO-001 — `apr-model-graph-v1` algorithm-level PARTIAL discharge
// for FALSIFY-GRAPH-001..008.
//
// Contract: `contracts/apr-model-graph-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Eight model-graph invariants:
//
// - GRAPH-001 (forward shape preservation): down_proj must be [hidden, intermediate].
// - GRAPH-002 (attention softmax row sum 1): even all-zero QK^T → uniform distribution.
// - GRAPH-003 (FFN gate/up shape match): gate_proj.shape == up_proj.shape.
// - GRAPH-004 (KV cache immutability): can't overwrite already-written position.
// - GRAPH-005 (tensor name bijection): no two roles map to same tensor name.
// - GRAPH-006 (quantization preserves elements): n_elements unchanged.
// - GRAPH-007 (MoE top-k count exact): router selects exactly k experts.
// - GRAPH-008 (Q4_K round-trip error): max element error < 0.5.

/// GRAPH-008 — Q4_K round-trip max-error bound.
pub const AC_GRAPH_008_Q4K_MAX_ERROR: f32 = 0.5;

/// GRAPH-002 — softmax row-sum tolerance.
pub const AC_GRAPH_002_SOFTMAX_SUM_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: GRAPH-001 — forward shape preservation.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_forward_shape(
    down_proj_shape: (usize, usize),
    expected_hidden: usize,
    expected_intermediate: usize,
) -> GraphVerdict {
    if down_proj_shape == (expected_hidden, expected_intermediate) {
        GraphVerdict::Pass
    } else {
        GraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: GRAPH-002 — softmax row sums to 1.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_softmax_row_sum(row: &[f32]) -> GraphVerdict {
    if row.is_empty() {
        return GraphVerdict::Fail;
    }
    let mut sum = 0.0_f32;
    for &v in row {
        if !v.is_finite() || v < 0.0 {
            return GraphVerdict::Fail;
        }
        sum += v;
    }
    if (sum - 1.0).abs() < AC_GRAPH_002_SOFTMAX_SUM_EPS {
        GraphVerdict::Pass
    } else {
        GraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: GRAPH-003 — FFN gate/up shape match.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_ffn_gate_up_shape_match(
    gate_shape: (usize, usize),
    up_shape: (usize, usize),
) -> GraphVerdict {
    if gate_shape == up_shape {
        GraphVerdict::Pass
    } else {
        GraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: GRAPH-004 — KV cache immutability.
// -----------------------------------------------------------------------------

/// `position_already_written` is true iff the slot has been filled
/// previously. Pass iff overwrite is rejected (overwrite=false) when
/// position_already_written=true.
#[must_use]
pub fn verdict_from_kv_cache_immutability(
    position_already_written: bool,
    overwrite_was_accepted: bool,
) -> GraphVerdict {
    if position_already_written && overwrite_was_accepted {
        GraphVerdict::Fail
    } else {
        GraphVerdict::Pass
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: GRAPH-005 — tensor name bijection.
// -----------------------------------------------------------------------------

/// `role_to_name` is a list of (role, tensor_name) pairs. Pass iff
/// every name appears exactly once across all roles.
#[must_use]
pub fn verdict_from_tensor_name_bijection(
    role_to_name: &[(&str, &str)],
) -> GraphVerdict {
    let mut name_set = std::collections::HashSet::new();
    for (_, name) in role_to_name {
        if !name_set.insert(*name) {
            return GraphVerdict::Fail;
        }
    }
    GraphVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 6: GRAPH-006 — quantization preserves element count.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_quantization_element_count(
    pre_quant_elements: usize,
    post_quant_elements: usize,
) -> GraphVerdict {
    if pre_quant_elements == post_quant_elements {
        GraphVerdict::Pass
    } else {
        GraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 7: GRAPH-007 — MoE router top-k count.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_moe_topk_count(
    selected_per_token: &[usize],
    num_experts_per_tok: usize,
) -> GraphVerdict {
    if num_experts_per_tok == 0 {
        return GraphVerdict::Fail;
    }
    if selected_per_token.is_empty() {
        return GraphVerdict::Fail;
    }
    for &k in selected_per_token {
        if k != num_experts_per_tok {
            return GraphVerdict::Fail;
        }
    }
    GraphVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 8: GRAPH-008 — Q4_K round-trip max error.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_q4k_roundtrip_error(max_abs_error: f32) -> GraphVerdict {
    if !max_abs_error.is_finite() || max_abs_error < 0.0 {
        return GraphVerdict::Fail;
    }
    if max_abs_error < AC_GRAPH_008_Q4K_MAX_ERROR {
        GraphVerdict::Pass
    } else {
        GraphVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_q4k_max_error_05() {
        assert_eq!(AC_GRAPH_008_Q4K_MAX_ERROR, 0.5);
    }

    #[test]
    fn provenance_softmax_eps() {
        assert_eq!(AC_GRAPH_002_SOFTMAX_SUM_EPS, 1e-5);
    }

    // -------------------------------------------------------------------------
    // Section 2: GRAPH-001 — forward shape.
    // -------------------------------------------------------------------------
    #[test]
    fn graph001_pass_correct_shape() {
        // Qwen2.5 7B: hidden=4096, intermediate=11008.
        assert_eq!(
            verdict_from_forward_shape((4096, 11008), 4096, 11008),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph001_fail_hidden_off_by_one() {
        // The contract failure: down_proj [H+1, I].
        assert_eq!(
            verdict_from_forward_shape((4097, 11008), 4096, 11008),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph001_fail_intermediate_wrong() {
        assert_eq!(
            verdict_from_forward_shape((4096, 1024), 4096, 11008),
            GraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: GRAPH-002 — softmax row sum.
    // -------------------------------------------------------------------------
    #[test]
    fn graph002_pass_uniform_4_classes() {
        let row = vec![0.25_f32; 4];
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Pass);
    }

    #[test]
    fn graph002_pass_skewed() {
        let row = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Pass);
    }

    #[test]
    fn graph002_fail_negative_entry() {
        let row = vec![1.5_f32, -0.5];
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Fail);
    }

    #[test]
    fn graph002_fail_does_not_sum_to_one() {
        let row = vec![0.5_f32, 0.3, 0.1]; // sum 0.9
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Fail);
    }

    #[test]
    fn graph002_fail_nan() {
        let row = vec![0.5_f32, f32::NAN];
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Fail);
    }

    #[test]
    fn graph002_fail_empty() {
        let row: Vec<f32> = vec![];
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: GRAPH-003 — FFN gate/up shape match.
    // -------------------------------------------------------------------------
    #[test]
    fn graph003_pass_match() {
        assert_eq!(
            verdict_from_ffn_gate_up_shape_match((11008, 4096), (11008, 4096)),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph003_fail_dim_swap() {
        // gate is [I, H], up is [H, I] — bug.
        assert_eq!(
            verdict_from_ffn_gate_up_shape_match((11008, 4096), (4096, 11008)),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph003_fail_size_mismatch() {
        assert_eq!(
            verdict_from_ffn_gate_up_shape_match((11008, 4096), (11008, 4097)),
            GraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: GRAPH-004 — KV cache immutability.
    // -------------------------------------------------------------------------
    #[test]
    fn graph004_pass_first_write_to_unwritten() {
        assert_eq!(
            verdict_from_kv_cache_immutability(false, true),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph004_pass_overwrite_rejected() {
        assert_eq!(
            verdict_from_kv_cache_immutability(true, false),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph004_fail_overwrite_accepted() {
        // The exact regression: position 0 overwritten.
        assert_eq!(
            verdict_from_kv_cache_immutability(true, true),
            GraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: GRAPH-005 — tensor name bijection.
    // -------------------------------------------------------------------------
    #[test]
    fn graph005_pass_unique_names() {
        let map = vec![
            ("attn.q", "blk.0.attn_q.weight"),
            ("attn.k", "blk.0.attn_k.weight"),
            ("attn.v", "blk.0.attn_v.weight"),
        ];
        assert_eq!(
            verdict_from_tensor_name_bijection(&map),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph005_pass_empty() {
        let map: Vec<(&str, &str)> = vec![];
        assert_eq!(
            verdict_from_tensor_name_bijection(&map),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph005_fail_duplicate_name() {
        // Two roles point to same tensor.
        let map = vec![
            ("attn.q", "shared.weight"),
            ("attn.k", "shared.weight"),
        ];
        assert_eq!(
            verdict_from_tensor_name_bijection(&map),
            GraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: GRAPH-006 — quantization element count.
    // -------------------------------------------------------------------------
    #[test]
    fn graph006_pass_4096x4096() {
        let n = 4096_usize * 4096;
        assert_eq!(
            verdict_from_quantization_element_count(n, n),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph006_fail_post_count_smaller() {
        // Quantization dropped elements (block padding bug).
        let pre = 4096_usize * 4096;
        let post = pre - 32; // dropped one block
        assert_eq!(
            verdict_from_quantization_element_count(pre, post),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph006_fail_post_count_larger() {
        let pre = 1024_usize;
        let post = 1056;
        assert_eq!(
            verdict_from_quantization_element_count(pre, post),
            GraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: GRAPH-007 — MoE top-k count.
    // -------------------------------------------------------------------------
    #[test]
    fn graph007_pass_all_tokens_select_2() {
        let selected = vec![2_usize; 100];
        assert_eq!(
            verdict_from_moe_topk_count(&selected, 2),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph007_pass_all_tokens_select_8() {
        let selected = vec![8_usize; 50];
        assert_eq!(
            verdict_from_moe_topk_count(&selected, 8),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph007_fail_one_token_too_few() {
        let mut selected = vec![2_usize; 100];
        selected[42] = 1;
        assert_eq!(
            verdict_from_moe_topk_count(&selected, 2),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph007_fail_one_token_too_many() {
        let mut selected = vec![2_usize; 100];
        selected[42] = 3;
        assert_eq!(
            verdict_from_moe_topk_count(&selected, 2),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph007_fail_zero_k() {
        assert_eq!(
            verdict_from_moe_topk_count(&[0_usize; 5], 0),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph007_fail_empty() {
        assert_eq!(verdict_from_moe_topk_count(&[], 2), GraphVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 9: GRAPH-008 — Q4_K round-trip error.
    // -------------------------------------------------------------------------
    #[test]
    fn graph008_pass_typical_error() {
        // Q4_K round-trip on N(0,1) typically gives error ~0.05.
        assert_eq!(
            verdict_from_q4k_roundtrip_error(0.05),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph008_pass_just_below_bound() {
        assert_eq!(
            verdict_from_q4k_roundtrip_error(0.499),
            GraphVerdict::Pass
        );
    }

    #[test]
    fn graph008_fail_at_bound() {
        // Strict <.
        assert_eq!(verdict_from_q4k_roundtrip_error(0.5), GraphVerdict::Fail);
    }

    #[test]
    fn graph008_fail_above_bound() {
        assert_eq!(verdict_from_q4k_roundtrip_error(1.0), GraphVerdict::Fail);
    }

    #[test]
    fn graph008_fail_negative() {
        assert_eq!(
            verdict_from_q4k_roundtrip_error(-0.1),
            GraphVerdict::Fail
        );
    }

    #[test]
    fn graph008_fail_nan() {
        assert_eq!(
            verdict_from_q4k_roundtrip_error(f32::NAN),
            GraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — full Qwen2.5-Coder-7B graph validation.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_qwen25_post_fix_full_pipeline() {
        // Synthesize a Qwen2.5-Coder-7B graph validation pass:
        let hidden = 4096_usize;
        let intermediate = 11008_usize;

        // GRAPH-001:
        assert_eq!(
            verdict_from_forward_shape((hidden, intermediate), hidden, intermediate),
            GraphVerdict::Pass
        );
        // GRAPH-002:
        let row = vec![0.25_f32; 4];
        assert_eq!(verdict_from_softmax_row_sum(&row), GraphVerdict::Pass);
        // GRAPH-003:
        assert_eq!(
            verdict_from_ffn_gate_up_shape_match(
                (intermediate, hidden),
                (intermediate, hidden)
            ),
            GraphVerdict::Pass
        );
        // GRAPH-004:
        assert_eq!(
            verdict_from_kv_cache_immutability(false, true),
            GraphVerdict::Pass
        );
        // GRAPH-005:
        let map = vec![
            ("attn.q", "blk.0.attn_q.weight"),
            ("attn.k", "blk.0.attn_k.weight"),
        ];
        assert_eq!(
            verdict_from_tensor_name_bijection(&map),
            GraphVerdict::Pass
        );
        // GRAPH-006:
        let n = hidden * intermediate;
        assert_eq!(
            verdict_from_quantization_element_count(n, n),
            GraphVerdict::Pass
        );
        // GRAPH-007 (Qwen2 dense, no MoE — synthetic top-k=1 trivial):
        assert_eq!(
            verdict_from_moe_topk_count(&[1_usize; 10], 1),
            GraphVerdict::Pass
        );
        // GRAPH-008:
        assert_eq!(
            verdict_from_q4k_roundtrip_error(0.045),
            GraphVerdict::Pass
        );
    }
}
