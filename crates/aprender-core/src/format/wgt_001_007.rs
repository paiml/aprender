// SHIP-TWO-001 — `qwen2-weight-loading-v1` algorithm-level PARTIAL
// discharge for FALSIFY-WGT-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qwen2-weight-loading-v1.yaml`.

// ===========================================================================
// Canonical Qwen2.5-Coder-0.5B shape constants
// ===========================================================================

pub const AC_WGT_NUM_LAYERS: u64 = 24;
pub const AC_WGT_HIDDEN_DIM: u64 = 896;
pub const AC_WGT_VOCAB_SIZE: u64 = 151_936;
pub const AC_WGT_NUM_ATTN_HEADS: u64 = 14;
pub const AC_WGT_NUM_KV_HEADS: u64 = 2;
pub const AC_WGT_GQA_RATIO: u64 = 7;
pub const AC_WGT_HEAD_DIM: u64 = 64;
pub const AC_WGT_KV_DIM: u64 = AC_WGT_NUM_KV_HEADS * AC_WGT_HEAD_DIM;

// ===========================================================================
// WGT-001 — All N layers populated with non-zero weights
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt001Verdict { Pass, Fail }

/// `nonzero_per_layer[i] == true` iff layer `i` has at least one non-zero
/// weight in its critical projections (Q/K/V/O/FFN gate/up/down).
#[must_use]
pub fn verdict_from_all_layers_populated(nonzero_per_layer: &[bool]) -> Wgt001Verdict {
    if nonzero_per_layer.is_empty() { return Wgt001Verdict::Fail; }
    if nonzero_per_layer.len() != AC_WGT_NUM_LAYERS as usize { return Wgt001Verdict::Fail; }
    if nonzero_per_layer.iter().all(|x| *x) { Wgt001Verdict::Pass } else { Wgt001Verdict::Fail }
}

// ===========================================================================
// WGT-002 — No NaN / Inf in any loaded tensor
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_nan_inf(values: &[f32]) -> Wgt002Verdict {
    if values.is_empty() { return Wgt002Verdict::Fail; }
    if values.iter().all(|v| v.is_finite()) { Wgt002Verdict::Pass } else { Wgt002Verdict::Fail }
}

// ===========================================================================
// WGT-003 — Q projection shape == [hidden_dim, hidden_dim]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_q_proj_shape(observed: [u64; 2]) -> Wgt003Verdict {
    if observed == [AC_WGT_HIDDEN_DIM, AC_WGT_HIDDEN_DIM] { Wgt003Verdict::Pass } else { Wgt003Verdict::Fail }
}

// ===========================================================================
// WGT-004 — Embedding shape == [vocab, hidden]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_embedding_shape(rows: u64, cols: u64) -> Wgt004Verdict {
    if rows == AC_WGT_VOCAB_SIZE && cols == AC_WGT_HIDDEN_DIM {
        Wgt004Verdict::Pass
    } else {
        Wgt004Verdict::Fail
    }
}

// ===========================================================================
// WGT-005 — Layer count == 24
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_layer_count(observed: u64) -> Wgt005Verdict {
    if observed == AC_WGT_NUM_LAYERS { Wgt005Verdict::Pass } else { Wgt005Verdict::Fail }
}

// ===========================================================================
// WGT-006 — GQA ratio: num_attention_heads % num_kv_heads == 0 AND ratio == 7
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt006Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_gqa_ratio(num_heads: u64, num_kv_heads: u64) -> Wgt006Verdict {
    if num_kv_heads == 0 { return Wgt006Verdict::Fail; }
    if !num_heads.is_multiple_of(num_kv_heads) { return Wgt006Verdict::Fail; }
    if num_heads / num_kv_heads == AC_WGT_GQA_RATIO { Wgt006Verdict::Pass } else { Wgt006Verdict::Fail }
}

// ===========================================================================
// WGT-007 — KV projection rows < Q projection rows (GQA compression)
//                 AND k_rows == 128 AND q_rows == 896
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wgt007Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_kv_smaller_than_q(q_rows: u64, k_rows: u64) -> Wgt007Verdict {
    if k_rows >= q_rows { return Wgt007Verdict::Fail; }
    if k_rows == AC_WGT_KV_DIM && q_rows == AC_WGT_HIDDEN_DIM {
        Wgt007Verdict::Pass
    } else {
        Wgt007Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // WGT-001
    #[test] fn wgt001_pass_all_populated() {
        let layers = vec![true; 24];
        assert_eq!(verdict_from_all_layers_populated(&layers), Wgt001Verdict::Pass);
    }
    #[test] fn wgt001_fail_one_zero_layer() {
        let mut layers = vec![true; 24];
        layers[5] = false;
        assert_eq!(verdict_from_all_layers_populated(&layers), Wgt001Verdict::Fail);
    }
    #[test] fn wgt001_fail_wrong_count() {
        let layers = vec![true; 23];
        assert_eq!(verdict_from_all_layers_populated(&layers), Wgt001Verdict::Fail);
    }
    #[test] fn wgt001_fail_empty() {
        assert_eq!(verdict_from_all_layers_populated(&[]), Wgt001Verdict::Fail);
    }

    // WGT-002
    #[test] fn wgt002_pass_finite() {
        let v = vec![1.0_f32, -2.0, 0.001, 100.0];
        assert_eq!(verdict_from_no_nan_inf(&v), Wgt002Verdict::Pass);
    }
    #[test] fn wgt002_fail_nan() {
        let v = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_no_nan_inf(&v), Wgt002Verdict::Fail);
    }
    #[test] fn wgt002_fail_inf() {
        let v = vec![1.0_f32, f32::INFINITY];
        assert_eq!(verdict_from_no_nan_inf(&v), Wgt002Verdict::Fail);
    }
    #[test] fn wgt002_fail_neg_inf() {
        let v = vec![f32::NEG_INFINITY];
        assert_eq!(verdict_from_no_nan_inf(&v), Wgt002Verdict::Fail);
    }

    // WGT-003
    #[test] fn wgt003_pass() {
        assert_eq!(verdict_from_q_proj_shape([896, 896]), Wgt003Verdict::Pass);
    }
    #[test] fn wgt003_fail_swapped() {
        assert_eq!(verdict_from_q_proj_shape([128, 896]), Wgt003Verdict::Fail);
    }
    #[test] fn wgt003_fail_wrong_dim() {
        assert_eq!(verdict_from_q_proj_shape([512, 512]), Wgt003Verdict::Fail);
    }

    // WGT-004
    #[test] fn wgt004_pass_canonical() {
        assert_eq!(verdict_from_embedding_shape(151_936, 896), Wgt004Verdict::Pass);
    }
    #[test] fn wgt004_fail_truncated() {
        assert_eq!(verdict_from_embedding_shape(151_900, 896), Wgt004Verdict::Fail);
    }
    #[test] fn wgt004_fail_wrong_hidden() {
        assert_eq!(verdict_from_embedding_shape(151_936, 1024), Wgt004Verdict::Fail);
    }

    // WGT-005
    #[test] fn wgt005_pass() { assert_eq!(verdict_from_layer_count(24), Wgt005Verdict::Pass); }
    #[test] fn wgt005_fail_short() { assert_eq!(verdict_from_layer_count(23), Wgt005Verdict::Fail); }
    #[test] fn wgt005_fail_long() { assert_eq!(verdict_from_layer_count(28), Wgt005Verdict::Fail); }
    #[test] fn wgt005_fail_zero() { assert_eq!(verdict_from_layer_count(0), Wgt005Verdict::Fail); }

    // WGT-006
    #[test] fn wgt006_pass_canonical() {
        // 14 / 2 == 7.
        assert_eq!(verdict_from_gqa_ratio(14, 2), Wgt006Verdict::Pass);
    }
    #[test] fn wgt006_fail_indivisible() {
        assert_eq!(verdict_from_gqa_ratio(14, 3), Wgt006Verdict::Fail);
    }
    #[test] fn wgt006_fail_wrong_ratio() {
        // 8 / 2 = 4 (not 7).
        assert_eq!(verdict_from_gqa_ratio(8, 2), Wgt006Verdict::Fail);
    }
    #[test] fn wgt006_fail_zero_kv() {
        assert_eq!(verdict_from_gqa_ratio(14, 0), Wgt006Verdict::Fail);
    }

    // WGT-007
    #[test] fn wgt007_pass_canonical() {
        assert_eq!(verdict_from_kv_smaller_than_q(896, 128), Wgt007Verdict::Pass);
    }
    #[test] fn wgt007_fail_kv_eq_q() {
        assert_eq!(verdict_from_kv_smaller_than_q(896, 896), Wgt007Verdict::Fail);
    }
    #[test] fn wgt007_fail_kv_larger() {
        assert_eq!(verdict_from_kv_smaller_than_q(128, 896), Wgt007Verdict::Fail);
    }
    #[test] fn wgt007_fail_wrong_kv_value() {
        // KV smaller but not the canonical 128.
        assert_eq!(verdict_from_kv_smaller_than_q(896, 256), Wgt007Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_constants() {
        assert_eq!(AC_WGT_NUM_LAYERS, 24);
        assert_eq!(AC_WGT_HIDDEN_DIM, 896);
        assert_eq!(AC_WGT_VOCAB_SIZE, 151_936);
        assert_eq!(AC_WGT_NUM_ATTN_HEADS, 14);
        assert_eq!(AC_WGT_NUM_KV_HEADS, 2);
        assert_eq!(AC_WGT_GQA_RATIO, 7);
        assert_eq!(AC_WGT_HEAD_DIM, 64);
        assert_eq!(AC_WGT_KV_DIM, 128);
    }

    #[test] fn provenance_self_consistency() {
        assert_eq!(AC_WGT_NUM_ATTN_HEADS * AC_WGT_HEAD_DIM, AC_WGT_HIDDEN_DIM);
        assert_eq!(AC_WGT_NUM_KV_HEADS * AC_WGT_HEAD_DIM, AC_WGT_KV_DIM);
        assert_eq!(AC_WGT_NUM_ATTN_HEADS / AC_WGT_NUM_KV_HEADS, AC_WGT_GQA_RATIO);
    }
}
