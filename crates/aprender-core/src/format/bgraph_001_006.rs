// SHIP-TWO-001 — `cuda-graph-batched-inference-v1` algorithm-level
// PARTIAL discharge for FALSIFY-BGRAPH-001..006.
//
// Contract: `contracts/cuda-graph-batched-inference-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Six gates from per-batch CUDA-graph inference (Yu 2022 Orca + vLLM):
//
// - BGRAPH-001 (graph parity): active-slot output max-diff < 1e-5.
// - BGRAPH-002 (c=1 no-regression): post/pre throughput >= 0.98.
// - BGRAPH-003 (c=4 throughput): post/pre throughput >= 1.20.
// - BGRAPH-004 (memory bound): total graph VRAM <= 7.5 GB envelope.
// - BGRAPH-005 (efficiency target): realizar tok/s/GB >= 1.5x vLLM.
// - BGRAPH-006 (padding isolation): seq_lens[i] = 0 for padding slots
//   AND padding slot KV cache untouched.
//
// All six are pure properties of (active_diffs, throughputs, vram,
// padding mask). No GPU dispatch wired at this layer.

/// Active-slot output equivalence threshold (graph vs eager).
pub const AC_BGRAPH_001_OUTPUT_TOLERANCE: f32 = 1e-5;

/// Post/pre throughput ratio floor at c=1 (no-regression).
pub const AC_BGRAPH_002_NO_REGRESSION_RATIO: f32 = 0.98;

/// Post/pre throughput ratio floor at c=4 (improvement).
pub const AC_BGRAPH_003_C4_IMPROVEMENT_RATIO: f32 = 1.20;

/// Maximum total VRAM envelope (model weights + KV + 6 graphs + margin).
pub const AC_BGRAPH_004_VRAM_BOUND_GB: f32 = 7.5;

/// realizar/vLLM tok/s/GB efficiency multiple at c=4.
pub const AC_BGRAPH_005_EFFICIENCY_MULTIPLE: f32 = 1.50;

/// Bucket set: only these batch sizes get pre-captured graphs.
pub const BGRAPH_BUCKET_SET: [usize; 6] = [1, 2, 4, 8, 16, 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BgraphVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// Smallest member of `BGRAPH_BUCKET_SET` that is ≥ `actual_m`.
/// Returns `None` when `actual_m` exceeds the largest bucket.
#[must_use]
pub fn next_bucket(actual_m: usize) -> Option<usize> {
    if actual_m == 0 {
        return None;
    }
    BGRAPH_BUCKET_SET.iter().copied().find(|&b| b >= actual_m)
}

/// Maximum elementwise absolute difference between two slices.
#[must_use]
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut max = 0.0_f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if !ai.is_finite() || !bi.is_finite() {
            return None;
        }
        let d = (ai - bi).abs();
        if d > max {
            max = d;
        }
    }
    Some(max)
}

// -----------------------------------------------------------------------------
// Verdict 1: BGRAPH-001 — graph output parity.
// -----------------------------------------------------------------------------

/// Pass iff the maximum absolute difference between `output_graph` and
/// `output_eager` over **active slots only** is < `1e-5`.
///
/// `n_active * hidden_dim` = expected length; padding slots are not
/// included in the diff.
#[must_use]
pub fn verdict_from_graph_parity(
    output_graph_active: &[f32],
    output_eager_active: &[f32],
) -> BgraphVerdict {
    match max_abs_diff(output_graph_active, output_eager_active) {
        Some(d) if d < AC_BGRAPH_001_OUTPUT_TOLERANCE => BgraphVerdict::Pass,
        _ => BgraphVerdict::Fail,
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: BGRAPH-002 — c=1 no-regression.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_c1_no_regression(post_tok_s: f32, pre_tok_s: f32) -> BgraphVerdict {
    if !post_tok_s.is_finite() || !pre_tok_s.is_finite() {
        return BgraphVerdict::Fail;
    }
    if pre_tok_s <= 0.0 {
        return BgraphVerdict::Fail;
    }
    let ratio = post_tok_s / pre_tok_s;
    if ratio >= AC_BGRAPH_002_NO_REGRESSION_RATIO {
        BgraphVerdict::Pass
    } else {
        BgraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: BGRAPH-003 — c=4 throughput improvement.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_c4_improvement(post_tok_s: f32, pre_tok_s: f32) -> BgraphVerdict {
    if !post_tok_s.is_finite() || !pre_tok_s.is_finite() {
        return BgraphVerdict::Fail;
    }
    if pre_tok_s <= 0.0 {
        return BgraphVerdict::Fail;
    }
    let ratio = post_tok_s / pre_tok_s;
    if ratio >= AC_BGRAPH_003_C4_IMPROVEMENT_RATIO {
        BgraphVerdict::Pass
    } else {
        BgraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: BGRAPH-004 — memory overhead bound.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_memory_bound(total_vram_gb: f32) -> BgraphVerdict {
    if !total_vram_gb.is_finite() || total_vram_gb < 0.0 {
        return BgraphVerdict::Fail;
    }
    if total_vram_gb <= AC_BGRAPH_004_VRAM_BOUND_GB {
        BgraphVerdict::Pass
    } else {
        BgraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: BGRAPH-005 — efficiency target.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_efficiency_target(
    realizar_tok_s_per_gb: f32,
    vllm_tok_s_per_gb: f32,
) -> BgraphVerdict {
    if !realizar_tok_s_per_gb.is_finite() || !vllm_tok_s_per_gb.is_finite() {
        return BgraphVerdict::Fail;
    }
    if vllm_tok_s_per_gb <= 0.0 || realizar_tok_s_per_gb < 0.0 {
        return BgraphVerdict::Fail;
    }
    let ratio = realizar_tok_s_per_gb / vllm_tok_s_per_gb;
    if ratio >= AC_BGRAPH_005_EFFICIENCY_MULTIPLE {
        BgraphVerdict::Pass
    } else {
        BgraphVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 6: BGRAPH-006 — padding slot isolation.
// -----------------------------------------------------------------------------

/// Pass iff every padding slot has `seq_lens[i] = 0` AND its
/// `kv_cache_modified[i] = false` (KV cache untouched).
///
/// `n_active` is the count of active (real) slots; slots in
/// `[n_active, m_padded)` must be zero-length and have unmodified KV.
#[must_use]
pub fn verdict_from_padding_isolation(
    seq_lens: &[u32],
    kv_cache_modified: &[bool],
    n_active: usize,
) -> BgraphVerdict {
    let m_padded = seq_lens.len();
    if m_padded == 0 {
        return BgraphVerdict::Fail;
    }
    if kv_cache_modified.len() != m_padded {
        return BgraphVerdict::Fail;
    }
    if n_active > m_padded {
        return BgraphVerdict::Fail;
    }
    for i in n_active..m_padded {
        if seq_lens[i] != 0 {
            return BgraphVerdict::Fail;
        }
        if kv_cache_modified[i] {
            return BgraphVerdict::Fail;
        }
    }
    BgraphVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_output_tolerance_1e_5() {
        assert_eq!(AC_BGRAPH_001_OUTPUT_TOLERANCE, 1e-5);
    }

    #[test]
    fn provenance_c1_ratio_098() {
        assert_eq!(AC_BGRAPH_002_NO_REGRESSION_RATIO, 0.98);
    }

    #[test]
    fn provenance_c4_ratio_120() {
        assert_eq!(AC_BGRAPH_003_C4_IMPROVEMENT_RATIO, 1.20);
    }

    #[test]
    fn provenance_vram_bound_75() {
        assert_eq!(AC_BGRAPH_004_VRAM_BOUND_GB, 7.5);
    }

    #[test]
    fn provenance_efficiency_multiple_15() {
        assert_eq!(AC_BGRAPH_005_EFFICIENCY_MULTIPLE, 1.50);
    }

    #[test]
    fn provenance_bucket_set_powers_of_2() {
        assert_eq!(BGRAPH_BUCKET_SET, [1, 2, 4, 8, 16, 32]);
    }

    // -------------------------------------------------------------------------
    // Section 2: Bucket selection round-trip.
    // -------------------------------------------------------------------------
    #[test]
    fn bucket_selection_exact_match() {
        for &b in &BGRAPH_BUCKET_SET {
            assert_eq!(next_bucket(b), Some(b));
        }
    }

    #[test]
    fn bucket_selection_round_up() {
        assert_eq!(next_bucket(3), Some(4));
        assert_eq!(next_bucket(5), Some(8));
        assert_eq!(next_bucket(9), Some(16));
        assert_eq!(next_bucket(17), Some(32));
    }

    #[test]
    fn bucket_selection_zero_returns_none() {
        assert_eq!(next_bucket(0), None);
    }

    #[test]
    fn bucket_selection_above_max_returns_none() {
        assert_eq!(next_bucket(33), None);
        assert_eq!(next_bucket(100), None);
    }

    // -------------------------------------------------------------------------
    // Section 3: BGRAPH-001 — graph parity.
    // -------------------------------------------------------------------------
    #[test]
    fn bgraph001_pass_identical_outputs() {
        let g = vec![1.0_f32, 2.0, 3.0];
        let e = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Pass);
    }

    #[test]
    fn bgraph001_pass_within_tolerance() {
        let g = vec![1.000005_f32];
        let e = vec![1.0_f32];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Pass);
    }

    #[test]
    fn bgraph001_fail_above_tolerance() {
        let g = vec![1.0001_f32];
        let e = vec![1.0_f32];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Fail);
    }

    #[test]
    fn bgraph001_fail_length_mismatch() {
        let g = vec![1.0_f32, 2.0];
        let e = vec![1.0_f32];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Fail);
    }

    #[test]
    fn bgraph001_fail_nan() {
        let g = vec![f32::NAN];
        let e = vec![1.0_f32];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Fail);
    }

    #[test]
    fn bgraph001_fail_padding_corruption_simulation() {
        // Simulated padding bug: graph M=8 with active M=3 produces
        // wrong values for active slots due to padding contaminating
        // attention.
        let g = vec![1.0_f32, 2.0, 3.5]; // slot 2 corrupted (should be 3.0)
        let e = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: BGRAPH-002 — c=1 no-regression.
    // -------------------------------------------------------------------------
    #[test]
    fn bgraph002_pass_unchanged_throughput() {
        // 369.9 → 369.9 (pre-graph baseline).
        assert_eq!(
            verdict_from_c1_no_regression(369.9, 369.9),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph002_pass_within_2_percent_drop() {
        // 369.9 → 365 = ratio 0.987 ≥ 0.98.
        assert_eq!(
            verdict_from_c1_no_regression(365.0, 369.9),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph002_pass_actually_improved() {
        assert_eq!(
            verdict_from_c1_no_regression(400.0, 369.9),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph002_fail_3_percent_drop() {
        // 369.9 → 358 = ratio 0.967 < 0.98.
        assert_eq!(
            verdict_from_c1_no_regression(358.0, 369.9),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph002_fail_zero_pre() {
        assert_eq!(
            verdict_from_c1_no_regression(369.9, 0.0),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph002_fail_nan_post() {
        assert_eq!(
            verdict_from_c1_no_regression(f32::NAN, 369.9),
            BgraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: BGRAPH-003 — c=4 throughput improvement.
    // -------------------------------------------------------------------------
    #[test]
    fn bgraph003_pass_at_threshold() {
        // 634 → 761 = ratio ≈ 1.2003 ≥ 1.20.
        // (760.8/634 underflows the exact 1.20 in f32 so we use 761.)
        assert_eq!(
            verdict_from_c4_improvement(761.0, 634.0),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph003_pass_well_above() {
        // 634 → 1268 = 2x.
        assert_eq!(
            verdict_from_c4_improvement(1268.0, 634.0),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph003_fail_just_below() {
        // 634 → 759 = 1.197 < 1.20.
        assert_eq!(
            verdict_from_c4_improvement(759.0, 634.0),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph003_fail_no_improvement() {
        assert_eq!(
            verdict_from_c4_improvement(634.0, 634.0),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph003_fail_regression() {
        assert_eq!(
            verdict_from_c4_improvement(500.0, 634.0),
            BgraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: BGRAPH-004 — memory bound.
    // -------------------------------------------------------------------------
    #[test]
    fn bgraph004_pass_at_envelope() {
        assert_eq!(verdict_from_memory_bound(7.5), BgraphVerdict::Pass);
    }

    #[test]
    fn bgraph004_pass_well_under() {
        assert_eq!(verdict_from_memory_bound(5.5), BgraphVerdict::Pass);
    }

    #[test]
    fn bgraph004_fail_above_envelope() {
        assert_eq!(verdict_from_memory_bound(8.0), BgraphVerdict::Fail);
    }

    #[test]
    fn bgraph004_fail_negative() {
        assert_eq!(verdict_from_memory_bound(-1.0), BgraphVerdict::Fail);
    }

    #[test]
    fn bgraph004_fail_nan() {
        assert_eq!(verdict_from_memory_bound(f32::NAN), BgraphVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: BGRAPH-005 — efficiency target.
    // -------------------------------------------------------------------------
    #[test]
    fn bgraph005_pass_at_threshold() {
        // realizar 150 / vLLM 100 = 1.50
        assert_eq!(
            verdict_from_efficiency_target(150.0, 100.0),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph005_pass_well_above() {
        assert_eq!(
            verdict_from_efficiency_target(300.0, 100.0),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph005_fail_below_threshold() {
        assert_eq!(
            verdict_from_efficiency_target(140.0, 100.0),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph005_fail_zero_vllm() {
        assert_eq!(
            verdict_from_efficiency_target(150.0, 0.0),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph005_fail_negative_realizar() {
        assert_eq!(
            verdict_from_efficiency_target(-1.0, 100.0),
            BgraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: BGRAPH-006 — padding slot isolation.
    // -------------------------------------------------------------------------
    #[test]
    fn bgraph006_pass_no_padding() {
        // M_actual = M_padded ⇒ all slots active.
        let seq_lens = vec![5_u32, 7, 3, 9];
        let kv_modified = vec![true, true, true, true];
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 4),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph006_pass_3_active_in_4_bucket() {
        let seq_lens = vec![5_u32, 7, 3, 0]; // slot 3 is padding
        let kv_modified = vec![true, true, true, false];
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 3),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph006_pass_5_active_in_8_bucket() {
        let seq_lens = vec![5_u32, 7, 3, 9, 4, 0, 0, 0];
        let kv_modified = vec![true, true, true, true, true, false, false, false];
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 5),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn bgraph006_fail_padding_with_nonzero_seq_len() {
        let seq_lens = vec![5_u32, 7, 3, 1]; // slot 3 should be padding (0)
        let kv_modified = vec![true, true, true, false];
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 3),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph006_fail_padding_kv_modified() {
        let seq_lens = vec![5_u32, 7, 3, 0];
        let kv_modified = vec![true, true, true, true]; // bug: slot 3 KV touched
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 3),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph006_fail_n_active_exceeds_padded() {
        let seq_lens = vec![5_u32, 7];
        let kv_modified = vec![true, true];
        // n_active=3 > M_padded=2 — illegal.
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 3),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph006_fail_kv_modified_length_mismatch() {
        let seq_lens = vec![5_u32, 0, 0];
        let kv_modified = vec![true, false]; // length 2 vs 3
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 1),
            BgraphVerdict::Fail
        );
    }

    #[test]
    fn bgraph006_fail_empty() {
        let seq_lens: Vec<u32> = vec![];
        let kv_modified: Vec<bool> = vec![];
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_modified, 0),
            BgraphVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 9: Sweep — bucket selection.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_bucket_invariants() {
        // For every actual_m in [1, 32], the bucket should be ≥ m and
        // ≤ 2m.
        for m in 1_usize..=32 {
            let b = next_bucket(m).unwrap();
            assert!(b >= m, "m={m} b={b}");
            assert!(b <= 2 * m, "m={m} b={b}");
            assert!(BGRAPH_BUCKET_SET.contains(&b), "b={b} not in bucket set");
        }
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — full Phase-18 acceptance scenario.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_phase18_acceptance_scenario() {
        // Synthesize a Phase-18 result and check all six gates.

        // BGRAPH-001: 4 active slots, 3 hidden dims each, max diff = 8e-6.
        let g = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut e = g.clone();
        e[5] += 8e-6; // small drift well within 1e-5
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Pass);

        // BGRAPH-002: 369.9 → 365 = 0.987 ratio.
        assert_eq!(
            verdict_from_c1_no_regression(365.0, 369.9),
            BgraphVerdict::Pass
        );

        // BGRAPH-003: 634 → 800 = 1.262 ratio.
        assert_eq!(
            verdict_from_c4_improvement(800.0, 634.0),
            BgraphVerdict::Pass
        );

        // BGRAPH-004: total VRAM 7.0 GB.
        assert_eq!(verdict_from_memory_bound(7.0), BgraphVerdict::Pass);

        // BGRAPH-005: realizar 200, vLLM 100.
        assert_eq!(
            verdict_from_efficiency_target(200.0, 100.0),
            BgraphVerdict::Pass
        );

        // BGRAPH-006: 3 active in 4 bucket.
        let seq_lens = vec![5_u32, 7, 3, 0];
        let kv_mod = vec![true, true, true, false];
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_mod, 3),
            BgraphVerdict::Pass
        );
    }

    #[test]
    fn realistic_padding_corruption_failure_mode() {
        // BGRAPH-001 if_fails: "Padding slot corruption or KV cache
        // indexing bug in M>1 graph". Active slot output deviates by
        // 1e-3 — far above tolerance.
        let g = vec![1.001_f32, 2.0, 3.0];
        let e = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_graph_parity(&g, &e), BgraphVerdict::Fail);

        // BGRAPH-006 catches the root cause: padding slot KV got modified.
        let seq_lens = vec![5_u32, 7, 0];
        let kv_mod = vec![true, true, true]; // bug
        assert_eq!(
            verdict_from_padding_isolation(&seq_lens, &kv_mod, 2),
            BgraphVerdict::Fail
        );
    }
}
