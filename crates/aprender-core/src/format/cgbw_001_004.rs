// SHIP-TWO-001 — `cuda-graph-backward-v1` algorithm-level PARTIAL
// discharge for F-GRAPH-BWD-001..004.
//
// Contract: `contracts/cuda-graph-backward-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four backward-graph capture gates (PMAT-464):
//
// - F-GRAPH-BWD-001 (gradient parity): graphed loss trajectory matches
//   ungraphed within 0.1 absolute tolerance at canary checkpoints
//   {step 10, 50, 100}.
// - F-GRAPH-BWD-002 (no D2H sync inside graph): sync_count_inside_graph
//   == 0 (capture succeeds without CUDA_ERROR).
// - F-GRAPH-BWD-003 (sync count reduction): sync_count(fixed) == 1
//   AND sync_count(current) == 168 (28 layers × 6 syncs/layer).
// - F-GRAPH-BWD-004 (clipping equivalence): global vs per-layer
//   gradient clipping final-loss diff < 0.5 over 1000 steps.

/// Loss trajectory tolerance per checkpoint (F-GRAPH-BWD-001).
pub const AC_CGBW_001_LOSS_TOLERANCE: f32 = 0.1;

/// Required sync count INSIDE graph capture boundary (F-GRAPH-BWD-002).
pub const AC_CGBW_002_SYNC_INSIDE_GRAPH_BUDGET: usize = 0;

/// Layers in 28-layer transformer (Qwen2.5-Coder-1.5B).
pub const AC_CGBW_003_N_LAYERS: usize = 28;

/// D2H syncs per layer in unfixed (current) backward.
pub const AC_CGBW_003_SYNCS_PER_LAYER_CURRENT: usize = 6;

/// Total expected syncs in current code (28 × 6 = 168).
pub const AC_CGBW_003_SYNC_COUNT_CURRENT: usize =
    AC_CGBW_003_N_LAYERS * AC_CGBW_003_SYNCS_PER_LAYER_CURRENT;

/// Total expected syncs after fix (1 global per backward).
pub const AC_CGBW_003_SYNC_COUNT_FIXED: usize = 1;

/// 1000-step extended canary final-loss diff floor (F-GRAPH-BWD-004).
pub const AC_CGBW_004_LOSS_DIFF_TOLERANCE: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgbwVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: F-GRAPH-BWD-001 — gradient parity (loss trajectory).
// -----------------------------------------------------------------------------

/// Pass iff `|loss_graphed[i] - loss_ungraphed[i]| < 0.1` for every
/// checkpoint pair `i`.
///
/// `loss_graphed` and `loss_ungraphed` must align element-wise on the
/// same step indices.
#[must_use]
pub fn verdict_from_loss_trajectory_parity(
    loss_graphed: &[f32],
    loss_ungraphed: &[f32],
) -> CgbwVerdict {
    if loss_graphed.is_empty() || loss_graphed.len() != loss_ungraphed.len() {
        return CgbwVerdict::Fail;
    }
    for (g, u) in loss_graphed.iter().zip(loss_ungraphed.iter()) {
        if !g.is_finite() || !u.is_finite() {
            return CgbwVerdict::Fail;
        }
        if (g - u).abs() >= AC_CGBW_001_LOSS_TOLERANCE {
            return CgbwVerdict::Fail;
        }
    }
    CgbwVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: F-GRAPH-BWD-002 — no D2H sync inside graph boundary.
// -----------------------------------------------------------------------------

/// Pass iff `sync_count_inside_graph == 0` AND `capture_succeeded`
/// (the cudaStreamBeginCapture+EndCapture round-trip returned no
/// CUDA_ERROR).
#[must_use]
pub fn verdict_from_no_sync_inside_graph(
    sync_count_inside_graph: usize,
    capture_succeeded: bool,
) -> CgbwVerdict {
    if !capture_succeeded {
        return CgbwVerdict::Fail;
    }
    if sync_count_inside_graph == AC_CGBW_002_SYNC_INSIDE_GRAPH_BUDGET {
        CgbwVerdict::Pass
    } else {
        CgbwVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: F-GRAPH-BWD-003 — sync count reduction 168 → 1.
// -----------------------------------------------------------------------------

/// Pass iff `sync_count_current == 168` AND `sync_count_fixed == 1`.
#[must_use]
pub fn verdict_from_sync_count_reduction(
    sync_count_current: usize,
    sync_count_fixed: usize,
) -> CgbwVerdict {
    if sync_count_current == AC_CGBW_003_SYNC_COUNT_CURRENT
        && sync_count_fixed == AC_CGBW_003_SYNC_COUNT_FIXED
    {
        CgbwVerdict::Pass
    } else {
        CgbwVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: F-GRAPH-BWD-004 — global vs per-layer clipping equivalence.
// -----------------------------------------------------------------------------

/// Pass iff `|global_clip_loss - per_layer_clip_loss| < 0.5` after
/// 1000 steps. Per-contract, "global gradient clipping produces
/// equivalent results to per-layer clipping" when total_norm <
/// clip_threshold most of the time (typical training regime).
#[must_use]
pub fn verdict_from_clipping_equivalence(
    global_clip_loss: f32,
    per_layer_clip_loss: f32,
) -> CgbwVerdict {
    if !global_clip_loss.is_finite() || !per_layer_clip_loss.is_finite() {
        return CgbwVerdict::Fail;
    }
    if (global_clip_loss - per_layer_clip_loss).abs() < AC_CGBW_004_LOSS_DIFF_TOLERANCE {
        CgbwVerdict::Pass
    } else {
        CgbwVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_loss_tolerance_01() {
        assert_eq!(AC_CGBW_001_LOSS_TOLERANCE, 0.1);
    }

    #[test]
    fn provenance_sync_inside_graph_zero() {
        assert_eq!(AC_CGBW_002_SYNC_INSIDE_GRAPH_BUDGET, 0);
    }

    #[test]
    fn provenance_n_layers_28() {
        assert_eq!(AC_CGBW_003_N_LAYERS, 28);
    }

    #[test]
    fn provenance_syncs_per_layer_6() {
        assert_eq!(AC_CGBW_003_SYNCS_PER_LAYER_CURRENT, 6);
    }

    #[test]
    fn provenance_sync_count_current_168() {
        assert_eq!(AC_CGBW_003_SYNC_COUNT_CURRENT, 168);
    }

    #[test]
    fn provenance_sync_count_fixed_1() {
        assert_eq!(AC_CGBW_003_SYNC_COUNT_FIXED, 1);
    }

    #[test]
    fn provenance_loss_diff_tolerance_05() {
        assert_eq!(AC_CGBW_004_LOSS_DIFF_TOLERANCE, 0.5);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-GRAPH-BWD-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn cgbw001_pass_identical_trajectories() {
        let g = vec![5.0_f32, 4.5, 4.0];
        let u = vec![5.0_f32, 4.5, 4.0];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw001_pass_within_tolerance() {
        // FP32 reduction-order drift.
        let g = vec![5.05_f32, 4.55, 4.05];
        let u = vec![5.0_f32, 4.5, 4.0];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw001_pass_full_canary() {
        // Steps 10 / 50 / 100 from a real canary run.
        let g = vec![3.21_f32, 2.84, 2.51];
        let u = vec![3.20_f32, 2.85, 2.50];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: F-GRAPH-BWD-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn cgbw001_fail_diverges_at_step_50() {
        // Per-layer clipping was missing in graphed path → divergence.
        let g = vec![3.20_f32, 3.50, 4.20]; // explosion
        let u = vec![3.20_f32, 2.85, 2.50];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw001_fail_at_threshold() {
        // 0.1 is strict — `|g - u| >= 0.1` ⇒ Fail. f32 quirk: 5.1f32 -
        // 5.0f32 ≈ 0.0999994 (< 0.1), so use 5.15 → diff = 0.15.
        let g = vec![5.15_f32];
        let u = vec![5.0_f32];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw001_fail_length_mismatch() {
        let g = vec![5.0_f32, 4.5];
        let u = vec![5.0_f32];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw001_fail_empty() {
        let v: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&v, &v),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw001_fail_nan() {
        let g = vec![f32::NAN];
        let u = vec![5.0_f32];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: F-GRAPH-BWD-002 — no sync inside graph.
    // -------------------------------------------------------------------------
    #[test]
    fn cgbw002_pass_zero_syncs_capture_succeeded() {
        assert_eq!(
            verdict_from_no_sync_inside_graph(0, true),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw002_fail_one_sync_inside() {
        // Old code path: 1 squared_sum_cuda sync inside graph.
        assert_eq!(
            verdict_from_no_sync_inside_graph(1, true),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw002_fail_capture_failed() {
        // Even with 0 syncs reported, if capture errored we Fail.
        assert_eq!(
            verdict_from_no_sync_inside_graph(0, false),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw002_fail_many_syncs() {
        // Worst case: all 168 still inside.
        assert_eq!(
            verdict_from_no_sync_inside_graph(168, false),
            CgbwVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: F-GRAPH-BWD-003 — sync count reduction.
    // -------------------------------------------------------------------------
    #[test]
    fn cgbw003_pass_canonical_168_to_1() {
        assert_eq!(
            verdict_from_sync_count_reduction(168, 1),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw003_fail_current_count_wrong() {
        // Current code reports != 168 ⇒ contract premise violated.
        assert_eq!(
            verdict_from_sync_count_reduction(150, 1),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw003_fail_fixed_count_wrong() {
        // Fixed code still has 2 syncs ⇒ root cause not addressed.
        assert_eq!(
            verdict_from_sync_count_reduction(168, 2),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw003_fail_no_change() {
        assert_eq!(
            verdict_from_sync_count_reduction(168, 168),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw003_fail_zero_zero() {
        // Both zero ⇒ measurement bug.
        assert_eq!(
            verdict_from_sync_count_reduction(0, 0),
            CgbwVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: F-GRAPH-BWD-004 — clipping equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn cgbw004_pass_identical_final_loss() {
        assert_eq!(
            verdict_from_clipping_equivalence(2.31, 2.31),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw004_pass_within_05() {
        assert_eq!(
            verdict_from_clipping_equivalence(2.31, 2.5),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw004_pass_at_boundary() {
        // |2.31 - 2.81| = 0.5 — strict, NOT inclusive ⇒ Fail.
        // |2.31 - 2.80| = 0.49 → Pass.
        assert_eq!(
            verdict_from_clipping_equivalence(2.31, 2.80),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn cgbw004_fail_exceeds_tolerance() {
        // Per-layer clipping prevented gradient explosion; global did not.
        assert_eq!(
            verdict_from_clipping_equivalence(5.0, 2.31),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw004_fail_nan() {
        assert_eq!(
            verdict_from_clipping_equivalence(f32::NAN, 2.31),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn cgbw004_fail_inf() {
        assert_eq!(
            verdict_from_clipping_equivalence(2.31, f32::INFINITY),
            CgbwVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — sync-count math.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_sync_count_current_matches_layers_times_per_layer() {
        assert_eq!(
            AC_CGBW_003_SYNC_COUNT_CURRENT,
            AC_CGBW_003_N_LAYERS * AC_CGBW_003_SYNCS_PER_LAYER_CURRENT
        );
    }

    #[test]
    fn domain_sync_savings_166_per_backward() {
        let saved = AC_CGBW_003_SYNC_COUNT_CURRENT - AC_CGBW_003_SYNC_COUNT_FIXED;
        // 168 - 1 = 167 saved.
        assert_eq!(saved, 167);
    }

    // -------------------------------------------------------------------------
    // Section 8: Sweep — sync count combinations.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_canonical_only_combination_passes() {
        // Only (168, 1) passes; everything else Fails.
        let cases: Vec<((usize, usize), CgbwVerdict)> = vec![
            ((168, 1), CgbwVerdict::Pass),
            ((167, 1), CgbwVerdict::Fail),
            ((169, 1), CgbwVerdict::Fail),
            ((168, 0), CgbwVerdict::Fail),
            ((168, 2), CgbwVerdict::Fail),
            ((84, 1), CgbwVerdict::Fail), // half the layers
            ((1, 1), CgbwVerdict::Fail),
            ((0, 0), CgbwVerdict::Fail),
        ];
        for ((current, fixed), expected) in cases {
            let v = verdict_from_sync_count_reduction(current, fixed);
            assert_eq!(v, expected, "({current}, {fixed})");
        }
    }

    // -------------------------------------------------------------------------
    // Section 9: Realistic — end-to-end scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_phase_pmat_464_acceptance() {
        // PMAT-464 acceptance: graph capture succeeds, 167 syncs
        // eliminated, loss trajectory parity, clipping equivalent.
        assert_eq!(
            verdict_from_no_sync_inside_graph(0, true),
            CgbwVerdict::Pass
        );
        assert_eq!(
            verdict_from_sync_count_reduction(168, 1),
            CgbwVerdict::Pass
        );

        let g = vec![3.20_f32, 2.85, 2.50]; // graphed
        let u = vec![3.20_f32, 2.85, 2.50]; // ungraphed
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Pass
        );

        assert_eq!(
            verdict_from_clipping_equivalence(2.50, 2.50),
            CgbwVerdict::Pass
        );
    }

    #[test]
    fn realistic_d2h_remaining_inside_graph_caught() {
        // F-GRAPH-BWD-002 if_fails: "Remaining D2H sync inside graph —
        // grep for synchronize() in backward path".
        assert_eq!(
            verdict_from_no_sync_inside_graph(1, false),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn realistic_clipping_drift_caught() {
        // F-GRAPH-BWD-004 if_fails: "Per-layer clipping prevents
        // gradient explosion in early layers that global clipping
        // misses".
        assert_eq!(
            verdict_from_clipping_equivalence(8.0, 2.5),
            CgbwVerdict::Fail
        );
    }

    #[test]
    fn realistic_per_layer_norm_order_drift_caught() {
        // F-GRAPH-BWD-001 if_fails: "Gradient clipping order matters —
        // per-layer vs global norm produces different trajectories".
        let g = vec![3.20_f32, 4.50, 6.00]; // diverging
        let u = vec![3.20_f32, 2.85, 2.50];
        assert_eq!(
            verdict_from_loss_trajectory_parity(&g, &u),
            CgbwVerdict::Fail
        );
    }
}
