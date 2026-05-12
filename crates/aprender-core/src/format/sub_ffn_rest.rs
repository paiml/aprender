// SHIP-TWO-001 — `trace-ffn-sub-block-v1` algorithm-level PARTIAL
// discharge for FALSIFY-SUB-FFN-001..004 + 006..008 (closes 7
// remaining gates; SUB-FFN-005 already bound in `sub_ffn_005.rs`).
//
// Contract: `contracts/trace-ffn-sub-block-v1.yaml`.
// Spec: SHIP-007 layer-3 sub-FFN bisection.

// ===========================================================================
// Canonical contract constants
// ===========================================================================

pub const AC_SUB_FFN_BEFORE_FIELD_COUNT: u64 = 6;
pub const AC_SUB_FFN_AFTER_FIELD_COUNT: u64 = 10;
pub const AC_SUB_FFN_REFERENCE_LAYER_INDEX: u32 = 3;
pub const AC_SUB_FFN_REFERENCE_FFN_OUT_STD_LAYER_3: f64 = 11.459;
pub const AC_SUB_FFN_REFERENCE_FFN_OUT_STD_LAYER_2: f64 = 0.216;
/// Spike ratio threshold: layer-3 std must exceed 3× median over layers 5..=26.
pub const AC_SUB_FFN_001_SPIKE_RATIO: f64 = 3.0;
pub const AC_SUB_FFN_001_NEIGHBOR_RATIO_LIMIT: f64 = 2.0;
pub const AC_SUB_FFN_006_QWEN_INTERMEDIATE_DIM: usize = 18_944;

// ===========================================================================
// SUB-FFN-001 — Exactly one slot at layer 3 has std ≥ 3× median
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn001Verdict { Pass, Fail }

/// `slot_stds_layer_3[i]` is the std of slot `i` at layer 3.
/// `slot_medians[i]` is the median std of slot `i` across layers 5..=26.
/// Pass iff exactly one slot has std >= 3 × median AND every other slot
/// has std < 2 × median.
#[must_use]
pub fn verdict_from_layer3_spike_localized(
    slot_stds_layer_3: &[f64; 5],
    slot_medians: &[f64; 5],
) -> SubFfn001Verdict {
    if slot_medians.iter().any(|m| !m.is_finite() || *m <= 0.0) {
        return SubFfn001Verdict::Fail;
    }
    if slot_stds_layer_3.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return SubFfn001Verdict::Fail;
    }
    let mut spike_count = 0;
    let mut neighbor_violations = 0;
    for i in 0..5 {
        let ratio = slot_stds_layer_3[i] / slot_medians[i];
        if ratio >= AC_SUB_FFN_001_SPIKE_RATIO {
            spike_count += 1;
        } else if ratio >= AC_SUB_FFN_001_NEIGHBOR_RATIO_LIMIT {
            neighbor_violations += 1;
        }
    }
    if spike_count == 1 && neighbor_violations == 0 {
        SubFfn001Verdict::Pass
    } else {
        SubFfn001Verdict::Fail
    }
}

// ===========================================================================
// SUB-FFN-002 — Backward compat: existing ffn_out_stats byte-identical
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn002Verdict { Pass, Fail }

/// Pass iff `pre_bytes == post_bytes` for the canonical ffn_out_stats
/// snapshot from one of the regression tests.
#[must_use]
pub fn verdict_from_ffn_out_stats_byte_identical(pre: &[u8], post: &[u8]) -> SubFfn002Verdict {
    if pre.is_empty() || post.is_empty() { return SubFfn002Verdict::Fail; }
    if pre == post { SubFfn002Verdict::Pass } else { SubFfn002Verdict::Fail }
}

// ===========================================================================
// SUB-FFN-003 — Naming alignment with layer-parity-v1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn003Verdict { Pass, Fail }

/// Canonical mapping from sub-FFN trace field to layer-parity step.
/// Pass iff the supplied mapping equals the canonical mapping (4
/// pairs).
#[must_use]
pub fn verdict_from_naming_alignment(observed: &[(&str, &str)]) -> SubFfn003Verdict {
    let canonical: [(&str, &str); 4] = [
        ("ffn_gate", "gate_proj"),
        ("ffn_up", "up_proj"),
        ("ffn_swiglu", "swiglu"),
        ("ffn_out", "down_proj"),
    ];
    if observed.len() != canonical.len() { return SubFfn003Verdict::Fail; }
    for can in canonical {
        if !observed.iter().any(|p| *p == can) { return SubFfn003Verdict::Fail; }
    }
    SubFfn003Verdict::Pass
}

// ===========================================================================
// SUB-FFN-004 — GPU TracedForward: populated stats OR explicit incomplete flag
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTelemetryShape {
    /// All 4 new fields populated with non-zero values from real GPU.
    AllPopulated,
    /// Fields explicitly zero AND `gpu_telemetry_complete: false`.
    AllZeroFlagged,
    /// Fields zero with `gpu_telemetry_complete: true` (the regression).
    SilentlyZeroPretendingComplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gpu_telemetry_shape(shape: GpuTelemetryShape) -> SubFfn004Verdict {
    match shape {
        GpuTelemetryShape::AllPopulated | GpuTelemetryShape::AllZeroFlagged => SubFfn004Verdict::Pass,
        GpuTelemetryShape::SilentlyZeroPretendingComplete => SubFfn004Verdict::Fail,
    }
}

// ===========================================================================
// SUB-FFN-006 — Captured vector lengths == intermediate_dim
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn006Verdict { Pass, Fail }

/// Pass iff every captured-vector length equals `intermediate_dim` AND
/// `intermediate_dim > 0`.
#[must_use]
pub fn verdict_from_vector_length_correct(
    captured_lens: &[usize; 4],
    intermediate_dim: usize,
) -> SubFfn006Verdict {
    if intermediate_dim == 0 { return SubFfn006Verdict::Fail; }
    for &len in captured_lens {
        if len != intermediate_dim { return SubFfn006Verdict::Fail; }
    }
    SubFfn006Verdict::Pass
}

// ===========================================================================
// SUB-FFN-007 — Doc-comment cites the contract id (drift-prevention)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn007Verdict { Pass, Fail }

pub const AC_SUB_FFN_007_NEEDLE: &str = "contracts/trace-ffn-sub-block-v1";
pub const AC_SUB_FFN_007_REQUIRED_HITS: usize = 4;

/// Pass iff `source_lines` contains the contract path string at least
/// `AC_SUB_FFN_007_REQUIRED_HITS` (=4) times — one per new field.
#[must_use]
pub fn verdict_from_doc_citation_count(source_lines: &str) -> SubFfn007Verdict {
    let count = source_lines.matches(AC_SUB_FFN_007_NEEDLE).count();
    if count >= AC_SUB_FFN_007_REQUIRED_HITS { SubFfn007Verdict::Pass } else { SubFfn007Verdict::Fail }
}

// ===========================================================================
// SUB-FFN-008 — Coverage co-evolution: every new field covered by ≥1 test
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn008Verdict { Pass, Fail }

pub const AC_SUB_FFN_008_NEW_FIELDS: [&str; 4] = [
    "ffn_gate_stats",
    "ffn_up_stats",
    "ffn_silu_gate_stats",
    "ffn_swiglu_inner_stats",
];

/// `field_test_counts` is a parallel array of test counts per field.
/// Pass iff every entry is ≥ 1.
#[must_use]
pub fn verdict_from_coverage_coevolution(field_test_counts: &[u64; 4]) -> SubFfn008Verdict {
    for &count in field_test_counts {
        if count == 0 { return SubFfn008Verdict::Fail; }
    }
    SubFfn008Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- SUB-FFN-001 -------------------------------------------------------

    #[test] fn sf001_pass_one_spike() {
        // Slot 3 (ffn_out) at 11.459 vs median 0.216 ≈ 53× spike;
        // other slots near median.
        let l3 = [0.5, 0.4, 0.3, 11.459, 0.5];
        let med = [0.5, 0.5, 0.5, 0.216, 0.5];
        assert_eq!(verdict_from_layer3_spike_localized(&l3, &med), SubFfn001Verdict::Pass);
    }

    #[test] fn sf001_fail_no_spike() {
        let l3 = [0.5, 0.4, 0.3, 0.5, 0.5];
        let med = [0.5; 5];
        assert_eq!(verdict_from_layer3_spike_localized(&l3, &med), SubFfn001Verdict::Fail);
    }

    #[test] fn sf001_fail_two_spikes() {
        let l3 = [10.0, 0.4, 0.3, 11.0, 0.5];
        let med = [0.5; 5];
        assert_eq!(verdict_from_layer3_spike_localized(&l3, &med), SubFfn001Verdict::Fail);
    }

    #[test] fn sf001_fail_neighbor_above_2x() {
        // Slot 3 at 30× spike + slot 1 at 2.5× (above 2× neighbor limit).
        let l3 = [0.5, 1.25, 0.3, 15.0, 0.5];
        let med = [0.5; 5];
        assert_eq!(verdict_from_layer3_spike_localized(&l3, &med), SubFfn001Verdict::Fail);
    }

    #[test] fn sf001_fail_zero_median() {
        let l3 = [0.5, 0.4, 0.3, 11.0, 0.5];
        let med = [0.5, 0.5, 0.5, 0.0, 0.5];
        assert_eq!(verdict_from_layer3_spike_localized(&l3, &med), SubFfn001Verdict::Fail);
    }

    // ----- SUB-FFN-002 -------------------------------------------------------

    #[test] fn sf002_pass_match() {
        assert_eq!(verdict_from_ffn_out_stats_byte_identical(b"abc", b"abc"), SubFfn002Verdict::Pass);
    }

    #[test] fn sf002_fail_drift() {
        assert_eq!(verdict_from_ffn_out_stats_byte_identical(b"abc", b"abd"), SubFfn002Verdict::Fail);
    }

    #[test] fn sf002_fail_empty() {
        assert_eq!(verdict_from_ffn_out_stats_byte_identical(b"", b"abc"), SubFfn002Verdict::Fail);
    }

    // ----- SUB-FFN-003 -------------------------------------------------------

    #[test] fn sf003_pass_canonical() {
        let obs = [
            ("ffn_gate", "gate_proj"),
            ("ffn_up", "up_proj"),
            ("ffn_swiglu", "swiglu"),
            ("ffn_out", "down_proj"),
        ];
        assert_eq!(verdict_from_naming_alignment(&obs), SubFfn003Verdict::Pass);
    }

    #[test] fn sf003_pass_reordered() {
        let obs = [
            ("ffn_out", "down_proj"),
            ("ffn_gate", "gate_proj"),
            ("ffn_swiglu", "swiglu"),
            ("ffn_up", "up_proj"),
        ];
        assert_eq!(verdict_from_naming_alignment(&obs), SubFfn003Verdict::Pass);
    }

    #[test] fn sf003_fail_missing_pair() {
        let obs = [
            ("ffn_gate", "gate_proj"),
            ("ffn_up", "up_proj"),
            ("ffn_out", "down_proj"),
        ];
        assert_eq!(verdict_from_naming_alignment(&obs), SubFfn003Verdict::Fail);
    }

    #[test] fn sf003_fail_renamed_field() {
        let obs = [
            ("ffn_gate", "gate_proj"),
            ("ffn_up", "up_proj"),
            ("ffn_swiglu", "swiglu"),
            ("ffn_out", "WRONG_NAME"),
        ];
        assert_eq!(verdict_from_naming_alignment(&obs), SubFfn003Verdict::Fail);
    }

    // ----- SUB-FFN-004 -------------------------------------------------------

    #[test] fn sf004_pass_populated() {
        assert_eq!(verdict_from_gpu_telemetry_shape(GpuTelemetryShape::AllPopulated), SubFfn004Verdict::Pass);
    }
    #[test] fn sf004_pass_flagged_zero() {
        assert_eq!(verdict_from_gpu_telemetry_shape(GpuTelemetryShape::AllZeroFlagged), SubFfn004Verdict::Pass);
    }
    #[test] fn sf004_fail_silent_zero() {
        // The regression: GPU silently emits zeros while pretending complete.
        assert_eq!(
            verdict_from_gpu_telemetry_shape(GpuTelemetryShape::SilentlyZeroPretendingComplete),
            SubFfn004Verdict::Fail
        );
    }

    // ----- SUB-FFN-006 -------------------------------------------------------

    #[test] fn sf006_pass_qwen_intermediate() {
        let lens = [18944, 18944, 18944, 18944];
        assert_eq!(verdict_from_vector_length_correct(&lens, AC_SUB_FFN_006_QWEN_INTERMEDIATE_DIM), SubFfn006Verdict::Pass);
    }
    #[test] fn sf006_fail_one_short() {
        let lens = [18944, 18944, 100, 18944];
        assert_eq!(verdict_from_vector_length_correct(&lens, AC_SUB_FFN_006_QWEN_INTERMEDIATE_DIM), SubFfn006Verdict::Fail);
    }
    #[test] fn sf006_fail_zero_dim() {
        let lens = [0, 0, 0, 0];
        assert_eq!(verdict_from_vector_length_correct(&lens, 0), SubFfn006Verdict::Fail);
    }

    // ----- SUB-FFN-007 -------------------------------------------------------

    #[test] fn sf007_pass_four_citations() {
        let src = "fn a() {}\n/// contracts/trace-ffn-sub-block-v1\nfn b() {}\n\
                   /// contracts/trace-ffn-sub-block-v1\nfn c() {}\n\
                   /// contracts/trace-ffn-sub-block-v1\nfn d() {}\n\
                   /// contracts/trace-ffn-sub-block-v1\nfn e() {}\n";
        assert_eq!(verdict_from_doc_citation_count(src), SubFfn007Verdict::Pass);
    }
    #[test] fn sf007_pass_more_than_four() {
        let src = "contracts/trace-ffn-sub-block-v1 ".repeat(10);
        assert_eq!(verdict_from_doc_citation_count(&src), SubFfn007Verdict::Pass);
    }
    #[test] fn sf007_fail_three_citations() {
        let src = "contracts/trace-ffn-sub-block-v1 ".repeat(3);
        assert_eq!(verdict_from_doc_citation_count(&src), SubFfn007Verdict::Fail);
    }
    #[test] fn sf007_fail_no_citations() {
        assert_eq!(verdict_from_doc_citation_count("fn main() {}"), SubFfn007Verdict::Fail);
    }

    // ----- SUB-FFN-008 -------------------------------------------------------

    #[test] fn sf008_pass_all_covered() {
        assert_eq!(verdict_from_coverage_coevolution(&[1, 1, 1, 1]), SubFfn008Verdict::Pass);
    }
    #[test] fn sf008_pass_more_tests() {
        assert_eq!(verdict_from_coverage_coevolution(&[5, 3, 8, 2]), SubFfn008Verdict::Pass);
    }
    #[test] fn sf008_fail_one_uncovered() {
        assert_eq!(verdict_from_coverage_coevolution(&[1, 1, 0, 1]), SubFfn008Verdict::Fail);
    }
    #[test] fn sf008_fail_all_zero() {
        assert_eq!(verdict_from_coverage_coevolution(&[0, 0, 0, 0]), SubFfn008Verdict::Fail);
    }

    // ----- Provenance pins ---------------------------------------------------

    #[test] fn provenance_constants() {
        assert_eq!(AC_SUB_FFN_BEFORE_FIELD_COUNT, 6);
        assert_eq!(AC_SUB_FFN_AFTER_FIELD_COUNT, 10);
        assert_eq!(AC_SUB_FFN_REFERENCE_LAYER_INDEX, 3);
        assert!((AC_SUB_FFN_REFERENCE_FFN_OUT_STD_LAYER_3 - 11.459).abs() < 1e-9);
        assert!((AC_SUB_FFN_REFERENCE_FFN_OUT_STD_LAYER_2 - 0.216).abs() < 1e-9);
        assert!((AC_SUB_FFN_001_SPIKE_RATIO - 3.0).abs() < 1e-9);
        assert!((AC_SUB_FFN_001_NEIGHBOR_RATIO_LIMIT - 2.0).abs() < 1e-9);
        assert_eq!(AC_SUB_FFN_006_QWEN_INTERMEDIATE_DIM, 18_944);
        assert_eq!(AC_SUB_FFN_007_REQUIRED_HITS, 4);
        assert_eq!(AC_SUB_FFN_008_NEW_FIELDS.len(), 4);
    }
}
