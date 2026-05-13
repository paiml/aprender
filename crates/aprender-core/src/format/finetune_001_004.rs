// SHIP-TWO-001 — `apr-finetune-metrics-v1` algorithm-level PARTIAL
// discharge for FALSIFY-FINETUNE-001..004.
//
// Contract: `contracts/apr-finetune-metrics-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four `apr finetune --json` schema gates (GH-566):
//
// - FINETUNE-001 (JSON schema complete): all required fields present.
// - FINETUNE-002 (wall_time consistency): wall_time_sec ≈ total_time_ms / 1000.
// - FINETUNE-003 (epoch_metrics length): len(epoch_metrics) == total_epochs.
// - FINETUNE-004 (throughput positive): tokens_per_sec > 0 after training.

/// Required top-level JSON fields for `apr finetune --json` output.
pub const AC_FINETUNE_001_REQUIRED_FIELDS: &[&str] = &[
    "status",
    "final_loss",
    "best_val_loss",
    "wall_time_sec",
    "total_epochs",
    "tokens_per_sec",
    "samples_per_sec",
    "checkpoint_dir",
    "epoch_metrics",
];

/// FINETUNE-002 tolerance for ms-to-sec round-trip (FP32 wobble).
pub const AC_FINETUNE_002_TIME_TOLERANCE_SEC: f64 = 0.01;

/// FINETUNE-004 lower bound on tokens_per_sec.
pub const AC_FINETUNE_004_THROUGHPUT_LOWER: f64 = 0.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinetuneVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: FINETUNE-001 — JSON schema complete.
// -----------------------------------------------------------------------------

/// Pass iff `present_fields` is a SUPERSET of `required_fields` AND no
/// required field's value is null/empty.
#[must_use]
pub fn verdict_from_json_schema_complete(
    present_fields: &[&str],
    null_fields: &[&str],
) -> FinetuneVerdict {
    let present_set: std::collections::HashSet<&&str> = present_fields.iter().collect();
    let null_set: std::collections::HashSet<&&str> = null_fields.iter().collect();

    for &required in AC_FINETUNE_001_REQUIRED_FIELDS {
        if !present_set.contains(&required) {
            return FinetuneVerdict::Fail;
        }
        if null_set.contains(&required) {
            return FinetuneVerdict::Fail;
        }
    }
    FinetuneVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: FINETUNE-002 — wall_time consistency.
// -----------------------------------------------------------------------------

/// Pass iff `|wall_time_sec - total_time_ms / 1000| < 0.01`.
#[must_use]
pub fn verdict_from_wall_time_consistency(
    wall_time_sec: f64,
    total_time_ms: u64,
) -> FinetuneVerdict {
    if !wall_time_sec.is_finite() {
        return FinetuneVerdict::Fail;
    }
    if wall_time_sec <= 0.0 {
        return FinetuneVerdict::Fail;
    }
    let expected_sec = total_time_ms as f64 / 1000.0;
    if (wall_time_sec - expected_sec).abs() < AC_FINETUNE_002_TIME_TOLERANCE_SEC {
        FinetuneVerdict::Pass
    } else {
        FinetuneVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: FINETUNE-003 — epoch_metrics length matches total_epochs.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_epoch_metrics_length(
    epoch_metrics_len: usize,
    total_epochs: u64,
) -> FinetuneVerdict {
    if total_epochs == 0 {
        return FinetuneVerdict::Fail;
    }
    if epoch_metrics_len as u64 == total_epochs {
        FinetuneVerdict::Pass
    } else {
        FinetuneVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: FINETUNE-004 — throughput positive.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_throughput_positive(tokens_per_sec: f64) -> FinetuneVerdict {
    if !tokens_per_sec.is_finite() {
        return FinetuneVerdict::Fail;
    }
    if tokens_per_sec > AC_FINETUNE_004_THROUGHPUT_LOWER {
        FinetuneVerdict::Pass
    } else {
        FinetuneVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_fields_count() {
        assert_eq!(AC_FINETUNE_001_REQUIRED_FIELDS.len(), 9);
    }

    #[test]
    fn provenance_required_fields_includes_core() {
        for f in [
            "status",
            "final_loss",
            "wall_time_sec",
            "total_epochs",
            "tokens_per_sec",
            "epoch_metrics",
        ] {
            assert!(
                AC_FINETUNE_001_REQUIRED_FIELDS.contains(&f),
                "missing: {f}"
            );
        }
    }

    #[test]
    fn provenance_time_tolerance_001() {
        assert_eq!(AC_FINETUNE_002_TIME_TOLERANCE_SEC, 0.01);
    }

    #[test]
    fn provenance_throughput_lower_zero() {
        assert_eq!(AC_FINETUNE_004_THROUGHPUT_LOWER, 0.0);
    }

    // -------------------------------------------------------------------------
    // Section 2: FINETUNE-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn finetune001_pass_all_present_no_nulls() {
        let present: Vec<&str> = AC_FINETUNE_001_REQUIRED_FIELDS.to_vec();
        let nulls: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_json_schema_complete(&present, &nulls),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune001_pass_with_extra_fields() {
        let mut present: Vec<&str> = AC_FINETUNE_001_REQUIRED_FIELDS.to_vec();
        present.push("extra_metric_x");
        present.push("debug_info");
        let nulls: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_json_schema_complete(&present, &nulls),
            FinetuneVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: FINETUNE-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn finetune001_fail_missing_status() {
        let present: Vec<&str> = AC_FINETUNE_001_REQUIRED_FIELDS
            .iter()
            .copied()
            .filter(|&f| f != "status")
            .collect();
        let nulls: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_json_schema_complete(&present, &nulls),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune001_fail_missing_epoch_metrics() {
        let present: Vec<&str> = AC_FINETUNE_001_REQUIRED_FIELDS
            .iter()
            .copied()
            .filter(|&f| f != "epoch_metrics")
            .collect();
        assert_eq!(
            verdict_from_json_schema_complete(&present, &[]),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune001_fail_required_field_null() {
        let present: Vec<&str> = AC_FINETUNE_001_REQUIRED_FIELDS.to_vec();
        let nulls = vec!["tokens_per_sec"];
        assert_eq!(
            verdict_from_json_schema_complete(&present, &nulls),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune001_fail_empty() {
        let present: Vec<&str> = vec![];
        let nulls: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_json_schema_complete(&present, &nulls),
            FinetuneVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: FINETUNE-002 — wall_time consistency.
    // -------------------------------------------------------------------------
    #[test]
    fn finetune002_pass_exact_match() {
        // 5000ms = 5.0s
        assert_eq!(
            verdict_from_wall_time_consistency(5.0, 5000),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune002_pass_within_tolerance() {
        // 5000ms = 5.0s, wall_time=5.005 → 0.005 < 0.01 ⇒ Pass.
        assert_eq!(
            verdict_from_wall_time_consistency(5.005, 5000),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune002_pass_long_run() {
        // 1 hour = 3,600,000ms = 3600s.
        assert_eq!(
            verdict_from_wall_time_consistency(3600.0, 3_600_000),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune002_fail_above_tolerance() {
        // wall_time off by 0.5s.
        assert_eq!(
            verdict_from_wall_time_consistency(5.5, 5000),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune002_fail_zero_wall_time() {
        assert_eq!(
            verdict_from_wall_time_consistency(0.0, 5000),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune002_fail_negative() {
        assert_eq!(
            verdict_from_wall_time_consistency(-5.0, 5000),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune002_fail_nan() {
        assert_eq!(
            verdict_from_wall_time_consistency(f64::NAN, 5000),
            FinetuneVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: FINETUNE-003 — epoch_metrics length.
    // -------------------------------------------------------------------------
    #[test]
    fn finetune003_pass_match() {
        assert_eq!(
            verdict_from_epoch_metrics_length(3, 3),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune003_pass_one_epoch() {
        assert_eq!(
            verdict_from_epoch_metrics_length(1, 1),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune003_pass_many_epochs() {
        assert_eq!(
            verdict_from_epoch_metrics_length(100, 100),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune003_fail_off_by_one() {
        assert_eq!(
            verdict_from_epoch_metrics_length(2, 3),
            FinetuneVerdict::Fail
        );
        assert_eq!(
            verdict_from_epoch_metrics_length(4, 3),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune003_fail_total_zero() {
        // Contract precondition: total_epochs >= 1.
        assert_eq!(
            verdict_from_epoch_metrics_length(0, 0),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune003_fail_empty_array() {
        // 0 metrics with non-zero total.
        assert_eq!(
            verdict_from_epoch_metrics_length(0, 3),
            FinetuneVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: FINETUNE-004 — throughput positive.
    // -------------------------------------------------------------------------
    #[test]
    fn finetune004_pass_positive() {
        assert_eq!(
            verdict_from_throughput_positive(125.5),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune004_pass_small_positive() {
        assert_eq!(
            verdict_from_throughput_positive(0.001),
            FinetuneVerdict::Pass
        );
    }

    #[test]
    fn finetune004_fail_zero() {
        // Strict > 0.
        assert_eq!(
            verdict_from_throughput_positive(0.0),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune004_fail_negative() {
        assert_eq!(
            verdict_from_throughput_positive(-1.0),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune004_fail_nan() {
        assert_eq!(
            verdict_from_throughput_positive(f64::NAN),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn finetune004_fail_inf() {
        assert_eq!(
            verdict_from_throughput_positive(f64::INFINITY),
            FinetuneVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Sweep — wall_time around tolerance.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_wall_time_band() {
        // Reference: 1000ms = 1.0s.
        let test_cases: Vec<(f64, FinetuneVerdict)> = vec![
            (1.0, FinetuneVerdict::Pass),
            (1.005, FinetuneVerdict::Pass),
            (0.999, FinetuneVerdict::Pass),
            (1.011, FinetuneVerdict::Fail),
            (0.989, FinetuneVerdict::Fail),
            (5.0, FinetuneVerdict::Fail),
        ];
        for (wall_sec, expected) in test_cases {
            let v = verdict_from_wall_time_consistency(wall_sec, 1000);
            assert_eq!(v, expected, "wall_sec={wall_sec}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_missing_field_caught() {
        // FINETUNE-001 if_fails: "JSON output missing required fields".
        let present = vec!["status", "final_loss"]; // many missing
        assert_eq!(
            verdict_from_json_schema_complete(&present, &[]),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn realistic_wall_time_inconsistent_caught() {
        // FINETUNE-002 if_fails: "wall_time_sec inconsistent with
        // total_time_ms" — bug returns ms instead of sec.
        assert_eq!(
            verdict_from_wall_time_consistency(5000.0, 5000),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn realistic_epoch_array_too_short_caught() {
        // FINETUNE-003 if_fails: "epoch_metrics array length !=
        // total_epochs" — early termination.
        assert_eq!(
            verdict_from_epoch_metrics_length(2, 5),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn realistic_zero_throughput_caught() {
        // FINETUNE-004 if_fails: "tokens_per_sec is zero or negative".
        assert_eq!(
            verdict_from_throughput_positive(0.0),
            FinetuneVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_finetune_output_passes_all_4_gates() {
        // Synthesize a realistic apr finetune --json output.
        let present: Vec<&str> = AC_FINETUNE_001_REQUIRED_FIELDS.to_vec();
        let nulls: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_json_schema_complete(&present, &nulls),
            FinetuneVerdict::Pass
        );
        assert_eq!(
            verdict_from_wall_time_consistency(8.234, 8234),
            FinetuneVerdict::Pass
        );
        assert_eq!(
            verdict_from_epoch_metrics_length(3, 3),
            FinetuneVerdict::Pass
        );
        assert_eq!(
            verdict_from_throughput_positive(1234.5),
            FinetuneVerdict::Pass
        );
    }
}
