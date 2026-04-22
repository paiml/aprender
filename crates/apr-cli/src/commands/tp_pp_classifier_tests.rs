//! Unit tests for `tp_pp_classifier` (extracted from `tp_pp_classifier.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "tp_pp_classifier_tests.rs"] mod tests;` in the parent.

use super::*;
use serde_json::json;

// ── world_size ─────────────────────────────────────────────────────────

#[test]
fn world_size_ok_on_tp_times_pp() {
    assert_eq!(classify_world_size(2, 4, 8), WorldSizeOutcome::Ok);
    assert_eq!(classify_world_size(1, 1, 1), WorldSizeOutcome::Ok);
    assert_eq!(classify_world_size(8, 1, 8), WorldSizeOutcome::Ok);
}

#[test]
fn world_size_rejects_zero_tp() {
    assert_eq!(classify_world_size(0, 1, 0), WorldSizeOutcome::ZeroTp);
}

#[test]
fn world_size_rejects_zero_pp() {
    assert_eq!(classify_world_size(2, 0, 0), WorldSizeOutcome::ZeroPp);
}

#[test]
fn world_size_rejects_mismatch() {
    assert_eq!(
        classify_world_size(2, 4, 16),
        WorldSizeOutcome::Mismatch {
            tp: 2,
            pp: 4,
            expected: 8,
            got: 16
        }
    );
}

#[test]
fn world_size_classifier_is_deterministic() {
    for _ in 0..5 {
        assert_eq!(classify_world_size(4, 2, 8), WorldSizeOutcome::Ok);
    }
}

// ── divisibility ───────────────────────────────────────────────────────

#[test]
fn divisibility_ok_on_aligned_config() {
    assert_eq!(classify_divisibility(32, 32, 2, 2), DivisibilityOutcome::Ok);
    assert_eq!(classify_divisibility(32, 32, 8, 1), DivisibilityOutcome::Ok);
}

#[test]
fn divisibility_rejects_zero_tp() {
    assert_eq!(
        classify_divisibility(32, 32, 0, 1),
        DivisibilityOutcome::ZeroTp
    );
}

#[test]
fn divisibility_rejects_zero_pp() {
    assert_eq!(
        classify_divisibility(32, 32, 2, 0),
        DivisibilityOutcome::ZeroPp
    );
}

#[test]
fn divisibility_rejects_zero_num_heads() {
    assert_eq!(
        classify_divisibility(0, 32, 2, 2),
        DivisibilityOutcome::ZeroNumHeads
    );
}

#[test]
fn divisibility_rejects_zero_num_layers() {
    assert_eq!(
        classify_divisibility(32, 0, 2, 2),
        DivisibilityOutcome::ZeroNumLayers
    );
}

#[test]
fn divisibility_rejects_heads_not_divisible() {
    assert_eq!(
        classify_divisibility(32, 32, 3, 1),
        DivisibilityOutcome::HeadsNotDivisible {
            num_heads: 32,
            tp: 3
        }
    );
}

#[test]
fn divisibility_rejects_layers_not_divisible() {
    assert_eq!(
        classify_divisibility(32, 30, 2, 4),
        DivisibilityOutcome::LayersNotDivisible {
            num_layers: 30,
            pp: 4
        }
    );
}

#[test]
fn divisibility_classifier_is_deterministic() {
    for _ in 0..5 {
        assert_eq!(classify_divisibility(32, 32, 4, 2), DivisibilityOutcome::Ok);
    }
}

// ── tp parity ──────────────────────────────────────────────────────────

#[test]
fn tp_parity_ok_on_identical_token_streams() {
    assert_eq!(
        classify_tp_parity(&[1, 2, 3, 4], &[1, 2, 3, 4]),
        TpParityOutcome::Ok
    );
}

#[test]
fn tp_parity_ok_on_two_empty() {
    assert_eq!(classify_tp_parity(&[], &[]), TpParityOutcome::Ok);
}

#[test]
fn tp_parity_rejects_emptiness_mismatch_ref_empty() {
    assert_eq!(
        classify_tp_parity(&[], &[1]),
        TpParityOutcome::EmptinessMismatch {
            ref_empty: true,
            parallel_empty: false
        }
    );
}

#[test]
fn tp_parity_rejects_emptiness_mismatch_parallel_empty() {
    assert_eq!(
        classify_tp_parity(&[1], &[]),
        TpParityOutcome::EmptinessMismatch {
            ref_empty: false,
            parallel_empty: true
        }
    );
}

#[test]
fn tp_parity_rejects_length_mismatch() {
    assert_eq!(
        classify_tp_parity(&[1, 2, 3], &[1, 2, 3, 4]),
        TpParityOutcome::LengthMismatch {
            ref_len: 3,
            parallel_len: 4
        }
    );
}

#[test]
fn tp_parity_rejects_token_divergence_at_first_mismatch() {
    assert_eq!(
        classify_tp_parity(&[1, 2, 3, 4], &[1, 2, 9, 4]),
        TpParityOutcome::TokenDivergence {
            at_index: 2,
            ref_token: 3,
            parallel_token: 9,
        }
    );
}

#[test]
fn tp_parity_reports_first_divergence_not_last() {
    // Two divergences: indices 1 and 3 — must report index 1.
    assert_eq!(
        classify_tp_parity(&[0, 1, 2, 3], &[0, 9, 2, 9]),
        TpParityOutcome::TokenDivergence {
            at_index: 1,
            ref_token: 1,
            parallel_token: 9,
        }
    );
}

#[test]
fn tp_parity_classifier_is_deterministic() {
    for _ in 0..5 {
        assert_eq!(
            classify_tp_parity(&[1, 2, 3], &[1, 2, 3]),
            TpParityOutcome::Ok
        );
    }
}

// ── scaling efficiency ─────────────────────────────────────────────────

#[test]
fn scaling_ok_on_near_linear_uplift() {
    match classify_scaling_efficiency(100.0, 180.0, 2, 0.70) {
        ScalingEfficiencyOutcome::Ok { observed_alpha } => {
            assert!((observed_alpha - 0.90).abs() < 1e-9);
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn scaling_ok_exactly_at_threshold() {
    match classify_scaling_efficiency(100.0, 140.0, 2, 0.70) {
        ScalingEfficiencyOutcome::Ok { observed_alpha } => {
            assert!((observed_alpha - 0.70).abs() < 1e-9);
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_regression() {
    match classify_scaling_efficiency(100.0, 90.0, 2, 0.70) {
        ScalingEfficiencyOutcome::Regression {
            base_tps,
            parallel_tps,
            ..
        } => {
            assert_eq!(base_tps, 100.0);
            assert_eq!(parallel_tps, 90.0);
        }
        other => panic!("expected Regression, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_below_threshold() {
    match classify_scaling_efficiency(100.0, 120.0, 2, 0.70) {
        ScalingEfficiencyOutcome::BelowThreshold {
            observed_alpha,
            required_alpha,
            tp,
        } => {
            assert!((observed_alpha - 0.60).abs() < 1e-9);
            assert_eq!(required_alpha, 0.70);
            assert_eq!(tp, 2);
        }
        other => panic!("expected BelowThreshold, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_zero_base_tps() {
    match classify_scaling_efficiency(0.0, 1.0, 2, 0.70) {
        ScalingEfficiencyOutcome::InvalidInput { reason } => {
            assert!(reason.contains("base_tps"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_negative_parallel_tps() {
    match classify_scaling_efficiency(100.0, -1.0, 2, 0.70) {
        ScalingEfficiencyOutcome::InvalidInput { reason } => {
            assert!(reason.contains("parallel_tps"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_tp_less_than_2() {
    match classify_scaling_efficiency(100.0, 120.0, 1, 0.70) {
        ScalingEfficiencyOutcome::InvalidInput { reason } => {
            assert!(reason.contains("tp"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_min_alpha_out_of_range() {
    match classify_scaling_efficiency(100.0, 200.0, 2, 1.5) {
        ScalingEfficiencyOutcome::InvalidInput { reason } => {
            assert!(reason.contains("min_alpha"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn scaling_rejects_nan_input() {
    match classify_scaling_efficiency(f64::NAN, 1.0, 2, 0.70) {
        ScalingEfficiencyOutcome::InvalidInput { reason } => {
            assert!(reason.contains("non-finite"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn scaling_classifier_is_deterministic() {
    for _ in 0..5 {
        match classify_scaling_efficiency(100.0, 180.0, 2, 0.70) {
            ScalingEfficiencyOutcome::Ok { .. } => {}
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}

// ── distributed metadata ───────────────────────────────────────────────

#[test]
fn distributed_metadata_ok_on_well_formed() {
    let v = json!({"distributed": {"tp": 2, "pp": 2, "world_size": 4}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::Ok {
            tp: 2,
            pp: 2,
            world_size: 4
        }
    );
}

#[test]
fn distributed_metadata_rejects_non_object_root() {
    let v = json!([1, 2, 3]);
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::NotObject
    );
}

#[test]
fn distributed_metadata_rejects_missing_distributed() {
    let v = json!({"model": "x"});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::MissingDistributed
    );
}

#[test]
fn distributed_metadata_rejects_non_object_distributed() {
    let v = json!({"distributed": "wrong"});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::DistributedNotObject
    );
}

#[test]
fn distributed_metadata_rejects_missing_tp() {
    let v = json!({"distributed": {"pp": 2, "world_size": 2}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::MissingField { field: "tp" }
    );
}

#[test]
fn distributed_metadata_rejects_missing_pp() {
    let v = json!({"distributed": {"tp": 2, "world_size": 2}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::MissingField { field: "pp" }
    );
}

#[test]
fn distributed_metadata_rejects_missing_world_size() {
    let v = json!({"distributed": {"tp": 2, "pp": 2}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::MissingField {
            field: "world_size"
        }
    );
}

#[test]
fn distributed_metadata_rejects_string_tp() {
    let v = json!({"distributed": {"tp": "2", "pp": 2, "world_size": 4}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::TypeMismatch { field: "tp" }
    );
}

#[test]
fn distributed_metadata_rejects_negative_tp() {
    let v = json!({"distributed": {"tp": -1, "pp": 2, "world_size": 4}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::TypeMismatch { field: "tp" }
    );
}

#[test]
fn distributed_metadata_rejects_world_size_mismatch() {
    let v = json!({"distributed": {"tp": 2, "pp": 2, "world_size": 8}});
    assert_eq!(
        classify_distributed_metadata(&v),
        DistributedMetadataOutcome::WorldSizeMismatch {
            tp: 2,
            pp: 2,
            got: 8
        }
    );
}

#[test]
fn distributed_metadata_classifier_is_deterministic() {
    let v = json!({"distributed": {"tp": 4, "pp": 1, "world_size": 4}});
    for _ in 0..5 {
        assert_eq!(
            classify_distributed_metadata(&v),
            DistributedMetadataOutcome::Ok {
                tp: 4,
                pp: 1,
                world_size: 4
            }
        );
    }
}

// ── constants invariants ───────────────────────────────────────────────

#[test]
fn scaling_constants_are_canonical() {
    assert!(DIVISIBILITY_REQUIRED);
    assert!((MIN_TP_SCALING_ALPHA - 0.70).abs() < 1e-9);
}
