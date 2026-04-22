//! Unit tests for `multi_lora_classifier` (extracted from `multi_lora_classifier.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "multi_lora_classifier_tests.rs"] mod tests;` in the parent.

use super::*;

// ── batched parity ─────────────────────────────────────────────────────

#[test]
fn batched_parity_ok_on_identical_per_request() {
    let s: Vec<&[u32]> = vec![&[1, 2, 3], &[4, 5], &[6]];
    let b: Vec<&[u32]> = vec![&[1, 2, 3], &[4, 5], &[6]];
    assert_eq!(classify_batched_parity(&s, &b), BatchedParityOutcome::Ok);
}

#[test]
fn batched_parity_ok_on_two_empty_vectors() {
    let s: Vec<&[u32]> = vec![];
    let b: Vec<&[u32]> = vec![];
    assert_eq!(classify_batched_parity(&s, &b), BatchedParityOutcome::Ok);
}

#[test]
fn batched_parity_rejects_count_mismatch() {
    let s: Vec<&[u32]> = vec![&[1], &[2]];
    let b: Vec<&[u32]> = vec![&[1]];
    assert_eq!(
        classify_batched_parity(&s, &b),
        BatchedParityOutcome::CountMismatch {
            serial_len: 2,
            batched_len: 1
        }
    );
}

#[test]
fn batched_parity_rejects_per_request_emptiness_mismatch() {
    let s: Vec<&[u32]> = vec![&[1], &[]];
    let b: Vec<&[u32]> = vec![&[1], &[7]];
    assert_eq!(
        classify_batched_parity(&s, &b),
        BatchedParityOutcome::EmptinessMismatch {
            at_index: 1,
            serial_empty: true,
            batched_empty: false,
        }
    );
}

#[test]
fn batched_parity_rejects_per_request_length_mismatch() {
    let s: Vec<&[u32]> = vec![&[1, 2], &[3]];
    let b: Vec<&[u32]> = vec![&[1, 2], &[3, 4]];
    assert_eq!(
        classify_batched_parity(&s, &b),
        BatchedParityOutcome::LengthMismatch {
            at_index: 1,
            serial_len: 1,
            batched_len: 2,
        }
    );
}

#[test]
fn batched_parity_rejects_cross_contamination_first_index() {
    // Request 0 matches; request 1 diverges at token 0. Report request 1.
    let s: Vec<&[u32]> = vec![&[10, 20], &[30, 40]];
    let b: Vec<&[u32]> = vec![&[10, 20], &[99, 40]];
    assert_eq!(
        classify_batched_parity(&s, &b),
        BatchedParityOutcome::TokenDivergence {
            request_index: 1,
            at_token_index: 0,
            serial_token: 30,
            batched_token: 99,
        }
    );
}

#[test]
fn batched_parity_is_deterministic() {
    let s: Vec<&[u32]> = vec![&[1, 2], &[3, 4]];
    let b: Vec<&[u32]> = vec![&[1, 2], &[3, 4]];
    for _ in 0..5 {
        assert_eq!(classify_batched_parity(&s, &b), BatchedParityOutcome::Ok);
    }
}

// ── throughput floor ───────────────────────────────────────────────────

#[test]
fn multi_lora_throughput_ok_above_floor() {
    match classify_multi_lora_throughput(100.0, 85.0, 0.80) {
        MultiLoraThroughputOutcome::Ok { observed_alpha } => {
            assert!((observed_alpha - 0.85).abs() < 1e-9);
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_ok_exactly_at_floor() {
    match classify_multi_lora_throughput(100.0, 80.0, 0.80) {
        MultiLoraThroughputOutcome::Ok { observed_alpha } => {
            assert!((observed_alpha - 0.80).abs() < 1e-9);
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_rejects_below_floor() {
    match classify_multi_lora_throughput(100.0, 70.0, 0.80) {
        MultiLoraThroughputOutcome::BelowThreshold {
            observed_alpha,
            required_alpha,
        } => {
            assert!((observed_alpha - 0.70).abs() < 1e-9);
            assert_eq!(required_alpha, 0.80);
        }
        other => panic!("expected BelowThreshold, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_rejects_nan() {
    match classify_multi_lora_throughput(f64::NAN, 100.0, 0.80) {
        MultiLoraThroughputOutcome::InvalidInput { reason } => {
            assert!(reason.contains("non-finite"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_rejects_zero_base() {
    match classify_multi_lora_throughput(0.0, 10.0, 0.80) {
        MultiLoraThroughputOutcome::InvalidInput { reason } => {
            assert!(reason.contains("base_tps"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_rejects_negative_multi() {
    match classify_multi_lora_throughput(100.0, -1.0, 0.80) {
        MultiLoraThroughputOutcome::InvalidInput { reason } => {
            assert!(reason.contains("multi_tps"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_rejects_alpha_out_of_range() {
    match classify_multi_lora_throughput(100.0, 100.0, 1.5) {
        MultiLoraThroughputOutcome::InvalidInput { reason } => {
            assert!(reason.contains("min_alpha"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn multi_lora_throughput_is_deterministic() {
    for _ in 0..5 {
        match classify_multi_lora_throughput(100.0, 90.0, 0.80) {
            MultiLoraThroughputOutcome::Ok { .. } => {}
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}

// ── unknown-adapter response ───────────────────────────────────────────

#[test]
fn unknown_adapter_ok_on_404_with_name_in_body() {
    assert_eq!(
        classify_unknown_adapter_response(
            "missing",
            &["a1", "a2"],
            404,
            "adapter 'missing' not found",
        ),
        UnknownAdapterResponseOutcome::Ok
    );
}

#[test]
fn unknown_adapter_rejects_empty_name() {
    assert_eq!(
        classify_unknown_adapter_response("", &["a1"], 404, "not found"),
        UnknownAdapterResponseOutcome::EmptyAdapterName
    );
}

#[test]
fn unknown_adapter_rejects_when_actually_loaded() {
    match classify_unknown_adapter_response("a1", &["a1", "a2"], 404, "not found") {
        UnknownAdapterResponseOutcome::AdapterIsLoaded { adapter_name } => {
            assert_eq!(adapter_name, "a1");
        }
        other => panic!("expected AdapterIsLoaded, got {other:?}"),
    }
}

#[test]
fn unknown_adapter_rejects_wrong_status_500() {
    assert_eq!(
        classify_unknown_adapter_response("missing", &["a1"], 500, "server error"),
        UnknownAdapterResponseOutcome::WrongStatusCode {
            got: 500,
            expected: 404
        }
    );
}

#[test]
fn unknown_adapter_rejects_silent_fallback_200() {
    // Silent fallback to base model — a real defect, not a 404.
    assert_eq!(
        classify_unknown_adapter_response("missing", &["a1"], 200, "ok"),
        UnknownAdapterResponseOutcome::WrongStatusCode {
            got: 200,
            expected: 404
        }
    );
}

#[test]
fn unknown_adapter_rejects_missing_name_in_body() {
    assert_eq!(
        classify_unknown_adapter_response("missing", &["a1"], 404, "not found"),
        UnknownAdapterResponseOutcome::MissingNameInBody
    );
}

#[test]
fn unknown_adapter_is_deterministic() {
    for _ in 0..5 {
        assert_eq!(
            classify_unknown_adapter_response(
                "missing",
                &["a1"],
                404,
                "adapter 'missing' not found"
            ),
            UnknownAdapterResponseOutcome::Ok
        );
    }
}

// ── max_loras capacity ─────────────────────────────────────────────────

#[test]
fn max_loras_ok_at_capacity_with_429() {
    assert_eq!(
        classify_max_loras_capacity(2, 2, 429),
        MaxLorasCapacityOutcome::Ok
    );
}

#[test]
fn max_loras_ok_at_capacity_with_503() {
    assert_eq!(
        classify_max_loras_capacity(2, 2, 503),
        MaxLorasCapacityOutcome::Ok
    );
}

#[test]
fn max_loras_rejects_zero_max() {
    assert_eq!(
        classify_max_loras_capacity(0, 0, 429),
        MaxLorasCapacityOutcome::ZeroMaxLoras
    );
}

#[test]
fn max_loras_rejects_rejected_while_within_capacity() {
    assert_eq!(
        classify_max_loras_capacity(1, 2, 429),
        MaxLorasCapacityOutcome::AcceptedWithinCapacity {
            loaded_count: 1,
            max_loras: 2,
        }
    );
}

#[test]
fn max_loras_rejects_wrong_status_200_at_capacity() {
    // Silent acceptance above capacity — OOM risk.
    assert_eq!(
        classify_max_loras_capacity(2, 2, 200),
        MaxLorasCapacityOutcome::WrongStatusCode {
            got: 200,
            allowed: OVER_CAPACITY_STATUS_CODES,
        }
    );
}

#[test]
fn max_loras_rejects_wrong_status_500_at_capacity() {
    assert_eq!(
        classify_max_loras_capacity(3, 2, 500),
        MaxLorasCapacityOutcome::WrongStatusCode {
            got: 500,
            allowed: OVER_CAPACITY_STATUS_CODES,
        }
    );
}

#[test]
fn max_loras_is_deterministic() {
    for _ in 0..5 {
        assert_eq!(
            classify_max_loras_capacity(5, 5, 503),
            MaxLorasCapacityOutcome::Ok
        );
    }
}

// ── constants ──────────────────────────────────────────────────────────

#[test]
fn multi_lora_constants_are_canonical() {
    assert!((MIN_MULTI_LORA_THROUGHPUT_ALPHA - 0.80).abs() < 1e-9);
    assert_eq!(OVER_CAPACITY_STATUS_CODES, &[429, 503]);
}
