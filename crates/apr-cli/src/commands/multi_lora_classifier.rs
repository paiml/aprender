//! CRUX-C-17 — Multi-LoRA batched serving classifiers (S-LoRA / Punica).
//!
//! Pure, deterministic algorithm-level gates discharging FALSIFY-CRUX-C-17-001..004
//! at PARTIAL_ALGORITHM_LEVEL. Full discharge is blocked on `apr serve --enable-lora`
//! + per-request `X-LoRA-Adapter` header + the S-LoRA/Punica batched kernel.
//!
//! Design rule: no silent passes — every ill-formed input maps to a distinct
//! Outcome variant so defect classes cannot collapse into each other.

/// Contract-pinned multi-LoRA throughput floor (80% of base-only batched throughput).
pub const MIN_MULTI_LORA_THROUGHPUT_ALPHA: f64 = 0.80;

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 1: batched ≡ serial per-request parity
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchedParityOutcome {
    Ok,
    CountMismatch { serial_len: usize, batched_len: usize },
    EmptinessMismatch { at_index: usize, serial_empty: bool, batched_empty: bool },
    LengthMismatch { at_index: usize, serial_len: usize, batched_len: usize },
    TokenDivergence {
        request_index: usize,
        at_token_index: usize,
        serial_token: u32,
        batched_token: u32,
    },
}

pub fn classify_batched_parity(
    serial_outputs: &[&[u32]],
    batched_outputs: &[&[u32]],
) -> BatchedParityOutcome {
    if serial_outputs.len() != batched_outputs.len() {
        return BatchedParityOutcome::CountMismatch {
            serial_len: serial_outputs.len(),
            batched_len: batched_outputs.len(),
        };
    }
    for (i, (s, b)) in serial_outputs.iter().zip(batched_outputs.iter()).enumerate() {
        let s_empty = s.is_empty();
        let b_empty = b.is_empty();
        if s_empty != b_empty {
            return BatchedParityOutcome::EmptinessMismatch {
                at_index: i,
                serial_empty: s_empty,
                batched_empty: b_empty,
            };
        }
        if s.len() != b.len() {
            return BatchedParityOutcome::LengthMismatch {
                at_index: i,
                serial_len: s.len(),
                batched_len: b.len(),
            };
        }
        for (j, (sv, bv)) in s.iter().zip(b.iter()).enumerate() {
            if sv != bv {
                return BatchedParityOutcome::TokenDivergence {
                    request_index: i,
                    at_token_index: j,
                    serial_token: *sv,
                    batched_token: *bv,
                };
            }
        }
    }
    BatchedParityOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 2: multi-LoRA throughput floor (80% of base)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MultiLoraThroughputOutcome {
    Ok { observed_alpha: f64 },
    InvalidInput { reason: &'static str },
    BelowThreshold { observed_alpha: f64, required_alpha: f64 },
}

pub fn classify_multi_lora_throughput(
    base_tps: f64,
    multi_tps: f64,
    min_alpha: f64,
) -> MultiLoraThroughputOutcome {
    if !base_tps.is_finite() || !multi_tps.is_finite() || !min_alpha.is_finite() {
        return MultiLoraThroughputOutcome::InvalidInput { reason: "non-finite input" };
    }
    if base_tps <= 0.0 {
        return MultiLoraThroughputOutcome::InvalidInput { reason: "base_tps <= 0" };
    }
    if multi_tps < 0.0 {
        return MultiLoraThroughputOutcome::InvalidInput { reason: "multi_tps < 0" };
    }
    if !(0.0..=1.0).contains(&min_alpha) {
        return MultiLoraThroughputOutcome::InvalidInput {
            reason: "min_alpha out of [0.0, 1.0]",
        };
    }
    let observed_alpha = multi_tps / base_tps;
    if observed_alpha < min_alpha {
        return MultiLoraThroughputOutcome::BelowThreshold {
            observed_alpha,
            required_alpha: min_alpha,
        };
    }
    MultiLoraThroughputOutcome::Ok { observed_alpha }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 3: unknown-adapter HTTP status discipline
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownAdapterResponseOutcome {
    Ok,
    EmptyAdapterName,
    AdapterIsLoaded { adapter_name: String },
    WrongStatusCode { got: u16, expected: u16 },
    MissingNameInBody,
}

pub fn classify_unknown_adapter_response(
    requested_adapter: &str,
    loaded_adapters: &[&str],
    status_code: u16,
    error_body: &str,
) -> UnknownAdapterResponseOutcome {
    if requested_adapter.is_empty() {
        return UnknownAdapterResponseOutcome::EmptyAdapterName;
    }
    // If adapter is in fact loaded, the 404 discipline doesn't apply —
    // calling this classifier would be a test-harness bug.
    if loaded_adapters.iter().any(|a| *a == requested_adapter) {
        return UnknownAdapterResponseOutcome::AdapterIsLoaded {
            adapter_name: requested_adapter.to_string(),
        };
    }
    if status_code != 404 {
        return UnknownAdapterResponseOutcome::WrongStatusCode {
            got: status_code,
            expected: 404,
        };
    }
    if !error_body.contains(requested_adapter) {
        return UnknownAdapterResponseOutcome::MissingNameInBody;
    }
    UnknownAdapterResponseOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 4: max_loras capacity discipline
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxLorasCapacityOutcome {
    Ok,
    ZeroMaxLoras,
    AcceptedWithinCapacity { loaded_count: u32, max_loras: u32 },
    WrongStatusCode { got: u16, allowed: &'static [u16] },
}

/// Valid HTTP status codes for "over-capacity" rejection per contract v1.1.0:
/// 429 Too Many Requests or 503 Service Unavailable.
pub const OVER_CAPACITY_STATUS_CODES: &[u16] = &[429, 503];

pub fn classify_max_loras_capacity(
    loaded_count: u32,
    max_loras: u32,
    status_code: u16,
) -> MaxLorasCapacityOutcome {
    if max_loras == 0 {
        return MaxLorasCapacityOutcome::ZeroMaxLoras;
    }
    if loaded_count < max_loras {
        // Request to load another adapter should succeed (200/202), not be
        // rejected as over-capacity.
        return MaxLorasCapacityOutcome::AcceptedWithinCapacity {
            loaded_count,
            max_loras,
        };
    }
    // At/above capacity — MUST respond with 429 or 503.
    if !OVER_CAPACITY_STATUS_CODES.contains(&status_code) {
        return MaxLorasCapacityOutcome::WrongStatusCode {
            got: status_code,
            allowed: OVER_CAPACITY_STATUS_CODES,
        };
    }
    MaxLorasCapacityOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — pure, deterministic.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
            BatchedParityOutcome::CountMismatch { serial_len: 2, batched_len: 1 }
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
            MultiLoraThroughputOutcome::BelowThreshold { observed_alpha, required_alpha } => {
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
            UnknownAdapterResponseOutcome::WrongStatusCode { got: 500, expected: 404 }
        );
    }

    #[test]
    fn unknown_adapter_rejects_silent_fallback_200() {
        // Silent fallback to base model — a real defect, not a 404.
        assert_eq!(
            classify_unknown_adapter_response("missing", &["a1"], 200, "ok"),
            UnknownAdapterResponseOutcome::WrongStatusCode { got: 200, expected: 404 }
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
}
