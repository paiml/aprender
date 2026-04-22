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
    CountMismatch {
        serial_len: usize,
        batched_len: usize,
    },
    EmptinessMismatch {
        at_index: usize,
        serial_empty: bool,
        batched_empty: bool,
    },
    LengthMismatch {
        at_index: usize,
        serial_len: usize,
        batched_len: usize,
    },
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
    for (i, (s, b)) in serial_outputs
        .iter()
        .zip(batched_outputs.iter())
        .enumerate()
    {
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
    Ok {
        observed_alpha: f64,
    },
    InvalidInput {
        reason: &'static str,
    },
    BelowThreshold {
        observed_alpha: f64,
        required_alpha: f64,
    },
}

pub fn classify_multi_lora_throughput(
    base_tps: f64,
    multi_tps: f64,
    min_alpha: f64,
) -> MultiLoraThroughputOutcome {
    if !base_tps.is_finite() || !multi_tps.is_finite() || !min_alpha.is_finite() {
        return MultiLoraThroughputOutcome::InvalidInput {
            reason: "non-finite input",
        };
    }
    if base_tps <= 0.0 {
        return MultiLoraThroughputOutcome::InvalidInput {
            reason: "base_tps <= 0",
        };
    }
    if multi_tps < 0.0 {
        return MultiLoraThroughputOutcome::InvalidInput {
            reason: "multi_tps < 0",
        };
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
#[path = "multi_lora_classifier_tests.rs"]
mod tests;
