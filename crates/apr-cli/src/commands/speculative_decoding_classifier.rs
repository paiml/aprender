//! Speculative-decoding parity + uplift + compatibility classifier (CRUX-C-09).
//!
//! Four pure, deterministic classifiers that discharge FALSIFY-CRUX-C-09-{001..004}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! already-captured speculative-decoding observations:
//!
//!   * `classify_speculative_parity` — at temperature=0 top_k=1, the
//!     speculative and target-only decode paths emit byte-identical token
//!     sequences.
//!   * `classify_throughput_uplift` — at a fixed K, `spec_tps >= base_tps *
//!     (1 + alpha_min)`; we never accept a regression as "uplift".
//!   * `classify_tokenizer_compatibility` — before the first decode step,
//!     the draft and target tokenizers must have equal sha256 and equal
//!     vocab_size; mismatches are rejected with a specific reason.
//!   * `classify_acceptance_rate` — `--json` output contains a
//!     `speculative.acceptance_rate` field, numeric, in `[0.0, 1.0]`.
//!
//! Full discharge blocks on a live `apr serve --draft-model` / `apr run
//! --draft-model` surface and an extended `--json` schema emitting
//! `speculative.acceptance_rate`.

use serde_json::Value;

/// vLLM default value of K (speculative tokens per step).
pub const DEFAULT_SPEC_TOKENS_K: u32 = 5;
/// Minimum uplift alpha (30%) required on code/math workloads at K=5.
pub const MIN_SPEC_UPLIFT_ALPHA: f64 = 0.30;
/// Allowed range for `spec_tokens` per vLLM's SpecDecodeWorker.
pub const SPEC_TOKENS_MIN: u32 = 1;
pub const SPEC_TOKENS_MAX: u32 = 16;

/// Outcome of `classify_speculative_parity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecParityOutcome {
    /// Token sequences are byte-identical (same length, same IDs in order).
    Ok,
    /// Different lengths — spec path truncated or over-emitted.
    LengthMismatch { base_len: usize, spec_len: usize },
    /// Same length but a divergent token ID at some position.
    TokenDivergence {
        at_index: usize,
        base_token: u32,
        spec_token: u32,
    },
}

/// Greedy parity gate: `temp==0.0` `top_k==1` requires byte-identical token
/// sequences between the base (target-only) and speculative paths.
pub fn classify_speculative_parity(base: &[u32], spec: &[u32]) -> SpecParityOutcome {
    if base.len() != spec.len() {
        return SpecParityOutcome::LengthMismatch {
            base_len: base.len(),
            spec_len: spec.len(),
        };
    }
    for (i, (&b, &s)) in base.iter().zip(spec.iter()).enumerate() {
        if b != s {
            return SpecParityOutcome::TokenDivergence {
                at_index: i,
                base_token: b,
                spec_token: s,
            };
        }
    }
    SpecParityOutcome::Ok
}

/// Outcome of `classify_throughput_uplift`.
#[derive(Debug, Clone, PartialEq)]
pub enum ThroughputUpliftOutcome {
    /// Observed alpha ≥ `alpha_min`. Reports the observed alpha.
    Ok { observed_alpha: f64 },
    /// Observed alpha < `alpha_min` but ≥ 0.
    BelowThreshold {
        observed_alpha: f64,
        required_alpha: f64,
    },
    /// `spec_tps < base_tps` — a regression, never allowed.
    Regression {
        base_tps: f64,
        spec_tps: f64,
        observed_alpha: f64,
    },
    /// Inputs are non-finite or non-positive.
    InvalidInput { reason: &'static str },
}

/// Throughput-uplift gate. `spec_tps / base_tps >= 1 + alpha_min`, with the
/// explicit regression case split out so "barely-positive" never masks a
/// genuine regression.
pub fn classify_throughput_uplift(
    base_tps: f64,
    spec_tps: f64,
    alpha_min: f64,
) -> ThroughputUpliftOutcome {
    if !base_tps.is_finite() || !spec_tps.is_finite() || !alpha_min.is_finite() {
        return ThroughputUpliftOutcome::InvalidInput {
            reason: "non-finite input",
        };
    }
    if base_tps <= 0.0 {
        return ThroughputUpliftOutcome::InvalidInput {
            reason: "base_tps must be positive",
        };
    }
    if spec_tps < 0.0 {
        return ThroughputUpliftOutcome::InvalidInput {
            reason: "spec_tps must be non-negative",
        };
    }
    if alpha_min < 0.0 {
        return ThroughputUpliftOutcome::InvalidInput {
            reason: "alpha_min must be non-negative",
        };
    }
    let observed_alpha = spec_tps / base_tps - 1.0;
    if spec_tps < base_tps {
        return ThroughputUpliftOutcome::Regression {
            base_tps,
            spec_tps,
            observed_alpha,
        };
    }
    if observed_alpha + f64::EPSILON < alpha_min {
        return ThroughputUpliftOutcome::BelowThreshold {
            observed_alpha,
            required_alpha: alpha_min,
        };
    }
    ThroughputUpliftOutcome::Ok { observed_alpha }
}

/// Outcome of `classify_tokenizer_compatibility`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerCompatOutcome {
    /// sha256 and vocab_size match.
    Ok,
    /// Tokenizer hash mismatch — drafts cannot tokenize input consistently.
    TokenizerShaMismatch,
    /// Vocab size mismatch — draft logits can't remap onto target IDs.
    VocabSizeMismatch { draft: u32, target: u32 },
    /// One or both sha256 values are empty / obviously malformed.
    MalformedSha { reason: &'static str },
    /// Vocab size was reported as zero.
    ZeroVocab { which: &'static str },
}

/// Compatibility gate: draft+target must share tokenizer sha256 AND vocab size.
pub fn classify_tokenizer_compatibility(
    draft_tokenizer_sha256: &str,
    target_tokenizer_sha256: &str,
    draft_vocab_size: u32,
    target_vocab_size: u32,
) -> TokenizerCompatOutcome {
    if draft_tokenizer_sha256.is_empty() || target_tokenizer_sha256.is_empty() {
        return TokenizerCompatOutcome::MalformedSha {
            reason: "empty sha256",
        };
    }
    if draft_vocab_size == 0 {
        return TokenizerCompatOutcome::ZeroVocab { which: "draft" };
    }
    if target_vocab_size == 0 {
        return TokenizerCompatOutcome::ZeroVocab { which: "target" };
    }
    if !draft_tokenizer_sha256.eq_ignore_ascii_case(target_tokenizer_sha256) {
        return TokenizerCompatOutcome::TokenizerShaMismatch;
    }
    if draft_vocab_size != target_vocab_size {
        return TokenizerCompatOutcome::VocabSizeMismatch {
            draft: draft_vocab_size,
            target: target_vocab_size,
        };
    }
    TokenizerCompatOutcome::Ok
}

/// Outcome of `classify_acceptance_rate`.
#[derive(Debug, Clone, PartialEq)]
pub enum AcceptanceRateOutcome {
    /// `speculative.acceptance_rate` is present, numeric, and within `[0,1]`.
    Ok { rate: f64 },
    /// Top-level response is not a JSON object.
    NotAnObject,
    /// `speculative` key is absent.
    MissingSpeculative,
    /// `speculative` is present but not an object.
    SpeculativeNotAnObject,
    /// `speculative.acceptance_rate` is absent.
    MissingAcceptanceRate,
    /// `speculative.acceptance_rate` is present but not a number.
    AcceptanceRateNotNumeric,
    /// Value is a number but outside `[0.0, 1.0]`.
    OutOfRange { value: f64 },
    /// Value is NaN (parses as number but unusable).
    NaN,
}

/// `--json` output must include `speculative.acceptance_rate` ∈ `[0, 1]`.
pub fn classify_acceptance_rate(json: &Value) -> AcceptanceRateOutcome {
    let obj = match json.as_object() {
        Some(o) => o,
        None => return AcceptanceRateOutcome::NotAnObject,
    };
    let spec = match obj.get("speculative") {
        Some(v) => v,
        None => return AcceptanceRateOutcome::MissingSpeculative,
    };
    let spec_obj = match spec.as_object() {
        Some(o) => o,
        None => return AcceptanceRateOutcome::SpeculativeNotAnObject,
    };
    let ar_val = match spec_obj.get("acceptance_rate") {
        Some(v) => v,
        None => return AcceptanceRateOutcome::MissingAcceptanceRate,
    };
    let rate = match ar_val.as_f64() {
        Some(f) => f,
        None => return AcceptanceRateOutcome::AcceptanceRateNotNumeric,
    };
    if rate.is_nan() {
        return AcceptanceRateOutcome::NaN;
    }
    if !(0.0..=1.0).contains(&rate) {
        return AcceptanceRateOutcome::OutOfRange { value: rate };
    }
    AcceptanceRateOutcome::Ok { rate }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- parity -----------------------------------------------------------

    #[test]
    fn parity_ok_on_identical_sequences() {
        let base = [1, 2, 3, 4, 5];
        let spec = [1, 2, 3, 4, 5];
        assert_eq!(
            classify_speculative_parity(&base, &spec),
            SpecParityOutcome::Ok
        );
    }

    #[test]
    fn parity_ok_on_two_empty_sequences() {
        assert_eq!(
            classify_speculative_parity(&[], &[]),
            SpecParityOutcome::Ok
        );
    }

    #[test]
    fn parity_rejects_length_mismatch() {
        assert_eq!(
            classify_speculative_parity(&[1, 2, 3], &[1, 2]),
            SpecParityOutcome::LengthMismatch {
                base_len: 3,
                spec_len: 2
            }
        );
    }

    #[test]
    fn parity_rejects_token_divergence() {
        assert_eq!(
            classify_speculative_parity(&[1, 2, 3, 4], &[1, 2, 99, 4]),
            SpecParityOutcome::TokenDivergence {
                at_index: 2,
                base_token: 3,
                spec_token: 99,
            }
        );
    }

    #[test]
    fn parity_classifier_is_deterministic() {
        let base = [5, 6, 7];
        let spec = [5, 6, 7];
        let a = classify_speculative_parity(&base, &spec);
        let b = classify_speculative_parity(&base, &spec);
        assert_eq!(a, b);
    }

    // ---- throughput uplift ------------------------------------------------

    #[test]
    fn uplift_ok_at_exactly_30_percent() {
        let r = classify_throughput_uplift(100.0, 130.0, MIN_SPEC_UPLIFT_ALPHA);
        match r {
            ThroughputUpliftOutcome::Ok { observed_alpha } => {
                assert!((observed_alpha - 0.30).abs() < 1e-9);
            }
            _ => panic!("expected Ok, got {r:?}"),
        }
    }

    #[test]
    fn uplift_ok_above_threshold() {
        let r = classify_throughput_uplift(100.0, 200.0, MIN_SPEC_UPLIFT_ALPHA);
        assert!(matches!(r, ThroughputUpliftOutcome::Ok { .. }));
    }

    #[test]
    fn uplift_below_threshold_rejected() {
        let r = classify_throughput_uplift(100.0, 120.0, MIN_SPEC_UPLIFT_ALPHA);
        match r {
            ThroughputUpliftOutcome::BelowThreshold {
                observed_alpha,
                required_alpha,
            } => {
                assert!((observed_alpha - 0.20).abs() < 1e-9);
                assert!((required_alpha - 0.30).abs() < 1e-9);
            }
            _ => panic!("expected BelowThreshold, got {r:?}"),
        }
    }

    #[test]
    fn uplift_regression_is_never_ok() {
        let r = classify_throughput_uplift(100.0, 80.0, MIN_SPEC_UPLIFT_ALPHA);
        match r {
            ThroughputUpliftOutcome::Regression {
                base_tps,
                spec_tps,
                observed_alpha,
            } => {
                assert!((base_tps - 100.0).abs() < 1e-9);
                assert!((spec_tps - 80.0).abs() < 1e-9);
                assert!(observed_alpha < 0.0);
            }
            _ => panic!("expected Regression, got {r:?}"),
        }
    }

    #[test]
    fn uplift_rejects_nonfinite_base() {
        assert_eq!(
            classify_throughput_uplift(f64::INFINITY, 200.0, 0.3),
            ThroughputUpliftOutcome::InvalidInput {
                reason: "non-finite input"
            }
        );
        assert_eq!(
            classify_throughput_uplift(f64::NAN, 200.0, 0.3),
            ThroughputUpliftOutcome::InvalidInput {
                reason: "non-finite input"
            }
        );
    }

    #[test]
    fn uplift_rejects_zero_or_negative_base_tps() {
        assert_eq!(
            classify_throughput_uplift(0.0, 200.0, 0.3),
            ThroughputUpliftOutcome::InvalidInput {
                reason: "base_tps must be positive"
            }
        );
        assert_eq!(
            classify_throughput_uplift(-1.0, 200.0, 0.3),
            ThroughputUpliftOutcome::InvalidInput {
                reason: "base_tps must be positive"
            }
        );
    }

    #[test]
    fn uplift_rejects_negative_spec_tps() {
        assert_eq!(
            classify_throughput_uplift(100.0, -1.0, 0.3),
            ThroughputUpliftOutcome::InvalidInput {
                reason: "spec_tps must be non-negative"
            }
        );
    }

    #[test]
    fn uplift_rejects_negative_alpha_min() {
        assert_eq!(
            classify_throughput_uplift(100.0, 200.0, -0.1),
            ThroughputUpliftOutcome::InvalidInput {
                reason: "alpha_min must be non-negative"
            }
        );
    }

    #[test]
    fn uplift_classifier_is_deterministic() {
        let a = classify_throughput_uplift(100.0, 150.0, 0.3);
        let b = classify_throughput_uplift(100.0, 150.0, 0.3);
        assert_eq!(a, b);
    }

    // ---- tokenizer compatibility -----------------------------------------

    #[test]
    fn compat_ok_on_matching_sha_and_vocab() {
        assert_eq!(
            classify_tokenizer_compatibility("deadbeef", "deadbeef", 151936, 151936),
            TokenizerCompatOutcome::Ok
        );
    }

    #[test]
    fn compat_ok_case_insensitive_sha_match() {
        assert_eq!(
            classify_tokenizer_compatibility("DEADBEEF", "deadbeef", 1000, 1000),
            TokenizerCompatOutcome::Ok
        );
    }

    #[test]
    fn compat_rejects_empty_sha() {
        assert_eq!(
            classify_tokenizer_compatibility("", "deadbeef", 1000, 1000),
            TokenizerCompatOutcome::MalformedSha {
                reason: "empty sha256"
            }
        );
        assert_eq!(
            classify_tokenizer_compatibility("deadbeef", "", 1000, 1000),
            TokenizerCompatOutcome::MalformedSha {
                reason: "empty sha256"
            }
        );
    }

    #[test]
    fn compat_rejects_zero_draft_vocab() {
        assert_eq!(
            classify_tokenizer_compatibility("aa", "aa", 0, 1000),
            TokenizerCompatOutcome::ZeroVocab { which: "draft" }
        );
    }

    #[test]
    fn compat_rejects_zero_target_vocab() {
        assert_eq!(
            classify_tokenizer_compatibility("aa", "aa", 1000, 0),
            TokenizerCompatOutcome::ZeroVocab { which: "target" }
        );
    }

    #[test]
    fn compat_rejects_sha_mismatch() {
        assert_eq!(
            classify_tokenizer_compatibility("aa", "bb", 1000, 1000),
            TokenizerCompatOutcome::TokenizerShaMismatch
        );
    }

    #[test]
    fn compat_rejects_vocab_size_mismatch() {
        assert_eq!(
            classify_tokenizer_compatibility("aa", "aa", 32000, 151936),
            TokenizerCompatOutcome::VocabSizeMismatch {
                draft: 32000,
                target: 151936,
            }
        );
    }

    #[test]
    fn compat_classifier_is_deterministic() {
        let a = classify_tokenizer_compatibility("abc", "abc", 100, 100);
        let b = classify_tokenizer_compatibility("abc", "abc", 100, 100);
        assert_eq!(a, b);
    }

    // ---- acceptance rate --------------------------------------------------

    #[test]
    fn acceptance_ok_at_boundaries() {
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": {"acceptance_rate": 0.0}})),
            AcceptanceRateOutcome::Ok { rate: 0.0 }
        );
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": {"acceptance_rate": 1.0}})),
            AcceptanceRateOutcome::Ok { rate: 1.0 }
        );
    }

    #[test]
    fn acceptance_ok_at_midrange() {
        match classify_acceptance_rate(&json!({"speculative": {"acceptance_rate": 0.72}})) {
            AcceptanceRateOutcome::Ok { rate } => assert!((rate - 0.72).abs() < 1e-9),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn acceptance_rejects_non_object_top() {
        assert_eq!(
            classify_acceptance_rate(&json!([1, 2, 3])),
            AcceptanceRateOutcome::NotAnObject
        );
    }

    #[test]
    fn acceptance_rejects_missing_speculative() {
        assert_eq!(
            classify_acceptance_rate(&json!({"other": {}})),
            AcceptanceRateOutcome::MissingSpeculative
        );
    }

    #[test]
    fn acceptance_rejects_non_object_speculative() {
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": "0.5"})),
            AcceptanceRateOutcome::SpeculativeNotAnObject
        );
    }

    #[test]
    fn acceptance_rejects_missing_acceptance_rate() {
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": {"num_accepted": 1}})),
            AcceptanceRateOutcome::MissingAcceptanceRate
        );
    }

    #[test]
    fn acceptance_rejects_non_numeric_acceptance_rate() {
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": {"acceptance_rate": "0.5"}})),
            AcceptanceRateOutcome::AcceptanceRateNotNumeric
        );
    }

    #[test]
    fn acceptance_rejects_out_of_range_high() {
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": {"acceptance_rate": 1.5}})),
            AcceptanceRateOutcome::OutOfRange { value: 1.5 }
        );
    }

    #[test]
    fn acceptance_rejects_out_of_range_low() {
        assert_eq!(
            classify_acceptance_rate(&json!({"speculative": {"acceptance_rate": -0.1}})),
            AcceptanceRateOutcome::OutOfRange { value: -0.1 }
        );
    }

    #[test]
    fn acceptance_classifier_is_deterministic() {
        let v = json!({"speculative": {"acceptance_rate": 0.42}});
        let a = classify_acceptance_rate(&v);
        let b = classify_acceptance_rate(&v);
        assert_eq!(a, b);
    }

    // ---- constants sanity -------------------------------------------------

    #[test]
    fn constants_have_expected_values() {
        assert_eq!(DEFAULT_SPEC_TOKENS_K, 5);
        assert!((MIN_SPEC_UPLIFT_ALPHA - 0.30).abs() < 1e-12);
        assert_eq!(SPEC_TOKENS_MIN, 1);
        assert_eq!(SPEC_TOKENS_MAX, 16);
    }
}
