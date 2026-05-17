//! Flash-attention parity classifier (CRUX-L-02).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-L-02-{002,003,004}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on:
//!
//!   * a captured `apr kernel parity --impl flash2 --ref naive --json` body
//!     (numerical parity gate); and
//!   * a captured `apr run --attn flash2 --json` body (kernel-source
//!     provenance + fallback metadata); and
//!   * a captured error JSON from a head_dim-rejection path.
//!
//! Classifiers:
//!   * `classify_parity_numerics` — `max_abs_diff <= tol_abs` (default 5e-3)
//!     AND `cosine_sim >= tol_cos` (default 0.9999). Bounds from the
//!     FlashAttention-2 paper (Dao 2023, arXiv:2307.08691).
//!   * `classify_provenance` — `attn_impl == "flash2"` ⇒
//!     `kernel_source` matches
//!     `^hf-kernels-community:flash-attn2@[0-9a-f]{40}$`;
//!     `attn_impl != "flash2"` ⇒ `fallback` is a non-empty string.
//!   * `classify_head_dim_error` — error JSON has `error` containing
//!     `unsupported-head-dim` or `head_dim` (caller asserts exit_code=1).
//!
//! Full discharge requires a live `apr kernel parity` + `apr run --attn flash2`
//! pipeline — tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Default FlashAttention-2 numerical-parity tolerances (Dao 2023 bound).
pub const L02_DEFAULT_MAX_ABS_DIFF: f64 = 5e-3;
pub const L02_DEFAULT_MIN_COSINE_SIM: f64 = 0.9999;

/// Canonical kernel-source prefix (hf-kernels-community FA2 release).
pub const L02_KERNEL_SOURCE_PREFIX: &str = "hf-kernels-community:flash-attn2@";

/// Outcome of `classify_parity_numerics`.
#[derive(Debug, Clone, PartialEq)]
pub enum AttnParityNumericsOutcome {
    Ok { max_abs_diff: f64, cosine_sim: f64 },
    NotAnObject,
    MissingMaxAbsDiff,
    MissingCosineSim,
    NonFiniteMaxAbsDiff { got: f64 },
    NonFiniteCosineSim { got: f64 },
    MaxAbsDiffExceedsTolerance { got: f64, tolerance: f64 },
    CosineSimBelowFloor { got: f64, floor: f64 },
}

/// Outcome of `classify_provenance`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnProvenanceOutcome {
    OkFlash2 { sha: String },
    OkFallback { reason: String },
    NotAnObject,
    MissingAttnImpl,
    UnknownAttnImpl { got: String },
    KernelSourceMissing,
    KernelSourceMalformed { got: String },
    FallbackMissingWhenNotFlash2 { attn_impl: String },
}

/// Outcome of `classify_head_dim_error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnHeadDimErrorOutcome {
    Ok { error: String },
    NotAnObject,
    MissingErrorField,
    ErrorDoesNotMentionHeadDim { got: String },
}

/// FALSIFY-CRUX-L-02-002: numerical parity vs the naive reference.
pub fn classify_parity_numerics(
    body: &Value,
    tol_abs: f64,
    tol_cos: f64,
) -> AttnParityNumericsOutcome {
    let Some(obj) = body.as_object() else {
        return AttnParityNumericsOutcome::NotAnObject;
    };
    let Some(mad) = obj.get("max_abs_diff").and_then(Value::as_f64) else {
        return AttnParityNumericsOutcome::MissingMaxAbsDiff;
    };
    let Some(cos) = obj.get("cosine_sim").and_then(Value::as_f64) else {
        return AttnParityNumericsOutcome::MissingCosineSim;
    };
    if !mad.is_finite() {
        return AttnParityNumericsOutcome::NonFiniteMaxAbsDiff { got: mad };
    }
    if !cos.is_finite() {
        return AttnParityNumericsOutcome::NonFiniteCosineSim { got: cos };
    }
    if mad > tol_abs {
        return AttnParityNumericsOutcome::MaxAbsDiffExceedsTolerance {
            got: mad,
            tolerance: tol_abs,
        };
    }
    if cos < tol_cos {
        return AttnParityNumericsOutcome::CosineSimBelowFloor {
            got: cos,
            floor: tol_cos,
        };
    }
    AttnParityNumericsOutcome::Ok {
        max_abs_diff: mad,
        cosine_sim: cos,
    }
}

/// FALSIFY-CRUX-L-02-003: provenance & fallback metadata.
pub fn classify_provenance(body: &Value) -> AttnProvenanceOutcome {
    let Some(obj) = body.as_object() else {
        return AttnProvenanceOutcome::NotAnObject;
    };
    let Some(impl_) = obj.get("attn_impl").and_then(Value::as_str) else {
        return AttnProvenanceOutcome::MissingAttnImpl;
    };
    match impl_ {
        "flash2" => {
            let src = obj.get("kernel_source").and_then(Value::as_str);
            match src {
                None => AttnProvenanceOutcome::KernelSourceMissing,
                Some(s) => {
                    if !s.starts_with(L02_KERNEL_SOURCE_PREFIX) {
                        return AttnProvenanceOutcome::KernelSourceMalformed { got: s.to_string() };
                    }
                    let sha = &s[L02_KERNEL_SOURCE_PREFIX.len()..];
                    if sha.len() != 40
                        || !sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                    {
                        return AttnProvenanceOutcome::KernelSourceMalformed { got: s.to_string() };
                    }
                    AttnProvenanceOutcome::OkFlash2 {
                        sha: sha.to_string(),
                    }
                }
            }
        }
        "naive" | "fallback" => {
            let fb = obj.get("fallback").and_then(Value::as_str);
            match fb {
                Some(s) if !s.is_empty() => AttnProvenanceOutcome::OkFallback {
                    reason: s.to_string(),
                },
                _ => AttnProvenanceOutcome::FallbackMissingWhenNotFlash2 {
                    attn_impl: impl_.to_string(),
                },
            }
        }
        other => AttnProvenanceOutcome::UnknownAttnImpl {
            got: other.to_string(),
        },
    }
}

/// FALSIFY-CRUX-L-02-004: unsupported head_dim error JSON shape.
pub fn classify_head_dim_error(body: &Value) -> AttnHeadDimErrorOutcome {
    let Some(obj) = body.as_object() else {
        return AttnHeadDimErrorOutcome::NotAnObject;
    };
    let Some(err) = obj.get("error").and_then(Value::as_str) else {
        return AttnHeadDimErrorOutcome::MissingErrorField;
    };
    if err.contains("unsupported-head-dim") || err.contains("head_dim") || err.contains("head-dim")
    {
        return AttnHeadDimErrorOutcome::Ok {
            error: err.to_string(),
        };
    }
    AttnHeadDimErrorOutcome::ErrorDoesNotMentionHeadDim {
        got: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parity_ok_inside_tolerance() {
        let body = json!({"max_abs_diff": 0.002, "cosine_sim": 0.99999});
        match classify_parity_numerics(&body, 5e-3, 0.9999) {
            AttnParityNumericsOutcome::Ok { .. } => {}
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn parity_rejects_max_abs_diff_above_tolerance() {
        let body = json!({"max_abs_diff": 0.01, "cosine_sim": 0.99999});
        assert!(matches!(
            classify_parity_numerics(&body, 5e-3, 0.9999),
            AttnParityNumericsOutcome::MaxAbsDiffExceedsTolerance { .. }
        ));
    }

    #[test]
    fn parity_rejects_cosine_below_floor() {
        let body = json!({"max_abs_diff": 0.001, "cosine_sim": 0.99});
        assert!(matches!(
            classify_parity_numerics(&body, 5e-3, 0.9999),
            AttnParityNumericsOutcome::CosineSimBelowFloor { .. }
        ));
    }

    #[test]
    fn parity_rejects_missing_keys() {
        assert_eq!(
            classify_parity_numerics(&json!({"cosine_sim": 1.0}), 5e-3, 0.9999),
            AttnParityNumericsOutcome::MissingMaxAbsDiff
        );
        assert_eq!(
            classify_parity_numerics(&json!({"max_abs_diff": 0.0}), 5e-3, 0.9999),
            AttnParityNumericsOutcome::MissingCosineSim
        );
    }

    #[test]
    fn provenance_ok_on_pinned_flash2() {
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        let src = format!("hf-kernels-community:flash-attn2@{sha}");
        let body = json!({"attn_impl": "flash2", "kernel_source": src});
        match classify_provenance(&body) {
            AttnProvenanceOutcome::OkFlash2 { sha: got } => assert_eq!(got, sha),
            other => panic!("expected OkFlash2, got {other:?}"),
        }
    }

    #[test]
    fn provenance_rejects_malformed_sha() {
        let body = json!({"attn_impl": "flash2", "kernel_source": "hf-kernels-community:flash-attn2@short"});
        assert!(matches!(
            classify_provenance(&body),
            AttnProvenanceOutcome::KernelSourceMalformed { .. }
        ));
    }

    #[test]
    fn provenance_rejects_missing_kernel_source_when_flash2() {
        let body = json!({"attn_impl": "flash2"});
        assert_eq!(
            classify_provenance(&body),
            AttnProvenanceOutcome::KernelSourceMissing
        );
    }

    #[test]
    fn provenance_ok_on_naive_with_fallback_reason() {
        let body = json!({"attn_impl": "naive", "fallback": "no-gpu"});
        match classify_provenance(&body) {
            AttnProvenanceOutcome::OkFallback { reason } => assert_eq!(reason, "no-gpu"),
            other => panic!("expected OkFallback, got {other:?}"),
        }
    }

    #[test]
    fn provenance_rejects_naive_without_fallback() {
        let body = json!({"attn_impl": "naive"});
        assert!(matches!(
            classify_provenance(&body),
            AttnProvenanceOutcome::FallbackMissingWhenNotFlash2 { .. }
        ));
    }

    #[test]
    fn provenance_rejects_unknown_attn_impl() {
        let body = json!({"attn_impl": "unobtanium"});
        assert!(matches!(
            classify_provenance(&body),
            AttnProvenanceOutcome::UnknownAttnImpl { .. }
        ));
    }

    #[test]
    fn head_dim_error_ok_on_unsupported_message() {
        let body = json!({"error": "unsupported-head-dim: got 96, expected 64 or 128"});
        match classify_head_dim_error(&body) {
            AttnHeadDimErrorOutcome::Ok { error } => {
                assert!(error.contains("unsupported-head-dim"));
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn head_dim_error_rejects_irrelevant_error() {
        let body = json!({"error": "out of memory"});
        assert!(matches!(
            classify_head_dim_error(&body),
            AttnHeadDimErrorOutcome::ErrorDoesNotMentionHeadDim { .. }
        ));
    }

    #[test]
    fn head_dim_error_rejects_missing_error_field() {
        let body = json!({});
        assert_eq!(
            classify_head_dim_error(&body),
            AttnHeadDimErrorOutcome::MissingErrorField
        );
    }
}
