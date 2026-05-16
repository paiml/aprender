//! NaN/Inf activation-check classifier (CRUX-F-11).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-11-{002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on:
//!
//!   * a captured `apr trace --check-finite` stderr JSON describing the
//!     first non-finite activation; and
//!   * a captured `apr trace --check-finite --list` JSON enumerating every
//!     tensor-producing op in the forward pass.
//!
//! Classifiers:
//!   * `classify_error_json` — error body has the 6 required keys
//!     (`error`, `layer`, `shape`, `first_bad_index`, `value`, `op`),
//!     `error == "non_finite"`, and `value ∈ {"nan", "+inf", "-inf"}`.
//!   * `classify_layer_coverage` — list body's `layers` array contains
//!     at least `min_layers` entries and includes every transformer
//!     op prefix in `required_op_prefixes` (default: attention_q,
//!     attention_k, attention_v, attention_out, ffn_gate, ffn_up,
//!     ffn_down, layernorm).
//!
//! FALSIFY-CRUX-F-11-001 (clean exit 0) and -004 (parity with torch
//! anomaly mode) require live runs and are tracked as
//! BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Default transformer-op name prefixes that every layer-list emission
/// must cover (CRUX-F-11 `layer_coverage_complete`).
pub const F11_REQUIRED_OP_PREFIXES: &[&str] = &[
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_out",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "layernorm",
];

/// Required top-level keys on a CRUX-F-11 error JSON
/// (`apr trace --check-finite` stderr on a poisoned model).
pub const F11_ERROR_JSON_KEYS: &[&str] =
    &["error", "layer", "shape", "first_bad_index", "value", "op"];

/// Allowed values for the `value` field of an error JSON.
pub const F11_ALLOWED_VALUES: &[&str] = &["nan", "+inf", "-inf"];

/// Outcome of `classify_error_json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckFiniteErrorOutcome {
    Ok,
    NotAnObject,
    MissingKey { key: &'static str },
    ErrorTagWrong { got: String },
    ValueOutOfSet { got: String },
    ShapeNotIntArray,
    FirstBadIndexNotNonNegative { got: i64 },
}

/// Outcome of `classify_layer_coverage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckFiniteCoverageOutcome {
    Ok { count: usize },
    NotAnObject,
    LayersNotArray,
    TooFewLayers { got: usize, min: usize },
    MissingOpPrefixes { missing: Vec<String> },
}

/// Validate the error JSON body emitted on stderr when a non-finite
/// activation is detected (FALSIFY-CRUX-F-11-002).
pub fn classify_error_json(body: &Value) -> CheckFiniteErrorOutcome {
    let Some(obj) = body.as_object() else {
        return CheckFiniteErrorOutcome::NotAnObject;
    };
    for k in F11_ERROR_JSON_KEYS {
        if !obj.contains_key(*k) {
            return CheckFiniteErrorOutcome::MissingKey { key: k };
        }
    }
    let err = obj.get("error").and_then(Value::as_str).unwrap_or("");
    if err != "non_finite" {
        return CheckFiniteErrorOutcome::ErrorTagWrong { got: err.to_string() };
    }
    let value = obj.get("value").and_then(Value::as_str).unwrap_or("");
    if !F11_ALLOWED_VALUES.contains(&value) {
        return CheckFiniteErrorOutcome::ValueOutOfSet { got: value.to_string() };
    }
    let Some(shape) = obj.get("shape").and_then(Value::as_array) else {
        return CheckFiniteErrorOutcome::ShapeNotIntArray;
    };
    for dim in shape {
        if dim.as_i64().is_none() {
            return CheckFiniteErrorOutcome::ShapeNotIntArray;
        }
    }
    let idx = obj.get("first_bad_index").and_then(Value::as_i64).unwrap_or(-1);
    if idx < 0 {
        return CheckFiniteErrorOutcome::FirstBadIndexNotNonNegative { got: idx };
    }
    CheckFiniteErrorOutcome::Ok
}

/// Validate the layer-coverage JSON emitted on stdout when
/// `--check-finite --list` is requested (FALSIFY-CRUX-F-11-003).
pub fn classify_layer_coverage(
    body: &Value,
    min_layers: usize,
    required_op_prefixes: &[&str],
) -> CheckFiniteCoverageOutcome {
    let Some(obj) = body.as_object() else {
        return CheckFiniteCoverageOutcome::NotAnObject;
    };
    let Some(layers) = obj.get("layers").and_then(Value::as_array) else {
        return CheckFiniteCoverageOutcome::LayersNotArray;
    };
    let count = layers.len();
    if count < min_layers {
        return CheckFiniteCoverageOutcome::TooFewLayers { got: count, min: min_layers };
    }
    // Collect every layer name (`name` field, otherwise the layer entry
    // itself if it is a bare string).
    let mut names: Vec<String> = Vec::new();
    for l in layers {
        if let Some(n) = l.get("name").and_then(Value::as_str) {
            names.push(n.to_string());
        } else if let Some(n) = l.as_str() {
            names.push(n.to_string());
        }
    }
    let mut missing: Vec<String> = required_op_prefixes
        .iter()
        .filter(|p| !names.iter().any(|n| n.contains(**p)))
        .map(|s| (*s).to_string())
        .collect();
    if !missing.is_empty() {
        missing.sort();
        return CheckFiniteCoverageOutcome::MissingOpPrefixes { missing };
    }
    CheckFiniteCoverageOutcome::Ok { count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_error_json() -> Value {
        json!({
            "error": "non_finite",
            "layer": "blk.0.ffn_up",
            "shape": [1, 4096],
            "first_bad_index": 0,
            "value": "nan",
            "op": "ffn_up"
        })
    }

    fn good_layer_list(n_blocks: usize) -> Value {
        let mut layers: Vec<Value> = Vec::new();
        for b in 0..n_blocks {
            for op in &[
                "attention_q",
                "attention_k",
                "attention_v",
                "attention_out",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "layernorm_in",
                "layernorm_post",
            ] {
                layers.push(json!({"name": format!("blk.{b}.{op}"), "shape": [1, 4096]}));
            }
        }
        layers.push(json!({"name": "embed_tokens", "shape": [1, 4096]}));
        layers.push(json!({"name": "output_norm", "shape": [1, 4096]}));
        layers.push(json!({"name": "lm_head", "shape": [1, 32000]}));
        json!({"layers": layers})
    }

    #[test]
    fn error_json_ok_on_well_formed() {
        assert_eq!(classify_error_json(&good_error_json()), CheckFiniteErrorOutcome::Ok);
    }

    #[test]
    fn error_json_rejects_not_an_object() {
        assert_eq!(
            classify_error_json(&json!([1, 2])),
            CheckFiniteErrorOutcome::NotAnObject
        );
    }

    #[test]
    fn error_json_reports_missing_key() {
        let body = json!({"error": "non_finite", "shape": [1], "first_bad_index": 0, "value": "nan", "op": "x"});
        assert_eq!(
            classify_error_json(&body),
            CheckFiniteErrorOutcome::MissingKey { key: "layer" }
        );
    }

    #[test]
    fn error_json_rejects_wrong_error_tag() {
        let body = json!({"error": "weird_other", "layer": "L", "shape": [1], "first_bad_index": 0, "value": "nan", "op": "x"});
        assert!(matches!(
            classify_error_json(&body),
            CheckFiniteErrorOutcome::ErrorTagWrong { .. }
        ));
    }

    #[test]
    fn error_json_rejects_value_outside_allowed_set() {
        let body = json!({"error": "non_finite", "layer": "L", "shape": [1], "first_bad_index": 0, "value": "banana", "op": "x"});
        assert!(matches!(
            classify_error_json(&body),
            CheckFiniteErrorOutcome::ValueOutOfSet { .. }
        ));
    }

    #[test]
    fn error_json_rejects_non_int_shape() {
        let body = json!({"error": "non_finite", "layer": "L", "shape": ["wide"], "first_bad_index": 0, "value": "nan", "op": "x"});
        assert_eq!(
            classify_error_json(&body),
            CheckFiniteErrorOutcome::ShapeNotIntArray
        );
    }

    #[test]
    fn error_json_rejects_negative_first_bad_index() {
        let body = json!({"error": "non_finite", "layer": "L", "shape": [1], "first_bad_index": -1, "value": "nan", "op": "x"});
        assert!(matches!(
            classify_error_json(&body),
            CheckFiniteErrorOutcome::FirstBadIndexNotNonNegative { got: -1 }
        ));
    }

    #[test]
    fn layer_coverage_ok_on_well_formed_list() {
        // 28 blocks × 9 ops + 3 extras = 255 — easily clears 100.
        let body = good_layer_list(28);
        match classify_layer_coverage(&body, 100, F11_REQUIRED_OP_PREFIXES) {
            CheckFiniteCoverageOutcome::Ok { count } => assert_eq!(count, 28 * 9 + 3),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn layer_coverage_rejects_not_an_object() {
        assert_eq!(
            classify_layer_coverage(&json!([1, 2]), 100, F11_REQUIRED_OP_PREFIXES),
            CheckFiniteCoverageOutcome::NotAnObject
        );
    }

    #[test]
    fn layer_coverage_rejects_missing_layers_array() {
        assert_eq!(
            classify_layer_coverage(&json!({"otherKey": []}), 100, F11_REQUIRED_OP_PREFIXES),
            CheckFiniteCoverageOutcome::LayersNotArray
        );
    }

    #[test]
    fn layer_coverage_reports_too_few_layers() {
        let body = good_layer_list(2); // 21 layers
        assert!(matches!(
            classify_layer_coverage(&body, 100, F11_REQUIRED_OP_PREFIXES),
            CheckFiniteCoverageOutcome::TooFewLayers { got: 21, min: 100 }
        ));
    }

    #[test]
    fn layer_coverage_reports_missing_op_prefixes() {
        // Only attention layers; FFN/LN missing.
        let mut layers: Vec<Value> = Vec::new();
        for b in 0..20 {
            for op in &["attention_q", "attention_k", "attention_v", "attention_out"] {
                layers.push(json!({"name": format!("blk.{b}.{op}")}));
            }
            // Pad so we hit the 100-layer floor without exercising FFN/LN.
            for k in 0..5 {
                layers.push(json!({"name": format!("blk.{b}.extra_{k}")}));
            }
        }
        let body = json!({"layers": layers});
        match classify_layer_coverage(&body, 100, F11_REQUIRED_OP_PREFIXES) {
            CheckFiniteCoverageOutcome::MissingOpPrefixes { missing } => {
                assert!(missing.contains(&"ffn_gate".to_string()));
                assert!(missing.contains(&"layernorm".to_string()));
                assert!(!missing.contains(&"attention_q".to_string()));
            }
            other => panic!("expected MissingOpPrefixes, got {other:?}"),
        }
    }
}
