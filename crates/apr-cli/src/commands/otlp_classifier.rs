//! OpenTelemetry OTLP trace classifier (CRUX-K-08).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-K-08-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! an already-captured OTLP/JSON `ExportTraceServiceRequest` body:
//!
//!   * `classify_span_present` — at least one span with the given name exists
//!     anywhere in `resourceSpans[].scopeSpans[].spans[]`.
//!   * `classify_genai_attributes` — at least one span carries every required
//!     attribute key (e.g. `gen_ai.system`, `gen_ai.request.model`,
//!     `apr.tokens.prompt`, `apr.tokens.output`).
//!   * `classify_trace_propagation` — at least one span's `traceId` matches an
//!     expected 32-hex-char trace-id (the W3C `traceparent` upstream).
//!
//! All three operate on the **OTLP/JSON** encoding documented at
//! <https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding>. Body
//! shape:
//!
//! ```json
//! {
//!   "resourceSpans": [{
//!     "resource": {"attributes": [{"key":..., "value":{"stringValue":...}}]},
//!     "scopeSpans": [{
//!       "spans": [{
//!         "traceId": "<32-hex>",
//!         "spanId": "<16-hex>",
//!         "parentSpanId": "<16-hex>",
//!         "name": "apr.inference",
//!         "attributes": [{"key":"gen_ai.system","value":{"stringValue":"apr"}}, ...]
//!       }]
//!     }]
//!   }]
//! }
//! ```
//!
//! Full discharge blocks on a live `apr serve` OTLP exporter wired to
//! `OTEL_EXPORTER_OTLP_ENDPOINT` — tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Canonical OpenTelemetry GenAI + apr-specific attribute keys that an
/// `apr.inference` span must carry (CRUX-K-08 `gen_ai_semconv_attributes`).
pub const K08_REQUIRED_ATTRIBUTES: &[&str] = &[
    "gen_ai.system",
    "gen_ai.request.model",
    "apr.tokens.prompt",
    "apr.tokens.output",
];

/// Canonical span name emitted by `apr serve` on every inference request.
pub const K08_ROOT_SPAN_NAME: &str = "apr.inference";

/// Outcome of `classify_span_present`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtlpSpanPresentOutcome {
    /// At least one span with the requested name exists.
    Ok { count: usize },
    /// OTLP body root is not a JSON object.
    NotAnObject,
    /// `resourceSpans` is absent or not an array.
    MissingResourceSpans,
    /// Body parsed but no span matches the requested name.
    SpanNameNotFound {
        requested: String,
        names_seen: Vec<String>,
    },
}

/// Outcome of `classify_genai_attributes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtlpAttributesOutcome {
    /// Every required attribute key appeared on at least one span.
    Ok,
    /// At least one required attribute key is absent from every span.
    Missing { missing: Vec<String> },
}

/// Outcome of `classify_trace_propagation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtlpTracePropagationOutcome {
    /// At least one span's `traceId` matches the expected trace-id (lowercase hex).
    Ok,
    /// Expected trace-id is not 32 lowercase hex characters.
    InvalidExpectedTraceId { got: String },
    /// No span in the body carries the expected `traceId`.
    TraceIdNotFound {
        expected: String,
        ids_seen: Vec<String>,
    },
}

/// Return true if a span anywhere in `body` has `name == span_name`.
pub fn classify_span_present(body: &Value, span_name: &str) -> OtlpSpanPresentOutcome {
    let Some(obj) = body.as_object() else {
        return OtlpSpanPresentOutcome::NotAnObject;
    };
    let Some(resource_spans) = obj.get("resourceSpans").and_then(Value::as_array) else {
        return OtlpSpanPresentOutcome::MissingResourceSpans;
    };
    let mut count = 0;
    let mut names = Vec::new();
    for rs in resource_spans {
        for ss in rs
            .get("scopeSpans")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            for span in ss
                .get("spans")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(n) = span.get("name").and_then(Value::as_str) {
                    if n == span_name {
                        count += 1;
                    } else {
                        names.push(n.to_string());
                    }
                }
            }
        }
    }
    if count > 0 {
        OtlpSpanPresentOutcome::Ok { count }
    } else {
        OtlpSpanPresentOutcome::SpanNameNotFound {
            requested: span_name.to_string(),
            names_seen: names,
        }
    }
}

/// Verify that every key in `required` appears on at least one span's
/// `attributes[]` list anywhere in the body. (OTLP attributes are a
/// key-value array, not a map.)
pub fn classify_genai_attributes(body: &Value, required: &[&str]) -> OtlpAttributesOutcome {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(resource_spans) = body.get("resourceSpans").and_then(Value::as_array) {
        for rs in resource_spans {
            for ss in rs
                .get("scopeSpans")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                for span in ss
                    .get("spans")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(attrs) = span.get("attributes").and_then(Value::as_array) {
                        for kv in attrs {
                            if let Some(k) = kv.get("key").and_then(Value::as_str) {
                                seen.insert(k.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    let mut missing: Vec<String> = required
        .iter()
        .filter(|r| !seen.contains(**r))
        .map(|s| (*s).to_string())
        .collect();
    if missing.is_empty() {
        OtlpAttributesOutcome::Ok
    } else {
        missing.sort();
        OtlpAttributesOutcome::Missing { missing }
    }
}

/// Verify that at least one span in `body` has `traceId == expected_trace_id`.
///
/// `expected_trace_id` must be 32 lowercase hex characters (W3C trace-context
/// trace-id encoding). The comparison is byte-equal — OTLP/JSON traceId is
/// always lowercase hex per the spec.
pub fn classify_trace_propagation(
    body: &Value,
    expected_trace_id: &str,
) -> OtlpTracePropagationOutcome {
    if expected_trace_id.len() != 32
        || !expected_trace_id
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return OtlpTracePropagationOutcome::InvalidExpectedTraceId {
            got: expected_trace_id.to_string(),
        };
    }
    let mut ids = Vec::new();
    if let Some(resource_spans) = body.get("resourceSpans").and_then(Value::as_array) {
        for rs in resource_spans {
            for ss in rs
                .get("scopeSpans")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                for span in ss
                    .get("spans")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(tid) = span.get("traceId").and_then(Value::as_str) {
                        if tid == expected_trace_id {
                            return OtlpTracePropagationOutcome::Ok;
                        }
                        ids.push(tid.to_string());
                    }
                }
            }
        }
    }
    OtlpTracePropagationOutcome::TraceIdNotFound {
        expected: expected_trace_id.to_string(),
        ids_seen: ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_body() -> Value {
        json!({
            "resourceSpans": [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": "apr"},
                    "spans": [{
                        "traceId": "0af7651916cd43dd8448eb211c80319c",
                        "spanId": "1234567890abcdef",
                        "parentSpanId": "00f067aa0ba902b7",
                        "name": "apr.inference",
                        "attributes": [
                            {"key": "gen_ai.system", "value": {"stringValue": "apr"}},
                            {"key": "gen_ai.request.model", "value": {"stringValue": "qwen3-1.7b"}},
                            {"key": "apr.tokens.prompt", "value": {"intValue": "42"}},
                            {"key": "apr.tokens.output", "value": {"intValue": "10"}},
                        ]
                    }]
                }]
            }]
        })
    }

    #[test]
    fn span_present_ok_on_apr_inference() {
        let out = classify_span_present(&good_body(), K08_ROOT_SPAN_NAME);
        assert_eq!(out, OtlpSpanPresentOutcome::Ok { count: 1 });
    }

    #[test]
    fn span_present_rejects_not_an_object() {
        let body = json!([1, 2, 3]);
        assert_eq!(
            classify_span_present(&body, K08_ROOT_SPAN_NAME),
            OtlpSpanPresentOutcome::NotAnObject
        );
    }

    #[test]
    fn span_present_rejects_missing_resource_spans() {
        let body = json!({"otherTopLevel": []});
        assert_eq!(
            classify_span_present(&body, K08_ROOT_SPAN_NAME),
            OtlpSpanPresentOutcome::MissingResourceSpans
        );
    }

    #[test]
    fn span_present_reports_unrecognized_name() {
        let body =
            json!({"resourceSpans":[{"scopeSpans":[{"spans":[{"name":"http.server.request"}]}]}]});
        match classify_span_present(&body, K08_ROOT_SPAN_NAME) {
            OtlpSpanPresentOutcome::SpanNameNotFound { names_seen, .. } => {
                assert!(names_seen.contains(&"http.server.request".to_string()));
            }
            other => panic!("expected SpanNameNotFound, got {other:?}"),
        }
    }

    #[test]
    fn genai_attributes_ok_on_full_body() {
        assert_eq!(
            classify_genai_attributes(&good_body(), K08_REQUIRED_ATTRIBUTES),
            OtlpAttributesOutcome::Ok
        );
    }

    #[test]
    fn genai_attributes_reports_missing_subset() {
        let body = json!({
            "resourceSpans":[{"scopeSpans":[{"spans":[{
                "name": "apr.inference",
                "attributes":[
                    {"key":"gen_ai.system","value":{"stringValue":"apr"}}
                ]
            }]}]}]
        });
        match classify_genai_attributes(&body, K08_REQUIRED_ATTRIBUTES) {
            OtlpAttributesOutcome::Missing { missing } => {
                assert!(missing.contains(&"gen_ai.request.model".to_string()));
                assert!(missing.contains(&"apr.tokens.prompt".to_string()));
                assert!(missing.contains(&"apr.tokens.output".to_string()));
                assert!(!missing.contains(&"gen_ai.system".to_string()));
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn trace_propagation_ok_on_canonical_traceparent() {
        let out = classify_trace_propagation(&good_body(), "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(out, OtlpTracePropagationOutcome::Ok);
    }

    #[test]
    fn trace_propagation_rejects_short_expected_id() {
        let out = classify_trace_propagation(&good_body(), "short");
        assert!(matches!(
            out,
            OtlpTracePropagationOutcome::InvalidExpectedTraceId { .. }
        ));
    }

    #[test]
    fn trace_propagation_rejects_uppercase_expected_id() {
        // OTLP/JSON encodes traceId as lowercase hex; an uppercase request is invalid input.
        let out = classify_trace_propagation(&good_body(), "0AF7651916CD43DD8448EB211C80319C");
        assert!(matches!(
            out,
            OtlpTracePropagationOutcome::InvalidExpectedTraceId { .. }
        ));
    }

    #[test]
    fn trace_propagation_reports_mismatch() {
        let out = classify_trace_propagation(&good_body(), "ffffffffffffffffffffffffffffffff");
        match out {
            OtlpTracePropagationOutcome::TraceIdNotFound { ids_seen, .. } => {
                assert!(ids_seen.contains(&"0af7651916cd43dd8448eb211c80319c".to_string()));
            }
            other => panic!("expected TraceIdNotFound, got {other:?}"),
        }
    }

    #[test]
    fn span_present_counts_multiple_spans_in_multiple_scopes() {
        let body = json!({
            "resourceSpans":[
                {"scopeSpans":[
                    {"spans":[{"name":"apr.inference"},{"name":"apr.inference"}]},
                    {"spans":[{"name":"apr.inference"}]}
                ]}
            ]
        });
        assert_eq!(
            classify_span_present(&body, "apr.inference"),
            OtlpSpanPresentOutcome::Ok { count: 3 }
        );
    }
}
