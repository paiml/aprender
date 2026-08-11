//! aprender#2375 finding 2 — `/v1/explain` returned fabricated explanations.
//!
//! 0.63.0 answered every request with HTTP 200 and a body shaped exactly like a
//! real SHAP explanation. The values came from the feature INDEX:
//!
//! ```text
//! .map(|(i, _)| 0.1 - (i as f32 * 0.02))
//! ```
//!
//! so they did not depend on the feature VALUES; `prediction` was the literal
//! `0.95`; and `State` was bound as `_state`, so the answer was identical
//! whether a model was loaded or not. That is the most expensive shape a defect
//! can take — a caller integrates against numbers that were never computed.
//!
//! Kernel SHAP needs a background dataset to establish expected values and
//! `ExplainRequest` carries none, so the endpoint now fails and says so.

use super::super::test_helpers::create_test_app_shared;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

async fn explain(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let response = create_test_app_shared()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// The falsifier that would have caught this from the outside, without reading
/// a line of the handler: two DIFFERENT inputs must not yield the same answer.
///
/// Under 0.63.0 both of these returned byte-identical `shap_values`
/// (`[0.1, 0.08, 0.06]`) and `prediction: 0.95`, because neither depended on
/// the numbers sent.
#[tokio::test]
async fn two_different_feature_vectors_do_not_produce_the_same_explanation() {
    let a = serde_json::json!({
        "features": [1.0, 2.0, 3.0],
        "feature_names": ["x", "y", "z"],
    });
    let b = serde_json::json!({
        "features": [-40.0, 0.001, 999.0],
        "feature_names": ["x", "y", "z"],
    });

    let (status_a, body_a) = explain(a).await;
    let (status_b, body_b) = explain(b).await;

    // Whatever the endpoint does, it must not answer 200 with a body that is
    // independent of its input.
    if status_a == StatusCode::OK && status_b == StatusCode::OK {
        assert_ne!(
            body_a.get("explanation"),
            body_b.get("explanation"),
            "wildly different features produced an IDENTICAL explanation — the \
             values cannot be derived from the input: {body_a}"
        );
    }
}

/// An unimplemented explanation must be an ERROR, not a plausible 200.
#[tokio::test]
async fn explain_does_not_return_success_with_values_it_did_not_compute() {
    let (status, body) = explain(serde_json::json!({
        "features": [1.0, 2.0, 3.0],
        "feature_names": ["a", "b", "c"],
    }))
    .await;

    assert_ne!(
        status,
        StatusCode::OK,
        "explain answered 200 without computing anything: {body}"
    );
    assert!(
        status == StatusCode::NOT_IMPLEMENTED || status == StatusCode::SERVICE_UNAVAILABLE,
        "expected 501 (no implementation) or 503 (no model), got {status}: {body}"
    );
}

/// The 0.63.0 constants must not appear on the wire at all.
#[tokio::test]
async fn the_fabricated_constants_are_gone() {
    let (_status, body) = explain(serde_json::json!({
        "features": [1.0, 2.0, 3.0, 4.0],
        "feature_names": ["a", "b", "c", "d"],
    }))
    .await;

    let text = body.to_string();
    assert!(
        !text.contains("0.95"),
        "the hardcoded prediction 0.95 is still on the wire: {text}"
    );
    // 0.1, 0.08, 0.06, 0.04 — the index-derived series.
    assert!(
        !(text.contains("0.08") && text.contains("0.06")),
        "the index-derived SHAP series is still on the wire: {text}"
    );
}

/// Input validation must still happen, and BEFORE the not-implemented verdict —
/// a caller with a malformed request should be told that, not told the endpoint
/// is missing.
#[tokio::test]
async fn malformed_requests_are_still_rejected_as_bad_requests() {
    let (status, _) = explain(serde_json::json!({
        "features": [],
        "feature_names": [],
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "empty features must be 400");

    let (status, _) = explain(serde_json::json!({
        "features": [1.0, 2.0],
        "feature_names": ["only_one"],
    }))
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a name/feature count mismatch must be 400"
    );
}
