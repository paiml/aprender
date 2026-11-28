//! aprender#7484 (G-01): `POST /v1/messages` (Anthropic Messages API) on the serve router.
//!
//! Model-free checks: the request is refused with the Anthropic error envelope
//! (`{"type":"error",...}`, status 400) before any model work. A 200 needs a loaded
//! GGUF, and is checked against the real binary (`apr serve run`), not here.

use super::*;
use crate::api::create_router;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

async fn post_messages(state: AppState, body: &str) -> (StatusCode, serde_json::Value) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
async fn messages_without_max_tokens_is_an_anthropic_error_400() {
    let state = AppState::demo_mock().expect("model-free state");
    let (status, body) = post_messages(
        state,
        r#"{"model":"m","messages":[{"role":"user","content":"What is 2+2?"}]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "error", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("max_tokens")),
        "{body}"
    );
}

#[tokio::test]
async fn messages_without_messages_is_an_anthropic_error_400() {
    let state = AppState::demo_mock().expect("model-free state");
    let (status, body) = post_messages(state, r#"{"model":"m","max_tokens":64}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "error", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert!(body["error"]["message"].is_string(), "{body}");
}

#[tokio::test]
async fn messages_that_is_not_json_is_an_anthropic_error_400() {
    let state = AppState::demo_mock().expect("model-free state");
    let (status, body) = post_messages(state, "not json").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "error", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert!(body["error"]["message"].is_string(), "{body}");
}
