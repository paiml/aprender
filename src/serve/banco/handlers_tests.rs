//! Tests for Banco endpoint handlers via router oneshot (no TCP).

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use super::router::create_banco_router;
use super::state::BancoStateInner;
use super::types::{
    BancoChatResponse, ErrorResponse, HealthResponse, ModelsResponse, SystemResponse,
};

/// Build a default Banco router for testing.
fn test_app() -> axum::Router {
    create_banco_router(BancoStateInner::with_defaults())
}

/// Helper: parse JSON body from a response.
async fn json_body<T: serde::de::DeserializeOwned>(response: axum::http::Response<Body>) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("read body");
    serde_json::from_slice(&bytes).expect("parse json")
}

// ============================================================================
// BANCO_HDL_001: GET /health
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_001_health() {
    let app = test_app();
    let response =
        app.oneshot(Request::get("/health").body(Body::empty()).expect("req")).await.expect("resp");

    assert_eq!(response.status(), StatusCode::OK);
    let health: HealthResponse = json_body(response).await;
    assert_eq!(health.status, "ok");
    assert_eq!(health.circuit_breaker_state, "closed");
}

// ============================================================================
// BANCO_HDL_002: GET /api/v1/models
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_002_models() {
    let app = test_app();
    let response = app
        .oneshot(Request::get("/api/v1/models").body(Body::empty()).expect("req"))
        .await
        .expect("resp");

    assert_eq!(response.status(), StatusCode::OK);
    let models: ModelsResponse = json_body(response).await;
    assert_eq!(models.object, "list");
    assert!(!models.data.is_empty());
}

// ============================================================================
// BANCO_HDL_003: GET /api/v1/system
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_003_system() {
    let app = test_app();
    let response = app
        .oneshot(Request::get("/api/v1/system").body(Body::empty()).expect("req"))
        .await
        .expect("resp");

    assert_eq!(response.status(), StatusCode::OK);
    let sys: SystemResponse = json_body(response).await;
    assert_eq!(sys.privacy_tier, "Standard");
    assert!(!sys.version.is_empty());
}

// ============================================================================
// BANCO_HDL_004: POST /api/v1/chat/completions — non-streaming
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_004_chat_completions_sync() {
    let app = test_app();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello!"}]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");

    assert_eq!(response.status(), StatusCode::OK);
    let chat: BancoChatResponse = json_body(response).await;
    assert_eq!(chat.object, "chat.completion");
    assert_eq!(chat.choices.len(), 1);
    assert_eq!(chat.choices[0].finish_reason, "dry_run");
    assert!(chat.choices[0].message.content.contains("No model loaded"));
    assert!(chat.usage.total_tokens > 0);
}

// ============================================================================
// BANCO_HDL_005: POST /api/v1/chat/completions — with model
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_005_chat_completions_with_model() {
    let app = test_app();
    let body = serde_json::json!({
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hi!"}]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");

    assert_eq!(response.status(), StatusCode::OK);
    let chat: BancoChatResponse = json_body(response).await;
    assert_eq!(chat.model, "llama3");
}

// ============================================================================
// BANCO_HDL_006: POST /api/v1/chat/completions — empty messages rejected
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_006_empty_messages_rejected() {
    let app = test_app();
    let body = serde_json::json!({
        "messages": []
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let err: ErrorResponse = json_body(response).await;
    assert_eq!(err.error.type_, "invalid_request");
    assert!(err.error.message.contains("empty"));
}

// ============================================================================
// BANCO_HDL_007: POST /api/v1/chat/completions — streaming
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_007_chat_completions_streaming() {
    let app = test_app();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");

    assert_eq!(response.status(), StatusCode::OK);
    // SSE responses have text/event-stream content type
    let ct = response.headers().get("content-type").expect("content-type").to_str().expect("str");
    assert!(ct.contains("text/event-stream"));

    // Read full body and check it contains SSE data lines
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("data:"));
    assert!(text.contains("[DONE]"));
}

// ============================================================================
// BANCO_HDL_008: Privacy header present on all responses
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_BANCO_HDL_008_privacy_header_on_health() {
    let app = test_app();
    let response =
        app.oneshot(Request::get("/health").body(Body::empty()).expect("req")).await.expect("resp");

    let tier = response
        .headers()
        .get("x-privacy-tier")
        .expect("x-privacy-tier header")
        .to_str()
        .expect("str");
    assert_eq!(tier, "standard");
}
