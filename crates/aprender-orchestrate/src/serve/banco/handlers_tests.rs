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
// PP-27: the SSE stream declares its mechanism and ends with `usage`
// ============================================================================

/// The defect this pins: the stream ended `content chunk -> [DONE]` with no
/// terminal `usage`, and a measuring client refuses such a stream outright
/// (`jugar_probar`'s `LlmClientError::StreamNoUsage`) rather than counting
/// frames — so `banco_llm.rs::l2_chat_streaming_sse` panicked on the `unwrap`.
/// Asserted over the wire, on the frames a client actually parses.
#[tokio::test]
async fn stream_ends_with_a_usage_chunk_before_done() {
    let app = test_app();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Count to 3"}],
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
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let text = String::from_utf8_lossy(&bytes);

    let payloads: Vec<&str> =
        text.lines().filter_map(|line| line.strip_prefix("data: ")).map(str::trim).collect();
    assert_eq!(
        payloads.last().copied(),
        Some("[DONE]"),
        "the stream must still end with the sentinel:\n{text}"
    );

    let frames: Vec<serde_json::Value> = payloads
        .iter()
        .filter(|p| **p != "[DONE]")
        .map(|p| serde_json::from_str(p).expect("each SSE frame must be JSON"))
        .collect();

    // PP-27: the mechanism is declared on the FIRST chunk, and banco replays.
    assert_eq!(
        frames.first().expect("a first chunk")["stream_mode"].as_str(),
        Some("replayed"),
        "banco generates the whole completion before the first frame:\n{text}"
    );

    // ...and the LAST chunk before the sentinel carries the counts.
    let terminal = frames.last().expect("a terminal chunk");
    assert!(
        !terminal["usage"].is_null(),
        "the terminal chunk must carry `usage`, or a measuring client refuses \
         the whole stream:\n{text}"
    );
    for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        assert!(
            terminal["usage"][field].as_u64().is_some(),
            "`usage.{field}` must be a number:\n{terminal}"
        );
    }
    assert_eq!(
        terminal["usage"]["total_tokens"].as_u64(),
        Some(
            terminal["usage"]["prompt_tokens"].as_u64().unwrap_or_default()
                + terminal["usage"]["completion_tokens"].as_u64().unwrap_or_default()
        ),
        "the total must be the sum it claims to be:\n{terminal}"
    );
    assert!(
        terminal["usage"]["prompt_tokens"].as_u64().unwrap_or_default() > 0,
        "a non-empty prompt must not report 0 prompt tokens — that is the \
         placeholder a real empty prompt is indistinguishable from:\n{terminal}"
    );
}

/// `stream_usage` counts TOKENS, not frames: a generated stream's terminal
/// marker (`finish_reason` set, no text) is not a token, and the dry-run has no
/// generation to count so it falls back to the same estimate the non-streaming
/// path reports.
#[test]
fn stream_usage_counts_tokens_not_frames() {
    use super::handlers::{stream_usage, BANCO_STREAM_MODE};

    let frames = vec![
        ("one".to_string(), None),
        ("two".to_string(), None),
        (String::new(), Some("length".to_string())),
    ];
    let measured = stream_usage(9, &frames, true);
    assert_eq!(
        measured.completion_tokens, 2,
        "the terminal marker carries the reason, not a token"
    );
    assert_eq!(measured.prompt_tokens, 9);
    assert_eq!(measured.total_tokens, 11);

    // The dry-run counts characters, not the four canned frames.
    let canned = vec![
        ("12345678".to_string(), None),
        ("12345678".to_string(), None),
        (String::new(), Some("dry_run".to_string())),
    ];
    let estimated = stream_usage(3, &canned, false);
    assert_eq!(estimated.completion_tokens, 4, "16 chars / 4");
    assert_eq!(estimated.total_tokens, 7);
    assert_ne!(
        stream_usage(3, &canned, true).completion_tokens,
        estimated.completion_tokens,
        "a generated stream and a canned one must not be counted the same way"
    );

    assert_eq!(
        BANCO_STREAM_MODE, "replayed",
        "generation finishes before the first frame is written"
    );
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
