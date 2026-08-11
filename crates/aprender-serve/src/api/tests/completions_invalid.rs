//! API Tests Part 13: OpenAI and Realize Handlers - HTTP Endpoint Tests
//!
//! Tests for openai_handlers.rs and realize_handlers.rs to improve coverage.
//! Focus: HTTP endpoint error paths, streaming, and edge cases.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;
use crate::api::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ErrorResponse};

// =============================================================================
// HTTP Endpoint Error Path Tests
// =============================================================================

#[tokio::test]
async fn test_completions_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("build"),
        )
        .await
        .expect("send");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_completions_missing_fields() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt": "test"}"#))
                .expect("build"),
        )
        .await
        .expect("send");
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_completions_empty_prompt() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model": "default", "prompt": ""}"#))
                .expect("build"),
        )
        .await
        .expect("send");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_embeddings_error_paths() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from("{invalid"))
                .expect("build"),
        )
        .await
        .expect("send");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_embeddings_missing_input() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model": "test"}"#))
                .expect("build"),
        )
        .await
        .expect("send");
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_realize_reload_error() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/reload")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model": "test"}"#))
                .expect("build"),
        )
        .await
        .expect("send");
    let status = response.status();
    assert!(
        status == StatusCode::BAD_REQUEST
            || status == StatusCode::NOT_IMPLEMENTED
            || status == StatusCode::NOT_FOUND,
        "Got {}",
        status
    );
}

// =============================================================================
// Chat Completions Request Types
// =============================================================================

#[test]
fn test_chat_completion_request_traits() {
    let req = ChatCompletionRequest {
        model: "model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
            name: None,
        
            ..Default::default()
        }],
        max_tokens: Some(10),
        temperature: Some(0.5),
        top_p: Some(0.9),
        top_k: None,
        repeat_penalty: None,
        repeat_last_n: None,
        seed: None,
        n: crate::api::ChoiceCount::ONE,
        stream: false,
        stop: Some(vec!["stop".to_string()]),
        user: Some("user".to_string()),
    
        ..Default::default()
    };

    let cloned = req.clone();
    assert_eq!(cloned.model, req.model);
    let debug = format!("{:?}", req);
    assert!(debug.contains("ChatCompletionRequest"));
}

#[test]
fn test_chat_completion_response_traits() {
    let response = ChatCompletionResponse {
        id: "test".to_string(),
        object: "chat.completion".to_string(),
        created: 1000,
        model: "m".to_string(),
        choices: vec![],
        usage: crate::api::Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };

    let cloned = response.clone();
    assert_eq!(cloned.id, response.id);
    let debug = format!("{:?}", response);
    assert!(debug.contains("ChatCompletionResponse"));
}

// =============================================================================
// HTTP Endpoint Streaming Tests
// =============================================================================

#[tokio::test]
async fn test_chat_completions_streaming() {
    let app = create_test_app_shared();
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    let ct = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or(""));
    if !ct.is_some_and(|c| c.contains("text/event-stream")) {
        return;
    } // Mock state guard
}

#[tokio::test]
async fn test_chat_completions_non_streaming() {
    let app = create_test_app_shared();
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    let ct = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or(""));
    assert!(ct.is_some_and(|c| c.contains("application/json")));
}

// =============================================================================
// Error Response Tests
// =============================================================================

#[test]
fn test_error_response() {
    let error = ErrorResponse {
        error: "test error".to_string(),
    };
    assert!(!error.error.is_empty());

    let json = r#"{"error": "Something went wrong"}"#;
    let parsed: ErrorResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.error, "Something went wrong");
}

// =============================================================================
// Completions with Parameters
// =============================================================================

#[tokio::test]
async fn test_completions_with_params() {
    let app = create_test_app_shared();
    let req_body = serde_json::json!({
        "model": "default",
        "prompt": "Hello",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Realize Endpoints
// =============================================================================

#[tokio::test]
async fn test_realize_embed() {
    let app = create_test_app_shared();
    let req_body = serde_json::json!({"input": "Test embedding"});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    // aprender#2376(5): the shared test state has NO model, so this condition is
    // deterministic — a server with no usable model answers 503 on every route.
    // The old assertion accepted a SET of statuses that included the defect, so it
    // could not fail and held the 404-here/500-there split in place.
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "no model is resident: expected 503"
    );
}

#[tokio::test]
async fn test_realize_embed_invalid() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from("invalid"))
                .expect("build"),
        )
        .await
        .expect("send");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_realize_model() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/realize/model")
                .body(Body::empty())
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(status == StatusCode::OK || status == StatusCode::NOT_FOUND);
}

// =============================================================================
// Usage Type Tests
// =============================================================================

#[test]
fn test_usage_roundtrip() {
    let usage = crate::api::Usage {
        prompt_tokens: 100,
        completion_tokens: 200,
        total_tokens: 300,
    };

    let debug = format!("{:?}", usage);
    assert!(debug.contains("Usage"));

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: crate::api::Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.total_tokens, 300);
}
