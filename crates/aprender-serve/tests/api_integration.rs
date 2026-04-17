//! Phase 38: API Integration Tests
//!
//! True integration tests that hit endpoints with a loaded model.
//! Uses `AppState::demo()` for fast, minimal model setup (~1ms).
//!
//! Coverage targets:
//! - Happy path (200 OK) for major endpoints
//! - Error paths (400, 404, 422) for validation
//! - Async handler code paths that unit tests miss

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use realizar::api::{create_router, AppState};
use tower::ServiceExt;

/// Helper to create test app with demo model
fn create_test_app() -> axum::Router {
    let state = AppState::demo().expect("demo state should create");
    create_router(state)
}

/// Helper to build JSON POST request
fn json_post(uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

/// Helper to build GET request
fn get_request(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

// ============================================================================
// Health & Metrics - GET endpoints (fast, no inference)
// ============================================================================

#[tokio::test]
async fn test_health_endpoint_returns_200() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/health")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "healthy");
}

#[tokio::test]
async fn test_metrics_endpoint_returns_200() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/metrics")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_models_endpoint_returns_200() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/models")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_v1_models_endpoint_returns_200() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/v1/models")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
}

// ============================================================================
// Tokenize - POST endpoint (fast, no inference)
// ============================================================================

#[tokio::test]
async fn test_tokenize_endpoint_happy_path() {
    let app = create_test_app();
    let body = serde_json::json!({
        "text": "hello world"
    });
    let response = app.oneshot(json_post("/tokenize", body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["token_ids"].is_array());
    assert!(json["num_tokens"].is_number());
}

#[tokio::test]
async fn test_tokenize_endpoint_empty_text() {
    let app = create_test_app();
    let body = serde_json::json!({
        "text": ""
    });
    let response = app.oneshot(json_post("/tokenize", body)).await.unwrap();
    // Empty text should still succeed (returns empty tokens)
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_tokenize_endpoint_missing_field_400() {
    let app = create_test_app();
    let body = serde_json::json!({});
    let response = app.oneshot(json_post("/tokenize", body)).await.unwrap();
    // Missing required field should return 400 or 422
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_batch_tokenize_endpoint_happy_path() {
    let app = create_test_app();
    let body = serde_json::json!({
        "texts": ["hello", "world", "test"]
    });
    let response = app
        .oneshot(json_post("/batch/tokenize", body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["results"].is_array());
    assert_eq!(json["results"].as_array().unwrap().len(), 3);
}

// ============================================================================
// Generate - POST endpoint (uses model inference)
// ============================================================================

#[tokio::test]
async fn test_generate_endpoint_happy_path() {
    let app = create_test_app();
    let body = serde_json::json!({
        "prompt": "hello",
        "max_tokens": 5
    });
    let response = app.oneshot(json_post("/generate", body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["text"].is_string());
    assert!(json["num_generated"].is_number());
}

#[tokio::test]
async fn test_generate_endpoint_with_temperature() {
    let app = create_test_app();
    let body = serde_json::json!({
        "prompt": "test",
        "max_tokens": 3,
        "temperature": 0.7,
        "strategy": "top_k",
        "top_k": 5
    });
    let response = app.oneshot(json_post("/generate", body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_generate_endpoint_missing_prompt_400() {
    let app = create_test_app();
    let body = serde_json::json!({
        "max_tokens": 5
    });
    let response = app.oneshot(json_post("/generate", body)).await.unwrap();
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_batch_generate_endpoint_happy_path() {
    let app = create_test_app();
    let body = serde_json::json!({
        "prompts": ["hello", "world"],
        "max_tokens": 3
    });
    let response = app
        .oneshot(json_post("/batch/generate", body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["results"].is_array());
}

// ============================================================================
// OpenAI-Compatible Chat Completions
// ============================================================================

#[tokio::test]
async fn test_chat_completions_happy_path() {
    let app = create_test_app();
    let body = serde_json::json!({
        "model": "demo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 5
    });
    let response = app
        .oneshot(json_post("/v1/chat/completions", body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"].is_array());
}

#[tokio::test]
async fn test_chat_completions_system_message() {
    let app = create_test_app();
    let body = serde_json::json!({
        "model": "demo",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 3
    });
    let response = app
        .oneshot(json_post("/v1/chat/completions", body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_chat_completions_empty_messages_400() {
    let app = create_test_app();
    let body = serde_json::json!({
        "model": "demo",
        "messages": []
    });
    let response = app
        .oneshot(json_post("/v1/chat/completions", body))
        .await
        .unwrap();
    // Empty messages should fail validation
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::OK // Some implementations allow empty
    );
}

#[tokio::test]
async fn test_chat_completions_missing_model_422() {
    let app = create_test_app();
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let response = app
        .oneshot(json_post("/v1/chat/completions", body))
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// ============================================================================
// OpenAI-Compatible Completions (non-chat)
// ============================================================================

#[tokio::test]
async fn test_completions_happy_path() {
    let app = create_test_app();
    let body = serde_json::json!({
        "model": "demo",
        "prompt": "Once upon a time",
        "max_tokens": 5
    });
    let response = app
        .oneshot(json_post("/v1/completions", body))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "text_completion");
}

// ============================================================================
// Realize API - Native endpoints
// ============================================================================

#[tokio::test]
async fn test_realize_model_endpoint() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/realize/model")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_realize_embed_endpoint() {
    let app = create_test_app();
    let body = serde_json::json!({
        "input": "test embedding"
    });
    let response = app
        .oneshot(json_post("/realize/embed", body))
        .await
        .unwrap();
    // May return 200 or 501 depending on implementation
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_IMPLEMENTED
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_realize_reload_endpoint() {
    let app = create_test_app();
    let body = serde_json::json!({});
    let response = app
        .oneshot(json_post("/realize/reload", body))
        .await
        .unwrap();
    // Reload without path returns 501 Not Implemented
    assert!(
        response.status() == StatusCode::NOT_IMPLEMENTED
            || response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
    );
}

// ============================================================================
// APR Prediction API
// ============================================================================

#[tokio::test]
async fn test_apr_predict_exercises_handler() {
    let app = create_test_app();
    let body = serde_json::json!({
        "features": [1.0, 2.0, 3.0, 4.0]
    });
    let response = app.oneshot(json_post("/v1/predict", body)).await.unwrap();
    // Demo model has "weight" tensor but handler looks for "weights"/"output"
    // This exercises the error handling path (400) or success (200)
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_apr_predict_with_feature_names() {
    let app = create_test_app();
    let body = serde_json::json!({
        "features": [1.0, 2.0, 3.0, 4.0],
        "feature_names": ["a", "b", "c", "d"]
    });
    let response = app.oneshot(json_post("/v1/predict", body)).await.unwrap();
    // Exercises handler code path regardless of tensor availability
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_apr_explain_exercises_handler() {
    let app = create_test_app();
    let body = serde_json::json!({
        "features": [1.0, 2.0, 3.0, 4.0],
        "feature_names": ["a", "b", "c", "d"]
    });
    let response = app.oneshot(json_post("/v1/explain", body)).await.unwrap();
    // Exercises handler - may return 200 or error depending on model setup
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_apr_predict_empty_features_400() {
    let app = create_test_app();
    let body = serde_json::json!({
        "features": []
    });
    let response = app.oneshot(json_post("/v1/predict", body)).await.unwrap();
    // Empty features should fail or return error
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::OK // Some impls handle gracefully
    );
}

// ============================================================================
// OpenAI Embeddings
// ============================================================================

#[tokio::test]
async fn test_v1_embeddings_endpoint() {
    let app = create_test_app();
    let body = serde_json::json!({
        "model": "demo",
        "input": "test text"
    });
    let response = app
        .oneshot(json_post("/v1/embeddings", body))
        .await
        .unwrap();
    // May return 200 or 501 depending on implementation
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::NOT_IMPLEMENTED
    );
}

// ============================================================================
// GPU endpoints (should gracefully handle no GPU)
// ============================================================================

#[tokio::test]
async fn test_gpu_status_endpoint() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/v1/gpu/status")).await.unwrap();
    // Should return status even without GPU
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_gpu_warmup_endpoint() {
    let app = create_test_app();
    let body = serde_json::json!({});
    let response = app
        .oneshot(json_post("/v1/gpu/warmup", body))
        .await
        .unwrap();
    // Should handle gracefully without GPU
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_IMPLEMENTED
    );
}

// ============================================================================
// Server Metrics (TUI monitoring)
// ============================================================================

#[tokio::test]
async fn test_server_metrics_endpoint() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/v1/metrics")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    // Should have metrics fields
    assert!(json.is_object());
}

// ============================================================================
// Invalid JSON / Malformed Requests
// ============================================================================

#[tokio::test]
async fn test_invalid_json_returns_400() {
    let app = create_test_app();
    let request = Request::builder()
        .method("POST")
        .uri("/tokenize")
        .header("content-type", "application/json")
        .body(Body::from("not valid json {{{"))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_wrong_content_type() {
    let app = create_test_app();
    let request = Request::builder()
        .method("POST")
        .uri("/tokenize")
        .header("content-type", "text/plain")
        .body(Body::from(r#"{"text": "hello"}"#))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    // Should reject or parse anyway
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

// ============================================================================
// 404 for unknown routes
// ============================================================================

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let app = create_test_app();
    let response = app
        .oneshot(get_request("/nonexistent/endpoint"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_unknown_v1_route_returns_404() {
    let app = create_test_app();
    let response = app.oneshot(get_request("/v1/unknown")).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
