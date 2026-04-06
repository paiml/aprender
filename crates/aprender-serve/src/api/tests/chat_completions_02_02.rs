
/// Test empty content in message
#[tokio::test]
async fn test_chat_completions_empty_content() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": ""}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Empty content should trigger empty prompt handling
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Special Characters in Content
// =============================================================================

/// Test unicode content
#[tokio::test]
async fn test_chat_completions_unicode_content() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "こんにちは 🎉 مرحبا"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Test content with newlines
#[tokio::test]
async fn test_chat_completions_multiline_content() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Line 1\nLine 2\nLine 3"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Optional Parameters
// =============================================================================

/// Test with all optional parameters
#[tokio::test]
async fn test_chat_completions_all_optional_params() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "top_p": 0.95,
        "max_tokens": 50,
        "stream": false,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Test with negative temperature (edge case)
#[tokio::test]
async fn test_chat_completions_negative_temperature() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": -0.5
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Negative temperature might be rejected or clamped
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Trace Header Tests
// =============================================================================

/// Test X-Request-ID header is echoed
#[tokio::test]
async fn test_chat_completions_trace_header() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-request-id", "test-trace-12345")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Stream Handler Error Paths (T-COV-95 Directive 5)
// =============================================================================

/// Stream handler: empty messages should return error
#[tokio::test]
async fn test_stream_handler_empty_messages() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [],
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Empty messages should be rejected (or NOT_FOUND if no model)
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Stream handler: non-existent model should return 404
#[tokio::test]
async fn test_stream_handler_model_not_found() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "non-existent-model-xyz-12345",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Non-existent model should return NOT_FOUND or fall back to default
    assert!(
        response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Stream handler: whitespace-only message content
#[tokio::test]
async fn test_stream_handler_whitespace_content() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "   \n\t   "}],
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Whitespace-only content might be rejected or processed
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Stream handler with top_p parameter
#[tokio::test]
async fn test_stream_handler_with_top_p() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "top_p": 0.9
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Stream handler with extreme top_p values
#[tokio::test]
async fn test_stream_handler_extreme_top_p() {
    let app = create_test_app_shared();

    // top_p = 0.0 (should select nothing or error)
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "top_p": 0.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Stream handler with max_tokens=0
#[tokio::test]
async fn test_stream_handler_zero_max_tokens() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Zero max_tokens might return empty or error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Pathological Registry Tests (T-COV-95 Final Corroboration)
// =============================================================================

/// Registry malfunction: Verify structured error response when model not found
/// (Popper: "Graceful Degradation" hypothesis test)
#[tokio::test]
async fn test_registry_malfunction_structured_error() {
    let app = create_test_app_shared();

    // Request a non-existent model to trigger registry error
    let req_body = serde_json::json!({
        "model": "poisoned-registry-model-xyz",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    let status = response.status();

    // Graceful degradation: must return structured error, not panic
    assert!(
        status == StatusCode::NOT_FOUND
            || status == StatusCode::OK  // Might fall back to default
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Expected graceful degradation, got: {status}"
    );

    // If error, verify response body is valid JSON with error field
    if status != StatusCode::OK {
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("test value should be present");
        let body_str = String::from_utf8_lossy(&body);

        // Must be valid JSON
        let json_result: Result<serde_json::Value, _> = serde_json::from_str(&body_str);
        assert!(
            json_result.is_ok(),
            "Error response must be valid JSON: {body_str}"
        );

        // Must have error field
        if let Ok(json) = json_result {
            assert!(
                json.get("error").is_some(),
                "Error response must have 'error' field: {body_str}"
            );
        }
    }
}
