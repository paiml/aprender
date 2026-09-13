
/// Test message with only system role (no user message)
#[tokio::test]
async fn test_chat_completions_system_only() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

// =============================================================================
// top_p Parameter (Nucleus Sampling)
// =============================================================================

/// Test top_p parameter triggers nucleus sampling branch
#[tokio::test]
async fn test_chat_completions_with_top_p() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "top_p": 0.9,
        "temperature": 0.7
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // top_p should trigger request.top_p.is_some() branch
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

/// Test top_p=1.0 (full distribution)
#[tokio::test]
async fn test_chat_completions_top_p_one() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "top_p": 1.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

// =============================================================================
// Content-Type Variations
// =============================================================================

/// Test request without content-type header
#[tokio::test]
async fn test_chat_completions_no_content_type() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Should still work or return appropriate error
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

/// Test request with wrong content-type
#[tokio::test]
async fn test_chat_completions_wrong_content_type() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "text/plain")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Empty Body
// =============================================================================

/// Test request with empty body
#[tokio::test]
async fn test_chat_completions_empty_body() {
    let app = create_test_app_shared();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Empty body should fail parsing (or return NOT_FOUND if no model)
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::NOT_FOUND
    );
}

// =============================================================================
// Malformed Message Objects
// =============================================================================

/// Test message missing 'role' field
#[tokio::test]
async fn test_chat_completions_message_missing_role() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Missing role should return 422"
    );
}

/// Test message missing 'content' field
#[tokio::test]
async fn test_chat_completions_message_missing_content() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "Missing content should return 422"
    );
}

/// Test message with invalid role
#[tokio::test]
async fn test_chat_completions_invalid_role() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "invalid_role", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Invalid role might be accepted or rejected depending on strictness
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

// =============================================================================
// Combinatorial Matrix: stream × temperature × max_tokens
// =============================================================================

/// Combinatorial: stream=true, temp=0.0, max_tokens=1
#[tokio::test]
async fn test_combo_stream_greedy_min_tokens() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "temperature": 0.0,
        "max_tokens": 1
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

/// Combinatorial: stream=false, temp=1.5, max_tokens=100
#[tokio::test]
async fn test_combo_no_stream_creative_mid_tokens() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false,
        "temperature": 1.5,
        "max_tokens": 100
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

/// Combinatorial: stream=true, temp=0.7, max_tokens=256 (defaults equivalent)
#[tokio::test]
async fn test_combo_stream_default_params() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "temperature": 0.7,
        "max_tokens": 256
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

// =============================================================================
// HTTP Method Variations
// =============================================================================

/// Test GET method (should fail - POST only)
#[tokio::test]
async fn test_chat_completions_get_method() {
    let app = create_test_app_shared();

    let request = Request::builder()
        .method("GET")
        .uri("/v1/chat/completions")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "GET should return 405"
    );
}

/// Test PUT method (should fail)
#[tokio::test]
async fn test_chat_completions_put_method() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request = Request::builder()
        .method("PUT")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert_eq!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "PUT should return 405"
    );
}

// =============================================================================
// Large Message Content
// =============================================================================

/// Test long message content (moderate size to avoid timeout)
#[tokio::test]
async fn test_chat_completions_large_content() {
    let app = create_test_app_shared();

    // Use 1000 chars instead of 10000 to test large content handling without timeout
    let large_content = "x".repeat(1000);
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": large_content}],
        "max_tokens": 5
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Large content should be handled
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}
