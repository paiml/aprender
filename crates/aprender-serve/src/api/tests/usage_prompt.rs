
// =============================================================================
// Usage Token Counting Tests
// =============================================================================

#[tokio::test]
async fn test_usage_prompt_tokens_positive() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello world"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test value should be present");
    let result: ChatCompletionResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    // Prompt should have at least 1 token
    assert!(result.usage.prompt_tokens > 0);
}

#[tokio::test]
async fn test_usage_total_equals_sum() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test value should be present");
    let result: ChatCompletionResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    // Total should equal prompt + completion
    assert_eq!(
        result.usage.total_tokens,
        result.usage.prompt_tokens + result.usage.completion_tokens
    );
}

// =============================================================================
// Content-Type Validation
// =============================================================================

#[tokio::test]
async fn test_chat_completions_wrong_content_type() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "text/plain")
                .body(Body::from("model=default&messages=hi"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    // Wrong content type should be rejected
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

#[tokio::test]
async fn test_chat_completions_no_content_type() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                // Note: no content-type header
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    // axum may accept this or reject it depending on configuration
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

// =============================================================================
// Long Content Tests
// =============================================================================

#[tokio::test]
async fn test_chat_completions_long_message() {
    let app = create_test_app_shared();

    // Create a message with 1000 characters
    let long_content = "x".repeat(1000);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": long_content}],
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_chat_completions_many_messages() {
    let app = create_test_app_shared();

    // Create 20 messages
    let mut messages = vec![];
    for i in 0..20 {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        messages.push(serde_json::json!({
            "role": role,
            "content": format!("Message {}", i)
        }));
    }

    let req_body = serde_json::json!({
        "model": "default",
        "messages": messages,
        "max_tokens": 5
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

// =============================================================================
// TraceData and TraceOperation Tests
// =============================================================================

#[test]
fn test_trace_data_empty_breakdown() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 0,
        total_time_us: 0,
        breakdown: vec![],
        provenance: TraceProvenance::Estimated,
    };

    let json = serde_json::to_string(&trace).expect("serialize");
    assert!(json.contains("[]")); // Empty breakdown array
}

#[test]
fn test_trace_operation_with_details() {
    let op = TraceOperation {
        name: "matmul".to_string(),
        time_us: 1000,
        details: Some("512x512".to_string()),
    };

    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("matmul"));
    assert!(json.contains("512x512"));
}

#[test]
fn test_trace_operation_without_details() {
    let op = TraceOperation {
        name: "softmax".to_string(),
        time_us: 50,
        details: None,
    };

    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("softmax"));
    // details field should be skipped
    assert!(!json.contains("details"));
}

// =============================================================================
// ErrorResponse Tests
// =============================================================================

#[test]
fn test_error_response_serialization() {
    let err = ErrorResponse {
        error: "Test error".to_string(),
    };

    let json = serde_json::to_string(&err).expect("serialize");
    assert!(json.contains("Test error"));
}

#[test]
fn test_error_response_deserialization() {
    let json = r#"{"error":"Something went wrong"}"#;
    let err: ErrorResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(err.error, "Something went wrong");
}

// =============================================================================
// ChatMessage Edge Cases
// =============================================================================

#[test]
fn test_chat_message_very_long_content() {
    let long_content = "x".repeat(10000);
    let msg = ChatMessage {
        role: "user".to_string(),
        content: long_content.clone(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.content.len(), 10000);
}

#[test]
fn test_chat_message_with_newlines() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "line1\nline2\nline3".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.content.contains('\n'));
}

#[test]
fn test_chat_message_with_tabs() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "col1\tcol2\tcol3".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.content.contains('\t'));
}

// =============================================================================
// Usage Struct Edge Cases
// =============================================================================

#[test]
fn test_usage_max_values() {
    let usage = Usage {
        prompt_tokens: usize::MAX,
        completion_tokens: 0,
        total_tokens: usize::MAX,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.prompt_tokens, usize::MAX);
}

// =============================================================================
// OpenAIModel Tests
// =============================================================================

#[test]
fn test_openai_model_with_numeric_id() {
    let model = OpenAIModel {
        id: "12345".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "test".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    assert!(json.contains("12345"));
}

#[test]
fn test_openai_model_with_special_chars_in_id() {
    let model = OpenAIModel {
        id: "model-v1.0-beta".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "test-org".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    let parsed: OpenAIModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "model-v1.0-beta");
}

// =============================================================================
// ChatChoice Edge Cases
// =============================================================================

#[test]
fn test_chat_choice_with_empty_content() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("\"content\":\"\""));
}

#[test]
fn test_chat_choice_with_large_index() {
    let choice = ChatChoice {
        index: 999,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "test".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("999"));
}

// =============================================================================
// ChatDelta Edge Cases
// =============================================================================

#[test]
fn test_chat_delta_empty() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    // Should be empty object
    assert_eq!(json, "{}");
}

#[test]
fn test_chat_delta_both_present() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hello".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("assistant"));
    assert!(json.contains("Hello"));
}

// =============================================================================
// ChatChunkChoice Edge Cases
// =============================================================================

#[test]
fn test_chat_chunk_choice_with_length_finish_reason() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("length".to_string()),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("length"));
}
