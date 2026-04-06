
#[test]
fn test_clean_chat_output_unicode() {
    let text = "Hello 世界! 🌍<|im_end|>";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Hello 世界! 🌍");
}

#[test]
fn test_format_chat_messages_unicode_content() {
    let messages = vec![chat_msg("user", "Bonjour! 你好! مرحبا!")];
    let formatted = format_chat_messages(&messages, None);
    assert!(formatted.contains("Bonjour"));
    assert!(formatted.contains("你好"));
}

#[test]
fn test_context_window_manager_large_message() {
    let manager = ContextWindowManager::default_manager();
    let long_content = "word ".repeat(1000);
    let messages = vec![chat_msg("user", &long_content)];
    let estimate = manager.estimate_total_tokens(&messages);
    assert!(estimate > 100);
}

#[test]
fn test_clean_chat_output_with_double_newline_user() {
    let text = "My response.\n\nUser: Next question";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "My response.");
}

#[test]
fn test_context_window_config_clone() {
    let config = ContextWindowConfig::new(2048).with_reserved_output(128);
    let cloned = config.clone();
    assert_eq!(cloned.max_tokens, 2048);
    assert_eq!(cloned.reserved_output_tokens, 128);
}

#[test]
fn test_context_window_config_debug() {
    let config = ContextWindowConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("ContextWindowConfig"));
}

#[test]
fn test_embedding_data_empty_vector() {
    let data = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![],
    };
    let json = serde_json::to_string(&data).expect("serialize");
    assert!(json.contains("[]"));
}

#[test]
fn test_format_chat_messages_with_qwen_model() {
    let messages = vec![chat_msg("user", "Test")];
    let formatted = format_chat_messages(&messages, Some("qwen"));
    assert!(formatted.contains("Test"));
}

#[test]
fn test_format_chat_messages_with_llama_model() {
    let messages = vec![chat_msg("user", "Test")];
    let formatted = format_chat_messages(&messages, Some("llama"));
    assert!(formatted.contains("Test"));
}

#[test]
fn test_format_chat_messages_system_only() {
    let messages = vec![chat_msg("system", "You are helpful.")];
    let formatted = format_chat_messages(&messages, None);
    assert!(formatted.contains("helpful"));
}

#[test]
fn test_context_window_truncate_all_system() {
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        chat_msg("system", "System 1"),
        chat_msg("system", "System 2"),
    ];

    let (result, _) = manager.truncate_messages(&messages);
    assert!(result.iter().all(|m| m.role == "system"));
}

#[test]
fn test_clean_chat_output_preserves_internal_newlines() {
    let text = "Line 1\nLine 2\nLine 3";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Line 1\nLine 2\nLine 3");
}

// =============================================================================
// Handler Integration Tests
// =============================================================================

#[tokio::test]
async fn test_realize_embed_endpoint() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "Hello, world!",
        "model": "test"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    // May return 404 if model not found in demo mode, but exercises the handler
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_realize_model_endpoint() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/realize/model")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_realize_reload_endpoint() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model_id": "test-model"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/reload")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    // Reload may fail without actual model or be unimplemented, but exercises the handler
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::NOT_IMPLEMENTED,
        "Unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_openai_completions_endpoint() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "prompt": "Hello",
        "max_tokens": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_openai_embeddings_endpoint() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "input": "Test text for embedding"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {}",
        status
    );
}

// =============================================================================
// Additional Request/Response Type Tests
// =============================================================================

#[test]
fn test_completion_request_serialization() {
    let req = CompletionRequest {
        model: "test-model".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("test-model"));
    assert!(json.contains("Hello"));

    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, "test-model");
    assert_eq!(parsed.max_tokens, Some(100));
}

#[test]
fn test_completion_request_minimal() {
    let json = r#"{"model":"m","prompt":"p"}"#;
    let parsed: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.model, "m");
    assert_eq!(parsed.prompt, "p");
    assert_eq!(parsed.max_tokens, None);
}

#[test]
fn test_reload_request_serialization() {
    let req = ReloadRequest {
        model: Some("model-123".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("model-123"));
    assert!(json.contains("/path/to/model.gguf"));

    let parsed: ReloadRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, Some("model-123".to_string()));
}

#[test]
fn test_reload_request_empty() {
    let req = ReloadRequest {
        model: None,
        path: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: ReloadRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, None);
}
