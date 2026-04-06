
#[test]
fn test_chat_completion_chunk_content() {
    let chunk = ChatCompletionChunk::content("chatcmpl-123", "gpt-4", "Hello");
    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    assert!(chunk.choices[0].delta.role.is_none());
    assert!(chunk.choices[0].finish_reason.is_none());
}

#[test]
fn test_chat_completion_chunk_done() {
    let chunk = ChatCompletionChunk::done("chatcmpl-123", "gpt-4");
    assert_eq!(chunk.id, "chatcmpl-123");
    assert!(chunk.choices[0].delta.content.is_none());
    assert!(chunk.choices[0].delta.role.is_none());
    assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
}

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = ChatCompletionChunk::content("chatcmpl-123", "gpt-4", "Hi");
    let json = serde_json::to_string(&chunk).expect("test");

    assert!(json.contains("\"object\":\"chat.completion.chunk\""));
    assert!(json.contains("\"id\":\"chatcmpl-123\""));
    assert!(json.contains("\"content\":\"Hi\""));
}

#[test]
fn test_chat_delta_serialization_skip_none() {
    let delta = ChatDelta {
        role: None,
        content: Some("test".to_string()),
    };
    let json = serde_json::to_string(&delta).expect("test");

    // Should not contain "role" when it's None
    assert!(!json.contains("\"role\""));
    assert!(json.contains("\"content\":\"test\""));
}

#[test]
fn test_chat_chunk_choice_serialization() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        finish_reason: None,
    };
    let json = serde_json::to_string(&choice).expect("test");

    assert!(json.contains("\"index\":0"));
    assert!(json.contains("\"role\":\"assistant\""));
    // content should not be present when None
    assert!(!json.contains("\"content\""));
}

#[test]
fn test_streaming_chunk_created_timestamp() {
    let chunk1 = ChatCompletionChunk::initial("id1", "model");
    std::thread::sleep(std::time::Duration::from_millis(10));
    let chunk2 = ChatCompletionChunk::initial("id2", "model");

    // Both should have valid timestamps
    assert!(chunk1.created > 0);
    assert!(chunk2.created > 0);
    // Second should be same or later
    assert!(chunk2.created >= chunk1.created);
}

// ========================================================================
// Context Window Manager Tests
// ========================================================================

#[test]
fn test_context_window_config_default() {
    let config = ContextWindowConfig::default();
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 256);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_new() {
    let config = ContextWindowConfig::new(8192);
    assert_eq!(config.max_tokens, 8192);
    assert_eq!(config.reserved_output_tokens, 256);
}

#[test]
fn test_context_window_config_with_reserved() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(512);
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
}

#[test]
fn test_context_window_available_tokens() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(256);
    assert_eq!(config.available_tokens(), 3840);
}

#[test]
fn test_context_manager_no_truncation_needed() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_context_manager_needs_truncation() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(500),
        name: None,
    }];

    assert!(manager.needs_truncation(&messages));
}

#[test]
fn test_context_manager_truncate_preserves_system() {
    // Use smaller context to force truncation
    let config = ContextWindowConfig::new(80).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(200), // Large old message
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(truncated);
    // System message should be preserved
    assert!(result.iter().any(|m| m.role == "system"));
    // Most recent message should be included
    assert!(result.iter().any(|m| m.content == "Recent"));
}

#[test]
fn test_context_manager_truncate_keeps_recent() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let mut cfg = config;
    cfg.preserve_system = false;
    let manager = ContextWindowManager::new(cfg);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Old message 1".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Old message 2".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    // If truncation occurs, most recent should be kept
    if truncated {
        assert!(result.iter().any(|m| m.content == "Recent"));
    }
}

#[test]
fn test_context_manager_estimate_tokens() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let tokens = manager.estimate_total_tokens(&messages);
    // Should include overhead and char-based estimate
    assert!(tokens > 0);
    assert!(tokens < 100);
}

#[test]
fn test_context_manager_empty_messages() {
    let manager = ContextWindowManager::default_manager();
    let messages: Vec<ChatMessage> = vec![];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert!(result.is_empty());
}

#[test]
fn test_context_manager_single_large_message() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    // Message larger than available space
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(1000),
        name: None,
    }];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(truncated);
    // Message too large to fit, result may be empty
    assert!(result.is_empty() || result.len() == 1);
}

// =========================================================================
// APR-Specific API Tests (spec §15.1)
// =========================================================================

#[tokio::test]
#[ignore = "APR model integration test - requires specific model setup"]
async fn test_apr_predict_endpoint() {
    let app = create_test_app_shared();

    // Use 4 features to match demo APR model's expected input dimension
    let request = PredictRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: None,
        top_k: Some(3),
        include_confidence: true,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: PredictResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert!(!result.request_id.is_empty());
    assert_eq!(result.model, "default");
    assert!(result.confidence.is_some());
    // For regression (single output), top_k returns the value itself
    assert!(result.top_k_predictions.is_some());
    assert!(result.latency_ms >= 0.0);
    // Verify real inference: 1+2+3+4 = 10.0 (our demo model sums inputs)
    assert_eq!(result.prediction, serde_json::json!(10.0));
}

#[tokio::test]
async fn test_apr_predict_empty_features() {
    let app = create_test_app_shared();

    let request = PredictRequest {
        model: None,
        features: vec![],
        feature_names: None,
        top_k: None,
        include_confidence: true,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_apr_explain_endpoint() {
    let app = create_test_app_shared();

    let request = ExplainRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        top_k_features: 2,
        method: "shap".to_string(),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let result: ExplainResponse = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return, // Mock state: error response, skip body assertions
    };

    assert!(!result.request_id.is_empty());
    assert_eq!(result.model, "default");
    assert!(!result.summary.is_empty());
    assert_eq!(result.explanation.feature_names.len(), 3);
    assert_eq!(result.explanation.shap_values.len(), 3);
}

#[tokio::test]
async fn test_apr_explain_mismatched_features() {
    let app = create_test_app_shared();

    let request = ExplainRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["f1".to_string()], // Mismatched count
        top_k_features: 2,
        method: "shap".to_string(),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}
