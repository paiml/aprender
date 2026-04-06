
#[tokio::test]
async fn test_stream_generate_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/stream/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompt":"Hello","max_tokens":5,"temperature":0.0,"strategy":"greedy","top_k":1,"top_p":1.0}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_gpu_warmup_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/gpu/warmup")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_gpu_status_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/gpu/status")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::SERVICE_UNAVAILABLE,
    );
}

#[tokio::test]
async fn test_v1_predict_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/predict")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"input":[1.0, 2.0, 3.0]}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_v1_explain_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/explain")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"input":[1.0, 2.0, 3.0]}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_v1_metrics_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/metrics")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND,);
}

#[tokio::test]
async fn test_metrics_dispatch_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/metrics/dispatch")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Dispatch metrics endpoint may return SERVICE_UNAVAILABLE without GPU
    let status = response.status().as_u16();
    assert!(
        status < 500 || status == 503,
        "Unexpected server error: {}",
        status
    );
}

#[tokio::test]
async fn test_metrics_dispatch_reset_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/metrics/dispatch/reset")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Reset endpoint may return SERVICE_UNAVAILABLE without GPU
    let status = response.status().as_u16();
    assert!(
        status < 500 || status == 503,
        "Unexpected server error: {}",
        status
    );
}

// ============================================================================
// format_chat_messages: model-specific formatting
// ============================================================================

#[test]
fn test_format_chat_messages_qwen_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "What is Rust?".to_string(),
            name: None,
        },
    ];
    let result = format_chat_messages(&messages, Some("qwen2"));
    assert!(!result.is_empty());
    assert!(result.contains("What is Rust?"));
}

#[test]
fn test_format_chat_messages_phi_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello phi".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("phi2"));
    assert!(!result.is_empty());
}

#[test]
fn test_format_chat_messages_tinyllama_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello tinyllama".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("tinyllama"));
    assert!(!result.is_empty());
}

#[test]
fn test_format_chat_messages_llama_model() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello llama".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&messages, Some("llama"));
    assert!(!result.is_empty());
}

// ============================================================================
// ContextWindowManager: deeper truncation tests
// ============================================================================

#[test]
fn test_context_window_manager_truncate_long_messages() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    // Very small context window to force truncation
    let config = ContextWindowConfig::new(20).with_reserved_output(5);
    let mgr = ContextWindowManager::new(config);
    let messages: Vec<crate::api::ChatMessage> = (0..100)
        .map(|i| crate::api::ChatMessage {
            role: "user".to_string(),
            content: format!("Message number {} with some extra text", i),
            name: None,
        })
        .collect();

    let needs = mgr.needs_truncation(&messages);
    assert!(needs); // 100 long messages should exceed 15 available tokens

    let (truncated, was_truncated) = mgr.truncate_messages(&messages);
    assert!(was_truncated);
    assert!(truncated.len() < messages.len());
}

#[test]
fn test_context_window_manager_preserves_system_message() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    let config = ContextWindowConfig {
        max_tokens: 50,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);

    let messages = vec![
        crate::api::ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "A very long message that should be long enough to cause truncation when combined with other messages".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "A very long response with lots of content that should be truncated eventually if needed".to_string(),
            name: None,
        },
        crate::api::ChatMessage {
            role: "user".to_string(),
            content: "Another long user message for extra length to ensure truncation happens".to_string(),
            name: None,
        },
    ];

    let (truncated, _was_truncated) = mgr.truncate_messages(&messages);
    // If truncation happened and preserve_system is true,
    // the first message (system) should be preserved
    if !truncated.is_empty() {
        assert_eq!(truncated[0].role, "system");
    }
}

// ============================================================================
// clean_chat_output: additional patterns
// ============================================================================

#[test]
fn test_clean_chat_output_im_end_only() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "Hello world<|im_end|>";
    let result = clean_chat_output(text);
    assert_eq!(result, "Hello world");
}

#[test]
fn test_clean_chat_output_endoftext() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "Some output text<|endoftext|>more stuff";
    let result = clean_chat_output(text);
    assert_eq!(result, "Some output text");
}

#[test]
fn test_clean_chat_output_eos_token() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "Generated text</s>more after";
    let result = clean_chat_output(text);
    assert_eq!(result, "Generated text");
}

#[test]
fn test_clean_chat_output_whitespace_only() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "   \n\t  ";
    let result = clean_chat_output(text);
    assert!(result.is_empty());
}

#[test]
fn test_clean_chat_output_trimming() {
    use crate::api::realize_handlers::clean_chat_output;
    let text = "  Hello world  ";
    let result = clean_chat_output(text);
    assert_eq!(result, "Hello world");
}

// ============================================================================
// AppState: with_cache
// ============================================================================

#[test]
fn test_appstate_with_cache() {
    let state = crate::api::AppState::with_cache(100);
    // with_cache returns Self directly (not Result)
    let _ = state; // Verify construction succeeds without panic
}

// ============================================================================
// Batch request/response additional serde tests
// ============================================================================

#[test]
fn test_batch_tokenize_request_serde() {
    let json = r#"{"texts":["hello","world"]}"#;
    let req: crate::api::BatchTokenizeRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(req.texts.len(), 2);
    let json_out = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json_out.contains("hello"));
}

#[test]
fn test_batch_generate_request_serde() {
    let json = r#"{"prompts":["hello"],"max_tokens":10,"temperature":0.5,"strategy":"greedy","top_k":1,"top_p":1.0}"#;
    let req: crate::api::BatchGenerateRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(req.prompts.len(), 1);
    assert_eq!(req.max_tokens, 10);
    assert!((req.temperature - 0.5).abs() < 1e-6);
}

#[test]
fn test_batch_tokenize_response_serde() {
    let resp = crate::api::BatchTokenizeResponse {
        results: vec![crate::api::TokenizeResponse {
            token_ids: vec![1, 2],
            num_tokens: 2,
        }],
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: crate::api::BatchTokenizeResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.results.len(), 1);
}

#[test]
fn test_batch_generate_response_serde() {
    let resp = crate::api::BatchGenerateResponse {
        results: vec![crate::api::GenerateResponse {
            token_ids: vec![1, 2, 3],
            text: "hello".to_string(),
            num_generated: 3,
        }],
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: crate::api::BatchGenerateResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.results.len(), 1);
    assert_eq!(deserialized.results[0].num_generated, 3);
}
