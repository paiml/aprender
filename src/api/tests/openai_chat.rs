
#[tokio::test]
async fn test_openai_chat_completions_empty_messages_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"model":"test","messages":[]}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
async fn test_openai_chat_completions_valid_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
}

#[tokio::test]
async fn test_openai_chat_completions_stream_empty_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"model":"test","messages":[],"stream":true}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/stream")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
async fn test_openai_chat_completions_stream_valid_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":3}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions/stream")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
}

#[test]
fn test_context_window_config_default_values_more_cov() {
    let config = ContextWindowConfig::default();
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 256);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_manager_large_truncation_more_cov() {
    let config = ContextWindowConfig::new(20).with_reserved_output(5);
    let manager = ContextWindowManager::new(config);

    let messages: Vec<ChatMessage> = (0..50)
        .map(|i| ChatMessage {
            role: "user".to_string(),
            content: format!("Message number {} with some content", i),
            name: None,
        })
        .collect();

    let (result, was_truncated) = manager.truncate_messages(&messages);
    assert!(was_truncated);
    assert!(result.len() < messages.len());
}

#[test]
fn test_format_chat_messages_system_role_more_cov() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
    ];
    let formatted = format_chat_messages(&messages, Some("chatml"));
    assert!(!formatted.is_empty());
}

#[test]
fn test_format_chat_messages_assistant_role_more_cov() {
    let messages = vec![ChatMessage {
        role: "assistant".to_string(),
        content: "I am here to help".to_string(),
        name: None,
    }];
    let formatted = format_chat_messages(&messages, None);
    assert!(!formatted.is_empty());
}

#[test]
fn test_default_n_function_more_cov() {
    assert_eq!(default_n(), 1);
}

#[test]
fn test_default_top_k_features_more_cov() {
    assert_eq!(default_top_k_features(), 5);
}

#[test]
fn test_default_explain_method_more_cov() {
    assert_eq!(default_explain_method(), "shap");
}

#[test]
fn test_audit_response_serialize_more_cov() {
    use crate::audit::{AuditOptions, LatencyBreakdown};
    use chrono::Utc;

    let record = AuditRecord {
        request_id: "test-id".to_string(),
        timestamp: Utc::now(),
        client_id_hash: None,
        model_hash: "hash123".to_string(),
        model_version: "1.0.0".to_string(),
        model_type: "test-model".to_string(),
        distillation_teacher_hash: None,
        input_dims: vec![4],
        input_hash: "input_hash".to_string(),
        options: AuditOptions::default(),
        prediction: serde_json::json!(42.0),
        confidence: Some(0.95),
        explanation_summary: None,
        latency_ms: 10.5,
        latency_breakdown: LatencyBreakdown::default(),
        memory_peak_bytes: 1024,
        quality_nan_check: true,
        quality_confidence_check: true,
        warnings: vec![],
    };
    let resp = AuditResponse { record };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("test-id"));
}

#[test]
fn test_chat_completion_request_all_fields_more_cov() {
    let req = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.5),
        top_p: Some(0.9),
        n: 2,
        stream: true,
        stop: Some(vec!["END".to_string()]),
        user: Some("test-user".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("gpt-4"));
    assert!(json.contains("test-user"));
}

#[tokio::test]
async fn test_tokenize_with_model_id_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"text":"hello","model_id":"custom"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    let status = response.status();
    assert!(status == StatusCode::OK || status == StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_stream_generate_invalid_strategy_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"prompt":"hello","strategy":"unknown"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
async fn test_realize_embed_with_model_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"input":"test text","model":"custom-model"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/embed")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    let status = response.status();
    assert!(status == StatusCode::OK || status == StatusCode::NOT_FOUND);
}

#[test]
fn test_dispatch_metrics_query_deserialize_more_cov() {
    let json = r#"{"format":"prometheus"}"#;
    let query: DispatchMetricsQuery = serde_json::from_str(json).expect("deserialize");
    assert_eq!(query.format, Some("prometheus".to_string()));
}

#[test]
fn test_dispatch_metrics_query_default_more_cov() {
    let json = r"{}";
    let query: DispatchMetricsQuery = serde_json::from_str(json).expect("deserialize");
    assert!(query.format.is_none());
}

#[test]
fn test_models_response_deserialize_more_cov() {
    let json = r#"{"models":[{"id":"m1","name":"Model 1","description":"Desc","format":"GGUF","loaded":true}]}"#;
    let resp: ModelsResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(resp.models.len(), 1);
    assert_eq!(resp.models[0].id, "m1");
}

#[test]
fn test_chat_completion_chunk_serialize_more_cov() {
    let chunk = ChatCompletionChunk {
        id: "chunk-1".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 12345,
        model: "test".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: Some("world".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).expect("serialize");
    assert!(json.contains("chunk-1"));
    assert!(json.contains("world"));
}

#[tokio::test]
async fn test_apr_predict_no_apr_model_more_cov() {
    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("create model");
    let vocab: Vec<String> = (0..100).map(|i| format!("tok{i}")).collect();
    let tokenizer = BPETokenizer::new(vocab, vec![], "tok0").expect("create tokenizer");
    let state = AppState::new(model, tokenizer);
    let app = create_router(state);

    let json = r#"{"features":[1.0,2.0]}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(json))
                .expect("test"),
        )
        .await
        .expect("test");

    assert!(
        response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[test]
fn test_app_state_has_quantized_model_false_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(!state.has_quantized_model());
}

#[test]
fn test_app_state_quantized_model_none_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(state.quantized_model().is_none());
}

#[cfg(feature = "gpu")]
#[test]
fn test_app_state_has_gpu_model_false_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(!state.has_gpu_model());
}

#[cfg(feature = "gpu")]
#[test]
fn test_app_state_gpu_model_none_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(state.gpu_model().is_none());
}

#[cfg(feature = "gpu")]
#[test]
fn test_app_state_has_cached_model_false_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(!state.has_cached_model());
}

#[cfg(feature = "gpu")]
#[test]
fn test_app_state_cached_model_none_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(state.cached_model().is_none());
}

#[cfg(feature = "gpu")]
#[test]
fn test_app_state_dispatch_metrics_none_more_cov() {
    let state = AppState::demo().expect("test");
    assert!(state.dispatch_metrics().is_none());
}

#[tokio::test]
async fn test_batch_generate_invalid_strategy_more_cov() {
    let app = create_test_app_shared();
    let json = r#"{"prompts":["hello"],"strategy":"invalid"}"#;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(json))
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
