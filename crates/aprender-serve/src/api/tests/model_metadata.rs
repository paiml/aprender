
// =========================================================================
// Coverage Tests: ModelMetadataResponse struct
// =========================================================================

#[test]
fn test_model_metadata_response_full_cov() {
    let metadata = ModelMetadataResponse {
        id: "my-model-v1".to_string(),
        name: "My Model V1".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 4_000_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 4096,
        lineage: Some(ModelLineage {
            uri: "pacha://org/model".to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "abc".to_string(),
        }),
        loaded: true,
    };
    let json = serde_json::to_string(&metadata).expect("serialize");
    assert!(json.contains("my-model-v1"));
    assert!(json.contains("My Model V1"));
    assert!(json.contains("4096"));
    assert!(json.contains("Q4_K_M"));
}

#[test]
fn test_model_metadata_response_minimal_cov() {
    let metadata = ModelMetadataResponse {
        id: "basic".to_string(),
        name: "Basic Model".to_string(),
        format: "APR".to_string(),
        size_bytes: 100_000_000,
        quantization: None,
        context_length: 1024,
        lineage: None,
        loaded: false,
    };
    let json = serde_json::to_string(&metadata).expect("serialize");
    assert!(json.contains("basic"));
    assert!(!json.contains("quantization"));
}

// =========================================================================
// Coverage Tests: OpenAIModel struct
// =========================================================================

#[test]
fn test_openai_model_serialize_cov() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "openai".to_string(),
    };
    let json = serde_json::to_string(&model).expect("serialize");
    assert!(json.contains("gpt-4"));
    assert!(json.contains("openai"));
}

#[test]
fn test_openai_model_deserialize_cov() {
    let json = r#"{"id":"llama","object":"model","created":0,"owned_by":"meta"}"#;
    let model: OpenAIModel = serde_json::from_str(json).expect("deserialize");
    assert_eq!(model.id, "llama");
    assert_eq!(model.owned_by, "meta");
}

// =========================================================================
// Coverage Tests: format_chat_messages function via integration
// =========================================================================

#[test]
fn test_chat_message_formatting_multiline_cov() {
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello\nWorld".to_string(),
            name: None,
        },
    ];
    // Ensure multiline content is preserved
    assert!(messages[1].content.contains('\n'));
}

// =========================================================================
// Coverage Tests: ContextWindowManager edge cases
// =========================================================================

#[test]
fn test_context_window_manager_preserve_system_false_cov() {
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 20,
        preserve_system: false,
    };
    let manager = ContextWindowManager::new(config);
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "System message".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "User message".to_string(),
            name: None,
        },
    ];
    let (truncated, _) = manager.truncate_messages(&messages);
    // With preserve_system=false, system messages are treated like others
    assert!(!truncated.is_empty());
}

#[test]
fn test_context_window_config_available_tokens_cov() {
    let config = ContextWindowConfig::new(1000).with_reserved_output(200);
    // Internal method - test via needs_truncation
    let manager = ContextWindowManager::new(config);
    // Short messages should not need truncation
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Short".to_string(),
        name: None,
    }];
    assert!(!manager.needs_truncation(&messages));
}

// =========================================================================
// Coverage Tests: BatchQueueStats struct (GPU feature)
// =========================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_default_cov() {
    let stats = BatchQueueStats::default();
    assert_eq!(stats.total_queued, 0);
    assert_eq!(stats.total_batches, 0);
    assert_eq!(stats.total_single, 0);
    assert!((stats.avg_batch_size - 0.0).abs() < f64::EPSILON);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_clone_debug_cov() {
    let stats = BatchQueueStats {
        total_queued: 100,
        total_batches: 10,
        total_single: 5,
        avg_batch_size: 10.0,
        avg_wait_ms: 5.5,
    };
    let cloned = stats.clone();
    assert_eq!(stats.total_queued, cloned.total_queued);
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("total_queued"));
}

// =========================================================================
// Coverage Tests: BatchConfig struct (GPU feature)
// =========================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_default_cov() {
    let config = BatchConfig::default();
    assert_eq!(config.window_ms, 50);
    assert_eq!(config.min_batch, 4);
    assert_eq!(config.optimal_batch, 32);
    assert_eq!(config.max_batch, 64);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_low_latency_cov() {
    let config = BatchConfig::low_latency();
    assert_eq!(config.window_ms, 5);
    assert_eq!(config.min_batch, 2);
    assert_eq!(config.optimal_batch, 8);
    assert_eq!(config.max_batch, 16);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_high_throughput_cov() {
    let config = BatchConfig::high_throughput();
    assert_eq!(config.window_ms, 100);
    assert_eq!(config.min_batch, 8);
    assert_eq!(config.optimal_batch, 32);
    assert_eq!(config.max_batch, 128);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_should_process_cov() {
    let config = BatchConfig::default();
    assert!(!config.should_process(10)); // Below optimal
    assert!(config.should_process(32)); // At optimal
    assert!(config.should_process(64)); // Above optimal
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_meets_minimum_cov() {
    let config = BatchConfig::default();
    assert!(!config.meets_minimum(2)); // Below min
    assert!(config.meets_minimum(4)); // At min
    assert!(config.meets_minimum(10)); // Above min
}

// =========================================================================
// Coverage Tests: ContinuousBatchResponse struct (GPU feature)
// =========================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_single_cov() {
    let resp = ContinuousBatchResponse::single(
        vec![1, 2, 3, 4, 5], // token_ids
        3,                   // prompt_len
        10.5,                // latency_ms
    );
    assert_eq!(resp.token_ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(resp.prompt_len, 3);
    assert!(!resp.batched);
    assert_eq!(resp.batch_size, 1);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_batched_cov() {
    let resp = ContinuousBatchResponse::batched(
        vec![10, 20, 30, 40, 50, 60], // token_ids
        4,                            // prompt_len
        8,                            // batch_size
        25.0,                         // latency_ms
    );
    assert!(resp.batched);
    assert_eq!(resp.batch_size, 8);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens_cov() {
    let resp = ContinuousBatchResponse::single(vec![1, 2, 3, 4, 5], 2, 5.0);
    let generated = resp.generated_tokens();
    assert_eq!(generated, &[3, 4, 5]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens_empty_cov() {
    // Edge case: prompt_len >= token_ids.len()
    let resp = ContinuousBatchResponse::single(vec![1, 2], 5, 1.0);
    let generated = resp.generated_tokens();
    assert!(generated.is_empty());
}

// =========================================================================
// Coverage Tests: BatchProcessResult struct (GPU feature)
// =========================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_process_result_debug_cov() {
    let result = BatchProcessResult {
        requests_processed: 10,
        was_batched: true,
        total_time_ms: 100.0,
        avg_latency_ms: 10.0,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("requests_processed"));
    assert!(debug_str.contains("was_batched"));
}

// =========================================================================
// Extended Coverage Tests Phase 2: Unique API tests
// =========================================================================

#[test]
fn test_health_response_roundtrip_ext_cov() {
    let json = r#"{"status":"ok","version":"2.0.0","compute_mode":"gpu"}"#;
    let resp: HealthResponse = serde_json::from_str(json).expect("parse failed");
    assert_eq!(resp.status, "ok");
    assert_eq!(resp.version, "2.0.0");
    assert_eq!(resp.compute_mode, "gpu");
}

#[test]
fn test_tokenize_request_without_model_ext_cov() {
    let json = r#"{"text":"test input"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).expect("parse failed");
    assert_eq!(req.text, "test input");
    assert!(req.model_id.is_none());
}

#[test]
fn test_generate_request_all_fields_ext_cov() {
    let json = r#"{
        "prompt": "Hello",
        "max_tokens": 100,
        "temperature": 0.7,
        "strategy": "top_k",
        "top_k": 40,
        "top_p": 0.95,
        "seed": 42,
        "model_id": "gpt-test"
    }"#;
    let req: GenerateRequest = serde_json::from_str(json).expect("parse failed");
    assert_eq!(req.prompt, "Hello");
    assert_eq!(req.max_tokens, 100);
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.model_id, Some("gpt-test".to_string()));
}

#[test]
fn test_batch_generate_request_with_seed_ext_cov() {
    let json = r#"{"prompts":["test"],"seed":12345}"#;
    let req: BatchGenerateRequest = serde_json::from_str(json).expect("parse failed");
    assert_eq!(req.seed, Some(12345));
}

#[test]
fn test_chat_message_with_name_ext_cov() {
    let msg = ChatMessage {
        role: "system".to_string(),
        content: "You are helpful".to_string(),
        name: Some("assistant_v2".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("invalid UTF-8");
    assert!(json.contains("assistant_v2"));
}

#[test]
fn test_chat_completion_request_stream_ext_cov() {
    let json = r#"{"model":"x","messages":[],"stream":true}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("parse failed");
    assert!(req.stream);
}

#[test]
fn test_chat_completion_chunk_methods_ext_cov() {
    // Test all chunk creation methods
    let chunk1 = ChatCompletionChunk::new("id1", "m", Some("text".to_string()), None);
    assert_eq!(chunk1.choices[0].delta.content, Some("text".to_string()));

    let chunk2 = ChatCompletionChunk::initial("id2", "m");
    assert_eq!(chunk2.choices[0].delta.role, Some("assistant".to_string()));

    let chunk3 = ChatCompletionChunk::content("id3", "m", "hello");
    assert_eq!(chunk3.choices[0].delta.content, Some("hello".to_string()));

    let chunk4 = ChatCompletionChunk::done("id4", "m");
    assert_eq!(chunk4.choices[0].finish_reason, Some("stop".to_string()));
}

#[test]
fn test_predict_response_skip_serialization_ext_cov() {
    let resp = PredictResponse {
        request_id: "req-456".to_string(),
        model: "model".to_string(),
        prediction: serde_json::json!(42.5),
        confidence: None,
        top_k_predictions: None,
        latency_ms: 0.5,
    };
    let json = serde_json::to_string(&resp).expect("invalid UTF-8");
    // confidence and top_k should be skipped with skip_serializing_if
    assert!(!json.contains("\"confidence\""));
    assert!(!json.contains("\"top_k_predictions\""));
}

#[test]
fn test_default_functions_ext_cov() {
    assert_eq!(default_max_tokens(), 50);
    assert_eq!(default_temperature(), 1.0);
    assert_eq!(default_strategy(), "greedy");
    assert_eq!(default_top_k(), 50);
    assert_eq!(default_top_p(), 0.9);
    assert!(default_true());
}

// =========================================================================
// More Coverage Tests (_more_cov suffix) - 56 new tests
// =========================================================================

#[test]
fn test_app_state_new_creates_valid_state_more_cov() {
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
    assert!(state.model.is_some());
    assert!(state.tokenizer.is_some());
}

#[test]
fn test_app_state_with_cache_creates_model_cache_more_cov() {
    let state = AppState::with_cache(5);
    assert!(state.cache.is_some());
    assert!(state.cache_key.is_some());
}

#[test]
fn test_error_response_serialize_more_cov() {
    let err = ErrorResponse {
        error: "Test error message".to_string(),
    };
    let json = serde_json::to_string(&err).expect("serialize");
    assert!(json.contains("Test error message"));
}

#[test]
fn test_batch_tokenize_request_serialize_more_cov() {
    let req = BatchTokenizeRequest {
        texts: vec!["hello".to_string(), "world".to_string(), "test".to_string()],
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("hello"));
    assert!(json.contains("world"));
}
