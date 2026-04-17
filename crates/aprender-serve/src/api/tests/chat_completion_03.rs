
#[test]
fn test_chat_completion_response_serialize() {
    let response = ChatCompletionResponse {
        id: "chat-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "phi-2".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("chat-123"));
    assert!(json.contains("phi-2"));
}

#[test]
fn test_usage_serialize() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    let json = serde_json::to_string(&usage).expect("test");
    assert!(json.contains("150"));
}

#[test]
fn test_openai_model_serialize() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "openai".to_string(),
    };
    let json = serde_json::to_string(&model).expect("test");
    assert!(json.contains("gpt-4"));
}

#[test]
fn test_stream_token_event_serialize() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&event).expect("test");
    assert!(json.contains("42"));
    assert!(json.contains("hello"));
}

#[test]
fn test_stream_done_event_serialize() {
    let event = StreamDoneEvent { num_generated: 100 };
    let json = serde_json::to_string(&event).expect("test");
    assert!(json.contains("100"));
}

#[test]
fn test_models_response_serialize() {
    let response = ModelsResponse {
        models: vec![
            ModelInfo {
                id: "phi-2".to_string(),
                name: "Phi-2".to_string(),
                description: "Microsoft Phi-2".to_string(),
                format: "gguf".to_string(),
                loaded: true,
            },
            ModelInfo {
                id: "llama".to_string(),
                name: "LLaMA".to_string(),
                description: "Meta LLaMA".to_string(),
                format: "gguf".to_string(),
                loaded: false,
            },
        ],
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("phi-2"));
    assert!(json.contains("llama"));
}

// =========================================================================
// Coverage Tests: Request/Response Structs
// =========================================================================

#[test]
fn test_chat_message_fields_cov() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello, world!".to_string(),
        name: None,
    };
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, "Hello, world!");
    assert!(msg.name.is_none());
}

#[test]
fn test_chat_completion_request_defaults_cov() {
    let req = ChatCompletionRequest {
        model: "phi-2".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        }],
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        max_tokens: None,
        user: None,
    };
    assert_eq!(req.model, "phi-2");
    assert!(req.temperature.is_none());
    assert!(!req.stream);
}

#[test]
fn test_chat_choice_fields_cov() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Hello!".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    assert_eq!(choice.index, 0);
    assert_eq!(choice.message.role, "assistant");
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn test_usage_fields_cov() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
}

#[test]
fn test_openai_model_fields_cov() {
    let model = OpenAIModel {
        id: "phi-2".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "microsoft".to_string(),
    };
    assert_eq!(model.id, "phi-2");
    assert_eq!(model.object, "model");
    assert_eq!(model.owned_by, "microsoft");
}

#[test]
fn test_chat_delta_fields_cov() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hello".to_string()),
    };
    assert!(delta.role.is_some());
    assert!(delta.content.is_some());
}

#[test]
fn test_chat_chunk_choice_fields_cov() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        finish_reason: None,
    };
    assert_eq!(choice.index, 0);
    assert!(choice.finish_reason.is_none());
}

#[test]
fn test_predict_request_fields_cov() {
    let req = PredictRequest {
        model: Some("test-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]),
        top_k: None,
        include_confidence: true,
    };
    assert!(req.model.is_some());
    assert_eq!(req.features.len(), 3);
    assert!(req.top_k.is_none());
    assert!(req.include_confidence);
}

#[test]
fn test_explain_request_fields_cov() {
    let req = ExplainRequest {
        model: Some("test-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        top_k_features: 5,
        method: "shap".to_string(),
    };
    assert!(req.model.is_some());
    assert_eq!(req.features.len(), 3);
    assert_eq!(req.method, "shap");
    assert_eq!(req.top_k_features, 5);
}

#[test]
fn test_dispatch_metrics_query_default_cov() {
    let query = DispatchMetricsQuery { format: None };
    assert!(query.format.is_none());
}

#[test]
fn test_dispatch_reset_response_fields_cov() {
    let resp = DispatchResetResponse {
        success: true,
        message: "Reset successful".to_string(),
    };
    assert!(resp.success);
    assert!(resp.message.contains("Reset"));
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_request_fields_cov() {
    let req = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 100,
        temperature: 0.7,
        top_k: 50,
        stop: vec![],
    };
    assert_eq!(req.prompts.len(), 2);
    assert_eq!(req.max_tokens, 100);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_result_fields_cov() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![1, 2, 3],
        text: "Generated text".to_string(),
        num_generated: 3,
    };
    assert_eq!(result.index, 0);
    assert_eq!(result.num_generated, 3);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_stats_fields_cov() {
    let stats = GpuBatchStats {
        batch_size: 10,
        gpu_used: true,
        total_tokens: 500,
        processing_time_ms: 100.0,
        throughput_tps: 5000.0,
    };
    assert_eq!(stats.batch_size, 10);
    assert!(stats.gpu_used);
    assert_eq!(stats.total_tokens, 500);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_warmup_response_fields_cov() {
    let resp = GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024,
        num_layers: 32,
        message: "GPU warmed up".to_string(),
    };
    assert!(resp.success);
    assert!(resp.memory_bytes > 0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_status_response_fields_cov() {
    let resp = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1024 * 1024,
        batch_threshold: 8,
        recommended_min_batch: 4,
    };
    assert!(resp.cache_ready);
    assert!(resp.cache_memory_bytes > 0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_fields_cov() {
    let config = BatchConfig {
        window_ms: 100,
        min_batch: 2,
        optimal_batch: 8,
        max_batch: 32,
        queue_size: 128,
        gpu_threshold: 32,
    };
    assert!(config.max_batch > 0);
    assert!(config.window_ms > 0);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_fields_cov() {
    let stats = BatchQueueStats {
        total_queued: 100,
        total_batches: 10,
        total_single: 5,
        avg_batch_size: 10.0,
        avg_wait_ms: 50.0,
    };
    assert_eq!(stats.total_queued, 100);
    assert_eq!(stats.total_batches, 10);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_process_result_fields_cov() {
    let result = BatchProcessResult {
        requests_processed: 5,
        was_batched: true,
        total_time_ms: 500.0,
        avg_latency_ms: 100.0,
    };
    assert_eq!(result.requests_processed, 5);
    assert!(result.was_batched);
}

#[test]
fn test_context_window_config_fields_cov() {
    let config = ContextWindowConfig {
        max_tokens: 4096,
        reserved_output_tokens: 512,
        preserve_system: true,
    };
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_default_cov() {
    let config = ContextWindowConfig::default();
    assert_eq!(config.max_tokens, 4096);
    assert!(config.preserve_system);
}

#[test]
fn test_embedding_request_fields_cov() {
    let req = EmbeddingRequest {
        input: "Some text to embed".to_string(),
        model: Some("text-embedding".to_string()),
    };
    assert!(req.model.is_some());
    assert!(req.input.contains("embed"));
}

// =========================================================================
// Coverage Tests: HealthResponse
// =========================================================================

#[test]
fn test_health_response_serialize_cov() {
    let resp = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("healthy"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("cpu"));
}

// =========================================================================
// Coverage Tests: ErrorResponse
// =========================================================================

#[test]
fn test_error_response_serialize_cov() {
    let resp = ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("Something went wrong"));
}

// =========================================================================
// Coverage Tests: OpenAI Compatibility Structs
// =========================================================================

#[test]
fn test_openai_models_response_serialize_cov() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "gpt-3.5-turbo".to_string(),
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "openai".to_string(),
        }],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("list"));
    assert!(json.contains("gpt-3.5-turbo"));
}
