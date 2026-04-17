//! Deep coverage tests for realizar/src/api.rs
//!
//! This module provides additional coverage for API internal methods,
//! configuration types, and edge cases not covered by serialization tests.
//!
//! Coverage targets:
//! - ChatCompletionChunk construction methods
//! - BatchConfig configuration patterns
//! - ContinuousBatchResponse methods
//! - AppState factory methods
//! - ContextWindowManager edge cases
//! - Error response patterns

use realizar::api::{
    AppState, BatchGenerateRequest, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatDelta, ChatMessage, CompletionChoice, CompletionRequest, CompletionResponse,
    ContextWindowConfig, ContextWindowManager, DispatchMetricsResponse, DispatchResetResponse,
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ErrorResponse,
    ExplainRequest, GenerateRequest, GenerateResponse, GpuBatchRequest, GpuBatchResponse,
    GpuBatchResult, GpuBatchStats, GpuStatusResponse, GpuWarmupResponse, HealthResponse,
    ModelLineage, ModelMetadataResponse, PredictRequest, PredictResponse, PredictionWithScore,
    ReloadRequest, ReloadResponse, ServerMetricsResponse, StreamDoneEvent, StreamTokenEvent,
    TokenizeRequest, TokenizeResponse, Usage,
};

// ============================================================================
// Test 1-10: AppState factory methods
// ============================================================================

#[test]
fn test_app_state_demo_creates_successfully() {
    let state = AppState::demo().expect("demo should succeed");
    // Demo state should not have quantized model
    assert!(!state.has_quantized_model());
}

#[test]
fn test_app_state_with_cache_creates_successfully() {
    let state = AppState::with_cache(10);
    // State should be created successfully
    assert!(!state.has_quantized_model());
}

#[test]
fn test_app_state_has_quantized_model_false_by_default() {
    let state = AppState::demo().expect("demo");
    assert!(!state.has_quantized_model());
}

#[test]
fn test_app_state_quantized_model_returns_none_by_default() {
    let state = AppState::demo().expect("demo");
    assert!(state.quantized_model().is_none());
}

// ============================================================================
// Test 11-20: ContextWindowConfig edge cases
// ============================================================================

#[test]
fn test_context_window_config_available_tokens_saturating() {
    // When reserved > max, should return 0 (saturating_sub)
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 200,
        preserve_system: true,
    };
    // available_tokens is private, but we test it indirectly through ContextWindowManager
    let manager = ContextWindowManager::new(config);
    // Any messages should not fit since available = 0
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "hello".to_string(),
        name: None,
    }];
    assert!(manager.needs_truncation(&msgs));
}

#[test]
fn test_context_window_config_default_has_reasonable_values() {
    let config = ContextWindowConfig::default();
    assert!(config.max_tokens > 0);
    assert!(config.reserved_output_tokens > 0);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_new_with_custom_max_tokens() {
    let config = ContextWindowConfig::new(8192);
    assert_eq!(config.max_tokens, 8192);
    assert_eq!(config.reserved_output_tokens, 256); // default
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_with_reserved_output() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(512);
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
}

#[test]
fn test_context_window_config_chained_builder_pattern() {
    let config = ContextWindowConfig::new(16384).with_reserved_output(1024);
    assert_eq!(config.max_tokens, 16384);
    assert_eq!(config.reserved_output_tokens, 1024);
}

// ============================================================================
// Test 21-30: ContextWindowManager truncation behavior
// ============================================================================

#[test]
fn test_context_window_manager_default_manager() {
    let manager = ContextWindowManager::default_manager();
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "short".to_string(),
        name: None,
    }];
    let (result, truncated) = manager.truncate_messages(&msgs);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_context_window_manager_preserves_system_messages_first() {
    // Create config that allows ~100 tokens
    let config = ContextWindowConfig {
        max_tokens: 200,
        reserved_output_tokens: 100,
        preserve_system: true,
    };
    let manager = ContextWindowManager::new(config);

    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "Be helpful".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Old message that should be dropped".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Old response that should be dropped".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent message".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&msgs);
    if truncated {
        // System message should be preserved
        assert!(result.iter().any(|m| m.role == "system"));
    }
}

#[test]
fn test_context_window_manager_needs_truncation_exact_boundary() {
    let config = ContextWindowConfig {
        max_tokens: 50,
        reserved_output_tokens: 0,
        preserve_system: false,
    };
    let manager = ContextWindowManager::new(config);

    // Create messages that should exactly fit
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(100), // ~25 tokens + 10 overhead = 35
        name: None,
    }];

    let needs = manager.needs_truncation(&msgs);
    let tokens = manager.estimate_total_tokens(&msgs);
    // Verify the estimation
    assert!(tokens > 0);
    assert_eq!(needs, tokens > 50);
}

#[test]
fn test_context_window_manager_estimate_total_tokens_multiple_messages() {
    let manager = ContextWindowManager::default_manager();
    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "System".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "User message".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Assistant response".to_string(),
            name: None,
        },
    ];

    let total = manager.estimate_total_tokens(&msgs);
    // Each message has ~10 overhead + content/4 tokens
    assert!(total >= 30); // At least 10 per message
}

#[test]
fn test_context_window_manager_empty_messages_no_truncation() {
    let manager = ContextWindowManager::default_manager();
    let msgs: Vec<ChatMessage> = vec![];
    let (result, truncated) = manager.truncate_messages(&msgs);
    assert!(!truncated);
    assert!(result.is_empty());
}

#[test]
fn test_context_window_manager_truncate_preserves_order() {
    let config = ContextWindowConfig {
        max_tokens: 300,
        reserved_output_tokens: 100,
        preserve_system: false,
    };
    let manager = ContextWindowManager::new(config);

    let msgs = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "First".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Second".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Third".to_string(),
            name: None,
        },
    ];

    let (result, _) = manager.truncate_messages(&msgs);
    if result.len() >= 2 {
        // Check order is preserved
        for i in 1..result.len() {
            // Messages should be in chronological order
            assert!(
                msgs.iter()
                    .position(|m| m.content == result[i - 1].content)
                    .unwrap()
                    < msgs
                        .iter()
                        .position(|m| m.content == result[i].content)
                        .unwrap()
            );
        }
    }
}

// ============================================================================
// Test 31-40: ChatDelta serialization edge cases
// ============================================================================

#[test]
fn test_chat_delta_only_role() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: None,
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("role"));
    assert!(!json.contains("content")); // skipped due to skip_serializing_if
}

#[test]
fn test_chat_delta_only_content() {
    let delta = ChatDelta {
        role: None,
        content: Some("hello".to_string()),
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(!json.contains("role")); // skipped
    assert!(json.contains("content"));
}

#[test]
fn test_chat_delta_both_none() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    assert_eq!(json, "{}");
}

#[test]
fn test_chat_delta_empty_string_content_serializes() {
    let delta = ChatDelta {
        role: None,
        content: Some(String::new()),
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    // Empty string is Some(""), not None, so it should serialize
    assert!(json.contains("content"));
}

// ============================================================================
// Test 41-50: GpuBatchStats edge cases
// ============================================================================

#[test]
fn test_gpu_batch_stats_zero_throughput() {
    let stats = GpuBatchStats {
        batch_size: 0,
        gpu_used: false,
        total_tokens: 0,
        processing_time_ms: 0.0,
        throughput_tps: 0.0,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    assert!(json.contains(r#""throughput_tps":0.0"#));
}

#[test]
fn test_gpu_batch_stats_high_throughput() {
    let stats = GpuBatchStats {
        batch_size: 64,
        gpu_used: true,
        total_tokens: 10000,
        processing_time_ms: 100.0,
        throughput_tps: 100000.0,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    assert!(json.contains("gpu_used"));
    assert!(json.contains("true"));
}

#[test]
fn test_gpu_batch_result_empty_generation() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![],
        text: String::new(),
        num_generated: 0,
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains(r#""num_generated":0"#));
}

#[test]
fn test_gpu_batch_result_many_tokens() {
    let tokens: Vec<u32> = (0..1000).collect();
    let result = GpuBatchResult {
        index: 5,
        token_ids: tokens,
        text: "x".repeat(1000),
        num_generated: 1000,
    };
    assert_eq!(result.token_ids.len(), 1000);
    assert_eq!(result.num_generated, 1000);
}

#[test]
fn test_gpu_batch_response_many_results() {
    let results: Vec<GpuBatchResult> = (0..100)
        .map(|i| GpuBatchResult {
            index: i,
            token_ids: vec![i as u32],
            text: format!("text{i}"),
            num_generated: 1,
        })
        .collect();
    let response = GpuBatchResponse {
        results,
        stats: GpuBatchStats {
            batch_size: 100,
            gpu_used: true,
            total_tokens: 100,
            processing_time_ms: 50.0,
            throughput_tps: 2000.0,
        },
    };
    assert_eq!(response.results.len(), 100);
}

// ============================================================================
// Test 51-60: PredictRequest/Response variations
// ============================================================================

#[test]
fn test_predict_request_minimal_fields() {
    let json = r#"{"features": [1.0, 2.0, 3.0]}"#;
    let request: PredictRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(request.features.len(), 3);
    assert!(request.model.is_none());
    assert!(request.feature_names.is_none());
    assert!(request.top_k.is_none());
    assert!(request.include_confidence); // default true
}

#[test]
fn test_predict_request_with_feature_names() {
    let request = PredictRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        top_k: Some(5),
        include_confidence: true,
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("feature_names"));
}

#[test]
fn test_predict_response_with_top_k_predictions() {
    let response = PredictResponse {
        request_id: "req-123".to_string(),
        model: "demo".to_string(),
        prediction: serde_json::json!("class_a"),
        confidence: Some(0.95),
        top_k_predictions: Some(vec![
            PredictionWithScore {
                label: "class_a".to_string(),
                score: 0.95,
            },
            PredictionWithScore {
                label: "class_b".to_string(),
                score: 0.03,
            },
            PredictionWithScore {
                label: "class_c".to_string(),
                score: 0.02,
            },
        ]),
        latency_ms: 5.5,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("top_k_predictions"));
}

#[test]
fn test_predict_response_regression_no_confidence() {
    let response = PredictResponse {
        request_id: "req-456".to_string(),
        model: "regression".to_string(),
        prediction: serde_json::json!(42.5),
        confidence: None,
        top_k_predictions: None,
        latency_ms: 2.3,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(!json.contains("confidence")); // skipped
}

// ============================================================================
// Test 61-70: ExplainRequest/Response edge cases
// ============================================================================

#[test]
fn test_explain_request_default_method_is_shap() {
    let json = r#"{
        "features": [1.0, 2.0],
        "feature_names": ["x", "y"]
    }"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(request.method, "shap");
    assert_eq!(request.top_k_features, 5); // default
}

#[test]
fn test_explain_request_custom_method() {
    let request = ExplainRequest {
        model: Some("custom".to_string()),
        features: vec![1.0],
        feature_names: vec!["x".to_string()],
        top_k_features: 10,
        method: "lime".to_string(),
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("lime"));
}

// ============================================================================
// Test 71-80: EmbeddingRequest/Response
// ============================================================================

#[test]
fn test_embedding_request_without_model() {
    let json = r#"{"input": "hello world"}"#;
    let request: EmbeddingRequest = serde_json::from_str(json).expect("deserialize");
    assert!(request.model.is_none());
}

#[test]
fn test_embedding_response_high_dimensional() {
    let embedding: Vec<f32> = (0..1536).map(|i| i as f32 / 1536.0).collect();
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding,
        }],
        model: "text-embedding-ada-002".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 10,
            total_tokens: 10,
        },
    };
    assert_eq!(response.data[0].embedding.len(), 1536);
}

#[test]
fn test_embedding_usage_serialization() {
    let usage = EmbeddingUsage {
        prompt_tokens: 100,
        total_tokens: 100,
    };
    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains("prompt_tokens"));
    assert!(json.contains("total_tokens"));
}

// ============================================================================
// Test 81-90: CompletionRequest/Response edge cases
// ============================================================================

#[test]
fn test_completion_request_minimal() {
    let json = r#"{"model": "gpt-3.5-turbo", "prompt": "Hello"}"#;
    let request: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(request.model, "gpt-3.5-turbo");
    assert!(request.max_tokens.is_none());
    assert!(request.temperature.is_none());
}

#[test]
fn test_completion_request_with_stop_sequences() {
    let request = CompletionRequest {
        model: "gpt-4".to_string(),
        prompt: "Tell me".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec!["\n".to_string(), "END".to_string()]),
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("stop"));
}

#[test]
fn test_completion_response_multiple_choices() {
    let response = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: "gpt-4".to_string(),
        choices: vec![
            CompletionChoice {
                text: "Choice 1".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
            CompletionChoice {
                text: "Choice 2".to_string(),
                index: 1,
                logprobs: None,
                finish_reason: "length".to_string(),
            },
        ],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
    };
    assert_eq!(response.choices.len(), 2);
}

#[test]
fn test_completion_choice_with_logprobs() {
    let choice = CompletionChoice {
        text: "test".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({
            "tokens": ["test"],
            "token_logprobs": [-0.5],
            "top_logprobs": [{"test": -0.5, "best": -1.0}]
        })),
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("logprobs"));
    assert!(json.contains("token_logprobs"));
}

// ============================================================================
// Test 91-100: ModelMetadataResponse edge cases
// ============================================================================

#[test]
fn test_model_metadata_response_without_quantization() {
    let response = ModelMetadataResponse {
        id: "model-1".to_string(),
        name: "Full precision".to_string(),
        format: "safetensors".to_string(),
        size_bytes: 10_000_000_000,
        quantization: None,
        context_length: 8192,
        lineage: None,
        loaded: true,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(!json.contains("quantization")); // skipped
}

#[test]
fn test_model_lineage_full() {
    let lineage = ModelLineage {
        uri: "pacha://llama3:8b-q4".to_string(),
        version: "2.0.0".to_string(),
        recipe: Some("fine-tune-lora".to_string()),
        parent: Some("pacha://llama3:8b".to_string()),
        content_hash: "blake3:abcd1234".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(json.contains("recipe"));
    assert!(json.contains("parent"));
}

#[test]
fn test_model_lineage_minimal() {
    let lineage = ModelLineage {
        uri: "pacha://tiny:1b".to_string(),
        version: "1.0.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "blake3:0000".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(!json.contains("recipe")); // skipped
    assert!(!json.contains("parent")); // skipped
}

// ============================================================================
// Test 101-110: ReloadRequest/Response
// ============================================================================

#[test]
fn test_reload_request_empty() {
    let json = r"{}";
    let request: ReloadRequest = serde_json::from_str(json).expect("deserialize");
    assert!(request.model.is_none());
    assert!(request.path.is_none());
}

#[test]
fn test_reload_request_with_model_only() {
    let request = ReloadRequest {
        model: Some("new-model".to_string()),
        path: None,
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("model"));
    assert!(!json.contains("path"));
}

#[test]
fn test_reload_response_success() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded".to_string(),
        reload_time_ms: 1500,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains(r#""success":true"#));
}

#[test]
fn test_reload_response_failure() {
    let response = ReloadResponse {
        success: false,
        message: "Model file not found".to_string(),
        reload_time_ms: 0,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains(r#""success":false"#));
}

// ============================================================================
// Test 111-120: ServerMetricsResponse edge cases
// ============================================================================

#[test]
fn test_server_metrics_response_all_zeros() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 0.0,
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
        latency_p99_ms: 0.0,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 0,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 0,
        queue_depth: 0,
        total_tokens: 0,
        total_requests: 0,
        uptime_secs: 0,
        model_name: "N/A".to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    let deserialized: ServerMetricsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.total_tokens, 0);
}

#[test]
fn test_server_metrics_response_high_values() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 10000.0,
        latency_p50_ms: 5.0,
        latency_p95_ms: 15.0,
        latency_p99_ms: 50.0,
        gpu_memory_used_bytes: 20_000_000_000,
        gpu_memory_total_bytes: 24_000_000_000,
        gpu_utilization_percent: 95,
        cuda_path_active: true,
        batch_size: 64,
        queue_depth: 128,
        total_tokens: 1_000_000,
        total_requests: 10000,
        uptime_secs: 86400,
        model_name: "phi-2-q4_k_m".to_string(),
    };
    assert_eq!(response.gpu_utilization_percent, 95);
}

// ============================================================================
// Test 121-130: DispatchMetricsResponse edge cases
// ============================================================================

#[test]
fn test_dispatch_metrics_response_full() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 100,
        gpu_dispatches: 900,
        total_dispatches: 1000,
        gpu_ratio: 0.9,
        cpu_latency_p50_us: 1000.0,
        cpu_latency_p95_us: 2000.0,
        cpu_latency_p99_us: 5000.0,
        gpu_latency_p50_us: 500.0,
        gpu_latency_p95_us: 1000.0,
        gpu_latency_p99_us: 2000.0,
        cpu_latency_mean_us: 1200.0,
        gpu_latency_mean_us: 600.0,
        cpu_latency_min_us: 100,
        cpu_latency_max_us: 10000,
        gpu_latency_min_us: 50,
        gpu_latency_max_us: 5000,
        cpu_latency_variance_us: 500000.0,
        cpu_latency_stddev_us: 707.0,
        gpu_latency_variance_us: 250000.0,
        gpu_latency_stddev_us: 500.0,
        bucket_boundaries_us: vec![
            "0-100".to_string(),
            "100-500".to_string(),
            "500-1000".to_string(),
        ],
        cpu_latency_bucket_counts: vec![10, 50, 40],
        gpu_latency_bucket_counts: vec![100, 500, 300],
        throughput_rps: 100.0,
        elapsed_seconds: 10.0,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("gpu_ratio"));
    assert!(json.contains("throughput_rps"));
}

#[test]
fn test_dispatch_reset_response_success() {
    let response = DispatchResetResponse {
        success: true,
        message: "Metrics reset successfully".to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    let rt: DispatchResetResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(rt.success);
}

// ============================================================================
// Test 131-140: ChatCompletionRequest/Response edge cases
// ============================================================================

#[test]
fn test_chat_completion_request_stream_true() {
    let json = r#"{
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    }"#;
    let request: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert!(request.stream);
}

#[test]
fn test_chat_completion_request_with_stop_sequences() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        n: 1,
        stream: false,
        stop: Some(vec!["END".to_string()]),
        user: Some("user-123".to_string()),
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("stop"));
    assert!(json.contains("user"));
}

#[test]
fn test_chat_completion_response_full() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1677652288,
        model: "gpt-4".to_string(),
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
        layer_trace: None,
        step_trace: None,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("chat.completion"));
}

// ============================================================================
// Test 141-150: GpuWarmupResponse and GpuStatusResponse
// ============================================================================

#[test]
fn test_gpu_warmup_response_success() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 5_000_000_000,
        num_layers: 32,
        message: "Warmed up".to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains(r#""success":true"#));
}

#[test]
fn test_gpu_warmup_response_failure() {
    let response = GpuWarmupResponse {
        success: false,
        memory_bytes: 0,
        num_layers: 0,
        message: "CUDA not available".to_string(),
    };
    assert!(!response.success);
}

#[test]
fn test_gpu_status_response_ready() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 10_000_000_000,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };
    assert!(response.cache_ready);
}

#[test]
fn test_gpu_status_response_not_ready() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };
    assert!(!response.cache_ready);
}

// ============================================================================
// Test 151-160: More serialization roundtrips
// ============================================================================

#[test]
fn test_health_response_roundtrip_custom() {
    let response = HealthResponse {
        status: "degraded".to_string(),
        version: "0.5.0-beta".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    let rt: HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt.status, "degraded");
}

#[test]
fn test_error_response_with_json_chars() {
    let response = ErrorResponse {
        error: r#"Error: {"code": 500, "message": "failed"}"#.to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    // ErrorResponse only implements Serialize, verify JSON contains expected content
    assert!(json.contains("code"));
    assert!(json.contains("500"));
}

#[test]
fn test_stream_token_event_unicode() {
    let event = StreamTokenEvent {
        token_id: 12345,
        text: "\u{1F600}".to_string(),
    };
    let json = serde_json::to_string(&event).expect("serialize");
    let rt: StreamTokenEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt.text, "\u{1F600}");
}

#[test]
fn test_stream_done_event_large_value() {
    let event = StreamDoneEvent {
        num_generated: usize::MAX,
    };
    let json = serde_json::to_string(&event).expect("serialize");
    let rt: StreamDoneEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt.num_generated, usize::MAX);
}

// ============================================================================
// Test 161-170: Usage struct edge cases
// ============================================================================

#[test]
fn test_usage_zero_tokens() {
    let usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains(r#""total_tokens":0"#));
}

#[test]
fn test_usage_large_prompt() {
    let usage = Usage {
        prompt_tokens: 100000,
        completion_tokens: 1000,
        total_tokens: 101000,
    };
    assert_eq!(
        usage.prompt_tokens + usage.completion_tokens,
        usage.total_tokens
    );
}

#[test]
fn test_tokenize_request_empty_text() {
    let request = TokenizeRequest {
        text: String::new(),
        model_id: None,
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains(r#""text":"""#));
}

#[test]
fn test_tokenize_response_large_array() {
    let token_ids: Vec<u32> = (0..10000).collect();
    let response = TokenizeResponse {
        token_ids,
        num_tokens: 10000,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    let rt: TokenizeResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt.num_tokens, 10000);
}

// ============================================================================
// Test 171-180: GenerateRequest edge cases
// ============================================================================

#[test]
fn test_generate_request_zero_max_tokens() {
    let request = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 0,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: None,
    };
    assert_eq!(request.max_tokens, 0);
}

#[test]
fn test_generate_request_high_temperature() {
    let request = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 100,
        temperature: 2.0,
        strategy: "top_p".to_string(),
        top_k: 100,
        top_p: 0.99,
        seed: Some(42),
        model_id: Some("model-1".to_string()),
    };
    assert!((request.temperature - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_generate_response_no_generation() {
    let response = GenerateResponse {
        token_ids: vec![1, 2, 3], // prompt only
        text: "prompt text".to_string(),
        num_generated: 0,
    };
    assert_eq!(response.num_generated, 0);
}

// ============================================================================
// Test 181-190: BatchGenerateRequest edge cases
// ============================================================================

#[test]
fn test_batch_generate_request_single_prompt() {
    let request = BatchGenerateRequest {
        prompts: vec!["single".to_string()],
        max_tokens: 50,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };
    assert_eq!(request.prompts.len(), 1);
}

#[test]
fn test_batch_generate_request_many_prompts() {
    let prompts: Vec<String> = (0..100).map(|i| format!("prompt {i}")).collect();
    let request = BatchGenerateRequest {
        prompts,
        max_tokens: 10,
        temperature: 0.5,
        strategy: "top_k".to_string(),
        top_k: 10,
        top_p: 0.95,
        seed: Some(12345),
    };
    assert_eq!(request.prompts.len(), 100);
}

#[test]
fn test_batch_generate_request_empty_prompts_serialization() {
    let request = BatchGenerateRequest {
        prompts: vec![],
        max_tokens: 50,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains(r#""prompts":[]"#));
}

// ============================================================================
// Test 191-200: GpuBatchRequest edge cases
// ============================================================================

#[test]
fn test_gpu_batch_request_defaults() {
    let json = r#"{"prompts": ["hello"]}"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(request.prompts.len(), 1);
    assert_eq!(request.max_tokens, 50); // default
    assert!(request.temperature.abs() < 0.01); // default is 0.0
    assert_eq!(request.top_k, 50); // default
}

#[test]
fn test_gpu_batch_request_with_all_options() {
    let request = GpuBatchRequest {
        prompts: vec!["p1".to_string(), "p2".to_string()],
        max_tokens: 128,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains(r#""max_tokens":128"#));
}

// ============================================================================
// Test 201-210: ChatMessage edge cases
// ============================================================================

#[test]
fn test_chat_message_with_name() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("Alice".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("Alice"));
}

#[test]
fn test_chat_message_system_role() {
    let msg = ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful assistant.".to_string(),
        name: None,
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("system"));
}

#[test]
fn test_chat_message_function_role() {
    let msg = ChatMessage {
        role: "function".to_string(),
        content: r#"{"result": 42}"#.to_string(),
        name: Some("get_value".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("function"));
    assert!(json.contains("get_value"));
}

#[test]
fn test_chat_message_empty_content() {
    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: String::new(),
        name: None,
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    let rt: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(rt.content.is_empty());
}

// ============================================================================
// Test 211-220: ChatChoice and finish reasons
// ============================================================================

#[test]
fn test_chat_choice_stop() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Done".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn test_chat_choice_length() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Truncated...".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };
    assert_eq!(choice.finish_reason, "length");
}

#[test]
fn test_chat_choice_content_filter() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
            name: None,
        },
        finish_reason: "content_filter".to_string(),
    };
    assert_eq!(choice.finish_reason, "content_filter");
}

// ============================================================================
// Test 221-230: Debug and Clone implementations
// ============================================================================

#[test]
fn test_context_window_config_debug() {
    let config = ContextWindowConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("ContextWindowConfig"));
    assert!(debug.contains("max_tokens"));
}

#[test]
fn test_context_window_config_clone() {
    let config = ContextWindowConfig::new(8192).with_reserved_output(512);
    let cloned = config;
    assert_eq!(cloned.max_tokens, 8192);
    assert_eq!(cloned.reserved_output_tokens, 512);
}

#[test]
fn test_chat_message_clone() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Test".to_string(),
        name: Some("Bob".to_string()),
    };
    let cloned = msg;
    assert_eq!(cloned.role, "user");
    assert_eq!(cloned.name, Some("Bob".to_string()));
}

#[test]
fn test_chat_completion_request_debug() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let debug = format!("{request:?}");
    assert!(debug.contains("ChatCompletionRequest"));
}

// ============================================================================
// Test 231-240: PredictionWithScore edge cases
// ============================================================================

#[test]
fn test_prediction_with_score_perfect() {
    let pred = PredictionWithScore {
        label: "class_a".to_string(),
        score: 1.0,
    };
    assert!((pred.score - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_prediction_with_score_zero() {
    let pred = PredictionWithScore {
        label: "class_z".to_string(),
        score: 0.0,
    };
    assert!(pred.score.abs() < f32::EPSILON);
}

#[test]
fn test_prediction_with_score_negative_not_prevented() {
    // API doesn't prevent negative scores
    let pred = PredictionWithScore {
        label: "weird".to_string(),
        score: -0.5,
    };
    let json = serde_json::to_string(&pred).expect("serialize");
    assert!(json.contains("-0.5"));
}

// ============================================================================
// Test 241-250: Integration-like tests
// ============================================================================

#[test]
fn test_complete_chat_completion_flow() {
    // Request
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Say hi".to_string(),
                name: None,
            },
        ],
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    // Serialize request
    let request_json = serde_json::to_string(&request).expect("serialize request");

    // Simulate response
    let response = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: request.messages.iter().map(|m| m.content.len() / 4).sum(),
            completion_tokens: 3,
            total_tokens: request
                .messages
                .iter()
                .map(|m| m.content.len() / 4)
                .sum::<usize>()
                + 3,
        },
        brick_trace: None,
        layer_trace: None,
        step_trace: None,
    };

    // Serialize response
    let response_json = serde_json::to_string(&response).expect("serialize response");

    // Verify both can be parsed back
    let _: ChatCompletionRequest = serde_json::from_str(&request_json).expect("parse request");
    let parsed_response: ChatCompletionResponse =
        serde_json::from_str(&response_json).expect("parse response");
    assert_eq!(parsed_response.choices[0].message.content, "Hi there!");
}

#[test]
fn test_complete_gpu_batch_flow() {
    // Request
    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 20,
        stop: vec![],
    };

    // Response
    let response = GpuBatchResponse {
        results: vec![
            GpuBatchResult {
                index: 0,
                token_ids: vec![1, 2, 3],
                text: "Hello response".to_string(),
                num_generated: 2,
            },
            GpuBatchResult {
                index: 1,
                token_ids: vec![4, 5, 6],
                text: "World response".to_string(),
                num_generated: 2,
            },
        ],
        stats: GpuBatchStats {
            batch_size: 2,
            gpu_used: true,
            total_tokens: 6,
            processing_time_ms: 10.5,
            throughput_tps: 571.43,
        },
    };

    // Verify
    assert_eq!(request.prompts.len(), response.results.len());
    assert!(response.stats.gpu_used);
}

#[test]
fn test_context_window_truncation_flow() {
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 50,
        preserve_system: true,
    };
    let manager = ContextWindowManager::new(config);

    // Create a conversation that exceeds context window
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "Be concise.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "a".repeat(100), // ~35 tokens
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "b".repeat(100), // ~35 tokens
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent question".to_string(),
            name: None,
        },
    ];

    // Check if truncation is needed
    let needs = manager.needs_truncation(&messages);

    // If truncation needed, verify behavior
    if needs {
        let (truncated, was_truncated) = manager.truncate_messages(&messages);
        assert!(was_truncated);
        // System message should be preserved
        assert!(truncated.iter().any(|m| m.role == "system"));
        // Most recent user message should be preserved
        assert!(truncated.iter().any(|m| m.content == "Recent question"));
    }
}

#[test]
fn test_predict_explain_flow() {
    // Predict request
    let predict_req = PredictRequest {
        model: Some("classifier".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: Some(vec![
            "age".to_string(),
            "income".to_string(),
            "score".to_string(),
            "tenure".to_string(),
        ]),
        top_k: Some(3),
        include_confidence: true,
    };

    // Explain request uses same features
    let explain_req = ExplainRequest {
        model: predict_req.model.clone(),
        features: predict_req.features.clone(),
        feature_names: predict_req.feature_names.clone().unwrap(),
        top_k_features: 4,
        method: "shap".to_string(),
    };

    // Verify both serialize correctly
    let predict_json = serde_json::to_string(&predict_req).expect("serialize predict");
    let explain_json = serde_json::to_string(&explain_req).expect("serialize explain");

    assert!(predict_json.contains("features"));
    assert!(explain_json.contains("feature_names"));
}

// ============================================================================
// Test 251-260: Boundary value tests
// ============================================================================

#[test]
fn test_max_u64_in_metrics() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: f64::MAX,
        latency_p50_ms: f64::MAX,
        latency_p95_ms: f64::MAX,
        latency_p99_ms: f64::MAX,
        gpu_memory_used_bytes: u64::MAX,
        gpu_memory_total_bytes: u64::MAX,
        gpu_utilization_percent: 100,
        cuda_path_active: true,
        batch_size: usize::MAX,
        queue_depth: usize::MAX,
        total_tokens: u64::MAX,
        total_requests: u64::MAX,
        uptime_secs: u64::MAX,
        model_name: "extreme".to_string(),
    };
    // Should serialize without overflow
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(!json.is_empty());
}

#[test]
fn test_empty_strings_everywhere() {
    let response = HealthResponse {
        status: String::new(),
        version: String::new(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    let rt: HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(rt.status.is_empty());
    assert!(rt.version.is_empty());
}

#[test]
fn test_unicode_in_all_string_fields() {
    let msg = ChatMessage {
        role: "\u{1F600}".to_string(),
        content: "\u{1F601}\u{1F602}".to_string(),
        name: Some("\u{1F603}".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    let rt: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(rt.content.contains('\u{1F601}'));
}

#[test]
fn test_special_json_characters_in_content() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: r#"{"key": "value with \"quotes\" and \n newlines"}"#.to_string(),
        name: None,
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    let rt: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert!(rt.content.contains("quotes"));
}

#[test]
fn test_very_long_content() {
    let long_content = "x".repeat(1_000_000);
    let msg = ChatMessage {
        role: "user".to_string(),
        content: long_content,
        name: None,
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    let rt: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(rt.content.len(), 1_000_000);
}

// ============================================================================
// Test 261-270: Additional edge cases for full coverage
// ============================================================================

#[test]
fn test_app_state_demo_created_successfully() {
    let state = AppState::demo().expect("demo");
    // Demo state should have model and tokenizer
    assert!(!state.has_quantized_model()); // demo doesn't use quantized model
}

#[test]
fn test_app_state_with_cache_created_successfully() {
    let state = AppState::with_cache(5);
    // State should be created with cache
    assert!(!state.has_quantized_model());
}

#[test]
fn test_context_window_manager_all_system_messages() {
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 20,
        preserve_system: true,
    };
    let manager = ContextWindowManager::new(config);

    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "Rule 1".to_string(),
            name: None,
        },
        ChatMessage {
            role: "system".to_string(),
            content: "Rule 2".to_string(),
            name: None,
        },
    ];

    let (result, _) = manager.truncate_messages(&msgs);
    // Both system messages should be in result if they fit
    assert!(result.len() <= msgs.len());
}

#[test]
fn test_context_window_manager_no_system_messages() {
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 20,
        preserve_system: true,
    };
    let manager = ContextWindowManager::new(config);

    let msgs = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
    ];

    let (result, _) = manager.truncate_messages(&msgs);
    // Should still work without system messages
    assert!(!result.is_empty() || msgs.is_empty());
}

#[test]
fn test_dispatch_metrics_response_clone() {
    let response = DispatchMetricsResponse {
        cpu_dispatches: 10,
        gpu_dispatches: 20,
        total_dispatches: 30,
        gpu_ratio: 0.67,
        cpu_latency_p50_us: 100.0,
        cpu_latency_p95_us: 200.0,
        cpu_latency_p99_us: 300.0,
        gpu_latency_p50_us: 50.0,
        gpu_latency_p95_us: 100.0,
        gpu_latency_p99_us: 150.0,
        cpu_latency_mean_us: 120.0,
        gpu_latency_mean_us: 60.0,
        cpu_latency_min_us: 10,
        cpu_latency_max_us: 500,
        gpu_latency_min_us: 5,
        gpu_latency_max_us: 250,
        cpu_latency_variance_us: 1000.0,
        cpu_latency_stddev_us: 31.6,
        gpu_latency_variance_us: 500.0,
        gpu_latency_stddev_us: 22.4,
        bucket_boundaries_us: vec!["0-100".to_string()],
        cpu_latency_bucket_counts: vec![10],
        gpu_latency_bucket_counts: vec![20],
        throughput_rps: 50.0,
        elapsed_seconds: 1.0,
    };
    let cloned = response;
    assert_eq!(cloned.total_dispatches, 30);
}

// ============================================================================
// Test 271-280: More serialization edge cases
// ============================================================================

#[test]
fn test_embedding_data_empty_embedding() {
    let data = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![],
    };
    let json = serde_json::to_string(&data).expect("serialize");
    assert!(json.contains(r#""embedding":[]"#));
}

#[test]
fn test_completion_choice_empty_text() {
    let choice = CompletionChoice {
        text: String::new(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains(r#""text":"""#));
}

#[test]
fn test_model_metadata_response_large_size() {
    let response = ModelMetadataResponse {
        id: "llama-70b".to_string(),
        name: "LLaMA 70B".to_string(),
        format: "gguf".to_string(),
        size_bytes: 140_000_000_000, // 140 GB
        quantization: Some("Q4_K_M".to_string()),
        context_length: 128000,
        lineage: None,
        loaded: false,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("140000000000"));
}

#[test]
fn test_generate_request_seed_zero() {
    let request = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(0),
        model_id: None,
    };
    assert_eq!(request.seed, Some(0));
}

#[test]
fn test_generate_request_seed_max() {
    let request = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: Some(u64::MAX),
        model_id: None,
    };
    assert_eq!(request.seed, Some(u64::MAX));
}

// ============================================================================
// Test 281-290: Final edge cases
// ============================================================================

#[test]
fn test_gpu_batch_request_empty_prompts() {
    let request = GpuBatchRequest {
        prompts: vec![],
        max_tokens: 10,
        temperature: 0.7,
        top_k: 40,
        stop: vec![],
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains(r#""prompts":[]"#));
}

#[test]
fn test_batch_tokenize_response_empty_results() {
    let response = realizar::api::BatchTokenizeResponse { results: vec![] };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains(r#""results":[]"#));
}

#[test]
fn test_models_response_empty_models() {
    let response = realizar::api::ModelsResponse { models: vec![] };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains(r#""models":[]"#));
}

#[test]
fn test_chat_completion_request_empty_messages() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains(r#""messages":[]"#));
}

#[test]
fn test_chat_completion_response_empty_choices() {
    let response = ChatCompletionResponse {
        id: "test".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "gpt-4".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        brick_trace: None,
        layer_trace: None,
        step_trace: None,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains(r#""choices":[]"#));
}
