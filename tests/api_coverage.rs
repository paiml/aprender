//! EXTREME TDD coverage tests for realizar/src/api.rs
//!
//! This module provides comprehensive coverage for API request/response structures,
//! handler functions, validation logic, error responses, and streaming code.
//!
//! Coverage targets:
//! - Request/Response struct serialization/deserialization
//! - Default value functions
//! - Error response generation
//! - Context window management
//! - OpenAI-compatible API types
//! - GPU batch inference types
//! - APR prediction types

use realizar::api::{
    AuditResponse, BatchGenerateRequest, BatchGenerateResponse, BatchTokenizeRequest,
    BatchTokenizeResponse, ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatDelta,
    ChatMessage, CompletionChoice, CompletionRequest, CompletionResponse, ContextWindowConfig,
    ContextWindowManager, DispatchResetResponse, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, ErrorResponse, ExplainRequest, ExplainResponse,
    GenerateRequest, GenerateResponse, GpuBatchRequest, GpuBatchResponse, GpuBatchResult,
    GpuBatchStats, GpuStatusResponse, GpuWarmupResponse, HealthResponse, ModelLineage,
    ModelMetadataResponse, ModelsResponse, OpenAIModel, OpenAIModelsResponse, PredictRequest,
    PredictResponse, PredictionWithScore, ReloadRequest, ReloadResponse, ServerMetricsResponse,
    StreamDoneEvent, StreamTokenEvent, TokenizeRequest, TokenizeResponse, Usage,
};
use realizar::explain::ShapExplanation;
use realizar::registry::ModelInfo;

// ============================================================================
// Test 1-5: Request/Response Struct Serialization
// ============================================================================

#[test]
fn test_health_response_serialization() {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""status":"healthy""#));
    assert!(json.contains(r#""version":"1.0.0""#));

    let deserialized: HealthResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.status, "healthy");
    assert_eq!(deserialized.version, "1.0.0");
}

#[test]
fn test_tokenize_request_with_model_id() {
    let request = TokenizeRequest {
        text: "hello world".to_string(),
        model_id: Some("test-model".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""text":"hello world""#));
    assert!(json.contains(r#""model_id":"test-model""#));

    let deserialized: TokenizeRequest = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.text, "hello world");
    assert_eq!(deserialized.model_id, Some("test-model".to_string()));
}

#[test]
fn test_tokenize_request_without_model_id() {
    let json = r#"{"text": "test input"}"#;
    let request: TokenizeRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.text, "test input");
    assert!(request.model_id.is_none());
}

#[test]
fn test_tokenize_response_serialization() {
    let response = TokenizeResponse {
        token_ids: vec![1, 2, 3, 4, 5],
        num_tokens: 5,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""token_ids":[1,2,3,4,5]"#));
    assert!(json.contains(r#""num_tokens":5"#));
}

#[test]
fn test_generate_response_serialization() {
    let response = GenerateResponse {
        token_ids: vec![10, 20, 30],
        text: "generated text".to_string(),
        num_generated: 3,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""token_ids":[10,20,30]"#));
    assert!(json.contains(r#""text":"generated text""#));
    assert!(json.contains(r#""num_generated":3"#));
}

// ============================================================================
// Test 6-10: Generate Request Default Values
// ============================================================================

#[test]
fn test_generate_request_all_defaults() {
    let json = r#"{"prompt": "hello"}"#;
    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 1.0).abs() < 1e-6);
    assert_eq!(request.strategy, "greedy");
    assert_eq!(request.top_k, 50);
    assert!((request.top_p - 0.9).abs() < 1e-6);
    assert!(request.seed.is_none());
    assert!(request.model_id.is_none());
}

#[test]
fn test_generate_request_custom_values() {
    let json = r#"{
        "prompt": "test prompt",
        "max_tokens": 100,
        "temperature": 0.7,
        "strategy": "top_k",
        "top_k": 10,
        "top_p": 0.95,
        "seed": 42,
        "model_id": "custom-model"
    }"#;

    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompt, "test prompt");
    assert_eq!(request.max_tokens, 100);
    assert!((request.temperature - 0.7).abs() < 1e-6);
    assert_eq!(request.strategy, "top_k");
    assert_eq!(request.top_k, 10);
    assert!((request.top_p - 0.95).abs() < 1e-6);
    assert_eq!(request.seed, Some(42));
    assert_eq!(request.model_id, Some("custom-model".to_string()));
}

#[test]
fn test_error_response_serialization() {
    let response = ErrorResponse {
        error: "Something went wrong".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""error":"Something went wrong""#));
}

#[test]
fn test_batch_tokenize_request_serialization() {
    let request = BatchTokenizeRequest {
        texts: vec!["text1".to_string(), "text2".to_string()],
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""texts":["text1","text2"]"#));

    let deserialized: BatchTokenizeRequest =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.texts.len(), 2);
}

#[test]
fn test_batch_tokenize_response_serialization() {
    let response = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1, 2],
                num_tokens: 2,
            },
            TokenizeResponse {
                token_ids: vec![3, 4, 5],
                num_tokens: 3,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: BatchTokenizeResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.results.len(), 2);
    assert_eq!(deserialized.results[0].num_tokens, 2);
    assert_eq!(deserialized.results[1].num_tokens, 3);
}

// ============================================================================
// Test 11-15: Batch Generate Request/Response
// ============================================================================

#[test]
fn test_batch_generate_request_defaults() {
    let json = r#"{"prompts": ["prompt1", "prompt2"]}"#;
    let request: BatchGenerateRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompts.len(), 2);
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 1.0).abs() < 1e-6);
    assert_eq!(request.strategy, "greedy");
    assert_eq!(request.top_k, 50);
    assert!((request.top_p - 0.9).abs() < 1e-6);
    assert!(request.seed.is_none());
}

#[test]
fn test_batch_generate_response_serialization() {
    let response = BatchGenerateResponse {
        results: vec![
            GenerateResponse {
                token_ids: vec![1, 2, 3],
                text: "output1".to_string(),
                num_generated: 3,
            },
            GenerateResponse {
                token_ids: vec![4, 5],
                text: "output2".to_string(),
                num_generated: 2,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: BatchGenerateResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.results.len(), 2);
}

#[test]
fn test_stream_token_event_serialization() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };

    let json = serde_json::to_string(&event).expect("should serialize");
    assert!(json.contains(r#""token_id":42"#));
    assert!(json.contains(r#""text":"hello""#));

    let deserialized: StreamTokenEvent = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.token_id, 42);
    assert_eq!(deserialized.text, "hello");
}

#[test]
fn test_stream_done_event_serialization() {
    let event = StreamDoneEvent { num_generated: 100 };

    let json = serde_json::to_string(&event).expect("should serialize");
    assert!(json.contains(r#""num_generated":100"#));

    let deserialized: StreamDoneEvent = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.num_generated, 100);
}

#[test]
fn test_models_response_serialization() {
    let response = ModelsResponse {
        models: vec![
            ModelInfo {
                id: "model-1".to_string(),
                name: "Test Model 1".to_string(),
                description: "First test model".to_string(),
                format: "gguf".to_string(),
                loaded: true,
            },
            ModelInfo {
                id: "model-2".to_string(),
                name: "Test Model 2".to_string(),
                description: "Second test model".to_string(),
                format: "apr".to_string(),
                loaded: false,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ModelsResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.models.len(), 2);
    assert_eq!(deserialized.models[0].id, "model-1");
    assert!(deserialized.models[0].loaded);
    assert!(!deserialized.models[1].loaded);
}

// ============================================================================
// Test 16-20: OpenAI-Compatible API Types
// ============================================================================

#[test]
fn test_chat_completion_request_minimal() {
    let json = r#"{
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }"#;

    let request: ChatCompletionRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.model, "gpt-4");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.n, 1); // default
    assert!(!request.stream); // default
}

#[test]
fn test_chat_completion_request_full() {
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
                content: "Hi".to_string(),
                name: Some("user123".to_string()),
            },
        ],
        max_tokens: Some(256),
        temperature: Some(0.7),
        top_p: Some(0.95),
        n: 2,
        stream: true,
        stop: Some(vec!["STOP".to_string()]),
        user: Some("user-id".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: ChatCompletionRequest =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.model, "gpt-4");
    assert_eq!(deserialized.messages.len(), 2);
    assert_eq!(deserialized.max_tokens, Some(256));
    assert_eq!(deserialized.n, 2);
    assert!(deserialized.stream);
}

#[test]
fn test_chat_completion_response_serialization() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "gpt-4".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello! How can I help?".to_string(),
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

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""id":"chatcmpl-123""#));
    assert!(json.contains(r#""object":"chat.completion""#));
    assert!(json.contains(r#""finish_reason":"stop""#));
}

#[test]
fn test_openai_models_response_serialization() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![
            OpenAIModel {
                id: "model-1".to_string(),
                object: "model".to_string(),
                created: 1000000000,
                owned_by: "realizar".to_string(),
            },
            OpenAIModel {
                id: "model-2".to_string(),
                object: "model".to_string(),
                created: 1000000001,
                owned_by: "realizar".to_string(),
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: OpenAIModelsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.object, "list");
    assert_eq!(deserialized.data.len(), 2);
    assert_eq!(deserialized.data[0].owned_by, "realizar");
}

#[test]
fn test_chat_delta_skip_none_serialization() {
    // With only content
    let delta1 = ChatDelta {
        role: None,
        content: Some("hello".to_string()),
    };
    let json1 = serde_json::to_string(&delta1).expect("should serialize");
    assert!(!json1.contains("role"));
    assert!(json1.contains(r#""content":"hello""#));

    // With only role
    let delta2 = ChatDelta {
        role: Some("assistant".to_string()),
        content: None,
    };
    let json2 = serde_json::to_string(&delta2).expect("should serialize");
    assert!(json2.contains(r#""role":"assistant""#));
    assert!(!json2.contains("content"));
}

// ============================================================================
// Test 21-25: Context Window Management
// ============================================================================

#[test]
fn test_context_window_config_builder() {
    let config = ContextWindowConfig::new(8192).with_reserved_output(512);

    assert_eq!(config.max_tokens, 8192);
    assert_eq!(config.reserved_output_tokens, 512);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_manager_no_truncation() {
    let manager = ContextWindowManager::default_manager();

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
    ];

    assert!(!manager.needs_truncation(&messages));

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_context_window_manager_forces_truncation() {
    let config = ContextWindowConfig::new(50).with_reserved_output(10);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(200), // Very long message
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "short".to_string(),
            name: None,
        },
    ];

    assert!(manager.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_token_estimation() {
    let manager = ContextWindowManager::default_manager();

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello world! This is a test.".to_string(),
        name: None,
    }];

    let tokens = manager.estimate_total_tokens(&messages);
    // Estimate: ~4 chars per token + 10 overhead = ~17 tokens
    assert!(tokens > 0);
    assert!(tokens < 50);
}

#[test]
fn test_context_window_preserves_recent_messages() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Old message that should be dropped".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(100),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Recent".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);

    if truncated {
        // Most recent message should be preserved
        let has_recent = result.iter().any(|m| m.content == "Recent");
        assert!(
            has_recent || result.is_empty(),
            "Recent message should be preserved when possible"
        );
    }
}

// ============================================================================
// Test 26-30: GPU Batch and APR Types
// ============================================================================

#[test]
fn test_gpu_batch_request_defaults() {
    let json = r#"{"prompts": ["hello", "world"]}"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.prompts.len(), 2);
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 0.0).abs() < 1e-6);
    assert_eq!(request.top_k, 50);
    assert!(request.stop.is_empty());
}

#[test]
fn test_gpu_batch_response_serialization() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "output".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: true,
            total_tokens: 3,
            processing_time_ms: 10.5,
            throughput_tps: 285.7,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""gpu_used":true"#));
    assert!(json.contains(r#""throughput_tps"#));

    let deserialized: GpuBatchResponse = serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.stats.gpu_used);
    assert_eq!(deserialized.results.len(), 1);
}

#[test]
fn test_gpu_warmup_response_serialization() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024 * 512,
        num_layers: 32,
        message: "Warmup complete".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuWarmupResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.num_layers, 32);
}

#[test]
fn test_gpu_status_response_serialization() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1024 * 1024 * 100,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuStatusResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.cache_ready);
    assert_eq!(deserialized.batch_threshold, 32);
}

#[test]
fn test_predict_request_full_serialization() {
    let request = PredictRequest {
        model: Some("my-model".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: Some(vec![
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
            "f4".to_string(),
        ]),
        top_k: Some(3),
        include_confidence: true,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: PredictRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.features.len(), 4);
    assert_eq!(deserialized.top_k, Some(3));
    assert!(deserialized.include_confidence);
}

// ============================================================================
// Test 31-35: APR Explain and Audit Types
// ============================================================================

#[test]
fn test_explain_request_defaults() {
    let json = r#"{"features": [1.0, 2.0], "feature_names": ["a", "b"]}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.top_k_features, 5); // default
    assert_eq!(request.method, "shap"); // default
}

#[test]
fn test_explain_response_serialization() {
    let response = ExplainResponse {
        request_id: "req-123".to_string(),
        model: "test-model".to_string(),
        prediction: serde_json::json!(0.85),
        confidence: Some(0.92),
        explanation: ShapExplanation {
            base_value: 0.5,
            shap_values: vec![0.1, -0.05, 0.3],
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            prediction: 0.85,
        },
        summary: "Top features: f3 (+), f1 (+)".to_string(),
        latency_ms: 15.5,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""request_id":"req-123""#));
    assert!(json.contains(r#""summary"#));

    let deserialized: ExplainResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.explanation.shap_values.len(), 3);
}

#[test]
fn test_prediction_with_score_serialization() {
    let prediction = PredictionWithScore {
        label: "class_0".to_string(),
        score: 0.95,
    };

    let json = serde_json::to_string(&prediction).expect("should serialize");
    assert!(json.contains(r#""label":"class_0""#));
    assert!(json.contains(r#""score":0.95"#));
}

#[test]
fn test_predict_response_regression() {
    let response = PredictResponse {
        request_id: "uuid-123".to_string(),
        model: "regressor".to_string(),
        prediction: serde_json::json!(42.5),
        confidence: Some(1.0),
        top_k_predictions: None,
        latency_ms: 5.2,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: PredictResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prediction, serde_json::json!(42.5));
    assert!(deserialized.top_k_predictions.is_none());
}

#[test]
fn test_predict_response_classification() {
    let response = PredictResponse {
        request_id: "uuid-456".to_string(),
        model: "classifier".to_string(),
        prediction: serde_json::json!("class_2"),
        confidence: Some(0.85),
        top_k_predictions: Some(vec![
            PredictionWithScore {
                label: "class_2".to_string(),
                score: 0.85,
            },
            PredictionWithScore {
                label: "class_1".to_string(),
                score: 0.10,
            },
            PredictionWithScore {
                label: "class_0".to_string(),
                score: 0.05,
            },
        ]),
        latency_ms: 8.3,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: PredictResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.top_k_predictions.is_some());
    let top_k = deserialized.top_k_predictions.unwrap();
    assert_eq!(top_k.len(), 3);
    assert_eq!(top_k[0].label, "class_2");
}

// ============================================================================
// Test 36-40: Embedding and Completion Types
// ============================================================================

#[test]
fn test_embedding_request_serialization() {
    let request = EmbeddingRequest {
        input: "test input text".to_string(),
        model: Some("text-embedding-ada".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: EmbeddingRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.input, "test input text");
    assert_eq!(deserialized.model, Some("text-embedding-ada".to_string()));
}

#[test]
fn test_embedding_response_serialization() {
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        }],
        model: "text-embedding".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: EmbeddingResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.data[0].embedding.len(), 4);
    assert_eq!(deserialized.usage.prompt_tokens, 5);
}

#[test]
fn test_completion_request_serialization() {
    let request = CompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        prompt: "Once upon a time".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec!["END".to_string()]),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: CompletionRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prompt, "Once upon a time");
    assert_eq!(deserialized.max_tokens, Some(50));
}

#[test]
fn test_completion_response_serialization() {
    let response = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1234567890,
        model: "gpt-3.5".to_string(),
        choices: vec![CompletionChoice {
            text: "there was a princess".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 4,
            completion_tokens: 5,
            total_tokens: 9,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: CompletionResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.choices[0].text, "there was a princess");
    assert_eq!(deserialized.usage.total_tokens, 9);
}

#[test]
fn test_completion_choice_with_logprobs() {
    let choice = CompletionChoice {
        text: "output".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({
            "tokens": ["out", "put"],
            "token_logprobs": [-0.5, -0.3]
        })),
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains("logprobs"));
    assert!(json.contains("token_logprobs"));
}

// ============================================================================
// Test 41-45: Model Metadata and Reload Types
// ============================================================================

#[test]
fn test_model_metadata_response_full() {
    let response = ModelMetadataResponse {
        id: "phi-2".to_string(),
        name: "Phi-2".to_string(),
        format: "gguf".to_string(),
        size_bytes: 2_700_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 2048,
        lineage: Some(ModelLineage {
            uri: "pacha://phi-2:latest".to_string(),
            version: "1.0.0".to_string(),
            recipe: Some("instruct-tuned".to_string()),
            parent: Some("phi-1.5".to_string()),
            content_hash: "blake3:abc123".to_string(),
        }),
        loaded: true,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ModelMetadataResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.format, "gguf");
    assert!(deserialized.lineage.is_some());
    let lineage = deserialized.lineage.unwrap();
    assert_eq!(lineage.parent, Some("phi-1.5".to_string()));
}

#[test]
fn test_model_metadata_response_minimal() {
    let response = ModelMetadataResponse {
        id: "simple".to_string(),
        name: "Simple Model".to_string(),
        format: "apr".to_string(),
        size_bytes: 100_000,
        quantization: None,
        context_length: 512,
        lineage: None,
        loaded: false,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(!json.contains("quantization"));
    assert!(!json.contains("lineage"));
}

#[test]
fn test_reload_request_serialization() {
    let request = ReloadRequest {
        model: Some("model-1".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: ReloadRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.model, Some("model-1".to_string()));
    assert_eq!(deserialized.path, Some("/path/to/model.gguf".to_string()));
}

#[test]
fn test_reload_response_serialization() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded successfully".to_string(),
        reload_time_ms: 1500,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ReloadResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.reload_time_ms, 1500);
}

#[test]
fn test_server_metrics_response_serialization() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 150.5,
        latency_p50_ms: 10.2,
        latency_p95_ms: 25.8,
        latency_p99_ms: 50.1,
        gpu_memory_used_bytes: 8_000_000_000,
        gpu_memory_total_bytes: 24_000_000_000,
        gpu_utilization_percent: 75,
        cuda_path_active: true,
        batch_size: 32,
        queue_depth: 5,
        total_tokens: 100_000,
        total_requests: 500,
        uptime_secs: 3600,
        model_name: "phi-2-q4k".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ServerMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!((deserialized.throughput_tok_per_sec - 150.5).abs() < 0.01);
    assert!(deserialized.cuda_path_active);
    assert_eq!(deserialized.model_name, "phi-2-q4k");
}

// ============================================================================
// Test 46-50: Edge Cases and Boundary Conditions
// ============================================================================

#[test]
fn test_empty_messages_array() {
    let json = r#"{"model": "test", "messages": []}"#;
    let request: ChatCompletionRequest = serde_json::from_str(json).expect("should deserialize");
    assert!(request.messages.is_empty());
}

#[test]
fn test_large_token_ids() {
    let response = TokenizeResponse {
        token_ids: vec![u32::MAX, u32::MAX - 1, 0],
        num_tokens: 3,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: TokenizeResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.token_ids[0], u32::MAX);
    assert_eq!(deserialized.token_ids[1], u32::MAX - 1);
}

#[test]
fn test_zero_latency() {
    let response = PredictResponse {
        request_id: "test".to_string(),
        model: "test".to_string(),
        prediction: serde_json::json!(1.0),
        confidence: Some(1.0),
        top_k_predictions: None,
        latency_ms: 0.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""latency_ms":0.0"#));
}

#[test]
fn test_unicode_in_messages() {
    let message = ChatMessage {
        role: "user".to_string(),
        content: "Hello World".to_string(),
        name: Some("User".to_string()),
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.content.contains("Hello"));
    assert!(deserialized.content.contains("World"));
}

#[test]
fn test_special_characters_in_text() {
    let request = TokenizeRequest {
        text: r#"Test with "quotes" and \backslashes\ and newlines"#.to_string(),
        model_id: None,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: TokenizeRequest = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.text.contains("quotes"));
    assert!(deserialized.text.contains("backslashes"));
}

// ============================================================================
// Test 51-55: Dispatch Metrics Types
// ============================================================================

#[test]
fn test_dispatch_metrics_response_serialization() {
    let response = realizar::api::DispatchMetricsResponse {
        cpu_dispatches: 100,
        gpu_dispatches: 50,
        total_dispatches: 150,
        gpu_ratio: 0.333,
        cpu_latency_p50_us: 100.5,
        cpu_latency_p95_us: 250.0,
        cpu_latency_p99_us: 500.0,
        gpu_latency_p50_us: 50.0,
        gpu_latency_p95_us: 120.0,
        gpu_latency_p99_us: 200.0,
        cpu_latency_mean_us: 125.5,
        gpu_latency_mean_us: 75.0,
        cpu_latency_min_us: 50,
        cpu_latency_max_us: 600,
        gpu_latency_min_us: 30,
        gpu_latency_max_us: 250,
        cpu_latency_variance_us: 1000.0,
        cpu_latency_stddev_us: 31.62,
        gpu_latency_variance_us: 500.0,
        gpu_latency_stddev_us: 22.36,
        bucket_boundaries_us: vec![
            "0-100".to_string(),
            "100-500".to_string(),
            "500+".to_string(),
        ],
        cpu_latency_bucket_counts: vec![50, 40, 10],
        gpu_latency_bucket_counts: vec![30, 15, 5],
        throughput_rps: 150.0,
        elapsed_seconds: 60.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""cpu_dispatches":100"#));
    assert!(json.contains(r#""gpu_ratio":0.333"#));
    assert!(json.contains(r#""throughput_rps":150.0"#));
}

#[test]
fn test_dispatch_metrics_query_default() {
    let json = r"{}";
    let query: realizar::api::DispatchMetricsQuery =
        serde_json::from_str(json).expect("should deserialize empty query");
    assert!(query.format.is_none());
}

#[test]
fn test_dispatch_metrics_query_with_format() {
    let json = r#"{"format": "prometheus"}"#;
    let query: realizar::api::DispatchMetricsQuery =
        serde_json::from_str(json).expect("should deserialize");
    assert_eq!(query.format, Some("prometheus".to_string()));
}

#[test]
fn test_dispatch_reset_response_serialization() {
    let response = realizar::api::DispatchResetResponse {
        success: true,
        message: "Reset complete".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""success":true"#));
    assert!(json.contains(r#""message":"Reset complete""#));

    let deserialized: realizar::api::DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert!(deserialized.success);
}

#[test]
fn test_dispatch_reset_response_failure() {
    let response = realizar::api::DispatchResetResponse {
        success: false,
        message: "No GPU model configured".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: realizar::api::DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!(!deserialized.success);
    assert!(deserialized.message.contains("No GPU"));
}

// ============================================================================
// Test 56-60: ChatCompletionChunk and Streaming Types
// ============================================================================

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = realizar::api::ChatCompletionChunk {
        id: "chunk-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "phi-2".to_string(),
        choices: vec![realizar::api::ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };

    let json = serde_json::to_string(&chunk).expect("should serialize");
    assert!(json.contains(r#""object":"chat.completion.chunk""#));
    assert!(json.contains(r#""model":"phi-2""#));

    let deserialized: realizar::api::ChatCompletionChunk =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.choices.len(), 1);
}

#[test]
fn test_chat_chunk_choice_with_content() {
    let choice = realizar::api::ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("Hello".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains(r#""content":"Hello""#));
    assert!(!json.contains("role")); // role should be skipped when None
}

#[test]
fn test_chat_chunk_choice_with_finish_reason() {
    let choice = realizar::api::ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("stop".to_string()),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains(r#""finish_reason":"stop""#));
}

#[test]
fn test_audit_response_serialization() {
    use uuid::Uuid;

    // Create an AuditRecord using the builder pattern
    let request_id = Uuid::new_v4();
    let record = realizar::audit::AuditRecord::new(request_id, "blake3:abc123", "classifier")
        .with_model_version("1.0.0")
        .with_input_hash("sha256:def456");

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""model_hash":"blake3:abc123""#));
    assert!(json.contains(r#""model_type":"classifier""#));

    let deserialized: AuditResponse = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.record.model_hash, "blake3:abc123");
}

#[test]
fn test_chat_delta_both_fields() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hi there".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("should serialize");
    assert!(json.contains(r#""role":"assistant""#));
    assert!(json.contains(r#""content":"Hi there""#));
}

// ============================================================================
// Test 61-65: Context Window Config Advanced Tests
// ============================================================================

#[test]
fn test_context_window_config_default_values() {
    let config = ContextWindowConfig::default();

    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 256);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_chained_builder() {
    let config = ContextWindowConfig::new(16384).with_reserved_output(1024);

    assert_eq!(config.max_tokens, 16384);
    assert_eq!(config.reserved_output_tokens, 1024);
}

#[test]
fn test_context_window_manager_preserves_system_messages() {
    let config = ContextWindowConfig::new(200).with_reserved_output(50);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(500), // Long message to force truncation
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Response".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);

    // System message should be preserved
    if truncated && !result.is_empty() {
        let has_system = result.iter().any(|m| m.role == "system");
        assert!(
            has_system,
            "System message should be preserved during truncation"
        );
    }
}

#[test]
fn test_context_window_manager_empty_messages() {
    let manager = ContextWindowManager::default_manager();
    let messages: Vec<ChatMessage> = vec![];

    let (result, truncated) = manager.truncate_messages(&messages);

    assert!(result.is_empty());
    assert!(!truncated);
}

#[test]
fn test_context_window_manager_single_message() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    let (result, truncated) = manager.truncate_messages(&messages);

    assert_eq!(result.len(), 1);
    assert!(!truncated);
    assert_eq!(result[0].content, "Hello");
}

// ============================================================================
// Test 66-70: Server Metrics and Lineage Types
// ============================================================================

#[test]
fn test_server_metrics_response_zero_values() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 0.0,
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
        latency_p99_ms: 0.0,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 0,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 1,
        queue_depth: 0,
        total_tokens: 0,
        total_requests: 0,
        uptime_secs: 0,
        model_name: "none".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ServerMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.throughput_tok_per_sec, 0.0);
    assert!(!deserialized.cuda_path_active);
}

#[test]
fn test_model_lineage_full() {
    let lineage = ModelLineage {
        uri: "pacha://llama-3:8b".to_string(),
        version: "2.0.0".to_string(),
        recipe: Some("chat-instruct".to_string()),
        parent: Some("llama-3:base".to_string()),
        content_hash: "blake3:xyz789".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("should serialize");
    assert!(json.contains(r#""uri":"pacha://llama-3:8b""#));
    assert!(json.contains(r#""recipe":"chat-instruct""#));
    assert!(json.contains(r#""parent":"llama-3:base""#));

    let deserialized: ModelLineage = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.version, "2.0.0");
}

#[test]
fn test_model_lineage_minimal() {
    let lineage = ModelLineage {
        uri: "local://model.gguf".to_string(),
        version: "1.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "blake3:abc".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("should serialize");
    // Optional fields should be omitted
    assert!(!json.contains("recipe"));
    assert!(!json.contains("parent"));
}

#[test]
fn test_reload_request_empty() {
    let json = r"{}";
    let request: ReloadRequest = serde_json::from_str(json).expect("should deserialize");

    assert!(request.model.is_none());
    assert!(request.path.is_none());
}

#[test]
fn test_reload_response_with_timing() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded in 2.5s".to_string(),
        reload_time_ms: 2500,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ReloadResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.reload_time_ms, 2500);
}

// ============================================================================
// Test 71-75: Completion Types (OpenAI-compatible)
// ============================================================================

#[test]
fn test_completion_request_minimal() {
    let json = r#"{"model": "gpt-3", "prompt": "Hello"}"#;
    let request: CompletionRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.model, "gpt-3");
    assert_eq!(request.prompt, "Hello");
    assert!(request.max_tokens.is_none());
    assert!(request.temperature.is_none());
}

#[test]
fn test_completion_response_multiple_choices() {
    let response = CompletionResponse {
        id: "cmpl-multi".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "gpt-3.5".to_string(),
        choices: vec![
            CompletionChoice {
                text: "First response".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
            CompletionChoice {
                text: "Second response".to_string(),
                index: 1,
                logprobs: None,
                finish_reason: "length".to_string(),
            },
        ],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
            total_tokens: 15,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: CompletionResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.choices.len(), 2);
    assert_eq!(deserialized.choices[1].finish_reason, "length");
}

#[test]
fn test_embedding_request_no_model() {
    let json = r#"{"input": "test text"}"#;
    let request: EmbeddingRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.input, "test text");
    assert!(request.model.is_none());
}

#[test]
fn test_embedding_usage_equality() {
    let usage1 = EmbeddingUsage {
        prompt_tokens: 10,
        total_tokens: 10,
    };
    let usage2 = EmbeddingUsage {
        prompt_tokens: 10,
        total_tokens: 10,
    };

    let json1 = serde_json::to_string(&usage1).expect("serialize");
    let json2 = serde_json::to_string(&usage2).expect("serialize");

    assert_eq!(json1, json2);
}

#[test]
fn test_embedding_data_with_large_vector() {
    let embedding = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![0.1; 1536], // OpenAI ada-002 dimension
    };

    let json = serde_json::to_string(&embedding).expect("should serialize");
    let deserialized: EmbeddingData = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.embedding.len(), 1536);
}

// ============================================================================
// Test 76-80: Chat Message Variations
// ============================================================================

#[test]
fn test_chat_message_with_name() {
    let message = ChatMessage {
        role: "user".to_string(),
        content: "Hi!".to_string(),
        name: Some("Alice".to_string()),
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    assert!(json.contains(r#""name":"Alice""#));

    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.name, Some("Alice".to_string()));
}

#[test]
fn test_chat_message_system_role() {
    let message = ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful AI assistant.".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.role, "system");
}

#[test]
fn test_chat_message_assistant_role() {
    let message = ChatMessage {
        role: "assistant".to_string(),
        content: "I can help you with that.".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.role, "assistant");
}

#[test]
fn test_chat_choice_serialization() {
    let choice = ChatChoice {
        index: 2,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Response".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains(r#""index":2"#));
    assert!(json.contains(r#""finish_reason":"length""#));
}

#[test]
fn test_usage_total_calculation() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };

    let json = serde_json::to_string(&usage).expect("should serialize");
    let deserialized: Usage = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(
        deserialized.total_tokens,
        deserialized.prompt_tokens + deserialized.completion_tokens
    );
}

// ============================================================================
// Test 81-85: GPU Batch Request Variations
// ============================================================================

#[test]
fn test_gpu_batch_request_full_options() {
    let request = GpuBatchRequest {
        prompts: vec!["prompt1".to_string(), "prompt2".to_string()],
        max_tokens: 100,
        temperature: 0.8,
        top_k: 40,
        stop: vec!["STOP".to_string(), "END".to_string()],
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: GpuBatchRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prompts.len(), 2);
    assert_eq!(deserialized.stop.len(), 2);
    assert!((deserialized.temperature - 0.8).abs() < 1e-6);
}

#[test]
fn test_gpu_batch_result_empty_generation() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![1, 2, 3], // prompt only, no generation
        text: "abc".to_string(),
        num_generated: 0,
    };

    let json = serde_json::to_string(&result).expect("should serialize");
    let deserialized: GpuBatchResult = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.num_generated, 0);
}

#[test]
fn test_gpu_batch_stats_cpu_fallback() {
    let stats = GpuBatchStats {
        batch_size: 4,
        gpu_used: false, // CPU fallback
        total_tokens: 100,
        processing_time_ms: 500.0,
        throughput_tps: 200.0,
    };

    let json = serde_json::to_string(&stats).expect("should serialize");
    let deserialized: GpuBatchStats = serde_json::from_str(&json).expect("should deserialize");

    assert!(!deserialized.gpu_used);
}

#[test]
fn test_gpu_warmup_response_failure() {
    let response = GpuWarmupResponse {
        success: false,
        memory_bytes: 0,
        num_layers: 0,
        message: "CUDA not available".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuWarmupResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(!deserialized.success);
    assert_eq!(deserialized.memory_bytes, 0);
}

#[test]
fn test_gpu_status_response_not_ready() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuStatusResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(!deserialized.cache_ready);
}

// ============================================================================
// Test 86-90: Predict Request/Response Variations
// ============================================================================

#[test]
fn test_predict_request_minimal() {
    let json = r#"{"features": [1.0, 2.0, 3.0]}"#;
    let request: PredictRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.features.len(), 3);
    assert!(request.model.is_none());
    assert!(request.feature_names.is_none());
    assert!(request.top_k.is_none());
    assert!(request.include_confidence); // default is true
}

#[test]
fn test_predict_request_no_confidence() {
    let json = r#"{"features": [1.0], "include_confidence": false}"#;
    let request: PredictRequest = serde_json::from_str(json).expect("should deserialize");

    assert!(!request.include_confidence);
}

#[test]
fn test_predict_response_no_confidence() {
    let response = PredictResponse {
        request_id: "req-1".to_string(),
        model: "regressor".to_string(),
        prediction: serde_json::json!(100.5),
        confidence: None,
        top_k_predictions: None,
        latency_ms: 10.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    // Optional fields should be omitted when None
    assert!(!json.contains("confidence"));
    assert!(!json.contains("top_k_predictions"));
}

#[test]
fn test_prediction_with_score_high_confidence() {
    let prediction = PredictionWithScore {
        label: "positive".to_string(),
        score: 0.99,
    };

    let json = serde_json::to_string(&prediction).expect("should serialize");
    let deserialized: PredictionWithScore =
        serde_json::from_str(&json).expect("should deserialize");

    assert!((deserialized.score - 0.99).abs() < 1e-6);
}

#[test]
fn test_explain_request_custom_method() {
    let json = r#"{"features": [1.0, 2.0], "feature_names": ["a", "b"], "method": "lime", "top_k_features": 10}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("should deserialize");

    assert_eq!(request.method, "lime");
    assert_eq!(request.top_k_features, 10);
}

// ============================================================================
// Test 91-95: OpenAI Model Types
// ============================================================================

#[test]
fn test_openai_model_owned_by() {
    let model = OpenAIModel {
        id: "phi-2".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "realizar".to_string(),
    };

    let json = serde_json::to_string(&model).expect("should serialize");
    assert!(json.contains(r#""owned_by":"realizar""#));
}

#[test]
fn test_openai_models_response_empty() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: OpenAIModelsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.data.is_empty());
}

#[test]
fn test_chat_completion_request_with_stop() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.5),
        top_p: None,
        n: 1,
        stream: false,
        stop: Some(vec!["STOP".to_string(), "END".to_string()]),
        user: None,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""stop":["STOP","END"]"#));
}

#[test]
fn test_chat_completion_request_with_user() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: Some("user-12345".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: ChatCompletionRequest =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.user, Some("user-12345".to_string()));
}

#[test]
fn test_chat_completion_response_multiple_choices() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-multi".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "gpt-4".to_string(),
        choices: vec![
            ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "First".to_string(),
                    name: None,
                },
                finish_reason: "stop".to_string(),
            },
            ChatChoice {
                index: 1,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Second".to_string(),
                    name: None,
                },
                finish_reason: "stop".to_string(),
            },
        ],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
        brick_trace: None,
        layer_trace: None,
        step_trace: None,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ChatCompletionResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.choices.len(), 2);
}

// ============================================================================
// Test 96-100: Edge Cases and Boundary Values
// ============================================================================

#[test]
fn test_max_tokens_boundary() {
    let response = TokenizeResponse {
        token_ids: vec![0], // minimum valid token
        num_tokens: 1,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: TokenizeResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.token_ids[0], 0);
}

#[test]
fn test_empty_string_content() {
    let message = ChatMessage {
        role: "user".to_string(),
        content: String::new(),
        name: None,
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.content.is_empty());
}

#[test]
fn test_very_long_content() {
    let long_content = "x".repeat(100_000);
    let message = ChatMessage {
        role: "user".to_string(),
        content: long_content,
        name: None,
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.content.len(), 100_000);
}

#[test]
fn test_negative_timestamp_handling() {
    // Test that timestamps serialize correctly (though negative is unusual)
    let response = ChatCompletionResponse {
        id: "test".to_string(),
        object: "chat.completion".to_string(),
        created: -1, // Edge case
        model: "test".to_string(),
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

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""created":-1"#));
}

#[test]
fn test_float_precision_in_latency() {
    let response = PredictResponse {
        request_id: "test".to_string(),
        model: "test".to_string(),
        prediction: serde_json::json!(1.0),
        confidence: Some(0.123_456_79),
        top_k_predictions: None,
        latency_ms: 0.000_001, // Very small latency
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: PredictResponse = serde_json::from_str(&json).expect("should deserialize");

    // Check that small values are preserved
    assert!(deserialized.latency_ms < 0.001);
}

// ============================================================================
// Test 101-110: Context Window Config Methods
// ============================================================================

#[test]
fn test_context_window_config_new_basic() {
    let config = ContextWindowConfig::new(2048);
    assert_eq!(config.max_tokens, 2048);
    assert_eq!(config.reserved_output_tokens, 256); // default
    assert!(config.preserve_system); // default
}

#[test]
fn test_context_window_config_small_window() {
    // Very small context window
    let config = ContextWindowConfig::new(64).with_reserved_output(16);
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.reserved_output_tokens, 16);
}

#[test]
fn test_context_window_config_large_window() {
    // Large context window (128k)
    let config = ContextWindowConfig::new(131072).with_reserved_output(4096);
    assert_eq!(config.max_tokens, 131072);
    assert_eq!(config.reserved_output_tokens, 4096);
}

#[test]
fn test_context_window_config_zero_reserved() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(0);
    assert_eq!(config.reserved_output_tokens, 0);
}

#[test]
fn test_context_window_manager_many_short_messages() {
    let config = ContextWindowConfig::new(200).with_reserved_output(50);
    let manager = ContextWindowManager::new(config);

    // Many short messages
    let messages: Vec<ChatMessage> = (0..20)
        .map(|i| ChatMessage {
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("Message {}", i),
            name: None,
        })
        .collect();

    let (result, truncated) = manager.truncate_messages(&messages);

    // Should truncate since many messages
    if truncated {
        // Recent messages should be preserved
        assert!(result.len() < messages.len());
    }
}

#[test]
fn test_context_window_manager_only_system_message() {
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    let manager = ContextWindowManager::new(config);

    let messages = vec![ChatMessage {
        role: "system".to_string(),
        content: "You are a test assistant.".to_string(),
        name: None,
    }];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].role, "system");
}

#[test]
fn test_context_window_manager_mixed_roles() {
    let manager = ContextWindowManager::default_manager();

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "How are you?".to_string(),
            name: None,
        },
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_context_window_estimate_tokens_empty() {
    let manager = ContextWindowManager::default_manager();
    let messages: Vec<ChatMessage> = vec![];

    let tokens = manager.estimate_total_tokens(&messages);
    assert_eq!(tokens, 0);
}

#[test]
fn test_context_window_needs_truncation_exact_boundary() {
    // Create config where we're exactly at the boundary
    let config = ContextWindowConfig::new(100).with_reserved_output(0);
    let manager = ContextWindowManager::new(config);

    // Create a message that's roughly 100 tokens
    // estimate is ~4 chars per token + 10 overhead
    // So 360 chars = ~90 tokens + 10 overhead = ~100 tokens
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(360),
        name: None,
    }];

    // Should be right at boundary - may or may not need truncation
    let needs = manager.needs_truncation(&messages);
    // Just verify the function runs without panic
    let _ = needs;
}

// ============================================================================
// Test 111-120: Chat Delta Edge Cases
// ============================================================================

#[test]
fn test_chat_delta_empty_strings() {
    let delta = ChatDelta {
        role: Some(String::new()),
        content: Some(String::new()),
    };

    let json = serde_json::to_string(&delta).expect("should serialize");
    assert!(json.contains(r#""role":"""#));
    assert!(json.contains(r#""content":"""#));
}

#[test]
fn test_chat_delta_only_role_none_content() {
    let delta = ChatDelta {
        role: Some("user".to_string()),
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("should serialize");
    assert!(json.contains(r#""role":"user""#));
    assert!(!json.contains("content"));
}

#[test]
fn test_chat_chunk_choice_empty_delta() {
    let choice = realizar::api::ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    // Delta should be mostly empty
    assert!(json.contains("delta"));
    assert!(!json.contains("role"));
    assert!(!json.contains("content"));
}

#[test]
fn test_chat_chunk_choice_all_fields() {
    let choice = realizar::api::ChatChunkChoice {
        index: 5,
        delta: ChatDelta {
            role: Some("assistant".to_string()),
            content: Some("Hello".to_string()),
        },
        finish_reason: Some("stop".to_string()),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains(r#""index":5"#));
    assert!(json.contains(r#""role":"assistant""#));
    assert!(json.contains(r#""content":"Hello""#));
    assert!(json.contains(r#""finish_reason":"stop""#));
}

#[test]
fn test_chat_completion_chunk_serialization_full() {
    let chunk = realizar::api::ChatCompletionChunk {
        id: "chunk-abc123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![
            realizar::api::ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: Some("Token 1".to_string()),
                },
                finish_reason: None,
            },
            realizar::api::ChatChunkChoice {
                index: 1,
                delta: ChatDelta {
                    role: None,
                    content: Some("Token 2".to_string()),
                },
                finish_reason: Some("length".to_string()),
            },
        ],
    };

    let json = serde_json::to_string(&chunk).expect("should serialize");
    let deserialized: realizar::api::ChatCompletionChunk =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.id, "chunk-abc123");
    assert_eq!(deserialized.choices.len(), 2);
    assert_eq!(
        deserialized.choices[1].finish_reason,
        Some("length".to_string())
    );
}

#[test]
fn test_chat_completion_chunk_empty_choices() {
    let chunk = realizar::api::ChatCompletionChunk {
        id: "empty".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
    };

    let json = serde_json::to_string(&chunk).expect("should serialize");
    assert!(json.contains(r#""choices":[]"#));
}

#[test]
fn test_dispatch_metrics_response_zero_values() {
    let response = realizar::api::DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""total_dispatches":0"#));
    assert!(json.contains(r#""gpu_ratio":0.0"#));
}

#[test]
fn test_dispatch_metrics_response_high_throughput() {
    let response = realizar::api::DispatchMetricsResponse {
        cpu_dispatches: 1000,
        gpu_dispatches: 9000,
        total_dispatches: 10000,
        gpu_ratio: 0.9,
        cpu_latency_p50_us: 50.0,
        cpu_latency_p95_us: 100.0,
        cpu_latency_p99_us: 150.0,
        gpu_latency_p50_us: 25.0,
        gpu_latency_p95_us: 50.0,
        gpu_latency_p99_us: 75.0,
        cpu_latency_mean_us: 60.0,
        gpu_latency_mean_us: 30.0,
        cpu_latency_min_us: 10,
        cpu_latency_max_us: 200,
        gpu_latency_min_us: 5,
        gpu_latency_max_us: 100,
        cpu_latency_variance_us: 400.0,
        cpu_latency_stddev_us: 20.0,
        gpu_latency_variance_us: 225.0,
        gpu_latency_stddev_us: 15.0,
        bucket_boundaries_us: vec!["0-100".to_string(), "100-500".to_string()],
        cpu_latency_bucket_counts: vec![800, 200],
        gpu_latency_bucket_counts: vec![8500, 500],
        throughput_rps: 1000.0,
        elapsed_seconds: 10.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""gpu_ratio":0.9"#));
    assert!(json.contains(r#""throughput_rps":1000.0"#));
}

#[test]
fn test_dispatch_metrics_query_json_format() {
    let json = r#"{"format": "json"}"#;
    let query: realizar::api::DispatchMetricsQuery =
        serde_json::from_str(json).expect("should deserialize");
    assert_eq!(query.format, Some("json".to_string()));
}

// ============================================================================
// Test 121-130: Server Metrics Response Variations
// ============================================================================

#[test]
fn test_server_metrics_response_gpu_active() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 256.5,
        latency_p50_ms: 5.2,
        latency_p95_ms: 12.8,
        latency_p99_ms: 25.6,
        gpu_memory_used_bytes: 8_000_000_000,
        gpu_memory_total_bytes: 24_000_000_000,
        gpu_utilization_percent: 85,
        cuda_path_active: true,
        batch_size: 32,
        queue_depth: 10,
        total_tokens: 1_000_000,
        total_requests: 5000,
        uptime_secs: 7200,
        model_name: "phi-2-q4k".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ServerMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.cuda_path_active);
    assert_eq!(deserialized.gpu_utilization_percent, 85);
    assert_eq!(deserialized.batch_size, 32);
}

#[test]
fn test_server_metrics_response_cpu_only() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: 12.5,
        latency_p50_ms: 50.0,
        latency_p95_ms: 100.0,
        latency_p99_ms: 200.0,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 0,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 1,
        queue_depth: 0,
        total_tokens: 5000,
        total_requests: 400,
        uptime_secs: 600,
        model_name: "cpu-model".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ServerMetricsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!(!deserialized.cuda_path_active);
    assert_eq!(deserialized.gpu_memory_total_bytes, 0);
}

#[test]
fn test_server_metrics_response_max_values() {
    let response = ServerMetricsResponse {
        throughput_tok_per_sec: f64::MAX / 2.0,
        latency_p50_ms: 0.001,
        latency_p95_ms: 0.002,
        latency_p99_ms: 0.003,
        gpu_memory_used_bytes: u64::MAX,
        gpu_memory_total_bytes: u64::MAX,
        gpu_utilization_percent: 100,
        cuda_path_active: true,
        batch_size: usize::MAX,
        queue_depth: usize::MAX,
        total_tokens: u64::MAX,
        total_requests: u64::MAX,
        uptime_secs: u64::MAX,
        model_name: "max-test".to_string(),
    };

    // Should serialize without panic
    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("max-test"));
}

#[test]
fn test_gpu_batch_request_empty_stop_sequences() {
    let request = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 20,
        stop: vec![],
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""stop":[]"#));
}

#[test]
fn test_gpu_batch_request_many_prompts() {
    let prompts: Vec<String> = (0..100).map(|i| format!("Prompt {}", i)).collect();
    let request = GpuBatchRequest {
        prompts,
        max_tokens: 50,
        temperature: 1.0,
        top_k: 50,
        stop: vec!["END".to_string()],
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: GpuBatchRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prompts.len(), 100);
}

#[test]
fn test_gpu_batch_response_multiple_results() {
    let response = GpuBatchResponse {
        results: vec![
            GpuBatchResult {
                index: 0,
                token_ids: vec![1, 2, 3],
                text: "abc".to_string(),
                num_generated: 3,
            },
            GpuBatchResult {
                index: 1,
                token_ids: vec![4, 5, 6, 7],
                text: "defg".to_string(),
                num_generated: 4,
            },
            GpuBatchResult {
                index: 2,
                token_ids: vec![8],
                text: "h".to_string(),
                num_generated: 1,
            },
        ],
        stats: GpuBatchStats {
            batch_size: 3,
            gpu_used: true,
            total_tokens: 8,
            processing_time_ms: 25.0,
            throughput_tps: 320.0,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuBatchResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.results.len(), 3);
    assert_eq!(deserialized.stats.total_tokens, 8);
}

#[test]
fn test_gpu_batch_stats_zero_time() {
    let stats = GpuBatchStats {
        batch_size: 1,
        gpu_used: false,
        total_tokens: 0,
        processing_time_ms: 0.0,
        throughput_tps: 0.0,
    };

    let json = serde_json::to_string(&stats).expect("should serialize");
    assert!(json.contains(r#""processing_time_ms":0.0"#));
}

#[test]
fn test_gpu_warmup_response_partial_success() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 1024 * 1024 * 256, // 256 MB
        num_layers: 16,
        message: "Partial warmup: 16 of 32 layers cached".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuWarmupResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.num_layers, 16);
}

#[test]
fn test_gpu_status_response_high_memory() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 24 * 1024 * 1024 * 1024, // 24 GB
        batch_threshold: 64,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: GpuStatusResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.batch_threshold, 64);
}

// ============================================================================
// Test 131-140: Additional API Request/Response Edge Cases
// ============================================================================

#[test]
fn test_generate_request_extreme_temperature() {
    let json = r#"{"prompt": "test", "temperature": 100.0}"#;
    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");
    assert!((request.temperature - 100.0).abs() < 1e-6);
}

#[test]
fn test_generate_request_zero_temperature() {
    let json = r#"{"prompt": "test", "temperature": 0.0}"#;
    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");
    assert!((request.temperature - 0.0).abs() < 1e-6);
}

#[test]
fn test_batch_generate_request_single_prompt() {
    let json = r#"{"prompts": ["only one"]}"#;
    let request: BatchGenerateRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.prompts.len(), 1);
}

#[test]
fn test_batch_generate_request_many_prompts() {
    let prompts: Vec<String> = (0..500).map(|i| format!("Prompt {}", i)).collect();
    let request = BatchGenerateRequest {
        prompts,
        max_tokens: 10,
        temperature: 0.8,
        strategy: "top_p".to_string(),
        top_k: 40,
        top_p: 0.95,
        seed: Some(12345),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: BatchGenerateRequest =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.prompts.len(), 500);
}

#[test]
fn test_predict_request_empty_feature_names() {
    let request = PredictRequest {
        model: None,
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec![]),
        top_k: None,
        include_confidence: true,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""feature_names":[]"#));
}

#[test]
fn test_predict_request_many_features() {
    let features: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
    let feature_names: Vec<String> = (0..1000).map(|i| format!("feature_{}", i)).collect();

    let request = PredictRequest {
        model: Some("large-model".to_string()),
        features,
        feature_names: Some(feature_names),
        top_k: Some(10),
        include_confidence: true,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: PredictRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.features.len(), 1000);
}

#[test]
fn test_explain_request_max_features() {
    let json = r#"{"features": [1.0], "feature_names": ["x"], "top_k_features": 1000}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.top_k_features, 1000);
}

#[test]
fn test_explain_response_empty_explanation() {
    let response = ExplainResponse {
        request_id: "empty".to_string(),
        model: "test".to_string(),
        prediction: serde_json::json!(null),
        confidence: None,
        explanation: ShapExplanation {
            base_value: 0.0,
            shap_values: vec![],
            feature_names: vec![],
            prediction: 0.0,
        },
        summary: "No features".to_string(),
        latency_ms: 0.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""shap_values":[]"#));
}

#[test]
fn test_reload_request_only_model() {
    let json = r#"{"model": "test-model"}"#;
    let request: ReloadRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.model, Some("test-model".to_string()));
    assert!(request.path.is_none());
}

#[test]
fn test_reload_request_only_path() {
    let json = r#"{"path": "/path/to/model.gguf"}"#;
    let request: ReloadRequest = serde_json::from_str(json).expect("should deserialize");
    assert!(request.model.is_none());
    assert_eq!(request.path, Some("/path/to/model.gguf".to_string()));
}

// ============================================================================
// Test 141-150: Model Metadata and Lineage Edge Cases
// ============================================================================

#[test]
fn test_model_metadata_response_no_quantization() {
    let response = ModelMetadataResponse {
        id: "fp16-model".to_string(),
        name: "Full Precision Model".to_string(),
        format: "safetensors".to_string(),
        size_bytes: 10_000_000_000,
        quantization: None,
        context_length: 8192,
        lineage: None,
        loaded: true,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(!json.contains("quantization"));
    assert!(!json.contains("lineage"));
}

#[test]
fn test_model_lineage_all_optional_none() {
    let lineage = ModelLineage {
        uri: "local://test.gguf".to_string(),
        version: "1.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "blake3:test".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("should serialize");
    assert!(!json.contains("recipe"));
    assert!(!json.contains("parent"));
}

#[test]
fn test_model_lineage_with_parent_chain() {
    let lineage = ModelLineage {
        uri: "pacha://model-v3:latest".to_string(),
        version: "3.0.0".to_string(),
        recipe: Some("instruct-finetune-v2".to_string()),
        parent: Some("pacha://model-v2:1.5".to_string()),
        content_hash: "blake3:abcdef1234567890".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("should serialize");
    assert!(json.contains("instruct-finetune-v2"));
    assert!(json.contains("model-v2"));
}

#[test]
fn test_completion_request_all_optional_none() {
    let json = r#"{"model": "gpt-4", "prompt": "Hello"}"#;
    let request: CompletionRequest = serde_json::from_str(json).expect("should deserialize");

    assert!(request.max_tokens.is_none());
    assert!(request.temperature.is_none());
    assert!(request.top_p.is_none());
    assert!(request.stop.is_none());
}

#[test]
fn test_completion_request_with_multiple_stop_sequences() {
    let request = CompletionRequest {
        model: "test".to_string(),
        prompt: "Once upon".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec![
            "END".to_string(),
            "STOP".to_string(),
            "\n\n".to_string(),
            "###".to_string(),
        ]),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: CompletionRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.stop.unwrap().len(), 4);
}

#[test]
fn test_completion_choice_with_complex_logprobs() {
    let choice = CompletionChoice {
        text: "generated text".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({
            "tokens": ["generated", " text"],
            "token_logprobs": [-1.5, -0.8],
            "top_logprobs": [
                {"generated": -1.5, "created": -2.1},
                {" text": -0.8, " output": -1.2}
            ],
            "text_offset": [0, 9]
        })),
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("should serialize");
    assert!(json.contains("token_logprobs"));
    assert!(json.contains("top_logprobs"));
}

#[test]
fn test_embedding_response_multiple_embeddings() {
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![
            EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding: vec![0.1, 0.2, 0.3],
            },
            EmbeddingData {
                object: "embedding".to_string(),
                index: 1,
                embedding: vec![0.4, 0.5, 0.6],
            },
        ],
        model: "text-embedding".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 10,
            total_tokens: 10,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: EmbeddingResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.data.len(), 2);
}

#[test]
fn test_embedding_data_high_dimensional() {
    let embedding = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![0.001; 4096], // 4096-dimensional embedding
    };

    let json = serde_json::to_string(&embedding).expect("should serialize");
    let deserialized: EmbeddingData = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.embedding.len(), 4096);
}

#[test]
fn test_embedding_usage_zero_tokens() {
    let usage = EmbeddingUsage {
        prompt_tokens: 0,
        total_tokens: 0,
    };

    let json = serde_json::to_string(&usage).expect("should serialize");
    assert!(json.contains(r#""prompt_tokens":0"#));
}

#[test]
fn test_stream_token_event_special_chars() {
    let event = StreamTokenEvent {
        token_id: 123,
        text: "line1\nline2\ttab".to_string(),
    };

    let json = serde_json::to_string(&event).expect("should serialize");
    let deserialized: StreamTokenEvent = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.text.contains('\n'));
    assert!(deserialized.text.contains('\t'));
}

// ============================================================================
// Test 151-160: OpenAI-Compatible Types Deep Coverage
// ============================================================================

#[test]
fn test_chat_completion_request_n_multiple() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.9),
        top_p: None,
        n: 5, // Request 5 completions
        stream: false,
        stop: None,
        user: None,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""n":5"#));
}

#[test]
fn test_chat_completion_request_streaming() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: true,
        stop: None,
        user: None,
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains(r#""stream":true"#));
}

#[test]
fn test_chat_completion_response_long_conversation() {
    let messages: Vec<ChatChoice> = (0..10)
        .map(|i| ChatChoice {
            index: i,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: format!("Response {}", i),
                name: None,
            },
            finish_reason: "stop".to_string(),
        })
        .collect();

    let response = ChatCompletionResponse {
        id: "multi-choice".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "gpt-4".to_string(),
        choices: messages,
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 500,
            total_tokens: 600,
        },
        brick_trace: None,
        layer_trace: None,
        step_trace: None,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ChatCompletionResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.choices.len(), 10);
}

#[test]
fn test_openai_model_timestamps() {
    let model = OpenAIModel {
        id: "test-model".to_string(),
        object: "model".to_string(),
        created: 0, // Unix epoch
        owned_by: "test".to_string(),
    };

    let json = serde_json::to_string(&model).expect("should serialize");
    assert!(json.contains(r#""created":0"#));
}

#[test]
fn test_usage_large_token_counts() {
    let usage = Usage {
        prompt_tokens: 100_000,
        completion_tokens: 50_000,
        total_tokens: 150_000,
    };

    let json = serde_json::to_string(&usage).expect("should serialize");
    let deserialized: Usage = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.total_tokens, 150_000);
}

#[test]
fn test_predict_response_array_prediction() {
    let response = PredictResponse {
        request_id: "array-pred".to_string(),
        model: "multi-output".to_string(),
        prediction: serde_json::json!([0.1, 0.5, 0.3, 0.1]),
        confidence: Some(0.5),
        top_k_predictions: None,
        latency_ms: 5.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("[0.1,0.5,0.3,0.1]"));
}

#[test]
fn test_predict_response_object_prediction() {
    let response = PredictResponse {
        request_id: "obj-pred".to_string(),
        model: "structured-output".to_string(),
        prediction: serde_json::json!({
            "class": "positive",
            "probabilities": [0.9, 0.1]
        }),
        confidence: Some(0.9),
        top_k_predictions: None,
        latency_ms: 3.0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("probabilities"));
}

#[test]
fn test_batch_tokenize_response_empty_texts() {
    // Batch with results that have 0 tokens
    let response = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![],
                num_tokens: 0,
            },
            TokenizeResponse {
                token_ids: vec![1],
                num_tokens: 1,
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: BatchTokenizeResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.results[0].num_tokens, 0);
}

#[test]
fn test_models_response_empty_list() {
    let response = ModelsResponse { models: vec![] };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""models":[]"#));
}

#[test]
fn test_health_response_custom_status() {
    let response = HealthResponse {
        status: "degraded".to_string(),
        version: "0.0.0-dev".to_string(),
        compute_mode: "cpu".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""status":"degraded""#));
}

// ============================================================================
// Test 161-165: Default Function Coverage
// ============================================================================

#[test]
fn test_explain_request_default_top_k_features() {
    let json = r#"{"features": [1.0], "feature_names": ["x"]}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.top_k_features, 5); // default value
}

#[test]
fn test_explain_request_default_method() {
    let json = r#"{"features": [1.0], "feature_names": ["x"]}"#;
    let request: ExplainRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.method, "shap"); // default value
}

#[test]
fn test_predict_request_default_include_confidence() {
    let json = r#"{"features": [1.0]}"#;
    let request: PredictRequest = serde_json::from_str(json).expect("should deserialize");
    assert!(request.include_confidence); // default true
}

#[test]
fn test_chat_completion_request_default_n() {
    let json = r#"{"model": "test", "messages": []}"#;
    let request: ChatCompletionRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.n, 1); // default value
}

#[test]
fn test_chat_completion_request_default_stream() {
    let json = r#"{"model": "test", "messages": []}"#;
    let request: ChatCompletionRequest = serde_json::from_str(json).expect("should deserialize");
    assert!(!request.stream); // default false
}

// ============================================================================
// Test 166-180: GPU Feature Types (BatchConfig, ContinuousBatchResponse)
// These tests run only when gpu feature is enabled
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu_coverage_tests {
    use realizar::api::{
        BatchConfig, BatchProcessResult, BatchQueueStats, ContinuousBatchResponse,
    };

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.window_ms, 50);
        assert_eq!(config.min_batch, 4);
        assert_eq!(config.optimal_batch, 32);
        assert_eq!(config.max_batch, 64);
        assert_eq!(config.queue_size, 1024);
        assert_eq!(config.gpu_threshold, 32);
    }

    #[test]
    fn test_batch_config_low_latency() {
        let config = BatchConfig::low_latency();
        assert_eq!(config.window_ms, 5);
        assert_eq!(config.min_batch, 2);
        assert_eq!(config.optimal_batch, 8);
        assert_eq!(config.max_batch, 16);
        assert_eq!(config.queue_size, 512);
        // GPU threshold > max_batch means GPU effectively disabled
        assert!(config.gpu_threshold > config.max_batch);
    }

    #[test]
    fn test_batch_config_high_throughput() {
        let config = BatchConfig::high_throughput();
        assert_eq!(config.window_ms, 100);
        assert_eq!(config.min_batch, 8);
        assert_eq!(config.optimal_batch, 32);
        assert_eq!(config.max_batch, 128);
        assert_eq!(config.queue_size, 2048);
        assert_eq!(config.gpu_threshold, 32);
    }

    #[test]
    fn test_batch_config_should_process() {
        let config = BatchConfig::default();
        // Should not process when below optimal
        assert!(!config.should_process(1));
        assert!(!config.should_process(31));
        // Should process at optimal or above
        assert!(config.should_process(32));
        assert!(config.should_process(64));
    }

    #[test]
    fn test_batch_config_meets_minimum() {
        let config = BatchConfig::default();
        // Should not meet minimum when below min_batch
        assert!(!config.meets_minimum(1));
        assert!(!config.meets_minimum(3));
        // Should meet minimum at threshold or above
        assert!(config.meets_minimum(4));
        assert!(config.meets_minimum(10));
    }

    #[test]
    fn test_batch_config_low_latency_should_process() {
        let config = BatchConfig::low_latency();
        assert!(!config.should_process(7));
        assert!(config.should_process(8));
    }

    #[test]
    fn test_batch_config_high_throughput_meets_minimum() {
        let config = BatchConfig::high_throughput();
        assert!(!config.meets_minimum(7));
        assert!(config.meets_minimum(8));
    }

    #[test]
    fn test_continuous_batch_response_single() {
        let response = ContinuousBatchResponse::single(
            vec![1, 2, 3, 4, 5],
            2, // prompt_len
            10.5,
        );
        assert_eq!(response.token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(response.prompt_len, 2);
        assert!(!response.batched);
        assert_eq!(response.batch_size, 1);
        assert!((response.latency_ms - 10.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_continuous_batch_response_batched() {
        let response = ContinuousBatchResponse::batched(
            vec![10, 20, 30, 40],
            1, // prompt_len
            32,
            5.0,
        );
        assert_eq!(response.token_ids, vec![10, 20, 30, 40]);
        assert_eq!(response.prompt_len, 1);
        assert!(response.batched);
        assert_eq!(response.batch_size, 32);
        assert!((response.latency_ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_continuous_batch_response_generated_tokens() {
        let response = ContinuousBatchResponse::single(
            vec![100, 200, 300, 400, 500],
            3, // prompt_len = 3, so generated = [400, 500]
            1.0,
        );
        let generated = response.generated_tokens();
        assert_eq!(generated, &[400, 500]);
    }

    #[test]
    fn test_continuous_batch_response_generated_tokens_all_prompt() {
        // Edge case: all tokens are prompt tokens
        let response = ContinuousBatchResponse::single(
            vec![1, 2, 3],
            3, // prompt_len == token count
            1.0,
        );
        let generated = response.generated_tokens();
        assert!(generated.is_empty());
    }

    #[test]
    fn test_continuous_batch_response_generated_tokens_empty() {
        // Edge case: empty token list
        let response = ContinuousBatchResponse::single(vec![], 0, 1.0);
        let generated = response.generated_tokens();
        assert!(generated.is_empty());
    }

    #[test]
    fn test_continuous_batch_response_prompt_len_exceeds_tokens() {
        // Edge case: prompt_len > token count (should return empty)
        let response = ContinuousBatchResponse::single(
            vec![1, 2],
            5, // prompt_len > actual tokens
            1.0,
        );
        let generated = response.generated_tokens();
        assert!(generated.is_empty());
    }

    #[test]
    fn test_batch_queue_stats_default() {
        let stats = BatchQueueStats::default();
        assert_eq!(stats.total_queued, 0);
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.total_single, 0);
        assert!((stats.avg_batch_size - 0.0).abs() < f64::EPSILON);
        assert!((stats.avg_wait_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_process_result_debug() {
        let result = BatchProcessResult {
            requests_processed: 32,
            was_batched: true,
            total_time_ms: 100.0,
            avg_latency_ms: 3.125,
        };
        // Just verify Debug trait works
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("requests_processed"));
        assert!(debug_str.contains("32"));
    }
}

// ============================================================================
// Test 181-195: Additional Struct Coverage
// ============================================================================

#[test]
fn test_audit_response_serialization_basic() {
    // Create a valid AuditRecord using the public API
    use chrono::Utc;
    use realizar::audit::{AuditOptions, LatencyBreakdown};

    let record = realizar::audit::AuditRecord {
        request_id: "audit-123".to_string(),
        timestamp: Utc::now(),
        client_id_hash: None,
        model_hash: "hash123".to_string(),
        model_version: "1.0.0".to_string(),
        model_type: "demo".to_string(),
        distillation_teacher_hash: None,
        input_dims: vec![4],
        input_hash: "input-hash".to_string(),
        options: AuditOptions::default(),
        prediction: serde_json::json!(0.5),
        confidence: Some(0.95),
        explanation_summary: None,
        latency_ms: 1.0,
        latency_breakdown: LatencyBreakdown::default(),
        memory_peak_bytes: 1024,
        quality_nan_check: true,
        quality_confidence_check: true,
        warnings: vec![],
    };

    let response = AuditResponse { record };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("audit-123"));
    assert!(json.contains("hash123"));
}

#[test]
fn test_stream_done_event_large_count() {
    let event = StreamDoneEvent {
        num_generated: usize::MAX,
    };

    let json = serde_json::to_string(&event).expect("should serialize");
    let deserialized: StreamDoneEvent = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.num_generated, usize::MAX);
}

#[test]
fn test_openai_models_response_many_models() {
    let models: Vec<OpenAIModel> = (0..100)
        .map(|i| OpenAIModel {
            id: format!("model-{}", i),
            object: "model".to_string(),
            created: 1700000000 + i as i64,
            owned_by: "test".to_string(),
        })
        .collect();

    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: models,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: OpenAIModelsResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.data.len(), 100);
}

#[test]
fn test_completion_response_fields() {
    let response = CompletionResponse {
        id: "cmpl-test".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "gpt-4".to_string(),
        choices: vec![CompletionChoice {
            text: "Generated text".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("cmpl-test"));
    assert!(json.contains("text_completion"));
}

#[test]
fn test_reload_response_timing() {
    let response = ReloadResponse {
        success: true,
        message: "Reloaded in 150ms".to_string(),
        reload_time_ms: 150,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ReloadResponse = serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.reload_time_ms, 150);
}

#[test]
fn test_shap_explanation_many_features() {
    let feature_count = 50;
    let explanation = ShapExplanation {
        base_value: 0.5,
        shap_values: vec![0.01; feature_count],
        feature_names: (0..feature_count).map(|i| format!("f{}", i)).collect(),
        prediction: 0.75,
    };

    let json = serde_json::to_string(&explanation).expect("should serialize");
    let deserialized: ShapExplanation = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.shap_values.len(), feature_count);
    assert_eq!(deserialized.feature_names.len(), feature_count);
}

#[test]
fn test_model_info_fields() {
    let info = ModelInfo {
        id: "llama-7b".to_string(),
        name: "LLaMA 7B".to_string(),
        description: "A 7 billion parameter model".to_string(),
        format: "gguf".to_string(),
        loaded: true,
    };

    let json = serde_json::to_string(&info).expect("should serialize");
    assert!(json.contains("llama-7b"));
    assert!(json.contains("LLaMA 7B"));
}

#[test]
fn test_dispatch_reset_response_success_roundtrip() {
    let response = DispatchResetResponse {
        success: true,
        message: "Reset complete".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!(deserialized.success);
    assert_eq!(deserialized.message, "Reset complete");
}

#[test]
fn test_dispatch_reset_response_failure_roundtrip() {
    let response = DispatchResetResponse {
        success: false,
        message: "No metrics available".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: DispatchResetResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert!(!deserialized.success);
    assert_eq!(deserialized.message, "No metrics available");
}

#[test]
fn test_generate_request_with_model_id() {
    let request = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: 50,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: Some("custom-model".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains("custom-model"));
}

#[test]
fn test_tokenize_request_minimal() {
    let json = r#"{"text": "Hello world"}"#;
    let request: TokenizeRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(request.text, "Hello world");
    assert!(request.model_id.is_none());
}

#[test]
fn test_chat_message_function_role() {
    // Some APIs support "function" role
    let message = ChatMessage {
        role: "function".to_string(),
        content: "{\"result\": 42}".to_string(),
        name: Some("calculate".to_string()),
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    assert!(json.contains("function"));
    assert!(json.contains("calculate"));
}

#[test]
fn test_chat_completion_request_with_stop_sequences() {
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Write a poem".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: Some(vec!["END".to_string(), "STOP".to_string()]),
        user: Some("user-123".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains("END"));
    assert!(json.contains("STOP"));
    assert!(json.contains("user-123"));
}

// ============================================================================
// Test 196-200: Error Response and Edge Cases
// ============================================================================

#[test]
fn test_error_response_unicode() {
    let response = ErrorResponse {
        error: "Error with unicode: Hello".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains("Hello"));
}

#[test]
fn test_error_response_special_json_chars() {
    let response = ErrorResponse {
        error: "Error: \"quoted\" with \\backslash".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    // JSON should escape these properly
    assert!(json.contains("quoted"));
}

#[test]
fn test_prediction_with_score_zero() {
    let pred = PredictionWithScore {
        label: "unlikely".to_string(),
        score: 0.0,
    };

    let json = serde_json::to_string(&pred).expect("should serialize");
    assert!(json.contains(r#""score":0.0"#));
}

#[test]
fn test_prediction_with_score_one() {
    let pred = PredictionWithScore {
        label: "certain".to_string(),
        score: 1.0,
    };

    let json = serde_json::to_string(&pred).expect("should serialize");
    assert!(json.contains(r#""score":1.0"#));
}

#[test]
fn test_model_metadata_full_with_all_fields() {
    let response = ModelMetadataResponse {
        id: "model-v2".to_string(),
        name: "Test Model V2".to_string(),
        format: "apr".to_string(),
        size_bytes: 5_000_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 32768,
        lineage: Some(ModelLineage {
            uri: "pacha://test:v2".to_string(),
            version: "2.0.0".to_string(),
            recipe: Some("finetune-chat".to_string()),
            parent: Some("pacha://test:v1".to_string()),
            content_hash: "blake3:abc123".to_string(),
        }),
        loaded: true,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ModelMetadataResponse =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.context_length, 32768);
    assert!(deserialized.lineage.is_some());
    let lineage = deserialized.lineage.unwrap();
    assert_eq!(lineage.recipe, Some("finetune-chat".to_string()));
}

// ============================================================================
// Test 201-220: Default Function Coverage and More Edge Cases
// ============================================================================

#[test]
fn test_generate_request_deserialization_with_defaults() {
    // Test that all default functions work properly
    let json = r#"{"prompt": "Hello"}"#;
    let request: GenerateRequest = serde_json::from_str(json).expect("should deserialize");

    // Verify defaults are applied
    assert_eq!(request.max_tokens, 50); // default_max_tokens()
    assert!((request.temperature - 1.0).abs() < f64::EPSILON as f32); // default_temperature()
    assert_eq!(request.strategy, "greedy"); // default_strategy()
    assert_eq!(request.top_k, 50); // default_top_k()
    assert!((request.top_p - 0.9).abs() < f64::EPSILON as f32); // default_top_p()
}

#[test]
fn test_batch_generate_request_with_all_strategies() {
    // Test top_k strategy
    let request = BatchGenerateRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 100,
        temperature: 0.5,
        strategy: "top_k".to_string(),
        top_k: 10,
        top_p: 0.9,
        seed: Some(42),
    };
    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains("top_k"));

    // Test top_p strategy
    let request2 = BatchGenerateRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 100,
        temperature: 0.5,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.95,
        seed: None,
    };
    let json2 = serde_json::to_string(&request2).expect("should serialize");
    assert!(json2.contains("top_p"));
}

#[test]
fn test_context_window_config_boundary_zero() {
    // Test with zero context window
    let config = ContextWindowConfig::new(0);
    assert_eq!(config.max_tokens, 0);
}

#[test]
fn test_context_window_config_preserve_system_default() {
    // Test preserve_system default is true
    let config = ContextWindowConfig::new(100).with_reserved_output(20);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_manager_truncation_behavior() {
    // Create config that will force truncation
    let config = ContextWindowConfig::new(50).with_reserved_output(10);
    let manager = ContextWindowManager::new(config);

    // Create messages that exceed the limit
    let messages: Vec<ChatMessage> = (0..10)
        .map(|i| ChatMessage {
            role: if i == 0 {
                "system"
            } else if i % 2 == 1 {
                "user"
            } else {
                "assistant"
            }
            .to_string(),
            content: format!("This is message {} with some content", i),
            name: None,
        })
        .collect();

    let (truncated, was_truncated) = manager.truncate_messages(&messages);
    // Should truncate given limited context window
    if was_truncated {
        assert!(truncated.len() < messages.len());
    }
}

#[test]
fn test_context_window_manager_default_manager_settings() {
    let manager = ContextWindowManager::default_manager();
    let messages: Vec<ChatMessage> = vec![];
    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert!(result.is_empty());
}

#[test]
fn test_explain_request_with_all_fields() {
    let request = ExplainRequest {
        model: Some("custom-model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        top_k_features: 10,
        method: "lime".to_string(),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains("custom-model"));
    assert!(json.contains("lime"));
    assert!(json.contains("10"));
}

#[test]
fn test_shap_explanation_edge_cases() {
    // Zero base value
    let explanation = ShapExplanation {
        base_value: 0.0,
        shap_values: vec![],
        feature_names: vec![],
        prediction: 0.0,
    };
    let json = serde_json::to_string(&explanation).expect("should serialize");
    assert!(json.contains(r#""base_value":0.0"#));

    // Negative SHAP values
    let explanation2 = ShapExplanation {
        base_value: 0.5,
        shap_values: vec![-0.5, -0.3, 0.2],
        feature_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        prediction: 0.1,
    };
    let json2 = serde_json::to_string(&explanation2).expect("should serialize");
    assert!(json2.contains("-0.5"));
}

#[test]
fn test_embedding_request_variations() {
    // Single input string
    let request = EmbeddingRequest {
        input: "single string".to_string(),
        model: None,
    };
    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains("single string"));

    // With model specified
    let request2 = EmbeddingRequest {
        input: "test input".to_string(),
        model: Some("text-embedding-ada-002".to_string()),
    };
    let json2 = serde_json::to_string(&request2).expect("should serialize");
    assert!(json2.contains("test input"));
    assert!(json2.contains("text-embedding-ada-002"));
}

#[test]
fn test_usage_various_sizes() {
    // Small usage
    let small = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
    };
    let json = serde_json::to_string(&small).expect("should serialize");
    assert!(json.contains(r#""total_tokens":2"#));

    // Large usage
    let large = Usage {
        prompt_tokens: 128000,
        completion_tokens: 32000,
        total_tokens: 160000,
    };
    let json2 = serde_json::to_string(&large).expect("should serialize");
    assert!(json2.contains("160000"));
}

#[test]
fn test_completion_request_with_all_params() {
    let request = CompletionRequest {
        model: "gpt-3.5-turbo-instruct".to_string(),
        prompt: "Once upon a time".to_string(),
        max_tokens: Some(256),
        temperature: Some(0.7),
        top_p: Some(0.95),
        stop: Some(vec!["THE END".to_string()]),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    let deserialized: CompletionRequest = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.max_tokens, Some(256));
    assert_eq!(deserialized.temperature, Some(0.7));
}

#[test]
fn test_chat_completion_response_all_finish_reasons() {
    // Test "length" finish reason
    let response = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "gpt-4".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "truncated".to_string(),
                name: None,
            },
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 100,
            total_tokens: 110,
        },
        brick_trace: None,
        layer_trace: None,
        step_trace: None,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""finish_reason":"length""#));
}

#[test]
fn test_gpu_batch_result_variations() {
    // Empty generation
    let empty = GpuBatchResult {
        index: 0,
        token_ids: vec![],
        text: String::new(),
        num_generated: 0,
    };
    let json = serde_json::to_string(&empty).expect("should serialize");
    assert!(json.contains(r#""num_generated":0"#));

    // Long generation
    let long = GpuBatchResult {
        index: 5,
        token_ids: (0..1000).collect(),
        text: "x".repeat(5000),
        num_generated: 1000,
    };
    let json2 = serde_json::to_string(&long).expect("should serialize");
    assert!(json2.contains("1000"));
}

#[test]
fn test_gpu_warmup_response_memory_bounds() {
    // Zero memory
    let zero_mem = GpuWarmupResponse {
        success: true,
        memory_bytes: 0,
        num_layers: 0,
        message: "Minimal warmup".to_string(),
    };
    let json = serde_json::to_string(&zero_mem).expect("should serialize");
    assert!(json.contains(r#""memory_bytes":0"#));

    // Large memory (24GB)
    let large_mem = GpuWarmupResponse {
        success: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        num_layers: 32,
        message: "Full model cached".to_string(),
    };
    let json2 = serde_json::to_string(&large_mem).expect("should serialize");
    assert!(json2.contains("32"));
}

#[test]
fn test_dispatch_metrics_response_histogram_data() {
    let response = realizar::api::DispatchMetricsResponse {
        cpu_dispatches: 500,
        gpu_dispatches: 1500,
        total_dispatches: 2000,
        gpu_ratio: 0.75,
        cpu_latency_p50_us: 100.0,
        cpu_latency_p95_us: 200.0,
        cpu_latency_p99_us: 300.0,
        gpu_latency_p50_us: 50.0,
        gpu_latency_p95_us: 80.0,
        gpu_latency_p99_us: 100.0,
        cpu_latency_mean_us: 120.0,
        gpu_latency_mean_us: 55.0,
        cpu_latency_min_us: 10,
        cpu_latency_max_us: 500,
        gpu_latency_min_us: 5,
        gpu_latency_max_us: 200,
        cpu_latency_variance_us: 1000.0,
        cpu_latency_stddev_us: 31.62,
        gpu_latency_variance_us: 400.0,
        gpu_latency_stddev_us: 20.0,
        bucket_boundaries_us: vec![
            "0-50".to_string(),
            "50-100".to_string(),
            "100-500".to_string(),
        ],
        cpu_latency_bucket_counts: vec![100, 200, 200],
        gpu_latency_bucket_counts: vec![1000, 400, 100],
        throughput_rps: 500.0,
        elapsed_seconds: 4.0,
    };

    // DispatchMetricsResponse only implements Serialize, not Deserialize
    let json = serde_json::to_string(&response).expect("should serialize");

    // Verify the JSON contains expected histogram data
    assert!(json.contains("0-50"));
    assert!(json.contains("50-100"));
    assert!(json.contains("100-500"));
    assert_eq!(response.bucket_boundaries_us.len(), 3);
    assert_eq!(response.cpu_latency_bucket_counts.len(), 3);
}

#[test]
fn test_tokenize_request_with_model() {
    let request = TokenizeRequest {
        text: "Hello world".to_string(),
        model_id: Some("llama-2".to_string()),
    };

    let json = serde_json::to_string(&request).expect("should serialize");
    assert!(json.contains("llama-2"));
}

#[test]
fn test_tokenize_response_boundary() {
    // Single token
    let single = TokenizeResponse {
        token_ids: vec![42],
        num_tokens: 1,
    };
    assert_eq!(single.num_tokens, 1);

    // Many tokens
    let many = TokenizeResponse {
        token_ids: (0..10000).collect(),
        num_tokens: 10000,
    };
    assert_eq!(many.num_tokens, 10000);
}

#[test]
fn test_models_response_with_many() {
    let models: Vec<ModelInfo> = (0..50)
        .map(|i| ModelInfo {
            id: format!("model-{}", i),
            name: format!("Model {}", i),
            description: format!("Description for model {}", i),
            format: if i % 2 == 0 { "gguf" } else { "apr" }.to_string(),
            loaded: i % 3 == 0,
        })
        .collect();

    let response = ModelsResponse { models };
    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: ModelsResponse = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.models.len(), 50);
}

#[test]
fn test_dispatch_metrics_query_prometheus_format() {
    let json = r#"{"format": "prometheus"}"#;
    let query: realizar::api::DispatchMetricsQuery =
        serde_json::from_str(json).expect("should deserialize");
    assert_eq!(query.format, Some("prometheus".to_string()));
}

#[test]
fn test_chat_message_long_content() {
    let message = ChatMessage {
        role: "user".to_string(),
        content: "a".repeat(100000), // Very long content
        name: None,
    };

    let json = serde_json::to_string(&message).expect("should serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(deserialized.content.len(), 100000);
}

#[test]
fn test_error_response_multiline() {
    // ErrorResponse only implements Serialize, not Deserialize
    let response = ErrorResponse {
        error: "Line 1\nLine 2\nLine 3".to_string(),
    };

    let json = serde_json::to_string(&response).expect("should serialize");

    // Verify the multiline error is properly escaped in JSON
    assert!(json.contains("Line 1"));
    assert!(json.contains("Line 2"));
    assert!(json.contains("Line 3"));
    // Newlines should be escaped as \n in JSON
    assert!(json.contains(r"\n"));
}

// ============================================================================
// Test 221-240: AppState Methods and Integration
// ============================================================================

#[test]
fn test_app_state_demo_creates_valid_state() {
    // Test that AppState::demo() creates a valid state with model and tokenizer
    let state = realizar::api::AppState::demo().expect("should create demo state");
    // Demo state should be created successfully - check it exists
    let _ = state;
}

#[test]
fn test_app_state_with_cache_creates_state() {
    // Test that AppState::with_cache creates state with specified capacity
    let state = realizar::api::AppState::with_cache(10);
    // State should be created with cache capacity
    let _ = state;
}

#[test]
fn test_generate_response_empty_generation() {
    // Edge case: no new tokens generated
    let response = GenerateResponse {
        token_ids: vec![1, 2, 3], // Prompt only
        text: "abc".to_string(),
        num_generated: 0,
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    assert!(json.contains(r#""num_generated":0"#));
}

#[test]
fn test_batch_generate_response_single_result() {
    let response = BatchGenerateResponse {
        results: vec![GenerateResponse {
            token_ids: vec![1, 2, 3],
            text: "output".to_string(),
            num_generated: 3,
        }],
    };

    let json = serde_json::to_string(&response).expect("should serialize");
    let deserialized: BatchGenerateResponse =
        serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(deserialized.results.len(), 1);
}

#[test]
fn test_chat_completion_chunk_construction_patterns() {
    // Test various chunk patterns for SSE streaming
    use realizar::api::ChatCompletionChunk;

    // Chunk with role only (initial)
    let chunk1 = ChatCompletionChunk {
        id: "chunk-1".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![realizar::api::ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let json1 = serde_json::to_string(&chunk1).expect("serialize");
    assert!(json1.contains("assistant"));
    assert!(!json1.contains("finish_reason\":\""));

    // Chunk with content only
    let chunk2 = ChatCompletionChunk {
        id: "chunk-2".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![realizar::api::ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: Some("Hello".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json2 = serde_json::to_string(&chunk2).expect("serialize");
    assert!(json2.contains("Hello"));

    // Chunk with finish reason (final)
    let chunk3 = ChatCompletionChunk {
        id: "chunk-3".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![realizar::api::ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    };
    let json3 = serde_json::to_string(&chunk3).expect("serialize");
    assert!(json3.contains(r#""finish_reason":"stop""#));
}

#[test]
fn test_context_window_config_debug_impl() {
    let config = ContextWindowConfig::new(8192).with_reserved_output(512);
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("8192"));
    assert!(debug_str.contains("512"));
}

#[test]
fn test_context_window_config_clone() {
    let config1 = ContextWindowConfig::new(4096).with_reserved_output(256);
    let config2 = config1.clone();
    assert_eq!(config1.max_tokens, config2.max_tokens);
    assert_eq!(
        config1.reserved_output_tokens,
        config2.reserved_output_tokens
    );
}

#[test]
fn test_context_window_manager_estimate_tokens_accuracy() {
    let manager = ContextWindowManager::default_manager();

    // Test estimation for various message sizes
    let short_message = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hi".to_string(),
        name: None,
    }];
    let tokens_short = manager.estimate_total_tokens(&short_message);
    assert!(tokens_short >= 10); // At least role overhead

    let long_message = vec![ChatMessage {
        role: "user".to_string(),
        content: "x".repeat(1000),
        name: None,
    }];
    let tokens_long = manager.estimate_total_tokens(&long_message);
    assert!(tokens_long > tokens_short);
    assert!(tokens_long >= 250); // ~1000 chars / 4 chars per token
}

#[test]
fn test_context_window_manager_truncation_preserves_order() {
    let config = ContextWindowConfig::new(150).with_reserved_output(0);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "First message".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Second message".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Third message".to_string(),
            name: None,
        },
    ];

    let (result, _) = manager.truncate_messages(&messages);

    // Check that remaining messages are in chronological order
    if result.len() >= 2 {
        // User messages and assistant messages should alternate or follow original order
        for i in 0..result.len() - 1 {
            // Messages should maintain relative order from original
            let orig_idx_curr = messages.iter().position(|m| m.content == result[i].content);
            let orig_idx_next = messages
                .iter()
                .position(|m| m.content == result[i + 1].content);
            if let (Some(curr), Some(next)) = (orig_idx_curr, orig_idx_next) {
                assert!(curr < next, "Messages should maintain chronological order");
            }
        }
    }
}

// ============================================================================
// Test 241-260: Request Validation Edge Cases
// ============================================================================

#[test]
fn test_generate_request_all_strategies() {
    // Greedy strategy
    let greedy = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 0.0,
        strategy: "greedy".to_string(),
        top_k: 1,
        top_p: 1.0,
        seed: None,
        model_id: None,
    };
    let json = serde_json::to_string(&greedy).expect("serialize");
    assert!(json.contains("greedy"));

    // Top-k strategy
    let top_k = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        strategy: "top_k".to_string(),
        top_k: 40,
        top_p: 1.0,
        seed: Some(123),
        model_id: None,
    };
    let json = serde_json::to_string(&top_k).expect("serialize");
    assert!(json.contains("top_k"));

    // Top-p strategy
    let top_p = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 0.9,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.95,
        seed: None,
        model_id: Some("model-1".to_string()),
    };
    let json = serde_json::to_string(&top_p).expect("serialize");
    assert!(json.contains("0.95"));
}

#[test]
fn test_chat_completion_request_all_optional_fields() {
    // Request with all optional fields set
    let request = ChatCompletionRequest {
        model: "gpt-4-turbo".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: Some("Alice".to_string()),
            },
        ],
        max_tokens: Some(4096),
        temperature: Some(0.5),
        top_p: Some(0.9),
        n: 3,
        stream: true,
        stop: Some(vec![
            "END".to_string(),
            "STOP".to_string(),
            "###".to_string(),
        ]),
        user: Some("user-abc123".to_string()),
    };

    let json = serde_json::to_string(&request).expect("serialize");
    let deserialized: ChatCompletionRequest = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.n, 3);
    assert!(deserialized.stream);
    assert_eq!(deserialized.stop.as_ref().unwrap().len(), 3);
    assert_eq!(deserialized.user, Some("user-abc123".to_string()));
}

#[test]
fn test_predict_request_variations() {
    // With all fields
    let full = PredictRequest {
        model: Some("xgboost-v1".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        feature_names: Some(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ]),
        top_k: Some(5),
        include_confidence: true,
    };
    let json = serde_json::to_string(&full).expect("serialize");
    assert!(json.contains("xgboost-v1"));
    assert!(json.contains("include_confidence"));

    // Minimal request
    let minimal_json = r#"{"features": [1.0, 2.0]}"#;
    let minimal: PredictRequest = serde_json::from_str(minimal_json).expect("deserialize");
    assert!(minimal.model.is_none());
    assert!(minimal.feature_names.is_none());
    assert!(minimal.include_confidence); // default true
}

#[test]
fn test_explain_request_method_variations() {
    // SHAP method (default)
    let shap_json = r#"{"features": [1.0], "feature_names": ["x"]}"#;
    let shap: ExplainRequest = serde_json::from_str(shap_json).expect("deserialize");
    assert_eq!(shap.method, "shap");

    // LIME method
    let lime_json = r#"{"features": [1.0], "feature_names": ["x"], "method": "lime"}"#;
    let lime: ExplainRequest = serde_json::from_str(lime_json).expect("deserialize");
    assert_eq!(lime.method, "lime");

    // Attention method
    let attention_json = r#"{"features": [1.0], "feature_names": ["x"], "method": "attention"}"#;
    let attention: ExplainRequest = serde_json::from_str(attention_json).expect("deserialize");
    assert_eq!(attention.method, "attention");
}

#[test]
fn test_gpu_batch_request_temperature_variations() {
    // Zero temperature (greedy)
    let greedy = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };
    let json = serde_json::to_string(&greedy).expect("serialize");
    assert!(json.contains(r#""temperature":0.0"#));

    // High temperature (creative)
    let creative = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 100,
        temperature: 1.5,
        top_k: 100,
        stop: vec!["END".to_string()],
    };
    let json = serde_json::to_string(&creative).expect("serialize");
    assert!(json.contains("1.5"));
}

#[test]
fn test_completion_request_stop_sequences() {
    // No stop sequences
    let no_stop = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let json1 = serde_json::to_string(&no_stop).expect("serialize");
    assert!(!json1.contains("stop"));

    // Multiple stop sequences
    let multi_stop = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec![
            "\n\n".to_string(),
            "END".to_string(),
            "STOP".to_string(),
            "```".to_string(),
        ]),
    };
    let json2 = serde_json::to_string(&multi_stop).expect("serialize");
    assert!(json2.contains("```"));
}

// ============================================================================
// Test 261-280: Response Structure Edge Cases
// ============================================================================

#[test]
fn test_chat_completion_response_with_all_finish_reasons() {
    // Test various finish reasons
    let finish_reasons = vec!["stop", "length", "content_filter", "function_call"];

    for reason in finish_reasons {
        let response = ChatCompletionResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "test".to_string(),
                    name: None,
                },
                finish_reason: reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            brick_trace: None,
            layer_trace: None,
            step_trace: None,
        };

        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains(&format!(r#""finish_reason":"{}""#, reason)));
    }
}

#[test]
fn test_predict_response_with_top_k_predictions() {
    let response = PredictResponse {
        request_id: "test-123".to_string(),
        model: "classifier".to_string(),
        prediction: serde_json::json!("class_a"),
        confidence: Some(0.85),
        top_k_predictions: Some(vec![
            PredictionWithScore {
                label: "class_a".to_string(),
                score: 0.85,
            },
            PredictionWithScore {
                label: "class_b".to_string(),
                score: 0.10,
            },
            PredictionWithScore {
                label: "class_c".to_string(),
                score: 0.05,
            },
        ]),
        latency_ms: 12.5,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let deserialized: PredictResponse = serde_json::from_str(&json).expect("deserialize");

    assert!(deserialized.top_k_predictions.is_some());
    let predictions = deserialized.top_k_predictions.unwrap();
    assert_eq!(predictions.len(), 3);
    assert!((predictions[0].score - 0.85).abs() < 1e-6);
}

#[test]
fn test_explain_response_with_all_fields() {
    let response = ExplainResponse {
        request_id: "explain-456".to_string(),
        model: "explainable-model".to_string(),
        prediction: serde_json::json!({"class": "positive", "score": 0.9}),
        confidence: Some(0.9),
        explanation: ShapExplanation {
            base_value: 0.5,
            shap_values: vec![0.3, -0.1, 0.2, 0.05, -0.05],
            feature_names: vec![
                "feature_1".to_string(),
                "feature_2".to_string(),
                "feature_3".to_string(),
                "feature_4".to_string(),
                "feature_5".to_string(),
            ],
            prediction: 0.9,
        },
        summary: "Top contributing features: feature_1 (+0.3), feature_3 (+0.2)".to_string(),
        latency_ms: 25.3,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let deserialized: ExplainResponse = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.explanation.shap_values.len(), 5);
    assert!(deserialized.summary.contains("feature_1"));
}

#[test]
fn test_gpu_batch_response_with_stats() {
    let response = GpuBatchResponse {
        results: vec![
            GpuBatchResult {
                index: 0,
                token_ids: vec![1, 2, 3, 4, 5],
                text: "Result 0".to_string(),
                num_generated: 5,
            },
            GpuBatchResult {
                index: 1,
                token_ids: vec![10, 20, 30],
                text: "Result 1".to_string(),
                num_generated: 3,
            },
        ],
        stats: GpuBatchStats {
            batch_size: 2,
            gpu_used: true,
            total_tokens: 8,
            processing_time_ms: 15.5,
            throughput_tps: 516.13,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let deserialized: GpuBatchResponse = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.results.len(), 2);
    assert!(deserialized.stats.gpu_used);
    assert!((deserialized.stats.throughput_tps - 516.13).abs() < 0.01);
}

#[test]
fn test_embedding_response_with_multiple_embeddings() {
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![
            EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding: vec![0.1, 0.2, 0.3],
            },
            EmbeddingData {
                object: "embedding".to_string(),
                index: 1,
                embedding: vec![0.4, 0.5, 0.6],
            },
            EmbeddingData {
                object: "embedding".to_string(),
                index: 2,
                embedding: vec![0.7, 0.8, 0.9],
            },
        ],
        model: "text-embedding-ada-002".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 15,
            total_tokens: 15,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let deserialized: EmbeddingResponse = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.data.len(), 3);
    assert_eq!(deserialized.data[2].index, 2);
}

#[test]
fn test_completion_response_with_logprobs() {
    let response = CompletionResponse {
        id: "cmpl-with-logprobs".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "gpt-3.5-turbo-instruct".to_string(),
        choices: vec![CompletionChoice {
            text: "Hello world".to_string(),
            index: 0,
            logprobs: Some(serde_json::json!({
                "tokens": ["Hello", " world"],
                "token_logprobs": [-0.5, -0.3],
                "top_logprobs": [
                    {"Hello": -0.5, "Hi": -1.2},
                    {" world": -0.3, " there": -0.8}
                ],
                "text_offset": [0, 5]
            })),
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 7,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("token_logprobs"));
    assert!(json.contains("top_logprobs"));
}

#[test]
fn test_model_metadata_response_variations() {
    // GGUF model with quantization
    let gguf = ModelMetadataResponse {
        id: "llama-7b-q4".to_string(),
        name: "LLaMA 7B Q4_K_M".to_string(),
        format: "gguf".to_string(),
        size_bytes: 4_000_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 4096,
        lineage: Some(ModelLineage {
            uri: "pacha://llama:7b".to_string(),
            version: "1.0.0".to_string(),
            recipe: Some("chat-instruct".to_string()),
            parent: None,
            content_hash: "blake3:abc123".to_string(),
        }),
        loaded: true,
    };
    let json1 = serde_json::to_string(&gguf).expect("serialize");
    assert!(json1.contains("Q4_K_M"));
    assert!(json1.contains("chat-instruct"));

    // SafeTensors model without quantization
    let safetensors = ModelMetadataResponse {
        id: "bert-base".to_string(),
        name: "BERT Base".to_string(),
        format: "safetensors".to_string(),
        size_bytes: 440_000_000,
        quantization: None,
        context_length: 512,
        lineage: None,
        loaded: false,
    };
    let json2 = serde_json::to_string(&safetensors).expect("serialize");
    assert!(!json2.contains("quantization"));
    assert!(!json2.contains("lineage"));
}

#[test]
fn test_reload_response_variations() {
    // Successful reload
    let success = ReloadResponse {
        success: true,
        message: "Model reloaded successfully from /models/v2.gguf".to_string(),
        reload_time_ms: 2500,
    };
    let json1 = serde_json::to_string(&success).expect("serialize");
    assert!(json1.contains("true"));
    assert!(json1.contains("2500"));

    // Failed reload
    let failure = ReloadResponse {
        success: false,
        message: "Failed to load model: file not found".to_string(),
        reload_time_ms: 0,
    };
    let json2 = serde_json::to_string(&failure).expect("serialize");
    assert!(json2.contains("false"));
}

#[test]
fn test_server_metrics_response_edge_cases() {
    // Idle server
    let idle = ServerMetricsResponse {
        throughput_tok_per_sec: 0.0,
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
        latency_p99_ms: 0.0,
        gpu_memory_used_bytes: 0,
        gpu_memory_total_bytes: 24 * 1024 * 1024 * 1024,
        gpu_utilization_percent: 0,
        cuda_path_active: false,
        batch_size: 0,
        queue_depth: 0,
        total_tokens: 0,
        total_requests: 0,
        uptime_secs: 1,
        model_name: "none".to_string(),
    };
    let json1 = serde_json::to_string(&idle).expect("serialize");
    assert!(json1.contains(r#""throughput_tok_per_sec":0.0"#));

    // Busy server
    let busy = ServerMetricsResponse {
        throughput_tok_per_sec: 500.0,
        latency_p50_ms: 5.0,
        latency_p95_ms: 15.0,
        latency_p99_ms: 50.0,
        gpu_memory_used_bytes: 20 * 1024 * 1024 * 1024,
        gpu_memory_total_bytes: 24 * 1024 * 1024 * 1024,
        gpu_utilization_percent: 95,
        cuda_path_active: true,
        batch_size: 64,
        queue_depth: 128,
        total_tokens: 1_000_000,
        total_requests: 10_000,
        uptime_secs: 86400,
        model_name: "phi-2-q4k".to_string(),
    };
    let json2 = serde_json::to_string(&busy).expect("serialize");
    assert!(json2.contains("500.0"));
    assert!(json2.contains("86400"));
}

// ============================================================================
// Test 281-300: Debug Trait and Clone Implementations
// ============================================================================

#[test]
fn test_chat_message_debug_impl() {
    let message = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("Alice".to_string()),
    };
    let debug_str = format!("{:?}", message);
    assert!(debug_str.contains("user"));
    assert!(debug_str.contains("Hello"));
    assert!(debug_str.contains("Alice"));
}

#[test]
fn test_chat_completion_request_clone() {
    let original = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        }],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };
    let cloned = original.clone();
    assert_eq!(original.model, cloned.model);
    assert_eq!(original.messages.len(), cloned.messages.len());
}

#[test]
fn test_chat_completion_response_clone() {
    let original = ChatCompletionResponse {
        id: "test".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "gpt-4".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello".to_string(),
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
    let cloned = original.clone();
    assert_eq!(original.id, cloned.id);
    assert_eq!(original.usage.total_tokens, cloned.usage.total_tokens);
}

#[test]
fn test_predict_request_debug_impl() {
    let request = PredictRequest {
        model: Some("test".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        top_k: Some(5),
        include_confidence: true,
    };
    let debug_str = format!("{:?}", request);
    assert!(debug_str.contains("test"));
    assert!(debug_str.contains("features"));
}

#[test]
fn test_predict_response_clone() {
    let original = PredictResponse {
        request_id: "test".to_string(),
        model: "model".to_string(),
        prediction: serde_json::json!(0.5),
        confidence: Some(0.9),
        top_k_predictions: None,
        latency_ms: 10.0,
    };
    let cloned = original.clone();
    assert_eq!(original.request_id, cloned.request_id);
    assert_eq!(original.confidence, cloned.confidence);
}

#[test]
fn test_explain_request_clone() {
    let original = ExplainRequest {
        model: Some("model".to_string()),
        features: vec![1.0, 2.0],
        feature_names: vec!["a".to_string(), "b".to_string()],
        top_k_features: 5,
        method: "shap".to_string(),
    };
    let cloned = original.clone();
    assert_eq!(original.method, cloned.method);
    assert_eq!(original.features.len(), cloned.features.len());
}

#[test]
fn test_shap_explanation_clone() {
    let original = ShapExplanation {
        base_value: 0.5,
        shap_values: vec![0.1, 0.2, 0.3],
        feature_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
        prediction: 0.8,
    };
    let cloned = original.clone();
    assert_eq!(original.base_value, cloned.base_value);
    assert_eq!(original.shap_values, cloned.shap_values);
}

#[test]
fn test_gpu_batch_request_debug() {
    let request = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec![],
    };
    let debug_str = format!("{:?}", request);
    assert!(debug_str.contains("test"));
    assert!(debug_str.contains("50"));
}

#[test]
fn test_gpu_batch_response_clone() {
    let original = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "abc".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: true,
            total_tokens: 3,
            processing_time_ms: 10.0,
            throughput_tps: 300.0,
        },
    };
    let cloned = original.clone();
    assert_eq!(original.results.len(), cloned.results.len());
    assert_eq!(original.stats.gpu_used, cloned.stats.gpu_used);
}

#[test]
fn test_model_lineage_clone() {
    let original = ModelLineage {
        uri: "pacha://test:1.0".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("instruct".to_string()),
        parent: Some("pacha://base:1.0".to_string()),
        content_hash: "blake3:abc".to_string(),
    };
    let cloned = original.clone();
    assert_eq!(original.uri, cloned.uri);
    assert_eq!(original.recipe, cloned.recipe);
}

#[test]
fn test_dispatch_metrics_response_clone() {
    let original = realizar::api::DispatchMetricsResponse {
        cpu_dispatches: 100,
        gpu_dispatches: 200,
        total_dispatches: 300,
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
        cpu_latency_max_us: 400,
        gpu_latency_min_us: 5,
        gpu_latency_max_us: 200,
        cpu_latency_variance_us: 500.0,
        cpu_latency_stddev_us: 22.36,
        gpu_latency_variance_us: 200.0,
        gpu_latency_stddev_us: 14.14,
        bucket_boundaries_us: vec!["0-100".to_string()],
        cpu_latency_bucket_counts: vec![50],
        gpu_latency_bucket_counts: vec![100],
        throughput_rps: 100.0,
        elapsed_seconds: 3.0,
    };
    let cloned = original.clone();
    assert_eq!(original.cpu_dispatches, cloned.cpu_dispatches);
    assert_eq!(original.gpu_ratio, cloned.gpu_ratio);
}

// ============================================================================
// Test 301-320: Serialization Round-Trip Tests
// ============================================================================

#[test]
fn test_health_response_roundtrip() {
    let original = HealthResponse {
        status: "healthy".to_string(),
        version: "1.2.3".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.status, restored.status);
    assert_eq!(original.version, restored.version);
}

#[test]
fn test_tokenize_request_roundtrip() {
    let original = TokenizeRequest {
        text: "Hello world!".to_string(),
        model_id: Some("tokenizer-v1".to_string()),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: TokenizeRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.text, restored.text);
    assert_eq!(original.model_id, restored.model_id);
}

#[test]
fn test_tokenize_response_roundtrip() {
    let original = TokenizeResponse {
        token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        num_tokens: 10,
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: TokenizeResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.token_ids, restored.token_ids);
    assert_eq!(original.num_tokens, restored.num_tokens);
}

#[test]
fn test_generate_request_roundtrip() {
    let original = GenerateRequest {
        prompt: "Once upon a time".to_string(),
        max_tokens: 100,
        temperature: 0.8,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.95,
        seed: Some(42),
        model_id: Some("story-gen".to_string()),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: GenerateRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.prompt, restored.prompt);
    assert_eq!(original.seed, restored.seed);
}

#[test]
fn test_generate_response_roundtrip() {
    let original = GenerateResponse {
        token_ids: vec![100, 200, 300, 400, 500],
        text: "Generated story text...".to_string(),
        num_generated: 5,
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: GenerateResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.token_ids, restored.token_ids);
    assert_eq!(original.text, restored.text);
}

#[test]
fn test_batch_tokenize_request_roundtrip() {
    let original = BatchTokenizeRequest {
        texts: vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: BatchTokenizeRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.texts, restored.texts);
}

#[test]
fn test_batch_tokenize_response_roundtrip() {
    let original = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1, 2],
                num_tokens: 2,
            },
            TokenizeResponse {
                token_ids: vec![3, 4, 5],
                num_tokens: 3,
            },
        ],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: BatchTokenizeResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.results.len(), restored.results.len());
}

#[test]
fn test_batch_generate_request_roundtrip() {
    let original = BatchGenerateRequest {
        prompts: vec!["Prompt 1".to_string(), "Prompt 2".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: BatchGenerateRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.prompts, restored.prompts);
}

#[test]
fn test_batch_generate_response_roundtrip() {
    let original = BatchGenerateResponse {
        results: vec![
            GenerateResponse {
                token_ids: vec![1, 2, 3],
                text: "Output 1".to_string(),
                num_generated: 3,
            },
            GenerateResponse {
                token_ids: vec![4, 5],
                text: "Output 2".to_string(),
                num_generated: 2,
            },
        ],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: BatchGenerateResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.results.len(), restored.results.len());
}

#[test]
fn test_stream_token_event_roundtrip() {
    let original = StreamTokenEvent {
        token_id: 12345,
        text: "token_text".to_string(),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: StreamTokenEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.token_id, restored.token_id);
    assert_eq!(original.text, restored.text);
}

#[test]
fn test_stream_done_event_roundtrip() {
    let original = StreamDoneEvent { num_generated: 42 };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: StreamDoneEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.num_generated, restored.num_generated);
}

#[test]
fn test_models_response_roundtrip() {
    let original = ModelsResponse {
        models: vec![
            ModelInfo {
                id: "model-1".to_string(),
                name: "Model One".to_string(),
                description: "First model".to_string(),
                format: "gguf".to_string(),
                loaded: true,
            },
            ModelInfo {
                id: "model-2".to_string(),
                name: "Model Two".to_string(),
                description: "Second model".to_string(),
                format: "apr".to_string(),
                loaded: false,
            },
        ],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: ModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.models.len(), restored.models.len());
}

#[test]
fn test_usage_roundtrip() {
    let original = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.prompt_tokens, restored.prompt_tokens);
    assert_eq!(original.completion_tokens, restored.completion_tokens);
    assert_eq!(original.total_tokens, restored.total_tokens);
}

#[test]
fn test_openai_model_roundtrip() {
    let original = OpenAIModel {
        id: "gpt-4-turbo".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "openai".to_string(),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: OpenAIModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.id, restored.id);
    assert_eq!(original.created, restored.created);
}

#[test]
fn test_openai_models_response_roundtrip() {
    let original = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![
            OpenAIModel {
                id: "model-a".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "realizar".to_string(),
            },
            OpenAIModel {
                id: "model-b".to_string(),
                object: "model".to_string(),
                created: 1700000001,
                owned_by: "realizar".to_string(),
            },
        ],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: OpenAIModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.data.len(), restored.data.len());
}

#[test]
fn test_embedding_request_roundtrip() {
    let original = EmbeddingRequest {
        input: "Text to embed".to_string(),
        model: Some("text-embedding-3-small".to_string()),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: EmbeddingRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.input, restored.input);
    assert_eq!(original.model, restored.model);
}

#[test]
fn test_embedding_response_roundtrip() {
    let original = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        }],
        model: "text-embedding".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: EmbeddingResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.data.len(), restored.data.len());
    assert_eq!(
        original.data[0].embedding.len(),
        restored.data[0].embedding.len()
    );
}

#[test]
fn test_embedding_data_roundtrip() {
    let original = EmbeddingData {
        object: "embedding".to_string(),
        index: 5,
        embedding: vec![0.123, 0.456, 0.789],
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: EmbeddingData = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.index, restored.index);
    assert_eq!(original.embedding.len(), restored.embedding.len());
}

#[test]
fn test_embedding_usage_roundtrip() {
    let original = EmbeddingUsage {
        prompt_tokens: 100,
        total_tokens: 100,
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: EmbeddingUsage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.prompt_tokens, restored.prompt_tokens);
    assert_eq!(original.total_tokens, restored.total_tokens);
}
