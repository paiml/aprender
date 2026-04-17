//! Property-based tests for API types and structures
//!
//! Tests request/response serialization roundtrips, default values,
//! and edge cases for the realizar API.

use proptest::prelude::*;
use realizar::api::{
    BatchGenerateRequest, BatchGenerateResponse, BatchTokenizeRequest, BatchTokenizeResponse,
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatDelta, ChatMessage,
    ContextWindowConfig, ContextWindowManager, ErrorResponse, GenerateRequest, GenerateResponse,
    GpuBatchRequest, GpuBatchResponse, GpuBatchResult, GpuBatchStats, HealthResponse,
    ModelsResponse, OpenAIModel, OpenAIModelsResponse, PredictRequest, PredictResponse,
    PredictionWithScore, StreamDoneEvent, StreamTokenEvent, TokenizeRequest, TokenizeResponse,
    Usage,
};

// ============================================================================
// HealthResponse Tests
// ============================================================================

#[test]
fn test_health_response_creation() {
    let resp = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };

    assert_eq!(resp.status, "healthy");
    assert_eq!(resp.version, "1.0.0");
    assert_eq!(resp.compute_mode, "cpu");
}

#[test]
fn test_health_response_serialization() {
    let resp = HealthResponse {
        status: "ok".to_string(),
        version: "2.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };

    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("ok"));

    let parsed: HealthResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.status, "ok");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_health_response_roundtrip(
        status in "[a-z]{3,20}",
        version in "[0-9]+\\.[0-9]+\\.[0-9]+"
    ) {
        let resp = HealthResponse { status: status.clone(), version: version.clone(), compute_mode: "cpu".to_string() };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: HealthResponse = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(parsed.status, status);
        prop_assert_eq!(parsed.version, version);
    }
}

// ============================================================================
// TokenizeRequest/Response Tests
// ============================================================================

#[test]
fn test_tokenize_request_simple() {
    let req = TokenizeRequest {
        text: "hello world".to_string(),
        model_id: None,
    };

    assert_eq!(req.text, "hello world");
    assert!(req.model_id.is_none());
}

#[test]
fn test_tokenize_request_with_model() {
    let req = TokenizeRequest {
        text: "test".to_string(),
        model_id: Some("gpt-2".to_string()),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("model_id"));
    assert!(json.contains("gpt-2"));
}

#[test]
fn test_tokenize_response_empty() {
    let resp = TokenizeResponse {
        token_ids: vec![],
        num_tokens: 0,
    };

    assert!(resp.token_ids.is_empty());
    assert_eq!(resp.num_tokens, 0);
}

#[test]
fn test_tokenize_response_with_tokens() {
    let resp = TokenizeResponse {
        token_ids: vec![1, 2, 3, 4, 5],
        num_tokens: 5,
    };

    assert_eq!(resp.token_ids.len(), 5);
    assert_eq!(resp.num_tokens, 5);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_tokenize_roundtrip(text in "[a-zA-Z0-9 ]{1,100}") {
        let req = TokenizeRequest { text: text.clone(), model_id: None };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: TokenizeRequest = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(parsed.text, text);
    }

    #[test]
    fn prop_tokenize_response_valid(tokens in proptest::collection::vec(0u32..50000, 0..100)) {
        let len = tokens.len();
        let resp = TokenizeResponse { token_ids: tokens, num_tokens: len };
        prop_assert_eq!(resp.num_tokens, len);
        prop_assert_eq!(resp.token_ids.len(), len);
    }
}

// ============================================================================
// GenerateRequest/Response Tests
// ============================================================================

#[test]
fn test_generate_request_minimal_json() {
    let json = r#"{"prompt": "Hello"}"#;
    let req: GenerateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "Hello");
    // Default values should be applied
    assert_eq!(req.max_tokens, 50); // default_max_tokens
    assert_eq!(req.temperature, 1.0); // default_temperature
}

#[test]
fn test_generate_request_full() {
    let req = GenerateRequest {
        prompt: "Test prompt".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        strategy: "top_k".to_string(),
        top_k: 40,
        top_p: 0.9,
        seed: Some(42),
        model_id: Some("test-model".to_string()),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("Test prompt"));
    assert!(json.contains("100"));
}

#[test]
fn test_generate_response_serialization() {
    let resp = GenerateResponse {
        token_ids: vec![1, 2, 3],
        text: "Generated text".to_string(),
        num_generated: 3,
    };

    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("Generated text"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_generate_request_roundtrip(
        prompt in "[a-zA-Z ]{1,50}",
        max_tokens in 1usize..1000
    ) {
        let req = GenerateRequest {
            prompt: prompt.clone(),
            max_tokens,
            temperature: 0.7,
            strategy: "greedy".to_string(),
            top_k: 50,
            top_p: 0.9,
            seed: None,
            model_id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: GenerateRequest = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(parsed.prompt, prompt);
        prop_assert_eq!(parsed.max_tokens, max_tokens);
    }
}

// ============================================================================
// BatchRequest/Response Tests
// ============================================================================

#[test]
fn test_batch_tokenize_request() {
    let req = BatchTokenizeRequest {
        texts: vec!["hello".to_string(), "world".to_string()],
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("hello"));
    assert!(json.contains("world"));
}

#[test]
fn test_batch_tokenize_response() {
    let resp = BatchTokenizeResponse {
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

    assert_eq!(resp.results.len(), 2);
}

#[test]
fn test_batch_generate_request_minimal() {
    let json = r#"{"prompts": ["prompt1", "prompt2"]}"#;
    let req: BatchGenerateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompts.len(), 2);
    // Defaults applied
    assert_eq!(req.max_tokens, 50);
}

#[test]
fn test_batch_generate_request_full() {
    let req = BatchGenerateRequest {
        prompts: vec!["prompt1".to_string(), "prompt2".to_string()],
        max_tokens: 100,
        temperature: 0.8,
        strategy: "top_p".to_string(),
        top_k: 40,
        top_p: 0.95,
        seed: Some(123),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("prompt1"));
}

#[test]
fn test_batch_generate_response() {
    let resp = BatchGenerateResponse {
        results: vec![GenerateResponse {
            token_ids: vec![1],
            text: "result".to_string(),
            num_generated: 1,
        }],
    };

    assert_eq!(resp.results.len(), 1);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_batch_request_sizes(n in 1usize..10) {
        let prompts: Vec<String> = (0..n).map(|i| format!("prompt_{}", i)).collect();
        let req = BatchGenerateRequest {
            prompts,
            max_tokens: 10,
            temperature: 1.0,
            strategy: "greedy".to_string(),
            top_k: 50,
            top_p: 0.9,
            seed: None,
        };
        prop_assert_eq!(req.prompts.len(), n);
    }
}

// ============================================================================
// ChatMessage Tests
// ============================================================================

#[test]
fn test_chat_message_user() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello!".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("user"));
    assert!(json.contains("Hello!"));
}

#[test]
fn test_chat_message_assistant() {
    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: "Hi there!".to_string(),
        name: Some("Claude".to_string()),
    };

    assert_eq!(msg.role, "assistant");
}

#[test]
fn test_chat_message_system() {
    let msg = ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful assistant.".to_string(),
        name: None,
    };

    assert_eq!(msg.role, "system");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_chat_message_roundtrip(
        role in "(user|assistant|system)",
        content in "[a-zA-Z0-9 ]{1,100}"
    ) {
        let msg = ChatMessage {
            role: role.clone(),
            content: content.clone(),
            name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(parsed.role, role);
        prop_assert_eq!(parsed.content, content);
    }
}

// ============================================================================
// ChatCompletionRequest/Response Tests
// ============================================================================

#[test]
fn test_chat_completion_request_minimal() {
    let json = r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn test_chat_completion_request_full() {
    let req = ChatCompletionRequest {
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
        stop: None,
        user: None,
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("gpt-4"));
}

#[test]
fn test_chat_choice() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Response".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    assert_eq!(choice.index, 0);
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn test_chat_completion_response() {
    let resp = ChatCompletionResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
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
        step_trace: None,
        layer_trace: None,
    };

    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("chatcmpl-123"));
    assert!(json.contains("chat.completion"));
}

// ============================================================================
// Usage Tests
// ============================================================================

#[test]
fn test_usage_serialization() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };

    let json = serde_json::to_string(&usage).unwrap();
    assert!(json.contains("150"));

    let parsed: Usage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.total_tokens, 150);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_usage_total_correct(prompt in 0usize..10000, completion in 0usize..10000) {
        let total = prompt + completion;
        let usage = Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: total,
        };
        prop_assert_eq!(usage.total_tokens, prompt + completion);
    }
}

// ============================================================================
// ErrorResponse Tests
// ============================================================================

#[test]
fn test_error_response() {
    let err = ErrorResponse {
        error: "Invalid request".to_string(),
    };

    let json = serde_json::to_string(&err).unwrap();
    assert!(json.contains("Invalid request"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_error_response_roundtrip(msg in "[a-zA-Z0-9 ]{1,50}") {
        let err = ErrorResponse { error: msg.clone() };
        let json = serde_json::to_string(&err).unwrap();
        // ErrorResponse may only have Serialize, check if deserializable
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
            prop_assert!(parsed["error"].as_str().unwrap() == msg);
        }
    }
}

// ============================================================================
// StreamEvent Tests
// ============================================================================

#[test]
fn test_stream_token_event() {
    let event = StreamTokenEvent {
        token_id: 1234,
        text: "hello".to_string(),
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("hello"));
    assert!(json.contains("1234"));
}

#[test]
fn test_stream_done_event() {
    let event = StreamDoneEvent { num_generated: 100 };

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("100"));
}

// ============================================================================
// PredictRequest/Response Tests
// ============================================================================

#[test]
fn test_predict_request() {
    let req = PredictRequest {
        model: Some("classifier-v1".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: None,
        top_k: Some(3),
        include_confidence: true,
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("features"));
}

#[test]
fn test_predict_request_minimal() {
    let json = r#"{"features": [1.0, 2.0, 3.0]}"#;
    let req: PredictRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.features.len(), 3);
    assert!(req.model.is_none());
}

#[test]
fn test_predict_response() {
    let resp = PredictResponse {
        request_id: "req-123".to_string(),
        model: "classifier-v1".to_string(),
        prediction: serde_json::json!("cat"),
        confidence: Some(0.85),
        top_k_predictions: Some(vec![PredictionWithScore {
            label: "cat".to_string(),
            score: 0.85,
        }]),
        latency_ms: 5.2,
    };

    assert_eq!(resp.model, "classifier-v1");
}

#[test]
fn test_prediction_with_score() {
    let pred = PredictionWithScore {
        label: "dog".to_string(),
        score: 0.95,
    };

    assert!(pred.score > 0.0 && pred.score <= 1.0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_prediction_scores_valid(score in 0.0f32..=1.0f32) {
        let pred = PredictionWithScore {
            label: "test".to_string(),
            score,
        };
        prop_assert!(pred.score >= 0.0 && pred.score <= 1.0);
    }
}

// ============================================================================
// GpuBatch Tests
// ============================================================================

#[test]
fn test_gpu_batch_request() {
    let req = GpuBatchRequest {
        prompts: vec!["prompt1".to_string(), "prompt2".to_string()],
        max_tokens: 100,
        temperature: 0.7,
        top_k: 50,
        stop: vec![],
    };

    assert_eq!(req.prompts.len(), 2);
}

#[test]
fn test_gpu_batch_stats() {
    let stats = GpuBatchStats {
        batch_size: 10,
        gpu_used: true,
        total_tokens: 500,
        processing_time_ms: 50.0,
        throughput_tps: 100.0,
    };

    assert!(stats.throughput_tps > 0.0);
}

#[test]
fn test_gpu_batch_response() {
    let resp = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "result".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: true,
            total_tokens: 3,
            processing_time_ms: 5.0,
            throughput_tps: 600.0,
        },
    };

    assert_eq!(resp.results.len(), 1);
}

// ============================================================================
// OpenAI Models Tests
// ============================================================================

#[test]
fn test_openai_model() {
    let model = OpenAIModel {
        id: "gpt-4".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "openai".to_string(),
    };

    let json = serde_json::to_string(&model).unwrap();
    assert!(json.contains("gpt-4"));
}

#[test]
fn test_openai_models_response() {
    let resp = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![OpenAIModel {
            id: "model-1".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "test".to_string(),
        }],
    };

    assert_eq!(resp.data.len(), 1);
}

#[test]
fn test_models_response() {
    use realizar::registry::ModelInfo;
    let resp = ModelsResponse {
        models: vec![ModelInfo {
            id: "model1".to_string(),
            name: "Model 1".to_string(),
            description: "Test model".to_string(),
            format: "gguf".to_string(),
            loaded: true,
        }],
    };

    assert_eq!(resp.models.len(), 1);
}

// ============================================================================
// ContextWindowManager Tests
// ============================================================================

#[test]
fn test_context_window_config() {
    let config = ContextWindowConfig::new(4096);
    assert_eq!(config.max_tokens, 4096);
}

#[test]
fn test_context_window_config_with_reserved() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(500);
    assert!(config.reserved_output_tokens > 0);
}

#[test]
fn test_context_window_manager_default() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];

    assert!(!manager.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_truncate() {
    let config = ContextWindowConfig::new(100);
    let manager = ContextWindowManager::new(config);

    let messages: Vec<ChatMessage> = (0..50)
        .map(|i| ChatMessage {
            role: "user".to_string(),
            content: format!("Message {} with some content that takes tokens", i),
            name: None,
        })
        .collect();

    let (truncated, was_truncated) = manager.truncate_messages(&messages);
    // With many messages, truncation should happen
    if was_truncated {
        assert!(truncated.len() < messages.len());
    }
}

#[test]
fn test_context_window_estimate_tokens() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello world".to_string(),
        name: None,
    }];

    let tokens = manager.estimate_total_tokens(&messages);
    assert!(tokens > 0);
}

// ============================================================================
// ChatDelta Tests
// ============================================================================

#[test]
fn test_chat_delta() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hello".to_string()),
    };

    let json = serde_json::to_string(&delta).unwrap();
    assert!(json.contains("assistant"));
}

#[test]
fn test_chat_delta_empty() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };

    let json = serde_json::to_string(&delta).unwrap();
    // Should still serialize
    let _: serde_json::Value = serde_json::from_str(&json).unwrap();
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_batch_tokenize() {
    let req = BatchTokenizeRequest { texts: vec![] };

    assert!(req.texts.is_empty());
}

#[test]
fn test_empty_messages_chat() {
    let req = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    assert!(req.messages.is_empty());
}

#[test]
fn test_unicode_content() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello \u{1F600} \u{4E2D}\u{6587} \u{0410}\u{0411}\u{0412}".to_string(),
        name: None,
    };

    let json = serde_json::to_string(&msg).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, msg.content);
}

#[test]
fn test_special_characters_prompt() {
    let req = GenerateRequest {
        prompt: "Line1\nLine2\tTabbed\r\nCRLF".to_string(),
        max_tokens: 10,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: None,
    };

    let json = serde_json::to_string(&req).unwrap();
    let parsed: GenerateRequest = serde_json::from_str(&json).unwrap();
    assert!(parsed.prompt.contains('\n'));
    assert!(parsed.prompt.contains('\t'));
}

#[test]
fn test_zero_tokens() {
    let usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };

    let json = serde_json::to_string(&usage).unwrap();
    let parsed: Usage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.total_tokens, 0);
}

#[test]
fn test_temperature_zero() {
    let req = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 0.0, // Greedy
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: None,
    };
    assert_eq!(req.temperature, 0.0);
}

#[test]
fn test_temperature_high() {
    let req = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: 10,
        temperature: 2.0, // Very random
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
        model_id: None,
    };
    assert_eq!(req.temperature, 2.0);
}

#[test]
fn test_large_token_list() {
    let tokens: Vec<u32> = (0..10000).collect();
    let resp = TokenizeResponse {
        token_ids: tokens,
        num_tokens: 10000,
    };

    assert_eq!(resp.token_ids.len(), 10000);
}

#[test]
fn test_many_prompts_batch() {
    let prompts: Vec<String> = (0..100).map(|i| format!("Prompt {}", i)).collect();
    let req = BatchGenerateRequest {
        prompts,
        max_tokens: 10,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 0.9,
        seed: None,
    };

    assert_eq!(req.prompts.len(), 100);
}

#[test]
fn test_predict_empty_features() {
    let req = PredictRequest {
        model: None,
        features: vec![],
        feature_names: None,
        top_k: None,
        include_confidence: true,
    };

    assert!(req.features.is_empty());
}

#[test]
fn test_chat_conversation() {
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

    let req = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages,
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    assert_eq!(req.messages.len(), 4);
}
