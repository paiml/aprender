
// =========================================================================
// Coverage Tests: ModelsResponse
// =========================================================================

#[test]
fn test_models_response_serialize_cov() {
    use crate::registry::ModelInfo;

    let resp = ModelsResponse {
        models: vec![
            ModelInfo {
                id: "model-1".to_string(),
                name: "LLaMA 7B".to_string(),
                description: "A large language model".to_string(),
                format: "gguf".to_string(),
                loaded: true,
            },
            ModelInfo {
                id: "model-2".to_string(),
                name: "Phi-2".to_string(),
                description: "Small but capable".to_string(),
                format: "safetensors".to_string(),
                loaded: false,
            },
        ],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("model-1"));
    assert!(json.contains("LLaMA"));
    assert!(json.contains("gguf"));
}

// =========================================================================
// Coverage Tests: ChatCompletionRequest deserialization
// =========================================================================

#[test]
fn test_chat_completion_request_full_cov() {
    let req = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: Some("Alice".to_string()),
            },
        ],
        max_tokens: Some(100),
        temperature: Some(0.8),
        top_p: Some(0.95),
        n: 2,
        stream: true,
        stop: Some(vec!["###".to_string()]),
        user: Some("user-123".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: ChatCompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model, "gpt-4");
    assert_eq!(parsed.n, 2);
    assert!(parsed.stream);
}

// =========================================================================
// Coverage Tests: ChatChoice
// =========================================================================

#[test]
fn test_chat_choice_serialize_cov() {
    let choice = ChatChoice {
        index: 1,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Hello there!".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("assistant"));
    assert!(json.contains("length"));
    assert!(json.contains("Hello there!"));
}

// =========================================================================
// Coverage Tests: TokenizeRequest/Response
// =========================================================================

#[test]
fn test_tokenize_request_with_model_id_cov() {
    let req = TokenizeRequest {
        text: "test text".to_string(),
        model_id: Some("custom-tokenizer".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("test text"));
    assert!(json.contains("custom-tokenizer"));
}

#[test]
fn test_tokenize_response_serialize_cov() {
    let resp = TokenizeResponse {
        token_ids: vec![101, 102, 103, 104],
        num_tokens: 4,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("101"));
    assert!(json.contains("num_tokens"));
}

// =========================================================================
// Coverage Tests: GenerateRequest
// =========================================================================

#[test]
fn test_generate_request_top_p_cov() {
    let req = GenerateRequest {
        prompt: "Once upon a time".to_string(),
        max_tokens: 200,
        temperature: 0.5,
        strategy: "top_p".to_string(),
        top_k: 50,
        top_p: 0.85,
        seed: None,
        model_id: Some("story-model".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("top_p"));
    assert!(json.contains("0.85"));
    assert!(json.contains("story-model"));
}

// =========================================================================
// Coverage Tests: OpenAIModelsResponse
// =========================================================================

#[test]
fn test_openai_models_response_deserialize_cov() {
    let json = r#"{
        "object": "list",
        "data": [
            {
                "id": "gpt-4-turbo",
                "object": "model",
                "created": 1699000000,
                "owned_by": "openai"
            }
        ]
    }"#;
    let resp: OpenAIModelsResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(resp.object, "list");
    assert_eq!(resp.data.len(), 1);
    assert_eq!(resp.data[0].id, "gpt-4-turbo");
}

// =========================================================================
// Coverage Tests: InMemorySinkWrapper
// =========================================================================

#[test]
fn test_in_memory_sink_wrapper_flush_cov() {
    use crate::audit::AuditSink;

    let sink = Arc::new(InMemoryAuditSink::new());
    let wrapper = InMemorySinkWrapper(sink.clone());

    // Test flush returns Ok
    let result = wrapper.flush();
    assert!(result.is_ok());

    // Test write_batch returns Ok
    let result = wrapper.write_batch(&[]);
    assert!(result.is_ok());
}

// =========================================================================
// Coverage Tests: AppState methods
// =========================================================================

#[test]
fn test_app_state_has_quantized_model_cov() {
    let state = AppState::demo().expect("test");
    // Demo state doesn't have quantized model
    assert!(!state.has_quantized_model());
    assert!(state.quantized_model().is_none());
}

#[test]
fn test_app_state_with_cache_cov() {
    let state = AppState::with_cache(10);
    // Should have model and tokenizer
    assert!(state.model.is_some());
    assert!(state.tokenizer.is_some());
    // Should have cache
    assert!(state.cache.is_some());
}

#[test]
fn test_app_state_get_model_no_registry_cov() {
    let state = AppState::demo().expect("test");
    // Single model mode - should return model
    let result = state.get_model(None);
    assert!(result.is_ok());
}

// =========================================================================
// Coverage Tests: create_demo_apr_model
// =========================================================================

#[test]
fn test_create_demo_apr_model_cov() {
    let result = crate::api::create_demo_apr_model(8);
    assert!(result.is_ok());
    let model = result.expect("operation failed");
    assert_eq!(model.tensor_count(), 1);
}

// =========================================================================
// Coverage Tests: OpenAI completions types
// =========================================================================

#[test]
fn test_completion_request_cov() {
    let json = r#"{
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Say hello",
        "max_tokens": 50
    }"#;
    let req: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(req.model, "gpt-3.5-turbo-instruct");
    assert_eq!(req.prompt, "Say hello");
}

#[test]
fn test_completion_response_cov() {
    let resp = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "text-davinci-003".to_string(),
        choices: vec![CompletionChoice {
            text: "Hello!".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 2,
            total_tokens: 7,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("cmpl-123"));
    assert!(json.contains("text_completion"));
}

#[test]
fn test_completion_choice_cov() {
    let choice = CompletionChoice {
        text: "Generated text".to_string(),
        index: 2,
        logprobs: Some(serde_json::json!({"tokens": [], "token_logprobs": []})),
        finish_reason: "length".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("Generated text"));
    assert!(json.contains("logprobs"));
}

// =========================================================================
// Coverage Tests: Embedding types
// =========================================================================

#[test]
fn test_embedding_response_cov() {
    let resp = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        }],
        model: "text-embedding-ada-002".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 4,
            total_tokens: 4,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("text-embedding-ada-002"));
    assert!(json.contains("0.1"));
}

#[test]
fn test_embedding_data_cov() {
    let data = EmbeddingData {
        object: "embedding".to_string(),
        index: 5,
        embedding: vec![0.5; 768],
    };
    assert_eq!(data.index, 5);
    assert_eq!(data.embedding.len(), 768);
}

#[test]
fn test_embedding_usage_cov() {
    let usage = EmbeddingUsage {
        prompt_tokens: 100,
        total_tokens: 100,
    };
    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains("100"));
}

// =========================================================================
// Coverage Tests: ErrorResponse variations
// =========================================================================

#[test]
fn test_error_response_long_message_ext_cov() {
    let resp = ErrorResponse {
        error: "A".repeat(1000),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.len() > 1000);
}

#[test]
fn test_error_response_special_chars_ext_cov() {
    let resp = ErrorResponse {
        error: "Error with \"quotes\" and \\backslashes\\".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("\\\""));
}

// =========================================================================
// Coverage Tests: Usage struct
// =========================================================================

#[test]
fn test_usage_serialize_ext_cov() {
    let usage = Usage {
        prompt_tokens: 150,
        completion_tokens: 75,
        total_tokens: 225,
    };
    let json = serde_json::to_string(&usage).expect("serialize");
    assert!(json.contains("225"));
}

#[test]
fn test_usage_zero_values_ext_cov() {
    let usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    assert_eq!(usage.total_tokens, 0);
}

// =========================================================================
// Coverage Tests: TokenizeResponse
// =========================================================================

#[test]
fn test_tokenize_response_empty_ext_cov() {
    let resp = TokenizeResponse {
        token_ids: vec![],
        num_tokens: 0,
    };
    assert!(resp.token_ids.is_empty());
    assert_eq!(resp.num_tokens, 0);
}

#[test]
fn test_tokenize_response_large_ext_cov() {
    let resp = TokenizeResponse {
        token_ids: vec![1; 10000],
        num_tokens: 10000,
    };
    assert_eq!(resp.token_ids.len(), 10000);
}

// =========================================================================
// Coverage Tests: EmbeddingRequest
// =========================================================================

#[test]
fn test_embedding_request_serialize_ext_cov() {
    let req = EmbeddingRequest {
        input: "Embed this text".to_string(),
        model: Some("text-embedding-ada-002".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("Embed this text"));
}

#[test]
fn test_embedding_request_deserialize_ext_cov() {
    let json = r#"{"input":"Hello","model":"test-model"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(req.input, "Hello");
    assert_eq!(req.model, Some("test-model".to_string()));
}

// =========================================================================
// Coverage Tests: BatchGenerateRequest
// =========================================================================

#[test]
fn test_batch_generate_request_serialize_ext_cov() {
    let req = BatchGenerateRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.8,
        strategy: "greedy".to_string(),
        top_k: 40,
        top_p: 0.9,
        seed: None,
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("World"));
}

#[test]
fn test_batch_generate_request_single_prompt_ext_cov() {
    let req = BatchGenerateRequest {
        prompts: vec!["Single prompt".to_string()],
        max_tokens: 100,
        temperature: 1.0,
        strategy: "top_k".to_string(),
        top_k: 50,
        top_p: 1.0,
        seed: Some(42),
    };
    assert_eq!(req.prompts.len(), 1);
}
