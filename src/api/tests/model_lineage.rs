
// =========================================================================
// Coverage Tests: ModelLineage struct
// =========================================================================

#[test]
fn test_model_lineage_serialize_cov() {
    let lineage = ModelLineage {
        uri: "pacha://org/model".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("training-recipe-v1".to_string()),
        parent: Some("parent-model".to_string()),
        content_hash: "abc123def456".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(json.contains("pacha://org/model"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("abc123def456"));
}

#[test]
fn test_model_lineage_deserialize_cov() {
    let json = r#"{
        "uri": "pacha://test/model",
        "version": "2.0.0",
        "content_hash": "hash123"
    }"#;
    let lineage: ModelLineage = serde_json::from_str(json).expect("deserialize");
    assert_eq!(lineage.uri, "pacha://test/model");
    assert_eq!(lineage.version, "2.0.0");
    assert!(lineage.recipe.is_none());
    assert!(lineage.parent.is_none());
}

#[test]
fn test_model_lineage_clone_debug_cov() {
    let lineage = ModelLineage {
        uri: "pacha://x/y".to_string(),
        version: "0.1.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "xyz".to_string(),
    };
    let cloned = lineage.clone();
    assert_eq!(lineage.uri, cloned.uri);
    let debug_str = format!("{:?}", lineage);
    assert!(debug_str.contains("uri"));
    assert!(debug_str.contains("version"));
}

// =========================================================================
// Coverage Tests: ReloadRequest/ReloadResponse structs
// =========================================================================

#[test]
fn test_reload_request_full_cov() {
    let req = ReloadRequest {
        model: Some("my-model".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("my-model"));
    assert!(json.contains("/path/to/model.gguf"));
}

#[test]
fn test_reload_request_empty_cov() {
    let req = ReloadRequest {
        model: None,
        path: None,
    };
    let json = serde_json::to_string(&req).expect("serialize");
    // Should be minimal since both are None with skip_serializing_if
    assert_eq!(json, "{}");
}

#[test]
fn test_reload_request_deserialize_cov() {
    let json = r#"{"model": "test-model"}"#;
    let req: ReloadRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(req.model, Some("test-model".to_string()));
    assert!(req.path.is_none());
}

#[test]
fn test_reload_response_success_cov() {
    let resp = ReloadResponse {
        success: true,
        message: "Model reloaded successfully".to_string(),
        reload_time_ms: 150,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("150"));
}

#[test]
fn test_reload_response_failure_cov() {
    let resp = ReloadResponse {
        success: false,
        message: "Model file not found".to_string(),
        reload_time_ms: 0,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("false"));
    assert!(json.contains("not found"));
}

#[test]
fn test_reload_response_clone_debug_cov() {
    let resp = ReloadResponse {
        success: true,
        message: "OK".to_string(),
        reload_time_ms: 50,
    };
    let cloned = resp.clone();
    assert_eq!(resp.success, cloned.success);
    let debug_str = format!("{:?}", resp);
    assert!(debug_str.contains("success"));
}

// =========================================================================
// Coverage Tests: CompletionRequest struct
// =========================================================================

#[test]
fn test_completion_request_full_cov2() {
    let req = CompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        prompt: "Once upon a time".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: Some(vec!["\n".to_string(), "END".to_string()]),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("gpt-3.5-turbo"));
    assert!(json.contains("Once upon a time"));
    assert!(json.contains("100"));
}

#[test]
fn test_completion_request_minimal_cov() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("test"));
    // Optional fields should be omitted
    assert!(!json.contains("max_tokens"));
}

#[test]
fn test_completion_request_deserialize_cov() {
    let json = r#"{"model":"llama","prompt":"Test prompt","max_tokens":50}"#;
    let req: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(req.model, "llama");
    assert_eq!(req.prompt, "Test prompt");
    assert_eq!(req.max_tokens, Some(50));
}

// =========================================================================
// Coverage Tests: CompletionResponse struct
// =========================================================================

#[test]
fn test_completion_response_full_cov2() {
    let resp = CompletionResponse {
        id: "cmpl-123".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "gpt-3.5-turbo".to_string(),
        choices: vec![CompletionChoice {
            text: "Generated text here".to_string(),
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
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("cmpl-123"));
    assert!(json.contains("text_completion"));
    assert!(json.contains("Generated text here"));
}

#[test]
fn test_completion_response_multiple_choices_cov() {
    let resp = CompletionResponse {
        id: "cmpl-456".to_string(),
        object: "text_completion".to_string(),
        created: 1700000001,
        model: "llama".to_string(),
        choices: vec![
            CompletionChoice {
                text: "Choice A".to_string(),
                index: 0,
                logprobs: Some(serde_json::json!({"tokens": []})),
                finish_reason: "length".to_string(),
            },
            CompletionChoice {
                text: "Choice B".to_string(),
                index: 1,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
        ],
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
            total_tokens: 15,
        },
    };
    assert_eq!(resp.choices.len(), 2);
    assert!(resp.choices[0].logprobs.is_some());
    assert!(resp.choices[1].logprobs.is_none());
}

// =========================================================================
// Coverage Tests: CompletionChoice struct
// =========================================================================

#[test]
fn test_completion_choice_with_logprobs_cov() {
    let choice = CompletionChoice {
        text: "Hello world".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({
            "tokens": ["Hello", " world"],
            "token_logprobs": [-0.5, -0.3]
        })),
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("Hello world"));
    assert!(json.contains("logprobs"));
}

#[test]
fn test_completion_choice_finish_reasons_cov() {
    for reason in ["stop", "length", "content_filter"] {
        let choice = CompletionChoice {
            text: "test".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: reason.to_string(),
        };
        assert_eq!(choice.finish_reason, reason);
    }
}

#[test]
fn test_completion_choice_clone_debug_cov() {
    let choice = CompletionChoice {
        text: "Output".to_string(),
        index: 5,
        logprobs: None,
        finish_reason: "length".to_string(),
    };
    let cloned = choice.clone();
    assert_eq!(choice.index, cloned.index);
    let debug_str = format!("{:?}", choice);
    assert!(debug_str.contains("index"));
    assert!(debug_str.contains("finish_reason"));
}

// =========================================================================
// Coverage Tests: GenerateRequest variations
// =========================================================================

#[test]
fn test_generate_request_all_strategies_cov() {
    for strategy in ["greedy", "top_k", "top_p", "nucleus"] {
        let req = GenerateRequest {
            prompt: "Test".to_string(),
            max_tokens: 10,
            temperature: 0.5,
            strategy: strategy.to_string(),
            top_k: 40,
            top_p: 0.95,
            seed: Some(123),
            model_id: None,
        };
        assert_eq!(req.strategy, strategy);
    }
}

#[test]
fn test_generate_request_with_model_id_cov() {
    let req = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: 50,
        temperature: 1.0,
        strategy: "greedy".to_string(),
        top_k: 50,
        top_p: 1.0,
        seed: None,
        model_id: Some("custom-model-v2".to_string()),
    };
    assert_eq!(req.model_id, Some("custom-model-v2".to_string()));
}

#[test]
fn test_generate_request_temperature_extremes_cov() {
    let cold = GenerateRequest {
        prompt: "Cold".to_string(),
        max_tokens: 5,
        temperature: 0.0,
        strategy: "greedy".to_string(),
        top_k: 1,
        top_p: 1.0,
        seed: Some(1),
        model_id: None,
    };
    let hot = GenerateRequest {
        prompt: "Hot".to_string(),
        max_tokens: 5,
        temperature: 2.0,
        strategy: "top_p".to_string(),
        top_k: 100,
        top_p: 0.99,
        seed: Some(2),
        model_id: None,
    };
    assert!(cold.temperature < 0.1);
    assert!(hot.temperature > 1.5);
}

// =========================================================================
// Coverage Tests: GenerateResponse struct
// =========================================================================

#[test]
fn test_generate_response_serialize_cov() {
    let resp = GenerateResponse {
        token_ids: vec![1, 2, 3, 4],
        text: "The quick brown fox".to_string(),
        num_generated: 4,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("quick brown fox"));
    assert!(json.contains("4"));
}

#[test]
fn test_generate_response_multiple_tokens_cov() {
    let resp = GenerateResponse {
        token_ids: vec![100, 200, 300],
        text: "Hello world".to_string(),
        num_generated: 3,
    };
    assert_eq!(resp.token_ids.len(), 3);
    assert_eq!(resp.num_generated, 3);
}

#[test]
fn test_generate_response_empty_cov() {
    let resp = GenerateResponse {
        token_ids: vec![],
        text: String::new(),
        num_generated: 0,
    };
    assert!(resp.text.is_empty());
    assert_eq!(resp.num_generated, 0);
    assert!(resp.token_ids.is_empty());
}

#[test]
fn test_generate_response_deserialize_cov() {
    let json = r#"{"token_ids":[1,2,3],"text":"abc","num_generated":3}"#;
    let resp: GenerateResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(resp.token_ids, vec![1, 2, 3]);
    assert_eq!(resp.text, "abc");
}

// =========================================================================
// Coverage Tests: ChatMessage struct
// =========================================================================

#[test]
fn test_chat_message_all_roles_cov() {
    for role in ["system", "user", "assistant", "function"] {
        let msg = ChatMessage {
            role: role.to_string(),
            content: format!("Content for {}", role),
            name: None,
        };
        assert_eq!(msg.role, role);
    }
}

#[test]
fn test_chat_message_long_content_cov() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "X".repeat(10000),
        name: None,
    };
    assert_eq!(msg.content.len(), 10000);
}

#[test]
fn test_chat_message_empty_content_cov() {
    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: String::new(),
        name: None,
    };
    assert!(msg.content.is_empty());
}

#[test]
fn test_chat_message_serialize_roundtrip_cov() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello with \"quotes\" and \\backslash\\".to_string(),
        name: Some("test_user".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("serialize");
    let deserialized: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(msg.role, deserialized.role);
    assert_eq!(msg.content, deserialized.content);
    assert_eq!(msg.name, deserialized.name);
}

#[test]
fn test_chat_message_with_name_cov() {
    let msg = ChatMessage {
        role: "function".to_string(),
        content: "Function result".to_string(),
        name: Some("my_function".to_string()),
    };
    assert_eq!(msg.name, Some("my_function".to_string()));
    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("my_function"));
}
