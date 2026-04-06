
    #[test]
    fn test_completion_request_full() {
        let request = CompletionRequest {
            model: "llama2".to_string(),
            prompt: "Once upon a time".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop: Some(vec!["\n".to_string(), "END".to_string()]),
        };
        let json = serde_json::to_string(&request).expect("serialize");
        assert!(json.contains("llama2"));
        assert!(json.contains("Once upon a time"));
        assert!(json.contains("100"));
        assert!(json.contains("0.7"));
        assert!(json.contains("0.9"));
        assert!(json.contains("END"));
    }

    #[test]
    fn test_completion_request_optional_fields_skipped() {
        let request = CompletionRequest {
            model: "test".to_string(),
            prompt: "hi".to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
        };
        let json = serde_json::to_string(&request).expect("serialize");
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("temperature"));
        assert!(!json.contains("top_p"));
        assert!(!json.contains("stop"));
    }

    #[test]
    fn test_completion_request_debug() {
        let request = CompletionRequest {
            model: "debug_test".to_string(),
            prompt: "test".to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
        };
        let debug = format!("{:?}", request);
        assert!(debug.contains("CompletionRequest"));
        assert!(debug.contains("debug_test"));
    }

    #[test]
    fn test_completion_request_clone() {
        let request = CompletionRequest {
            model: "test".to_string(),
            prompt: "hello".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.5),
            top_p: None,
            stop: None,
        };
        let cloned = request.clone();
        assert_eq!(cloned.model, "test");
        assert_eq!(cloned.prompt, "hello");
        assert_eq!(cloned.max_tokens, Some(50));
    }

    // =========================================================================
    // CompletionResponse serialization/deserialization
    // =========================================================================

    #[test]
    fn test_completion_response_basic() {
        let response = CompletionResponse {
            id: "cmpl-123".to_string(),
            object: "text_completion".to_string(),
            created: 1234567890,
            model: "llama2".to_string(),
            choices: vec![CompletionChoice {
                text: "generated text".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 10,
                total_tokens: 15,
            },
        };
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].text, "generated text");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn test_completion_response_serialization() {
        let response = CompletionResponse {
            id: "cmpl-test".to_string(),
            object: "text_completion".to_string(),
            created: 1000,
            model: "test".to_string(),
            choices: vec![CompletionChoice {
                text: "output".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "length".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 7,
                total_tokens: 10,
            },
        };
        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("text_completion"));
        assert!(json.contains("output"));
        assert!(json.contains("length"));
    }

    #[test]
    fn test_completion_response_clone() {
        let response = CompletionResponse {
            id: "test".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let cloned = response.clone();
        assert_eq!(cloned.id, "test");
    }

    #[test]
    fn test_completion_response_debug() {
        let response = CompletionResponse {
            id: "debug-id".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let debug = format!("{:?}", response);
        assert!(debug.contains("CompletionResponse"));
        assert!(debug.contains("debug-id"));
    }

    // =========================================================================
    // CompletionChoice
    // =========================================================================

    #[test]
    fn test_completion_choice_with_logprobs() {
        let choice = CompletionChoice {
            text: "hello".to_string(),
            index: 0,
            logprobs: Some(serde_json::json!({"tokens": ["hello"], "token_logprobs": [-0.5]})),
            finish_reason: "stop".to_string(),
        };
        assert!(choice.logprobs.is_some());
        let json = serde_json::to_string(&choice).expect("serialize");
        assert!(json.contains("logprobs"));
        assert!(json.contains("token_logprobs"));
    }

    #[test]
    fn test_completion_choice_no_logprobs() {
        let choice = CompletionChoice {
            text: "world".to_string(),
            index: 1,
            logprobs: None,
            finish_reason: "length".to_string(),
        };
        let json = serde_json::to_string(&choice).expect("serialize");
        assert!(!json.contains("logprobs"));
        assert!(json.contains("length"));
    }

    #[test]
    fn test_completion_choice_clone() {
        let choice = CompletionChoice {
            text: "test".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        };
        let cloned = choice.clone();
        assert_eq!(cloned.text, "test");
        assert_eq!(cloned.index, 0);
    }

    #[test]
    fn test_completion_choice_debug() {
        let choice = CompletionChoice {
            text: "debug".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        };
        let debug = format!("{:?}", choice);
        assert!(debug.contains("CompletionChoice"));
    }

    // =========================================================================
    // epoch_secs / epoch_millis
    // =========================================================================

    #[test]
    fn test_epoch_secs_returns_reasonable_value() {
        let secs = epoch_secs();
        // Should be after Jan 1, 2020 (1577836800)
        assert!(secs > 1_577_836_800);
    }

    #[test]
    fn test_epoch_millis_returns_reasonable_value() {
        let millis = epoch_millis();
        // Should be after Jan 1, 2020 in millis
        assert!(millis > 1_577_836_800_000);
    }

    #[test]
    fn test_epoch_millis_greater_than_secs() {
        let secs = epoch_secs() as u128;
        let millis = epoch_millis();
        assert!(millis >= secs * 1000);
        assert!(millis < (secs + 2) * 1000);
    }

    // =========================================================================
    // completion_resp helper
    // =========================================================================

    #[test]
    fn test_completion_resp_stop_reason() {
        let resp = completion_resp(
            "cmpl-test",
            "model-x".to_string(),
            "output text".to_string(),
            10,
            5,
            100, // max_tokens = 100, completion_tokens = 5 < 100 => "stop"
        );
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
        assert!(resp.id.starts_with("cmpl-test-"));
        assert_eq!(resp.model, "model-x");
        assert_eq!(resp.object, "text_completion");
    }

    #[test]
    fn test_completion_resp_length_reason() {
        let resp = completion_resp(
            "cmpl-len",
            "model-y".to_string(),
            "long output".to_string(),
            5,
            100,
            100, // max_tokens = 100, completion_tokens = 100 >= 100 => "length"
        );
        assert_eq!(resp.choices[0].finish_reason, "length");
    }

    #[test]
    fn test_completion_resp_length_reason_exceeds() {
        let resp = completion_resp(
            "cmpl",
            "m".to_string(),
            "text".to_string(),
            1,
            200,
            100, // completion_tokens = 200 > max_tokens = 100 => "length"
        );
        assert_eq!(resp.choices[0].finish_reason, "length");
    }

    #[test]
    fn test_completion_resp_zero_tokens() {
        let resp = completion_resp("cmpl", "m".to_string(), String::new(), 0, 0, 100);
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert_eq!(resp.usage.total_tokens, 0);
        assert!(resp.choices[0].text.is_empty());
    }

    #[test]
    fn test_completion_resp_single_choice() {
        let resp = completion_resp("prefix", "model".to_string(), "text".to_string(), 1, 1, 10);
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].index, 0);
        assert!(resp.choices[0].logprobs.is_none());
    }

    // =========================================================================
    // EmbeddingRequest edge cases
    // =========================================================================

    #[test]
    fn test_embedding_request_empty_input() {
        let request = EmbeddingRequest {
            input: String::new(),
            model: None,
        };
        let json = serde_json::to_string(&request).expect("serialize");
        let parsed: EmbeddingRequest = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.input.is_empty());
    }

    #[test]
    fn test_embedding_request_long_input() {
        let request = EmbeddingRequest {
            input: "word ".repeat(1000),
            model: Some("ada".to_string()),
        };
        assert_eq!(request.input.len(), 5000);
    }

    #[test]
    fn test_embedding_request_debug() {
        let request = EmbeddingRequest {
            input: "test".to_string(),
            model: None,
        };
        let debug = format!("{:?}", request);
        assert!(debug.contains("EmbeddingRequest"));
    }

    #[test]
    fn test_embedding_request_clone() {
        let request = EmbeddingRequest {
            input: "clone test".to_string(),
            model: Some("model-a".to_string()),
        };
        let cloned = request.clone();
        assert_eq!(cloned.input, "clone test");
        assert_eq!(cloned.model, Some("model-a".to_string()));
    }

    // =========================================================================
    // ReloadRequest/Response deserialization edge cases
    // =========================================================================

    #[test]
    fn test_reload_request_deserialization_empty_json() {
        let json = "{}";
        let request: ReloadRequest = serde_json::from_str(json).expect("deserialize");
        assert!(request.model.is_none());
        assert!(request.path.is_none());
    }

    #[test]
    fn test_reload_request_full_deserialization() {
        let json = r#"{"model": "llama3", "path": "/models/llama3.gguf"}"#;
        let request: ReloadRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(request.model, Some("llama3".to_string()));
        assert_eq!(request.path, Some("/models/llama3.gguf".to_string()));
    }

    #[test]
    fn test_reload_response_deserialization() {
        let json = r#"{"success": true, "message": "OK", "reload_time_ms": 42}"#;
        let response: ReloadResponse = serde_json::from_str(json).expect("deserialize");
        assert!(response.success);
        assert_eq!(response.message, "OK");
        assert_eq!(response.reload_time_ms, 42);
    }

    #[test]
    fn test_reload_request_debug() {
        let request = ReloadRequest {
            model: Some("test".to_string()),
            path: Some("/path".to_string()),
        };
        let debug = format!("{:?}", request);
        assert!(debug.contains("ReloadRequest"));
    }

    // =========================================================================
    // ModelMetadataResponse edge cases
    // =========================================================================

    #[test]
    fn test_model_metadata_response_deserialization() {
        let json = r#"{"id":"m1","name":"Model 1","format":"GGUF","size_bytes":100,"context_length":4096,"loaded":true}"#;
        let response: ModelMetadataResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(response.id, "m1");
        assert_eq!(response.name, "Model 1");
        assert!(response.loaded);
        assert!(response.quantization.is_none());
        assert!(response.lineage.is_none());
    }

    #[test]
    fn test_model_metadata_response_debug() {
        let response = ModelMetadataResponse {
            id: "debug".to_string(),
            name: "Debug Model".to_string(),
            format: "APR".to_string(),
            size_bytes: 0,
            quantization: None,
            context_length: 2048,
            lineage: None,
            loaded: false,
        };
        let debug = format!("{:?}", response);
        assert!(debug.contains("ModelMetadataResponse"));
    }

    #[test]
    fn test_model_metadata_response_clone() {
        let response = ModelMetadataResponse {
            id: "c".to_string(),
            name: "Clone".to_string(),
            format: "GGUF".to_string(),
            size_bytes: 42,
            quantization: Some("Q4_K_M".to_string()),
            context_length: 4096,
            lineage: None,
            loaded: true,
        };
        let cloned = response.clone();
        assert_eq!(cloned.id, "c");
        assert_eq!(cloned.quantization, Some("Q4_K_M".to_string()));
    }
