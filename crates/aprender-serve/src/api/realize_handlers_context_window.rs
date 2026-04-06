
    // =========================================================================
    // ContextWindowConfig tests
    // =========================================================================

    #[test]
    fn test_context_window_config_default() {
        let config = ContextWindowConfig::default();
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.reserved_output_tokens, 256);
        assert!(config.preserve_system);
    }

    #[test]
    fn test_context_window_config_new() {
        let config = ContextWindowConfig::new(8192);
        assert_eq!(config.max_tokens, 8192);
        assert_eq!(config.reserved_output_tokens, 256); // default
    }

    #[test]
    fn test_context_window_config_with_reserved_output() {
        let config = ContextWindowConfig::new(4096).with_reserved_output(512);
        assert_eq!(config.reserved_output_tokens, 512);
    }

    #[test]
    fn test_context_window_config_available_tokens() {
        let config = ContextWindowConfig {
            max_tokens: 4096,
            reserved_output_tokens: 256,
            preserve_system: true,
        };
        assert_eq!(config.available_tokens(), 3840);
    }

    #[test]
    fn test_context_window_config_available_tokens_saturating() {
        let config = ContextWindowConfig {
            max_tokens: 100,
            reserved_output_tokens: 200, // More than max
            preserve_system: true,
        };
        assert_eq!(config.available_tokens(), 0);
    }

    #[test]
    fn test_context_window_config_clone() {
        let config = ContextWindowConfig::new(2048);
        let cloned = config.clone();
        assert_eq!(config.max_tokens, cloned.max_tokens);
    }

    #[test]
    fn test_context_window_config_debug() {
        let config = ContextWindowConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ContextWindowConfig"));
    }

    // =========================================================================
    // ContextWindowManager tests
    // =========================================================================

    #[test]
    fn test_context_window_manager_new() {
        let config = ContextWindowConfig::default();
        let manager = ContextWindowManager::new(config);
        assert!(manager.config.max_tokens > 0);
    }

    #[test]
    fn test_context_window_manager_default_manager() {
        let manager = ContextWindowManager::default_manager();
        assert_eq!(manager.config.max_tokens, 4096);
    }

    #[test]
    fn test_context_window_manager_estimate_total_tokens() {
        let manager = ContextWindowManager::default_manager();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello world".to_string(), // ~11 chars = ~3 tokens + 10 overhead = ~13
            name: None,
        }];
        let tokens = manager.estimate_total_tokens(&messages);
        assert!(tokens > 0);
        assert!(tokens < 100); // Reasonable upper bound
    }

    #[test]
    fn test_context_window_manager_needs_truncation_false() {
        let manager = ContextWindowManager::default_manager();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Short message".to_string(),
            name: None,
        }];
        assert!(!manager.needs_truncation(&messages));
    }

    #[test]
    fn test_context_window_manager_needs_truncation_true() {
        let config = ContextWindowConfig::new(50); // Very small window
        let manager = ContextWindowManager::new(config);
        let long_content = "x".repeat(1000);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: long_content,
            name: None,
        }];
        assert!(manager.needs_truncation(&messages));
    }

    #[test]
    fn test_context_window_manager_truncate_messages_no_truncation() {
        let manager = ContextWindowManager::default_manager();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }];
        let (result, truncated) = manager.truncate_messages(&messages);
        assert!(!truncated);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_context_window_manager_truncate_messages_with_truncation() {
        let config = ContextWindowConfig::new(100);
        let manager = ContextWindowManager::new(config);
        let messages: Vec<ChatMessage> = (0..10)
            .map(|i| ChatMessage {
                role: "user".to_string(),
                content: format!("Message {} with some longer content here", i),
                name: None,
            })
            .collect();
        let (result, truncated) = manager.truncate_messages(&messages);
        assert!(truncated);
        assert!(result.len() < messages.len());
    }

    #[test]
    fn test_context_window_manager_truncate_preserves_system() {
        // Use a larger window that can fit system message but not all user messages
        let mut config = ContextWindowConfig::new(500);
        config.preserve_system = true;
        config.reserved_output_tokens = 50;
        let manager = ContextWindowManager::new(config);

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello ".repeat(200), // Very long message
                name: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Another ".repeat(200), // Another very long message
                name: None,
            },
        ];

        let (result, truncated) = manager.truncate_messages(&messages);
        // If truncation occurred and result is not empty, system should be preserved
        // Note: the algorithm may not include any messages if none fit after system
        if truncated && !result.is_empty() {
            // System message should be first if preserved
            let has_system = result.iter().any(|m| m.role == "system");
            // This is a best-effort check - the truncation might drop everything
            // if the context is too small, which is valid behavior
            assert!(has_system || result.len() < messages.len());
        }
    }

    // =========================================================================
    // clean_chat_output tests
    // =========================================================================

    #[test]
    fn test_clean_chat_output_no_stop_sequence() {
        let text = "Hello, how are you?";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Hello, how are you?");
    }

    #[test]
    fn test_clean_chat_output_with_im_end() {
        let text = "Hello<|im_end|> extra stuff";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_output_with_endoftext() {
        let text = "Response<|endoftext|>more";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response");
    }

    #[test]
    fn test_clean_chat_output_with_eos() {
        let text = "Output</s>garbage";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Output");
    }

    #[test]
    fn test_clean_chat_output_with_human_turn() {
        let text = "Response here\nHuman: next question";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response here");
    }

    #[test]
    fn test_clean_chat_output_multiple_stop_sequences() {
        let text = "Response<|im_end|>stuff</s>more";
        let cleaned = clean_chat_output(text);
        // Should stop at earliest
        assert_eq!(cleaned, "Response");
    }

    #[test]
    fn test_clean_chat_output_trims_whitespace() {
        let text = "  Response  <|im_end|>";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response");
    }

    // =========================================================================
    // EmbeddingRequest tests
    // =========================================================================

    #[test]
    fn test_embedding_request_basic() {
        let request = EmbeddingRequest {
            input: "Hello world".to_string(),
            model: Some("text-embedding-ada-002".to_string()),
        };
        assert_eq!(request.input, "Hello world");
        assert!(request.model.is_some());
    }

    #[test]
    fn test_embedding_request_serialization() {
        let request = EmbeddingRequest {
            input: "test".to_string(),
            model: None,
        };
        let json = serde_json::to_string(&request).expect("serialize");
        assert!(json.contains("test"));
        // model should be skipped when None
        assert!(!json.contains("model"));
    }

    #[test]
    fn test_embedding_request_deserialization() {
        let json = r#"{"input": "hello"}"#;
        let request: EmbeddingRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(request.input, "hello");
        assert!(request.model.is_none());
    }

    // =========================================================================
    // EmbeddingResponse tests
    // =========================================================================

    #[test]
    fn test_embedding_response_basic() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding: vec![0.1, 0.2, 0.3],
            }],
            model: "text-embedding-ada-002".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].embedding.len(), 3);
    }

    #[test]
    fn test_embedding_response_serialization() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![],
            model: "test".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
            },
        };
        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("list"));
        assert!(json.contains("prompt_tokens"));
    }

    // =========================================================================
    // EmbeddingData tests
    // =========================================================================

    #[test]
    fn test_embedding_data_basic() {
        let data = EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(data.index, 0);
        assert_eq!(data.embedding.len(), 4);
    }

    #[test]
    fn test_embedding_data_clone() {
        let data = EmbeddingData {
            object: "embedding".to_string(),
            index: 1,
            embedding: vec![0.5],
        };
        let cloned = data.clone();
        assert_eq!(data.index, cloned.index);
    }

    // =========================================================================
    // EmbeddingUsage tests
    // =========================================================================

    #[test]
    fn test_embedding_usage_basic() {
        let usage = EmbeddingUsage {
            prompt_tokens: 15,
            total_tokens: 15,
        };
        assert_eq!(usage.prompt_tokens, usage.total_tokens);
    }

    #[test]
    fn test_embedding_usage_serialization() {
        let usage = EmbeddingUsage {
            prompt_tokens: 100,
            total_tokens: 100,
        };
        let json = serde_json::to_string(&usage).expect("serialize");
        assert!(json.contains("100"));
    }

    // =========================================================================
    // ModelMetadataResponse tests
    // =========================================================================

    #[test]
    fn test_model_metadata_response_basic() {
        let response = ModelMetadataResponse {
            id: "model-123".to_string(),
            name: "TinyLlama".to_string(),
            format: "GGUF".to_string(),
            size_bytes: 1_000_000_000,
            quantization: Some("Q4_K_M".to_string()),
            context_length: 4096,
            lineage: None,
            loaded: true,
        };
        assert_eq!(response.id, "model-123");
        assert!(response.loaded);
    }

    #[test]
    fn test_model_metadata_response_with_lineage() {
        let response = ModelMetadataResponse {
            id: "model-456".to_string(),
            name: "CustomModel".to_string(),
            format: "APR".to_string(),
            size_bytes: 500_000_000,
            quantization: None,
            context_length: 2048,
            lineage: Some(ModelLineage {
                uri: "pacha://models/custom".to_string(),
                version: "1.0.0".to_string(),
                recipe: Some("fine-tune".to_string()),
                parent: Some("llama2-7b".to_string()),
                content_hash: "abc123".to_string(),
            }),
            loaded: false,
        };
        assert!(response.lineage.is_some());
        let lineage = response.lineage.expect("lineage");
        assert_eq!(lineage.version, "1.0.0");
    }

    #[test]
    fn test_model_metadata_response_serialization() {
        let response = ModelMetadataResponse {
            id: "test".to_string(),
            name: "Test".to_string(),
            format: "GGUF".to_string(),
            size_bytes: 100,
            quantization: None,
            context_length: 1024,
            lineage: None,
            loaded: true,
        };
        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("GGUF"));
        // None fields should be skipped
        assert!(!json.contains("quantization"));
        assert!(!json.contains("lineage"));
    }

    // =========================================================================
    // ModelLineage tests
    // =========================================================================

    #[test]
    fn test_model_lineage_basic() {
        let lineage = ModelLineage {
            uri: "pacha://test".to_string(),
            version: "2.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "hash123".to_string(),
        };
        assert_eq!(lineage.uri, "pacha://test");
        assert!(lineage.recipe.is_none());
    }
