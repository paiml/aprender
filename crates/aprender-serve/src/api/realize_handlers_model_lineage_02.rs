
    #[test]
    fn test_model_lineage_full() {
        let lineage = ModelLineage {
            uri: "pacha://models/llama".to_string(),
            version: "3.0.0".to_string(),
            recipe: Some("rlhf".to_string()),
            parent: Some("base-llama".to_string()),
            content_hash: "blake3hash".to_string(),
        };
        assert_eq!(lineage.recipe, Some("rlhf".to_string()));
        assert_eq!(lineage.parent, Some("base-llama".to_string()));
    }

    #[test]
    fn test_model_lineage_clone() {
        let lineage = ModelLineage {
            uri: "uri".to_string(),
            version: "1.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "hash".to_string(),
        };
        let cloned = lineage.clone();
        assert_eq!(lineage.uri, cloned.uri);
    }

    // =========================================================================
    // ReloadRequest tests
    // =========================================================================

    #[test]
    fn test_reload_request_empty() {
        let request = ReloadRequest {
            model: None,
            path: None,
        };
        assert!(request.model.is_none());
        assert!(request.path.is_none());
    }

    #[test]
    fn test_reload_request_with_model() {
        let request = ReloadRequest {
            model: Some("llama2".to_string()),
            path: None,
        };
        assert_eq!(request.model, Some("llama2".to_string()));
    }

    #[test]
    fn test_reload_request_with_path() {
        let request = ReloadRequest {
            model: None,
            path: Some("/path/to/model.gguf".to_string()),
        };
        assert!(request.path.is_some());
    }

    #[test]
    fn test_reload_request_serialization() {
        let request = ReloadRequest {
            model: Some("test".to_string()),
            path: Some("/path".to_string()),
        };
        let json = serde_json::to_string(&request).expect("serialize");
        assert!(json.contains("test"));
        assert!(json.contains("/path"));
    }

    // =========================================================================
    // ReloadResponse tests
    // =========================================================================

    #[test]
    fn test_reload_response_success() {
        let response = ReloadResponse {
            success: true,
            message: "Model reloaded".to_string(),
            reload_time_ms: 1500,
        };
        assert!(response.success);
        assert_eq!(response.reload_time_ms, 1500);
    }

    #[test]
    fn test_reload_response_failure() {
        let response = ReloadResponse {
            success: false,
            message: "Model not found".to_string(),
            reload_time_ms: 0,
        };
        assert!(!response.success);
    }

    #[test]
    fn test_reload_response_serialization() {
        let response = ReloadResponse {
            success: true,
            message: "OK".to_string(),
            reload_time_ms: 100,
        };
        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("success"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_reload_response_clone() {
        let response = ReloadResponse {
            success: true,
            message: "Done".to_string(),
            reload_time_ms: 50,
        };
        let cloned = response.clone();
        assert_eq!(response.message, cloned.message);
    }

    // =========================================================================
    // format_chat_messages tests
    // =========================================================================

    #[test]
    fn test_format_chat_messages_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let result = format_chat_messages(&messages, None);
        // Should handle empty gracefully
        assert!(result.is_empty() || !result.is_empty()); // Just shouldn't panic
    }

    #[test]
    fn test_format_chat_messages_single() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, None);
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_format_chat_messages_with_model() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Test".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, Some("llama2"));
        // Should format without panic
        assert!(!result.is_empty());
    }

    // =========================================================================
    // format_chat_messages: multi-role conversations
    // =========================================================================

    #[test]
    fn test_format_chat_messages_system_user_assistant() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
                name: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
                name: None,
            },
        ];
        let result = format_chat_messages(&messages, None);
        assert!(result.contains("helpful assistant"));
        assert!(result.contains("2+2"));
        assert!(result.contains("4"));
    }

    #[test]
    fn test_format_chat_messages_multi_turn() {
        let messages = vec![
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
        let result = format_chat_messages(&messages, None);
        assert!(result.contains("Hello"));
        assert!(result.contains("Hi there!"));
        assert!(result.contains("How are you?"));
    }

    #[test]
    fn test_format_chat_messages_with_qwen_model() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Test".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, Some("qwen2"));
        assert!(!result.is_empty());
    }

    #[test]
    fn test_format_chat_messages_with_unknown_model() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Test prompt".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, Some("unknown_model_xyz"));
        assert!(result.contains("Test prompt"));
    }

    #[test]
    fn test_format_chat_messages_only_system() {
        let messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "System prompt only".to_string(),
            name: None,
        }];
        let result = format_chat_messages(&messages, None);
        assert!(result.contains("System prompt only"));
    }

    // =========================================================================
    // clean_chat_output: remaining stop sequences
    // =========================================================================

    #[test]
    fn test_clean_chat_output_with_end_tag() {
        let text = "Some output<|end|>extra stuff";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Some output");
    }

    #[test]
    fn test_clean_chat_output_with_user_turn() {
        let text = "Response here\nUser: next question";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response here");
    }

    #[test]
    fn test_clean_chat_output_with_double_newline_human() {
        let text = "Response here\n\nHuman: next question";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response here");
    }

    #[test]
    fn test_clean_chat_output_with_double_newline_user() {
        let text = "Response here\n\nUser: next question";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response here");
    }

    #[test]
    fn test_clean_chat_output_with_im_start() {
        let text = "Response here<|im_start|>user\nAnother message";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Response here");
    }

    #[test]
    fn test_clean_chat_output_empty_before_stop() {
        let text = "<|im_end|>stuff";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_output_only_whitespace_before_stop() {
        let text = "   \n  </s>garbage";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_output_no_content() {
        let text = "";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_output_only_whitespace() {
        let text = "   \n\t  ";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_output_preserves_internal_newlines() {
        let text = "Line 1\nLine 2\nLine 3";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "Line 1\nLine 2\nLine 3");
    }

    #[test]
    fn test_clean_chat_output_earliest_of_multiple() {
        // <|end|> is at index 2, </s> is at index 10
        let text = "OK<|end|>middle</s>end";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "OK");
    }

    #[test]
    fn test_clean_chat_output_endoftext_earliest() {
        let text = "A<|endoftext|>B<|im_end|>C";
        let cleaned = clean_chat_output(text);
        assert_eq!(cleaned, "A");
    }

    // =========================================================================
    // ContextWindowManager: truncation edge cases
    // =========================================================================

    #[test]
    fn test_context_window_manager_truncate_preserves_recent_messages() {
        let config = ContextWindowConfig::new(200).with_reserved_output(50);
        let manager = ContextWindowManager::new(config);

        let messages: Vec<ChatMessage> = (0..20)
            .map(|i| ChatMessage {
                role: "user".to_string(),
                content: format!("Message number {}", i),
                name: None,
            })
            .collect();

        let (result, truncated) = manager.truncate_messages(&messages);
        assert!(truncated);
        // Most recent messages should be preserved
        if !result.is_empty() {
            let last_result = &result[result.len() - 1];
            let last_original = &messages[messages.len() - 1];
            assert_eq!(last_result.content, last_original.content);
        }
    }

    #[test]
    fn test_context_window_manager_truncate_no_messages() {
        let manager = ContextWindowManager::default_manager();
        let messages: Vec<ChatMessage> = vec![];
        let (result, truncated) = manager.truncate_messages(&messages);
        assert!(!truncated);
        assert!(result.is_empty());
    }

    #[test]
    fn test_context_window_manager_truncate_single_huge_message() {
        let config = ContextWindowConfig::new(50).with_reserved_output(10);
        let manager = ContextWindowManager::new(config);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "x".repeat(10000),
            name: None,
        }];
        let (result, truncated) = manager.truncate_messages(&messages);
        assert!(truncated);
        // The single message doesn't fit, so result should be empty
        assert!(result.is_empty());
    }

    #[test]
    fn test_context_window_manager_needs_truncation_empty() {
        let manager = ContextWindowManager::default_manager();
        assert!(!manager.needs_truncation(&[]));
    }

    #[test]
    fn test_context_window_manager_estimate_total_tokens_empty() {
        let manager = ContextWindowManager::default_manager();
        assert_eq!(manager.estimate_total_tokens(&[]), 0);
    }

    #[test]
    fn test_context_window_manager_estimate_total_tokens_multiple() {
        let manager = ContextWindowManager::default_manager();
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: None,
            },
        ];
        let total = manager.estimate_total_tokens(&messages);
        // Each message: len/4 + 10 overhead
        // "You are helpful." = 16 chars -> 4 tokens + 10 = 14
        // "Hi" = 2 chars -> 1 token + 10 = 11
        // Total ~= 25
        assert!(total > 20);
        assert!(total < 50);
    }

    #[test]
    fn test_context_window_config_zero_max_tokens() {
        let config = ContextWindowConfig::new(0);
        assert_eq!(config.available_tokens(), 0);
    }

    #[test]
    fn test_context_window_config_chained_builder() {
        let config = ContextWindowConfig::new(8192).with_reserved_output(1024);
        assert_eq!(config.max_tokens, 8192);
        assert_eq!(config.reserved_output_tokens, 1024);
        assert_eq!(config.available_tokens(), 7168);
    }

    // =========================================================================
    // CompletionRequest serialization/deserialization
    // =========================================================================

    #[test]
    fn test_completion_request_minimal() {
        let json = r#"{"model": "gpt-4", "prompt": "Hello"}"#;
        let request: CompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.prompt, "Hello");
        assert!(request.max_tokens.is_none());
        assert!(request.temperature.is_none());
        assert!(request.top_p.is_none());
        assert!(request.stop.is_none());
    }
