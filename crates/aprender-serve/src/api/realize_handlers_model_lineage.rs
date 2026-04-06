
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ModelLineage serialization edge cases
    // =========================================================================

    #[test]
    fn test_model_lineage_serialization_skip_none() {
        let lineage = ModelLineage {
            uri: "pacha://test".to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "hash".to_string(),
        };
        let json = serde_json::to_string(&lineage).expect("serialize");
        assert!(!json.contains("recipe"));
        assert!(!json.contains("parent"));
    }

    #[test]
    fn test_model_lineage_serialization_with_all_fields() {
        let lineage = ModelLineage {
            uri: "pacha://test".to_string(),
            version: "2.0".to_string(),
            recipe: Some("sft".to_string()),
            parent: Some("base".to_string()),
            content_hash: "abc123".to_string(),
        };
        let json = serde_json::to_string(&lineage).expect("serialize");
        assert!(json.contains("sft"));
        assert!(json.contains("base"));
    }

    #[test]
    fn test_model_lineage_debug() {
        let lineage = ModelLineage {
            uri: "test".to_string(),
            version: "1.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "h".to_string(),
        };
        let debug = format!("{:?}", lineage);
        assert!(debug.contains("ModelLineage"));
    }

    // =========================================================================
    // EmbeddingUsage / EmbeddingData edge cases
    // =========================================================================

    #[test]
    fn test_embedding_usage_deserialization() {
        let json = r#"{"prompt_tokens": 42, "total_tokens": 42}"#;
        let usage: EmbeddingUsage = serde_json::from_str(json).expect("deserialize");
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.total_tokens, 42);
    }

    #[test]
    fn test_embedding_usage_debug() {
        let usage = EmbeddingUsage {
            prompt_tokens: 10,
            total_tokens: 10,
        };
        let debug = format!("{:?}", usage);
        assert!(debug.contains("EmbeddingUsage"));
    }

    #[test]
    fn test_embedding_usage_clone() {
        let usage = EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        };
        let cloned = usage.clone();
        assert_eq!(cloned.prompt_tokens, 5);
    }

    #[test]
    fn test_embedding_data_debug() {
        let data = EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2],
        };
        let debug = format!("{:?}", data);
        assert!(debug.contains("EmbeddingData"));
    }

    #[test]
    fn test_embedding_data_serialization() {
        let data = EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![1.0, 2.0, 3.0],
        };
        let json = serde_json::to_string(&data).expect("serialize");
        assert!(json.contains("embedding"));
        assert!(json.contains("1.0"));
    }

    #[test]
    fn test_embedding_response_debug() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![],
            model: "test".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        let debug = format!("{:?}", response);
        assert!(debug.contains("EmbeddingResponse"));
    }

    #[test]
    fn test_embedding_response_clone() {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                index: 0,
                embedding: vec![0.5],
            }],
            model: "m".to_string(),
            usage: EmbeddingUsage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
        };
        let cloned = response.clone();
        assert_eq!(cloned.data.len(), 1);
    }

    #[test]
    fn test_embedding_response_deserialization() {
        let json = r#"{"object":"list","data":[],"model":"test","usage":{"prompt_tokens":0,"total_tokens":0}}"#;
        let response: EmbeddingResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(response.object, "list");
        assert!(response.data.is_empty());
    }

    // =========================================================================
    // ContextWindowManager: estimate_tokens static method
    // =========================================================================

    #[test]
    fn test_estimate_tokens_short() {
        // "Hi" = 2 chars => ceil(2/4) + 10 = 1 + 10 = 11
        let manager = ContextWindowManager::default_manager();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            name: None,
        }];
        let tokens = manager.estimate_total_tokens(&messages);
        assert_eq!(tokens, 11);
    }

    #[test]
    fn test_estimate_tokens_empty_content() {
        // "" = 0 chars => ceil(0/4) + 10 = 0 + 10 = 10
        let manager = ContextWindowManager::default_manager();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: String::new(),
            name: None,
        }];
        let tokens = manager.estimate_total_tokens(&messages);
        assert_eq!(tokens, 10);
    }

    #[test]
    fn test_estimate_tokens_exact_multiple_of_four() {
        // "abcd" = 4 chars => ceil(4/4) + 10 = 1 + 10 = 11
        let manager = ContextWindowManager::default_manager();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "abcd".to_string(),
            name: None,
        }];
        let tokens = manager.estimate_total_tokens(&messages);
        assert_eq!(tokens, 11);
    }

    #[test]
    fn test_context_window_truncate_system_not_preserved() {
        let mut config = ContextWindowConfig::new(100);
        config.preserve_system = false;
        config.reserved_output_tokens = 10;
        let manager = ContextWindowManager::new(config);

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "x".repeat(500),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "short".to_string(),
                name: None,
            },
        ];

        let (result, truncated) = manager.truncate_messages(&messages);
        assert!(truncated);
        // With preserve_system=false, system is just another message
        // The user message should be included (it's the most recent)
        if !result.is_empty() {
            // The most recent non-system message is "short"
            let has_user = result.iter().any(|m| m.content == "short");
            assert!(has_user);
        }
    }
include!("realize_handlers_context_window.rs");
include!("realize_handlers_model_lineage_02.rs");
include!("realize_handlers_completion_request.rs");
include!("realize_handlers_clean_chat.rs");
}
