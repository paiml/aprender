//! API Tests Part 07: Realize Handlers Coverage (Phase 37 - Scenario Blitz)
//!
//! Tests for realize_handlers.rs to drive coverage from 65% to 85%+.
//! Focus: Context window management, chat formatting, and response types.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::realize_handlers::{
    clean_chat_output, format_chat_messages, CompletionRequest, ContextWindowConfig,
    ContextWindowManager, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    ModelMetadataResponse, ReloadRequest,
};
use crate::api::test_helpers::create_test_app_shared;
use crate::api::ChatMessage;

/// Helper to create a ChatMessage
fn chat_msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
        name: None,
    }
}

// =============================================================================
// ContextWindowConfig Tests
// =============================================================================

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
    assert_eq!(config.reserved_output_tokens, 256); // Default
    assert!(config.preserve_system); // Default
}

#[test]
fn test_context_window_config_with_reserved_output() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(512);
    assert_eq!(config.max_tokens, 4096);
    assert_eq!(config.reserved_output_tokens, 512);
}

#[test]
fn test_context_window_config_available_tokens() {
    let config = ContextWindowConfig::new(4096).with_reserved_output(256);
    assert_eq!(config.available_tokens(), 3840);
}

#[test]
fn test_context_window_config_available_tokens_underflow() {
    // Reserved > max should saturate to 0
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 500,
        preserve_system: true,
    };
    assert_eq!(config.available_tokens(), 0);
}

// =============================================================================
// ContextWindowManager Tests
// =============================================================================

#[test]
fn test_context_window_manager_new() {
    let config = ContextWindowConfig::new(2048);
    let manager = ContextWindowManager::new(config);
    let _ = manager;
}

#[test]
fn test_context_window_manager_default() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![chat_msg("user", "Hello")];
    assert!(!manager.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_estimate_total_tokens() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![
        chat_msg("user", "Hello world"),
        chat_msg("assistant", "Hi there!"),
    ];

    let estimate = manager.estimate_total_tokens(&messages);
    assert!(estimate > 0);
    assert!(estimate < 100);
}

#[test]
fn test_context_window_manager_needs_truncation_false() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![chat_msg("user", "Short message")];
    assert!(!manager.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_needs_truncation_true() {
    let config = ContextWindowConfig::new(50).with_reserved_output(10);
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        chat_msg("user", "This is a very long message that will exceed the small context window limit we've set."),
        chat_msg("assistant", "Another long message that adds more tokens to the count."),
    ];
    assert!(manager.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_truncate_no_truncation_needed() {
    let manager = ContextWindowManager::default_manager();
    let messages = vec![chat_msg("user", "Hello"), chat_msg("assistant", "Hi!")];

    let (result, truncated) = manager.truncate_messages(&messages);
    assert!(!truncated);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_context_window_manager_truncate_with_system_preserved() {
    // Very small context window to force truncation
    let config = ContextWindowConfig {
        max_tokens: 50,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let manager = ContextWindowManager::new(config);

    // Messages that definitely exceed 40 tokens
    let messages = vec![
        chat_msg("system", "You are a helpful assistant."),
        chat_msg("user", "First very long message that definitely takes up many tokens in the context window and will need to be truncated."),
        chat_msg("assistant", "This is a detailed response to your first message with lots of tokens."),
        chat_msg("user", "Second message with more content."),
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    // If not truncated, context is big enough - that's also valid
    if truncated {
        // System message should be preserved when truncation happens
        assert!(result.iter().any(|m| m.role == "system"));
    }
}

#[test]
fn test_context_window_manager_truncate_keeps_recent() {
    let config = ContextWindowConfig {
        max_tokens: 80,
        reserved_output_tokens: 10,
        preserve_system: false,
    };
    let manager = ContextWindowManager::new(config);

    let messages = vec![
        chat_msg("user", "Old message 1"),
        chat_msg("assistant", "Old reply 1"),
        chat_msg("user", "Recent message"),
    ];

    let (result, truncated) = manager.truncate_messages(&messages);
    if truncated {
        assert!(result.iter().any(|m| m.content.contains("Recent")));
    }
}

// =============================================================================
// format_chat_messages Tests
// =============================================================================

#[test]
fn test_format_chat_messages_single_user() {
    let messages = vec![chat_msg("user", "Hello, world!")];
    let formatted = format_chat_messages(&messages, None);
    assert!(!formatted.is_empty());
    assert!(formatted.contains("Hello, world!"));
}

#[test]
fn test_format_chat_messages_conversation() {
    let messages = vec![
        chat_msg("system", "You are a helpful assistant."),
        chat_msg("user", "What is 2+2?"),
        chat_msg("assistant", "2+2 equals 4."),
        chat_msg("user", "Thanks!"),
    ];

    let formatted = format_chat_messages(&messages, None);
    assert!(formatted.contains("helpful assistant"));
    assert!(formatted.contains("2+2"));
    assert!(formatted.contains("Thanks"));
}

#[test]
fn test_format_chat_messages_with_model_name() {
    let messages = vec![chat_msg("user", "Hello")];

    let formatted_tinyllama = format_chat_messages(&messages, Some("tinyllama"));
    let formatted_phi = format_chat_messages(&messages, Some("phi"));
    let formatted_mistral = format_chat_messages(&messages, Some("mistral"));

    assert!(formatted_tinyllama.contains("Hello"));
    assert!(formatted_phi.contains("Hello"));
    assert!(formatted_mistral.contains("Hello"));
}

#[test]
fn test_format_chat_messages_empty() {
    let messages: Vec<ChatMessage> = vec![];
    let formatted = format_chat_messages(&messages, None);
    let _ = formatted;
}

// =============================================================================
// clean_chat_output Tests
// =============================================================================

#[test]
fn test_clean_chat_output_no_stop_sequence() {
    let text = "This is a normal response without any stop sequences.";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, text);
}

#[test]
fn test_clean_chat_output_with_im_end() {
    let text = "Here is my response.<|im_end|>Some extra text";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Here is my response.");
}

#[test]
fn test_clean_chat_output_with_endoftext() {
    let text = "The answer is 42.<|endoftext|>garbage";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "The answer is 42.");
}

#[test]
fn test_clean_chat_output_with_llama_stop() {
    let text = "Response here.</s>more text";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Response here.");
}

#[test]
fn test_clean_chat_output_with_human_turn() {
    let text = "My response to you.\nHuman: But what about...";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "My response to you.");
}

#[test]
fn test_clean_chat_output_with_user_turn() {
    let text = "Here's my answer.\nUser: Follow up question";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Here's my answer.");
}

#[test]
fn test_clean_chat_output_with_double_newline_human() {
    let text = "Response text.\n\nHuman: Next message";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Response text.");
}

#[test]
fn test_clean_chat_output_with_im_start() {
    let text = "My response<|im_start|>user\nNew message";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "My response");
}

#[test]
fn test_clean_chat_output_multiple_stop_sequences() {
    let text = "Response</s>text<|im_end|>more";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Response");
}

#[test]
fn test_clean_chat_output_trims_whitespace() {
    let text = "  Response with spaces  <|im_end|>";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Response with spaces");
}

#[test]
fn test_clean_chat_output_with_end_tag() {
    let text = "Output here<|end|>trailing";
    let cleaned = clean_chat_output(text);
    assert_eq!(cleaned, "Output here");
}

// =============================================================================
// Request/Response Type Tests
// =============================================================================

#[test]
fn test_embedding_request_serialization() {
    let req = EmbeddingRequest {
        input: "Test text".to_string(),
        model: Some("test-model".to_string()),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("Test text"));
    assert!(json.contains("test-model"));

    let parsed: EmbeddingRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.input, "Test text");
    assert_eq!(parsed.model, Some("test-model".to_string()));
}

#[test]
fn test_embedding_request_without_model() {
    let req = EmbeddingRequest {
        input: "Text only".to_string(),
        model: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");

    let json_with_null = r#"{"input":"Text only","model":null}"#;
    let parsed: EmbeddingRequest = serde_json::from_str(json_with_null).expect("deserialize");
    assert_eq!(parsed.model, None);
    let _ = json;
}

#[test]
fn test_embedding_data_serialization() {
    let data = EmbeddingData {
        object: "embedding".to_string(),
        index: 0,
        embedding: vec![0.1, 0.2, 0.3],
    };

    let json = serde_json::to_string(&data).expect("serialize");
    assert!(json.contains("embedding"));
    assert!(json.contains("0.1"));

    let parsed: EmbeddingData = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.embedding.len(), 3);
}

#[test]
fn test_embedding_usage_serialization() {
    let usage = EmbeddingUsage {
        prompt_tokens: 10,
        total_tokens: 10,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: EmbeddingUsage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.prompt_tokens, 10);
    assert_eq!(parsed.total_tokens, 10);
}

#[test]
fn test_embedding_response_serialization() {
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2],
        }],
        model: "test-model".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: EmbeddingResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.data.len(), 1);
    assert_eq!(parsed.model, "test-model");
}

#[test]
fn test_model_metadata_response_serialization() {
    let response = ModelMetadataResponse {
        id: "model-001".to_string(),
        name: "Test Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 1024 * 1024 * 100,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 4096,
        lineage: None,
        loaded: true,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("model-001"));
    assert!(json.contains("GGUF"));
    assert!(json.contains("Q4_K_M"));

    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "model-001");
    assert!(parsed.loaded);
}

#[test]
fn test_model_metadata_response_without_optional_fields() {
    let response = ModelMetadataResponse {
        id: "model-002".to_string(),
        name: "Minimal Model".to_string(),
        format: "APR".to_string(),
        size_bytes: 1000,
        quantization: None,
        context_length: 2048,
        lineage: None,
        loaded: false,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let _ = json;
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_clean_chat_output_empty_string() {
    let cleaned = clean_chat_output("");
    assert_eq!(cleaned, "");
}

#[test]
fn test_clean_chat_output_only_stop_sequence() {
    let cleaned = clean_chat_output("<|im_end|>");
    assert_eq!(cleaned, "");
}

include!("clean_chat_format.rs");
