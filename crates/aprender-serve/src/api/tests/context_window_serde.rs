//! T-COV-95 Deep Coverage Bridge: realize_handlers, openai_handlers, gpu_handlers
//!
//! Targets: ContextWindowConfig/Manager, format_chat_messages, clean_chat_output,
//! realize_embed_handler, realize_model_handler, realize_completions_handler,
//! openai_models_handler, openai_chat_completions_handler, HTTP error paths,
//! struct serde round-trips, GPU/batch handler routing.

use crate::api::realize_handlers::*;
use crate::api::test_helpers::create_test_app_shared;
use crate::api::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

// ============================================================================
// ContextWindowConfig
// ============================================================================

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
    assert!(config.preserve_system); // default
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
fn test_context_window_config_available_tokens_saturating() {
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 200, // more than max
        preserve_system: true,
    };
    assert_eq!(config.available_tokens(), 0);
}

// ============================================================================
// ContextWindowManager
// ============================================================================

#[test]
fn test_context_window_manager_default() {
    let mgr = ContextWindowManager::default_manager();
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];
    assert!(!mgr.needs_truncation(&msgs));
}

#[test]
fn test_context_window_manager_no_truncation() {
    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Short message".to_string(),
        name: None,
    }];
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(!truncated);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_context_window_manager_truncation_needed() {
    let config = ContextWindowConfig {
        max_tokens: 50,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);
    // Create messages that exceed 40 available tokens
    let msgs: Vec<ChatMessage> = (0..10)
        .map(|i| ChatMessage {
            role: "user".to_string(),
            content: format!("Message number {} with enough content to use tokens", i),
            name: None,
        })
        .collect();
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(truncated);
    assert!(result.len() < msgs.len());
}

#[test]
fn test_context_window_manager_preserves_system() {
    let config = ContextWindowConfig {
        max_tokens: 80,
        reserved_output_tokens: 10,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);
    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are helpful".to_string(),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "A".repeat(200),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "B".repeat(200),
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Latest question".to_string(),
            name: None,
        },
    ];
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(truncated);
    // System message should be preserved
    assert!(result.iter().any(|m| m.role == "system"));
}

#[test]
fn test_context_window_manager_needs_truncation() {
    let config = ContextWindowConfig {
        max_tokens: 4096,
        reserved_output_tokens: 256,
        preserve_system: true,
    };
    let mgr = ContextWindowManager::new(config);
    let short = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hi".to_string(),
        name: None,
    }];
    assert!(!mgr.needs_truncation(&short));

    // Create enough messages to exceed 3840 available tokens
    let long: Vec<ChatMessage> = (0..500)
        .map(|i| ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Long message number {} with content that takes many tokens to fill the window",
                i
            ),
            name: None,
        })
        .collect();
    assert!(mgr.needs_truncation(&long));
}

#[test]
fn test_context_window_manager_estimate_total_tokens() {
    let config = ContextWindowConfig::default();
    let mgr = ContextWindowManager::new(config);
    let msgs = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello world".to_string(),
            name: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "Hi there".to_string(),
            name: None,
        },
    ];
    let tokens = mgr.estimate_total_tokens(&msgs);
    assert!(tokens > 0);
    // Each message ~= len/4 + 10 overhead
    assert!(tokens >= 20);
}

#[test]
fn test_context_window_manager_empty_messages() {
    let config = ContextWindowConfig::default();
    let mgr = ContextWindowManager::new(config);
    let msgs: Vec<ChatMessage> = vec![];
    let (result, truncated) = mgr.truncate_messages(&msgs);
    assert!(!truncated);
    assert!(result.is_empty());
}

// ============================================================================
// format_chat_messages
// ============================================================================

#[test]
fn test_format_chat_messages_basic() {
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    }];
    let result = format_chat_messages(&msgs, None);
    assert!(!result.is_empty());
    assert!(result.contains("Hello"));
}

#[test]
fn test_format_chat_messages_multi_turn() {
    let msgs = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant".to_string(),
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
        ChatMessage {
            role: "user".to_string(),
            content: "Thanks!".to_string(),
            name: None,
        },
    ];
    let result = format_chat_messages(&msgs, Some("qwen"));
    assert!(!result.is_empty());
    assert!(result.contains("What is 2+2?"));
    assert!(result.contains("Thanks!"));
}

#[test]
fn test_format_chat_messages_empty() {
    let msgs: Vec<ChatMessage> = vec![];
    let result = format_chat_messages(&msgs, None);
    // Empty messages should still return something (template header/footer)
    let _ = result; // just ensure no panic
}

#[test]
fn test_format_chat_messages_model_names() {
    let msgs = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hi".to_string(),
        name: None,
    }];
    // Different model name templates
    for model in &["qwen", "llama", "mistral", "phi", "unknown-model"] {
        let result = format_chat_messages(&msgs, Some(model));
        assert!(result.contains("Hi"), "model={model}");
    }
}

// ============================================================================
// clean_chat_output
// ============================================================================

#[test]
fn test_clean_chat_output_no_markers() {
    let result = clean_chat_output("Hello world");
    assert_eq!(result, "Hello world");
}

#[test]
fn test_clean_chat_output_chatml_end() {
    let result = clean_chat_output("Hello<|im_end|>extra stuff");
    assert_eq!(result, "Hello");
}

#[test]
fn test_clean_chat_output_endoftext() {
    let result = clean_chat_output("Response here<|endoftext|>garbage");
    assert_eq!(result, "Response here");
}

#[test]
fn test_clean_chat_output_llama_eos() {
    let result = clean_chat_output("Output text</s>more");
    assert_eq!(result, "Output text");
}

#[test]
fn test_clean_chat_output_human_turn() {
    let result = clean_chat_output("My response\nHuman: next question");
    assert_eq!(result, "My response");
}

#[test]
fn test_clean_chat_output_user_turn() {
    let result = clean_chat_output("Answer text\nUser: follow up");
    assert_eq!(result, "Answer text");
}

#[test]
fn test_clean_chat_output_chatml_start() {
    let result = clean_chat_output("Response<|im_start|>user\nmore");
    assert_eq!(result, "Response");
}

#[test]
fn test_clean_chat_output_empty() {
    let result = clean_chat_output("");
    assert_eq!(result, "");
}

#[test]
fn test_clean_chat_output_only_marker() {
    let result = clean_chat_output("<|im_end|>");
    assert_eq!(result, "");
}

#[test]
fn test_clean_chat_output_multiple_markers() {
    let result = clean_chat_output("Text<|im_end|>more</s>even more");
    // Should stop at earliest marker
    assert_eq!(result, "Text");
}

#[test]
fn test_clean_chat_output_end_marker() {
    let result = clean_chat_output("Response<|end|>trailing");
    assert_eq!(result, "Response");
}

#[test]
fn test_clean_chat_output_double_newline_human() {
    let result = clean_chat_output("Output\n\nHuman: question");
    assert_eq!(result, "Output");
}

#[test]
fn test_clean_chat_output_double_newline_user() {
    let result = clean_chat_output("Output\n\nUser: question");
    assert_eq!(result, "Output");
}

// ============================================================================
// Struct serde round-trips
// ============================================================================

#[test]
fn test_embedding_request_serde() {
    let req = EmbeddingRequest {
        input: "test".to_string(),
        model: Some("default".to_string()),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: EmbeddingRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.input, "test");
    assert_eq!(parsed.model.as_deref(), Some("default"));
}

#[test]
fn test_embedding_request_no_model() {
    let json = r#"{"input":"hello"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(req.input, "hello");
    assert!(req.model.is_none());
}

#[test]
fn test_embedding_response_serde() {
    let resp = EmbeddingResponse {
        object: "list".to_string(),
        data: vec![EmbeddingData {
            object: "embedding".to_string(),
            index: 0,
            embedding: vec![0.1, 0.2, 0.3],
        }],
        model: "test".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: EmbeddingResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.data.len(), 1);
    assert_eq!(parsed.data[0].embedding.len(), 3);
}

#[test]
fn test_model_metadata_response_serde() {
    let resp = ModelMetadataResponse {
        id: "m1".to_string(),
        name: "Test Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 1024,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 2048,
        lineage: None,
        loaded: true,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.id, "m1");
    assert!(parsed.loaded);
}

#[test]
fn test_model_lineage_serde() {
    let lineage = ModelLineage {
        uri: "pacha://model:v1".to_string(),
        version: "1.0".to_string(),
        recipe: Some("finetune".to_string()),
        parent: Some("base-model".to_string()),
        content_hash: "abc123".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("JSON serialization failed");
    let parsed: ModelLineage = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.uri, "pacha://model:v1");
    assert_eq!(parsed.recipe.as_deref(), Some("finetune"));
}

#[test]
fn test_reload_request_serde() {
    let req = ReloadRequest {
        model: Some("test".to_string()),
        path: Some("/path/to/model".to_string()),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: ReloadRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.model.as_deref(), Some("test"));
}

include!("reload_request_response.rs");
include!("batch_generate_03.rs");
