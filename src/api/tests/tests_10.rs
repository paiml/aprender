//! API Tests Part 10: Realize Handlers Extended Coverage
//!
//! Additional tests for realize_handlers.rs focusing on:
//! - ModelLineage serialization
//! - ReloadResponse serialization
//! - CompletionChoice tests
//! - Handler error paths
//! - Edge cases for embeddings and completions

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::realize_handlers::{
    CompletionChoice, CompletionRequest, CompletionResponse, ModelLineage, ModelMetadataResponse,
    ReloadRequest, ReloadResponse,
};
use crate::api::test_helpers::create_test_app_shared;
use crate::api::Usage;

// =============================================================================
// ModelLineage Tests
// =============================================================================

#[test]
fn test_model_lineage_serialization_full() {
    let lineage = ModelLineage {
        uri: "pacha://model-001:v1.0".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("llama2-finetune-chat".to_string()),
        parent: Some("llama2-7b-base".to_string()),
        content_hash: "blake3:abc123def456".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(json.contains("pacha://model-001:v1.0"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("llama2-finetune-chat"));
    assert!(json.contains("llama2-7b-base"));
    assert!(json.contains("blake3:abc123def456"));

    let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.uri, "pacha://model-001:v1.0");
    assert_eq!(parsed.version, "1.0.0");
    assert_eq!(parsed.recipe, Some("llama2-finetune-chat".to_string()));
    assert_eq!(parsed.parent, Some("llama2-7b-base".to_string()));
}

#[test]
fn test_model_lineage_serialization_minimal() {
    let lineage = ModelLineage {
        uri: "pacha://test:latest".to_string(),
        version: "0.0.1".to_string(),
        recipe: None,
        parent: None,
        content_hash: "blake3:000000".to_string(),
    };

    let json = serde_json::to_string(&lineage).expect("serialize");
    // recipe and parent should be skipped when None
    assert!(!json.contains("recipe"));
    assert!(!json.contains("parent"));

    let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.recipe, None);
    assert_eq!(parsed.parent, None);
}

#[test]
fn test_model_lineage_clone() {
    let lineage = ModelLineage {
        uri: "pacha://test:v1".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("recipe".to_string()),
        parent: Some("parent".to_string()),
        content_hash: "hash".to_string(),
    };

    let cloned = lineage.clone();
    assert_eq!(cloned.uri, lineage.uri);
    assert_eq!(cloned.version, lineage.version);
    assert_eq!(cloned.recipe, lineage.recipe);
    assert_eq!(cloned.parent, lineage.parent);
    assert_eq!(cloned.content_hash, lineage.content_hash);
}

#[test]
fn test_model_lineage_debug() {
    let lineage = ModelLineage {
        uri: "pacha://debug:latest".to_string(),
        version: "1.0.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "hash".to_string(),
    };

    let debug_str = format!("{:?}", lineage);
    assert!(debug_str.contains("ModelLineage"));
    assert!(debug_str.contains("pacha://debug:latest"));
}

// =============================================================================
// ReloadResponse Tests
// =============================================================================

#[test]
fn test_reload_response_serialization() {
    let response = ReloadResponse {
        success: true,
        message: "Model reloaded successfully".to_string(),
        reload_time_ms: 1500,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("Model reloaded successfully"));
    assert!(json.contains("1500"));

    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.success);
    assert_eq!(parsed.message, "Model reloaded successfully");
    assert_eq!(parsed.reload_time_ms, 1500);
}

#[test]
fn test_reload_response_failure() {
    let response = ReloadResponse {
        success: false,
        message: "Model file not found".to_string(),
        reload_time_ms: 0,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ReloadResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(!parsed.success);
    assert_eq!(parsed.reload_time_ms, 0);
}

#[test]
fn test_reload_response_clone() {
    let response = ReloadResponse {
        success: true,
        message: "OK".to_string(),
        reload_time_ms: 100,
    };

    let cloned = response.clone();
    assert_eq!(cloned.success, response.success);
    assert_eq!(cloned.message, response.message);
    assert_eq!(cloned.reload_time_ms, response.reload_time_ms);
}

#[test]
fn test_reload_response_debug() {
    let response = ReloadResponse {
        success: true,
        message: "Debug test".to_string(),
        reload_time_ms: 42,
    };

    let debug_str = format!("{:?}", response);
    assert!(debug_str.contains("ReloadResponse"));
    assert!(debug_str.contains("Debug test"));
}

// =============================================================================
// CompletionChoice Tests
// =============================================================================

#[test]
fn test_completion_choice_serialization() {
    let choice = CompletionChoice {
        text: "Generated text output".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("Generated text output"));
    assert!(json.contains("stop"));

    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.text, "Generated text output");
    assert_eq!(parsed.index, 0);
    assert!(parsed.logprobs.is_none());
    assert_eq!(parsed.finish_reason, "stop");
}

#[test]
fn test_completion_choice_with_logprobs() {
    let logprobs = serde_json::json!({
        "tokens": [" Hello", " world"],
        "token_logprobs": [-0.5, -0.3]
    });

    let choice = CompletionChoice {
        text: "Hello world".to_string(),
        index: 0,
        logprobs: Some(logprobs),
        finish_reason: "length".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("token_logprobs"));

    let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.logprobs.is_some());
}

#[test]
fn test_completion_choice_multiple_indices() {
    for i in 0..5 {
        let choice = CompletionChoice {
            text: format!("Choice {i}"),
            index: i,
            logprobs: None,
            finish_reason: "stop".to_string(),
        };

        let json = serde_json::to_string(&choice).expect("serialize");
        let parsed: CompletionChoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.index, i);
    }
}

#[test]
fn test_completion_choice_clone() {
    let choice = CompletionChoice {
        text: "Test".to_string(),
        index: 1,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let cloned = choice.clone();
    assert_eq!(cloned.text, choice.text);
    assert_eq!(cloned.index, choice.index);
}

#[test]
fn test_completion_choice_debug() {
    let choice = CompletionChoice {
        text: "Debug".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };

    let debug_str = format!("{:?}", choice);
    assert!(debug_str.contains("CompletionChoice"));
}

// =============================================================================
// CompletionResponse Tests
// =============================================================================

#[test]
fn test_completion_response_serialization() {
    let response = CompletionResponse {
        id: "cmpl-test-123".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![CompletionChoice {
            text: "Hello, world!".to_string(),
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

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("cmpl-test-123"));
    assert!(json.contains("text_completion"));
    assert!(json.contains("Hello, world!"));

    let parsed: CompletionResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "cmpl-test-123");
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.usage.total_tokens, 15);
}

#[test]
fn test_completion_response_multiple_choices() {
    let response = CompletionResponse {
        id: "cmpl-multi".to_string(),
        object: "text_completion".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![
            CompletionChoice {
                text: "First".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
            CompletionChoice {
                text: "Second".to_string(),
                index: 1,
                logprobs: None,
                finish_reason: "stop".to_string(),
            },
        ],
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 6,
            total_tokens: 9,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("First"));
    assert!(json.contains("Second"));

    let parsed: CompletionResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.choices.len(), 2);
}

#[test]
fn test_completion_response_clone() {
    let response = CompletionResponse {
        id: "test".to_string(),
        object: "text_completion".to_string(),
        created: 123,
        model: "model".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    let cloned = response.clone();
    assert_eq!(cloned.id, response.id);
    assert_eq!(cloned.created, response.created);
}

#[test]
fn test_completion_response_debug() {
    let response = CompletionResponse {
        id: "debug-test".to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    let debug_str = format!("{:?}", response);
    assert!(debug_str.contains("CompletionResponse"));
}

// =============================================================================
// CompletionRequest Additional Tests
// =============================================================================

#[test]
fn test_completion_request_with_stop_tokens() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: Some(vec!["</s>".to_string(), "<|endoftext|>".to_string()]),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("</s>"));
    assert!(json.contains("<|endoftext|>"));

    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.stop.as_ref().expect("test value should be present").len(), 2);
}

#[test]
fn test_completion_request_temperature_extremes() {
    // Test temperature = 0.0 (greedy)
    let req_zero = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: Some(0.0),
        top_p: None,
        stop: None,
    };
    let json = serde_json::to_string(&req_zero).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.temperature, Some(0.0));

    // Test temperature = 2.0 (high randomness)
    let req_high = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: Some(2.0),
        top_p: None,
        stop: None,
    };
    let json = serde_json::to_string(&req_high).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.temperature, Some(2.0));
}

#[test]
fn test_completion_request_top_p_values() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: Some(0.95),
        stop: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert!((parsed.top_p.expect("test value should be present") - 0.95).abs() < 0.01);
}

include!("completion_request_02.rs");
include!("openai_completions.rs");
