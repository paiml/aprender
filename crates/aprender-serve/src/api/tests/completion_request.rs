//! API Tests Part 12: OpenAI and Realize Handlers - Request/Response Types
//!
//! Tests for openai_handlers.rs and realize_handlers.rs to improve coverage.
//! Focus: Request and response type serialization/deserialization.

use crate::api::realize_handlers::{
    CompletionChoice, CompletionRequest, CompletionResponse, ModelLineage, ModelMetadataResponse,
    ReloadRequest, ReloadResponse,
};

// =============================================================================
// CompletionRequest Tests
// =============================================================================

#[test]
fn test_completion_request_with_stop_sequences() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: None,
        stop: Some(vec!["</s>".to_string(), "<|end|>".to_string()]),
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("</s>"));
    let parsed: CompletionRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.stop.as_ref().expect("test value should be present").len(), 2);
}

#[test]
fn test_completion_request_optional_fields() {
    let json = r#"{"model":"m","prompt":"p"}"#;
    let req: CompletionRequest = serde_json::from_str(json).expect("deserialize");
    assert!(req.max_tokens.is_none() && req.temperature.is_none());
}

#[test]
fn test_completion_request_traits() {
    let req = CompletionRequest {
        model: "test".to_string(),
        prompt: "hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.5),
        top_p: Some(0.9),
        stop: Some(vec!["stop".to_string()]),
    };

    let cloned = req.clone();
    assert_eq!(cloned.model, req.model);
    let debug = format!("{:?}", req);
    assert!(debug.contains("CompletionRequest"));
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
        model: "phi-2".to_string(),
        choices: vec![CompletionChoice {
            text: "Generated text".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: crate::api::Usage {
            prompt_tokens: 5,
            completion_tokens: 10,
            total_tokens: 15,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("cmpl-test-123"));
}

#[test]
fn test_completion_response_deserialization() {
    let json = r#"{
        "id": "cmpl-001", "object": "text_completion", "created": 12345,
        "model": "test", "choices": [{"text": "output", "index": 0, "logprobs": null, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    }"#;
    let response: CompletionResponse = serde_json::from_str(json).expect("deserialize");
    assert_eq!(response.choices[0].finish_reason, "length");
}

// =============================================================================
// CompletionChoice Tests
// =============================================================================

#[test]
fn test_completion_choice_variants() {
    let choice_with = CompletionChoice {
        text: "hello".to_string(),
        index: 0,
        logprobs: Some(serde_json::json!({"tokens": ["hello"]})),
        finish_reason: "stop".to_string(),
    };
    assert!(serde_json::to_string(&choice_with)
        .expect("test value should be present")
        .contains("logprobs"));

    let choice_without = CompletionChoice {
        text: "test".to_string(),
        index: 1,
        logprobs: None,
        finish_reason: "length".to_string(),
    };
    let debug = format!("{:?}", choice_without.clone());
    assert!(debug.contains("CompletionChoice"));
}

// =============================================================================
// ModelLineage Tests
// =============================================================================

#[test]
fn test_model_lineage_full() {
    let lineage = ModelLineage {
        uri: "pacha://model:v1".to_string(),
        version: "1.0.0".to_string(),
        recipe: Some("finetune-lora".to_string()),
        parent: Some("base-model".to_string()),
        content_hash: "blake3:abc123".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(json.contains("pacha://model:v1") && json.contains("finetune-lora"));
}

#[test]
fn test_model_lineage_minimal() {
    let lineage = ModelLineage {
        uri: "pacha://m:latest".to_string(),
        version: "2.0.0".to_string(),
        recipe: None,
        parent: None,
        content_hash: "blake3:xyz".to_string(),
    };
    let json = serde_json::to_string(&lineage).expect("serialize");
    assert!(!json.contains("recipe") && !json.contains("parent"));

    let debug = format!("{:?}", lineage.clone());
    assert!(debug.contains("ModelLineage"));
}

// =============================================================================
// ReloadRequest/Response Tests
// =============================================================================

#[test]
fn test_reload_request_variants() {
    let req_both = ReloadRequest {
        model: Some("model-id".to_string()),
        path: Some("/path/to/model.gguf".to_string()),
    };
    let json = serde_json::to_string(&req_both).expect("serialize");
    assert!(json.contains("model-id") && json.contains("model.gguf"));

    let req_none = ReloadRequest {
        model: None,
        path: None,
    };
    let cloned = req_none.clone();
    assert!(cloned.model.is_none());
}

#[test]
fn test_reload_response_variants() {
    let success = ReloadResponse {
        success: true,
        message: "OK".to_string(),
        reload_time_ms: 150,
    };
    let json = serde_json::to_string(&success).expect("serialize");
    assert!(json.contains("true") && json.contains("150"));

    let failure = ReloadResponse {
        success: false,
        message: "Error".to_string(),
        reload_time_ms: 5,
    };
    let debug = format!("{:?}", failure.clone());
    assert!(debug.contains("ReloadResponse"));
}

// =============================================================================
// ModelMetadataResponse Tests
// =============================================================================

#[test]
fn test_model_metadata_with_lineage() {
    let response = ModelMetadataResponse {
        id: "model-with-lineage".to_string(),
        name: "Test Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 1_000_000,
        quantization: Some("Q4_K_M".to_string()),
        context_length: 4096,
        lineage: Some(ModelLineage {
            uri: "pacha://test:v1".to_string(),
            version: "1.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "hash".to_string(),
        }),
        loaded: true,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("lineage") && json.contains("pacha://test:v1"));
}

#[test]
fn test_model_metadata_traits() {
    let response = ModelMetadataResponse {
        id: "debug-id".to_string(),
        name: "Debug".to_string(),
        format: "APR".to_string(),
        size_bytes: 500,
        quantization: None,
        context_length: 2048,
        lineage: None,
        loaded: false,
    };
    let debug = format!("{:?}", response.clone());
    assert!(debug.contains("ModelMetadataResponse") && debug.contains("debug-id"));
}
