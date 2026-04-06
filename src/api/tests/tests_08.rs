//! API Tests Part 08: OpenAI Handlers Coverage
//!
//! Tests for openai_handlers.rs to improve coverage from ~25% to 50%+.
//! Focus: Error paths, edge cases, model validation, streaming, token counting.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;
use crate::api::{
    build_trace_data, ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatMessage, ErrorResponse, OpenAIModel,
    OpenAIModelsResponse, TraceData, TraceOperation, TraceProvenance, Usage,
};

// =============================================================================
// ChatCompletionRequest Tests
// =============================================================================

#[test]
fn test_chat_completion_request_deserialization_minimal() {
    let json = r#"{"model": "test", "messages": [{"role": "user", "content": "Hi"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(req.model, "test");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
    assert_eq!(req.messages[0].content, "Hi");
    // Defaults
    assert_eq!(req.max_tokens, None);
    assert_eq!(req.temperature, None);
    assert_eq!(req.top_p, None);
    assert_eq!(req.n, 1);
    assert!(!req.stream);
    assert_eq!(req.stop, None);
    assert_eq!(req.user, None);
}

#[test]
fn test_chat_completion_request_deserialization_full() {
    let json = r#"{
        "model": "phi-2",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stream": true,
        "stop": ["</s>", "<|im_end|>"],
        "user": "test-user"
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(req.model, "phi-2");
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.max_tokens, Some(100));
    assert!((req.temperature.expect("test value should be present") - 0.7).abs() < 0.01);
    assert!((req.top_p.expect("test value should be present") - 0.9).abs() < 0.01);
    assert_eq!(req.n, 1);
    assert!(req.stream);
    assert_eq!(req.stop.as_ref().expect("test value should be present").len(), 2);
    assert_eq!(req.user, Some("test-user".to_string()));
}

#[test]
fn test_chat_completion_request_with_name() {
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "Hi", "name": "John"}]
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(req.messages[0].name, Some("John".to_string()));
}

#[test]
fn test_chat_completion_request_serialization() {
    let req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
        max_tokens: Some(50),
        temperature: Some(0.8),
        top_p: None,
        n: 1,
        stream: false,
        stop: None,
        user: None,
    };

    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("test-model"));
    assert!(json.contains("Hello"));
    assert!(json.contains("50"));
}

// =============================================================================
// ChatMessage Tests
// =============================================================================

#[test]
fn test_chat_message_all_roles() {
    let roles = ["system", "user", "assistant"];
    for role in roles {
        let msg = ChatMessage {
            role: role.to_string(),
            content: "test".to_string(),
            name: None,
        };
        assert_eq!(msg.role, role);
    }
}

#[test]
fn test_chat_message_with_optional_name() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("Alice".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("serialize");
    assert!(json.contains("Alice"));

    let parsed: ChatMessage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.name, Some("Alice".to_string()));
}

#[test]
fn test_chat_message_clone() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("Bob".to_string()),
    };
    let cloned = msg.clone();
    assert_eq!(cloned.role, msg.role);
    assert_eq!(cloned.content, msg.content);
    assert_eq!(cloned.name, msg.name);
}

#[test]
fn test_chat_message_debug() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: None,
    };
    let debug = format!("{:?}", msg);
    assert!(debug.contains("ChatMessage"));
    assert!(debug.contains("user"));
}

// =============================================================================
// ChatCompletionResponse Tests
// =============================================================================

#[test]
fn test_chat_completion_response_serialization() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-test-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "phi-2".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
        brick_trace: None,
        step_trace: None,
        layer_trace: None,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("chatcmpl-test-123"));
    assert!(json.contains("chat.completion"));
    assert!(json.contains("phi-2"));
    assert!(json.contains("Hello!"));
    assert!(json.contains("stop"));
    // Trace fields should not be present when None
    assert!(!json.contains("brick_trace"));
}

#[test]
fn test_chat_completion_response_with_traces() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![TraceOperation {
            name: "matmul".to_string(),
            time_us: 500,
            details: Some("4x4".to_string()),
        }],
        provenance: TraceProvenance::Estimated,
    };

    let response = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        brick_trace: Some(trace),
        step_trace: None,
        layer_trace: None,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("brick_trace"));
    assert!(json.contains("matmul"));
}

// =============================================================================
// ChatCompletionChunk Tests (Streaming)
// =============================================================================

#[test]
fn test_chat_completion_chunk_initial() {
    let chunk = ChatCompletionChunk::initial("chatcmpl-123", "phi-2");

    assert_eq!(chunk.id, "chatcmpl-123");
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert_eq!(chunk.model, "phi-2");
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
    assert_eq!(chunk.choices[0].delta.content, None);
    assert_eq!(chunk.choices[0].finish_reason, None);
}

#[test]
fn test_chat_completion_chunk_content() {
    let chunk = ChatCompletionChunk::content("chatcmpl-456", "phi-2", "Hello");

    assert_eq!(chunk.id, "chatcmpl-456");
    assert_eq!(chunk.choices[0].delta.role, None);
    assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    assert_eq!(chunk.choices[0].finish_reason, None);
}

#[test]
fn test_chat_completion_chunk_done() {
    let chunk = ChatCompletionChunk::done("chatcmpl-789", "phi-2");

    assert_eq!(chunk.id, "chatcmpl-789");
    assert_eq!(chunk.choices[0].delta.role, None);
    assert_eq!(chunk.choices[0].delta.content, None);
    assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
}

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = ChatCompletionChunk::content("test-id", "test-model", "Hi");
    let json = serde_json::to_string(&chunk).expect("serialize");

    assert!(json.contains("chat.completion.chunk"));
    assert!(json.contains("test-id"));
    assert!(json.contains("test-model"));
    assert!(json.contains("Hi"));
}

#[test]
fn test_chat_completion_chunk_created_timestamp() {
    let chunk1 = ChatCompletionChunk::initial("id1", "model");
    let chunk2 = ChatCompletionChunk::initial("id2", "model");

    // Both should have valid timestamps (>0)
    assert!(chunk1.created > 0);
    assert!(chunk2.created > 0);
}

// =============================================================================
// ChatDelta Tests
// =============================================================================

#[test]
fn test_chat_delta_role_only() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("assistant"));
    // Content should be skipped when None
    assert!(!json.contains("content"));
}

#[test]
fn test_chat_delta_content_only() {
    let delta = ChatDelta {
        role: None,
        content: Some("Hello world".to_string()),
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    assert!(json.contains("Hello world"));
    // Role should be skipped when None
    assert!(!json.contains("role"));
}

#[test]
fn test_chat_delta_both_none() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    // Should produce empty object
    assert_eq!(json, "{}");
}

// =============================================================================
// ChatChunkChoice Tests
// =============================================================================

#[test]
fn test_chat_chunk_choice_with_finish_reason() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("length".to_string()),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("length"));
}

#[test]
fn test_chat_chunk_choice_without_finish_reason() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("token".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("token"));
}

// =============================================================================
// OpenAIModelsResponse Tests
// =============================================================================

#[test]
fn test_openai_models_response_serialization() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![
            OpenAIModel {
                id: "phi-2".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "realizar".to_string(),
            },
            OpenAIModel {
                id: "llama-7b".to_string(),
                object: "model".to_string(),
                created: 1700000001,
                owned_by: "realizar".to_string(),
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("list"));
    assert!(json.contains("phi-2"));
    assert!(json.contains("llama-7b"));
    assert!(json.contains("realizar"));
}

#[test]
fn test_openai_models_response_empty() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("[]"));
}

// =============================================================================
// OpenAIModel Tests
// =============================================================================

#[test]
fn test_openai_model_serialization() {
    let model = OpenAIModel {
        id: "gpt-4-turbo".to_string(),
        object: "model".to_string(),
        created: 1699999999,
        owned_by: "openai".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    let parsed: OpenAIModel = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.id, "gpt-4-turbo");
    assert_eq!(parsed.object, "model");
    assert_eq!(parsed.created, 1699999999);
    assert_eq!(parsed.owned_by, "openai");
}

#[test]
fn test_openai_model_clone() {
    let model = OpenAIModel {
        id: "test".to_string(),
        object: "model".to_string(),
        created: 12345,
        owned_by: "test-org".to_string(),
    };

    let cloned = model.clone();
    assert_eq!(cloned.id, model.id);
    assert_eq!(cloned.created, model.created);
}

include!("openai_model.rs");
include!("openai_chat_02.rs");
include!("chat_completion_04.rs");
include!("chat_completions.rs");
