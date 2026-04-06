//! Tests for Banco API types — serde roundtrip, defaults, error format.

use super::types::*;
use crate::serve::templates::{ChatMessage, Role};

// ============================================================================
// BANCO_TYP_001: BancoChatRequest serde roundtrip
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_001_chat_request_roundtrip() {
    let json = r#"{
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": false
    }"#;
    let req: BancoChatRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(req.model, Some("llama3".to_string()));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, Role::User);
    assert_eq!(req.max_tokens, 100);
    assert!((req.temperature - 0.5).abs() < f32::EPSILON);
    assert!(!req.stream);

    // Roundtrip
    let serialized = serde_json::to_string(&req).expect("serialize");
    let req2: BancoChatRequest = serde_json::from_str(&serialized).expect("re-deserialize");
    assert_eq!(req2.model, req.model);
    assert_eq!(req2.messages.len(), 1);
}

// ============================================================================
// BANCO_TYP_002: Request defaults
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_002_request_defaults() {
    let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
    let req: BancoChatRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(req.model, None);
    assert_eq!(req.max_tokens, 256);
    assert!((req.temperature - 0.7).abs() < f32::EPSILON);
    assert!((req.top_p - 1.0).abs() < f32::EPSILON);
    assert!(!req.stream);
}

// ============================================================================
// BANCO_TYP_002b: Response format
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_002b_response_format_json_schema() {
    let json = r#"{
        "messages": [{"role": "user", "content": "Extract"}],
        "response_format": {
            "type": "json_schema",
            "name": "person",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
        }
    }"#;
    let req: BancoChatRequest = serde_json::from_str(json).expect("deserialize");
    assert!(req.response_format.is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_002b_response_format_json_object() {
    let json =
        r#"{"messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_object"}}"#;
    let req: BancoChatRequest = serde_json::from_str(json).expect("deserialize");
    assert!(req.response_format.is_some());
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_002b_response_format_none() {
    let json = r#"{"messages":[{"role":"user","content":"Hi"}]}"#;
    let req: BancoChatRequest = serde_json::from_str(json).expect("deserialize");
    assert!(req.response_format.is_none());
}

// ============================================================================
// BANCO_TYP_003: BancoChatResponse serde roundtrip
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_003_chat_response_roundtrip() {
    let resp = BancoChatResponse {
        id: "banco-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1_700_000_000,
        model: "echo".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::assistant("Hello!"),
            finish_reason: "dry_run".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            context_window: None,
            context_used_pct: None,
        },
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    let resp2: BancoChatResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(resp2.id, "banco-123");
    assert_eq!(resp2.choices.len(), 1);
    assert_eq!(resp2.choices[0].finish_reason, "dry_run");
    assert_eq!(resp2.usage.total_tokens, 15);
}

// ============================================================================
// BANCO_TYP_004: BancoChatChunk serde roundtrip
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_004_chat_chunk_roundtrip() {
    let chunk = BancoChatChunk {
        id: "banco-456".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1_700_000_001,
        model: "echo".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta { role: Some(Role::Assistant), content: None },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).expect("serialize");
    assert!(!json.contains("finish_reason"));
    assert!(json.contains("assistant"));

    let chunk2: BancoChatChunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(chunk2.choices[0].delta.role, Some(Role::Assistant));
    assert!(chunk2.choices[0].finish_reason.is_none());
}

// ============================================================================
// BANCO_TYP_005: ErrorResponse
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_005_error_response_new() {
    let err = ErrorResponse::new("bad request", "invalid_request", 400);
    assert_eq!(err.error.message, "bad request");
    assert_eq!(err.error.type_, "invalid_request");
    assert_eq!(err.error.code, 400);

    let json = serde_json::to_string(&err).expect("serialize");
    assert!(json.contains("bad request"));
    assert!(json.contains("invalid_request"));
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_005_error_response_roundtrip() {
    let err = ErrorResponse::new("not found", "not_found", 404);
    let json = serde_json::to_string(&err).expect("serialize");
    let err2: ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(err2.error.code, 404);
    assert_eq!(err2.error.message, "not found");
}

// ============================================================================
// BANCO_TYP_006: Health / Models / System responses
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_006_health_response_serde() {
    let health = HealthResponse {
        status: "ok".to_string(),
        circuit_breaker_state: "closed".to_string(),
        uptime_secs: 42,
    };
    let json = serde_json::to_string(&health).expect("serialize");
    let h2: HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(h2.status, "ok");
    assert_eq!(h2.uptime_secs, 42);
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_006_models_response_serde() {
    let models = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "realizar".to_string(),
            object: "model".to_string(),
            owned_by: "batuta".to_string(),
            local: true,
        }],
    };
    let json = serde_json::to_string(&models).expect("serialize");
    let m2: ModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(m2.data.len(), 1);
    assert!(m2.data[0].local);
}

#[test]
#[allow(non_snake_case)]
fn test_BANCO_TYP_006_system_response_serde() {
    let sys = SystemResponse {
        privacy_tier: "Standard".to_string(),
        backends: vec!["Realizar".to_string()],
        gpu_available: true,
        version: "0.1.0".to_string(),
        telemetry: false,
        model_loaded: false,
        model_id: None,
        hint: None,
        tokenizer: None,
        endpoints: 54,
        files: 0,
        conversations: 0,
        rag_indexed: false,
        rag_chunks: 0,
        training_runs: 0,
        audit_entries: 0,
    };
    let json = serde_json::to_string(&sys).expect("serialize");
    let s2: SystemResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(s2.privacy_tier, "Standard");
    assert!(s2.gpu_available);
}
