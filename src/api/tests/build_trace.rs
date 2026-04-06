//! T-COV-95 Extended Coverage: api/mod.rs
//!
//! Targets: build_trace_data for all trace levels, AppState accessors,
//! streaming types serde, error paths, edge cases.

use crate::api::test_helpers::create_test_app_shared;
use crate::api::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

// ============================================================================
// build_trace_data coverage for all branches
// ============================================================================

#[test]
fn test_build_trace_data_brick_level() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 1000, 10, 5, 4);
    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());

    let trace = brick.expect("test value should be present");
    assert_eq!(trace.level, "brick");
    assert_eq!(trace.operations, 5);
    assert_eq!(trace.total_time_us, 1000);
    // Single honest entry with wall-clock total — no fabricated per-op breakdown
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 1000);
    assert!(trace.breakdown[0]
        .details
        .as_ref()
        .expect("details should be present")
        .contains("apr profile"));
    // GH-92: provenance must be WallClockTotal, never Estimated or Measured
    assert_eq!(trace.provenance, TraceProvenance::WallClockTotal);
}

#[test]
fn test_build_trace_data_step_level() {
    let (brick, step, layer) = build_trace_data(Some("step"), 5000, 20, 10, 8);
    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());

    let trace = step.expect("test value should be present");
    assert_eq!(trace.level, "step");
    assert_eq!(trace.operations, 10);
    assert_eq!(trace.total_time_us, 5000);
    // Single honest entry — no fabricated tokenize/forward_pass/decode breakdown
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 5000);
    let details = trace.breakdown[0].details.as_ref().expect("details should be present");
    assert!(details.contains("20 prompt"));
    assert!(details.contains("10 completion"));
    // GH-92: provenance must be WallClockTotal
    assert_eq!(trace.provenance, TraceProvenance::WallClockTotal);
}

#[test]
fn test_build_trace_data_layer_level() {
    let (brick, step, layer) = build_trace_data(Some("layer"), 8000, 15, 8, 12);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());

    let trace = layer.expect("test value should be present");
    assert_eq!(trace.level, "layer");
    assert_eq!(trace.operations, 12);
    assert_eq!(trace.total_time_us, 8000);
    // Single honest entry — no fabricated per-layer breakdown
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 8000);
    let details = trace.breakdown[0].details.as_ref().expect("details should be present");
    assert!(details.contains("12 layers"));
    // GH-92: provenance must be WallClockTotal
    assert_eq!(trace.provenance, TraceProvenance::WallClockTotal);
}

#[test]
fn test_build_trace_data_none_level() {
    let (brick, step, layer) = build_trace_data(None, 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = build_trace_data(Some("unknown"), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_empty_string_level() {
    let (brick, step, layer) = build_trace_data(Some(""), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_detailed_level() {
    // "detailed" is not a recognized level, should return none
    let (brick, step, layer) = build_trace_data(Some("detailed"), 1000, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

#[test]
fn test_build_trace_data_zero_values() {
    let (brick, _step, _layer) = build_trace_data(Some("brick"), 0, 0, 0, 0);
    assert!(brick.is_some());
    let trace = brick.expect("test value should be present");
    assert_eq!(trace.total_time_us, 0);
    assert_eq!(trace.operations, 0);
}

#[test]
fn test_build_trace_data_large_values() {
    let (_brick, _step, layer) = build_trace_data(Some("layer"), u64::MAX / 2, 100000, 50000, 1000);
    assert!(layer.is_some());
    let trace = layer.expect("test value should be present");
    // Single honest entry regardless of layer count
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].time_us, u64::MAX / 2);
}

// ============================================================================
// TraceData and TraceOperation struct coverage
// ============================================================================

#[test]
fn test_trace_data_debug() {
    let trace = TraceData {
        level: "test".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![],
        provenance: TraceProvenance::Estimated,
    };
    let debug = format!("{:?}", trace);
    assert!(debug.contains("test"));
    assert!(debug.contains("1000"));
}

#[test]
fn test_trace_data_clone() {
    let trace = TraceData {
        level: "test".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![TraceOperation {
            name: "op1".to_string(),
            time_us: 100,
            details: Some("detail".to_string()),
        }],
        provenance: TraceProvenance::Estimated,
    };
    let cloned = trace.clone();
    assert_eq!(cloned.level, "test");
    assert_eq!(cloned.breakdown.len(), 1);
}

#[test]
fn test_trace_data_serde() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 10,
        total_time_us: 5000,
        breakdown: vec![TraceOperation {
            name: "matmul".to_string(),
            time_us: 500,
            details: None,
        }],
        provenance: TraceProvenance::Estimated,
    };
    let json = serde_json::to_string(&trace).expect("JSON serialization failed");
    assert!(json.contains("brick"));
    assert!(json.contains("matmul"));
    let parsed: TraceData = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.level, "brick");
}

#[test]
fn test_trace_provenance_serde_round_trip() {
    // GH-92: Provenance must survive JSON serialization
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 1,
        total_time_us: 100,
        breakdown: vec![],
        provenance: TraceProvenance::WallClockTotal,
    };
    let json = serde_json::to_string(&trace).expect("serialize");
    assert!(json.contains("\"wall_clock_total\""));
    let parsed: TraceData = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.provenance, TraceProvenance::WallClockTotal);
}

#[test]
fn test_trace_provenance_defaults_to_estimated_on_missing_field() {
    // GH-92: Backwards compat — old JSON without provenance field defaults to Estimated
    let json = r#"{"level":"brick","operations":1,"total_time_us":100,"breakdown":[]}"#;
    let parsed: TraceData = serde_json::from_str(json).expect("deserialize");
    assert_eq!(parsed.provenance, TraceProvenance::Estimated);
}

#[test]
fn test_trace_provenance_all_variants_serde() {
    for (variant, expected_str) in [
        (TraceProvenance::Measured, "measured"),
        (TraceProvenance::WallClockTotal, "wall_clock_total"),
        (TraceProvenance::Estimated, "estimated"),
    ] {
        let json = serde_json::to_string(&variant).expect("serialize");
        assert!(json.contains(expected_str), "expected {expected_str} in {json}");
        let parsed: TraceProvenance = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, variant);
    }
}

#[test]
fn test_trace_operation_debug() {
    let op = TraceOperation {
        name: "test_op".to_string(),
        time_us: 500,
        details: Some("info".to_string()),
    };
    let debug = format!("{:?}", op);
    assert!(debug.contains("test_op"));
    assert!(debug.contains("info"));
}

#[test]
fn test_trace_operation_clone() {
    let op = TraceOperation {
        name: "test_op".to_string(),
        time_us: 500,
        details: None,
    };
    let cloned = op.clone();
    assert_eq!(cloned.name, "test_op");
    assert!(cloned.details.is_none());
}

#[test]
fn test_trace_operation_serde() {
    let op = TraceOperation {
        name: "softmax".to_string(),
        time_us: 250,
        details: Some("batch=32".to_string()),
    };
    let json = serde_json::to_string(&op).expect("JSON serialization failed");
    assert!(json.contains("softmax"));
    let parsed: TraceOperation = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.name, "softmax");
    assert_eq!(parsed.details.as_deref(), Some("batch=32"));
}

// ============================================================================
// Streaming types coverage
// ============================================================================

#[test]
fn test_chat_completion_chunk_serde() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1234567890,
        model: "default".to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: Some("Hello".to_string()),
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).expect("JSON serialization failed");
    assert!(json.contains("chatcmpl-123"));
    assert!(json.contains("chat.completion.chunk"));
    let parsed: ChatCompletionChunk = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.id, "chatcmpl-123");
}

#[test]
fn test_chat_completion_chunk_debug() {
    let chunk = ChatCompletionChunk {
        id: "id".to_string(),
        object: "obj".to_string(),
        created: 0,
        model: "mod".to_string(),
        choices: vec![],
    };
    let debug = format!("{:?}", chunk);
    assert!(debug.contains("ChatCompletionChunk"));
}

#[test]
fn test_chat_completion_chunk_clone() {
    let chunk = ChatCompletionChunk {
        id: "id".to_string(),
        object: "obj".to_string(),
        created: 123,
        model: "mod".to_string(),
        choices: vec![],
    };
    let cloned = chunk.clone();
    assert_eq!(cloned.id, "id");
    assert_eq!(cloned.created, 123);
}

#[test]
fn test_chat_chunk_choice_serde() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("world".to_string()),
        },
        finish_reason: Some("stop".to_string()),
    };
    let json = serde_json::to_string(&choice).expect("JSON serialization failed");
    assert!(json.contains("world"));
    assert!(json.contains("stop"));
    let parsed: ChatChunkChoice = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn test_chat_chunk_choice_debug() {
    let choice = ChatChunkChoice {
        index: 5,
        delta: ChatDelta {
            role: Some("user".to_string()),
            content: None,
        },
        finish_reason: None,
    };
    let debug = format!("{:?}", choice);
    assert!(debug.contains("ChatChunkChoice"));
    assert!(debug.contains("5"));
}

#[test]
fn test_chat_delta_serde() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("response".to_string()),
    };
    let json = serde_json::to_string(&delta).expect("JSON serialization failed");
    assert!(json.contains("assistant"));
    assert!(json.contains("response"));
    let parsed: ChatDelta = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.role.as_deref(), Some("assistant"));
}

#[test]
fn test_chat_delta_debug() {
    let delta = ChatDelta {
        role: None,
        content: Some("text".to_string()),
    };
    let debug = format!("{:?}", delta);
    assert!(debug.contains("ChatDelta"));
}

#[test]
fn test_chat_delta_clone() {
    let delta = ChatDelta {
        role: Some("user".to_string()),
        content: Some("hello".to_string()),
    };
    let cloned = delta.clone();
    assert_eq!(cloned.role.as_deref(), Some("user"));
    assert_eq!(cloned.content.as_deref(), Some("hello"));
}

// ============================================================================
// ChatMessage coverage
// ============================================================================

#[test]
fn test_chat_message_with_name() {
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("Alice".to_string()),
    };
    let json = serde_json::to_string(&msg).expect("JSON serialization failed");
    assert!(json.contains("Alice"));
    let parsed: ChatMessage = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.name.as_deref(), Some("Alice"));
}

#[test]
fn test_chat_message_debug() {
    let msg = ChatMessage {
        role: "system".to_string(),
        content: "Be helpful".to_string(),
        name: None,
    };
    let debug = format!("{:?}", msg);
    assert!(debug.contains("system"));
    assert!(debug.contains("Be helpful"));
}

#[test]
fn test_chat_message_clone() {
    let msg = ChatMessage {
        role: "assistant".to_string(),
        content: "Hi there".to_string(),
        name: Some("Bot".to_string()),
    };
    let cloned = msg.clone();
    assert_eq!(cloned.content, "Hi there");
}

// ============================================================================
// ChatChoice coverage
// ============================================================================

#[test]
fn test_chat_choice_serde() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Response".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("JSON serialization failed");
    assert!(json.contains("Response"));
    let parsed: ChatChoice = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.finish_reason, "stop");
}

#[test]
fn test_chat_choice_debug() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "test".to_string(),
            name: None,
        },
        finish_reason: "length".to_string(),
    };
    let debug = format!("{:?}", choice);
    assert!(debug.contains("ChatChoice"));
}

// ============================================================================
// Usage coverage
// ============================================================================

#[test]
fn test_usage_debug() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let debug = format!("{:?}", usage);
    assert!(debug.contains("30"));
}

#[test]
fn test_usage_clone() {
    let usage = Usage {
        prompt_tokens: 5,
        completion_tokens: 15,
        total_tokens: 20,
    };
    let cloned = usage.clone();
    assert_eq!(cloned.total_tokens, 20);
}

include!("openai_model_02.rs");
include!("chat_completion_05.rs");
