//! API Tests Part 14: Handler Coverage Extension
//!
//! Tests for additional coverage in realize_handlers.rs, openai_handlers.rs,
//! and gpu_handlers.rs. Focus: Streaming types, trace data, and edge cases.

use crate::api::{
    build_trace_data, ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatDelta, ChatMessage,
    CompletionChoice, OpenAIModel, OpenAIModelsResponse, ReloadResponse, TraceData, TraceOperation,
    TraceProvenance, Usage,
};

// =============================================================================
// ChatCompletionChunk Tests (Streaming Types)
// =============================================================================

#[test]
fn test_chat_completion_chunk_initial() {
    let chunk = ChatCompletionChunk::initial("req-123", "test-model");

    assert_eq!(chunk.id, "req-123");
    assert_eq!(chunk.model, "test-model");
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert!(chunk.created > 0);
    assert_eq!(chunk.choices.len(), 1);

    // Initial chunk has role but no content or finish_reason
    let choice = &chunk.choices[0];
    assert_eq!(choice.index, 0);
    assert_eq!(choice.delta.role, Some("assistant".to_string()));
    assert!(choice.delta.content.is_none());
    assert!(choice.finish_reason.is_none());
}

#[test]
fn test_chat_completion_chunk_content() {
    let chunk = ChatCompletionChunk::content("req-456", "model-v1", "Hello");

    assert_eq!(chunk.id, "req-456");
    assert_eq!(chunk.model, "model-v1");
    assert_eq!(chunk.choices.len(), 1);

    let choice = &chunk.choices[0];
    // Content chunk has content but no role
    assert!(choice.delta.role.is_none());
    assert_eq!(choice.delta.content, Some("Hello".to_string()));
    assert!(choice.finish_reason.is_none());
}

#[test]
fn test_chat_completion_chunk_done() {
    let chunk = ChatCompletionChunk::done("req-789", "model-v2");

    assert_eq!(chunk.id, "req-789");
    assert_eq!(chunk.model, "model-v2");
    assert_eq!(chunk.choices.len(), 1);

    let choice = &chunk.choices[0];
    // Done chunk has finish_reason but no role or content
    assert!(choice.delta.role.is_none());
    assert!(choice.delta.content.is_none());
    assert_eq!(choice.finish_reason, Some("stop".to_string()));
}

#[test]
fn test_chat_completion_chunk_serialization() {
    let chunk = ChatCompletionChunk::content("test-id", "test-model", "world");

    let json = serde_json::to_string(&chunk).expect("serialize");
    assert!(json.contains("test-id"));
    assert!(json.contains("test-model"));
    assert!(json.contains("chat.completion.chunk"));
    assert!(json.contains("world"));

    let parsed: ChatCompletionChunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "test-id");
    assert_eq!(parsed.choices[0].delta.content, Some("world".to_string()));
}

#[test]
fn test_chat_completion_chunk_clone_debug() {
    let chunk = ChatCompletionChunk::initial("clone-test", "model");
    let cloned = chunk.clone();
    assert_eq!(cloned.id, chunk.id);

    let debug = format!("{:?}", chunk);
    assert!(debug.contains("ChatCompletionChunk"));
    assert!(debug.contains("clone-test"));
}

// =============================================================================
// ChatChunkChoice and ChatDelta Tests
// =============================================================================

#[test]
fn test_chat_chunk_choice_serialization() {
    let choice = ChatChunkChoice {
        index: 2,
        delta: ChatDelta {
            role: Some("user".to_string()),
            content: Some("test content".to_string()),
        },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("\"index\":2"));
    assert!(json.contains("user"));
    assert!(json.contains("test content"));

    let parsed: ChatChunkChoice = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.index, 2);
    assert_eq!(parsed.delta.role, Some("user".to_string()));
}

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

    let parsed: ChatChunkChoice = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.finish_reason, Some("length".to_string()));
}

#[test]
fn test_chat_delta_empty() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };

    let json = serde_json::to_string(&delta).expect("serialize");
    // Empty delta should still serialize (with skip_serializing_if)
    let parsed: ChatDelta = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.role.is_none());
    assert!(parsed.content.is_none());
}

#[test]
fn test_chat_delta_clone_debug() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("response".to_string()),
    };
    let cloned = delta.clone();
    assert_eq!(cloned.role, delta.role);

    let debug = format!("{:?}", delta);
    assert!(debug.contains("ChatDelta"));
}

// =============================================================================
// build_trace_data Tests
// =============================================================================

#[test]
fn test_build_trace_data_brick_level() {
    let (brick, step, layer) = build_trace_data(Some("brick"), 10000, 50, 20, 12);

    assert!(brick.is_some());
    assert!(step.is_none());
    assert!(layer.is_none());

    let trace = brick.expect("test value should be present");
    assert_eq!(trace.level, "brick");
    assert_eq!(trace.operations, 20); // completion_tokens
    assert_eq!(trace.total_time_us, 10000);
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 10000);
    let details = trace.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("apr profile"));
    assert!(details.contains("50 prompt"));
    assert!(details.contains("20 completion"));
}

#[test]
fn test_build_trace_data_step_level() {
    let (brick, step, layer) = build_trace_data(Some("step"), 5000, 100, 50, 24);

    assert!(brick.is_none());
    assert!(step.is_some());
    assert!(layer.is_none());

    let trace = step.expect("test value should be present");
    assert_eq!(trace.level, "step");
    assert_eq!(trace.operations, 50); // completion_tokens
    assert_eq!(trace.total_time_us, 5000);
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 5000);
    let details = trace.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_layer_level() {
    let (brick, step, layer) = build_trace_data(Some("layer"), 12000, 30, 15, 6);

    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_some());

    let trace = layer.expect("test value should be present");
    assert_eq!(trace.level, "layer");
    assert_eq!(trace.operations, 6); // num_layers
    assert_eq!(trace.total_time_us, 12000);
    assert_eq!(trace.breakdown.len(), 1);
    assert_eq!(trace.breakdown[0].name, "total_inference");
    assert_eq!(trace.breakdown[0].time_us, 12000);
    let details = trace.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("6 layers"));
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_none() {
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

// =============================================================================
// TraceData and TraceOperation Tests
// =============================================================================

#[test]
fn test_trace_data_serialization() {
    let trace = TraceData {
        level: "test".to_string(),
        operations: 10,
        total_time_us: 5000,
        breakdown: vec![
            TraceOperation {
                name: "op1".to_string(),
                time_us: 2000,
                details: Some("detail1".to_string()),
            },
            TraceOperation {
                name: "op2".to_string(),
                time_us: 3000,
                details: None,
            },
        ],
        provenance: TraceProvenance::Estimated,
    };

    let json = serde_json::to_string(&trace).expect("serialize");
    assert!(json.contains("\"level\":\"test\""));
    assert!(json.contains("\"operations\":10"));
    assert!(json.contains("detail1"));

    let parsed: TraceData = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.level, "test");
    assert_eq!(parsed.breakdown.len(), 2);
}

#[test]
fn test_trace_data_clone_debug() {
    let trace = TraceData {
        level: "brick".to_string(),
        operations: 5,
        total_time_us: 1000,
        breakdown: vec![],
        provenance: TraceProvenance::Estimated,
    };
    let cloned = trace.clone();
    assert_eq!(cloned.level, trace.level);

    let debug = format!("{:?}", trace);
    assert!(debug.contains("TraceData"));
}

#[test]
fn test_trace_operation_serialization() {
    let op = TraceOperation {
        name: "matmul".to_string(),
        time_us: 500,
        details: Some("M=4096, K=4096, N=4096".to_string()),
    };

    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("matmul"));
    assert!(json.contains("500"));
    assert!(json.contains("M=4096"));

    let parsed: TraceOperation = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.name, "matmul");
    assert_eq!(parsed.time_us, 500);
}

#[test]
fn test_trace_operation_without_details() {
    let op = TraceOperation {
        name: "softmax".to_string(),
        time_us: 100,
        details: None,
    };

    let json = serde_json::to_string(&op).expect("serialize");
    // details should be skipped when None
    let parsed: TraceOperation = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.details.is_none());
}

#[test]
fn test_trace_operation_clone_debug() {
    let op = TraceOperation {
        name: "test_op".to_string(),
        time_us: 42,
        details: None,
    };
    let cloned = op.clone();
    assert_eq!(cloned.name, op.name);

    let debug = format!("{:?}", op);
    assert!(debug.contains("TraceOperation"));
}

// =============================================================================
// OpenAIModel and OpenAIModelsResponse Tests
// =============================================================================

#[test]
fn test_openai_model_serialization() {
    let model = OpenAIModel {
        id: "gpt-3.5-turbo".to_string(),
        object: "model".to_string(),
        created: 1677610602,
        owned_by: "openai".to_string(),
    };

    let json = serde_json::to_string(&model).expect("serialize");
    assert!(json.contains("gpt-3.5-turbo"));
    assert!(json.contains("model"));
    assert!(json.contains("openai"));

    let parsed: OpenAIModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, "gpt-3.5-turbo");
    assert_eq!(parsed.created, 1677610602);
}

#[test]
fn test_openai_model_clone_debug() {
    let model = OpenAIModel {
        id: "test".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "test".to_string(),
    };
    let cloned = model.clone();
    assert_eq!(cloned.id, model.id);

    let debug = format!("{:?}", model);
    assert!(debug.contains("OpenAIModel"));
}

#[test]
fn test_openai_models_response_serialization() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![
            OpenAIModel {
                id: "model-1".to_string(),
                object: "model".to_string(),
                created: 1000,
                owned_by: "realizar".to_string(),
            },
            OpenAIModel {
                id: "model-2".to_string(),
                object: "model".to_string(),
                created: 2000,
                owned_by: "realizar".to_string(),
            },
        ],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("list"));
    assert!(json.contains("model-1"));
    assert!(json.contains("model-2"));

    let parsed: OpenAIModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.data.len(), 2);
}

#[test]
fn test_openai_models_response_empty() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![],
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: OpenAIModelsResponse = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.data.is_empty());
}

#[test]
fn test_openai_models_response_clone_debug() {
    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![],
    };
    let cloned = response.clone();
    assert_eq!(cloned.object, response.object);

    let debug = format!("{:?}", response);
    assert!(debug.contains("OpenAIModelsResponse"));
}

// =============================================================================
// ChatChoice Tests
// =============================================================================

#[test]
fn test_chat_choice_serialization() {
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: "Hello, how can I help?".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("assistant"));
    assert!(json.contains("Hello, how can I help?"));
    assert!(json.contains("stop"));

    let parsed: ChatChoice = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.index, 0);
    assert_eq!(parsed.finish_reason, "stop");
}

include!("chat_choice.rs");
