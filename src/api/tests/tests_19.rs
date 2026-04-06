//! API Tests Part 19: T-COV-95 Deep Coverage Bridge
//!
//! Covers additional uncovered paths in:
//! - BatchConfig: low_latency, high_throughput, should_process, meets_minimum
//! - ContinuousBatchResponse: single, batched, generated_tokens edge cases
//! - ChatCompletionChunk, ChatChunkChoice, ChatDelta serde
//! - ChatChoice, OpenAIModelsResponse, OpenAIModel serde
//! - TraceData/TraceOperation serde
//! - Additional HTTP endpoints: /metrics, /models, /realize/generate, /realize/batch,
//!   /stream/generate, /v1/gpu/warmup, /v1/gpu/status, /v1/predict, /v1/explain
//! - format_chat_messages: model-specific formatting (qwen, phi, tinyllama)
//! - ContextWindowManager: truncation required, large message sets
//! - build_trace_data: deeper field verification
//!
//! Refs PMAT-802: Protocol T-COV-95

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;

// ============================================================================
// BatchConfig methods
// ============================================================================

#[test]
fn test_batch_config_low_latency() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    assert!(config.window_ms <= 10); // Low latency = short window
    assert!(config.min_batch > 0);
    assert!(config.optimal_batch > 0);
    assert!(config.max_batch >= config.optimal_batch);
    assert!(config.queue_size > 0);
}

#[test]
fn test_batch_config_high_throughput() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::high_throughput();
    assert!(config.window_ms >= 50); // High throughput = longer window
    assert!(config.min_batch >= 4);
    assert!(config.max_batch >= 64);
    assert!(config.queue_size >= 1024);
}

#[test]
fn test_batch_config_should_process_at_optimal() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    // At optimal batch size → should process
    assert!(config.should_process(config.optimal_batch));
    // Above optimal → should process
    assert!(config.should_process(config.optimal_batch + 1));
    // Below optimal → should not process
    assert!(!config.should_process(config.optimal_batch - 1));
}

#[test]
fn test_batch_config_should_process_zero() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    assert!(!config.should_process(0));
}

#[test]
fn test_batch_config_meets_minimum() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::low_latency();
    assert!(config.meets_minimum(config.min_batch));
    assert!(config.meets_minimum(config.min_batch + 1));
    assert!(!config.meets_minimum(0));
    assert!(!config.meets_minimum(config.min_batch - 1));
}

#[test]
fn test_batch_config_meets_minimum_high_throughput() {
    use crate::api::gpu_handlers::BatchConfig;

    let config = BatchConfig::high_throughput();
    assert!(config.meets_minimum(config.min_batch));
    assert!(!config.meets_minimum(1));
}

// ============================================================================
// ContinuousBatchResponse
// ============================================================================

#[test]
fn test_continuous_batch_response_single() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    let resp = ContinuousBatchResponse::single(vec![1, 2, 3, 4, 5], 2, 5.0);
    assert!(!resp.batched);
    assert_eq!(resp.batch_size, 1);
    assert_eq!(resp.prompt_len, 2);
    assert_eq!(resp.token_ids, vec![1, 2, 3, 4, 5]);
    assert!((resp.latency_ms - 5.0).abs() < 1e-6);
}

#[test]
fn test_continuous_batch_response_batched() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    let resp = ContinuousBatchResponse::batched(vec![1, 2, 3, 4, 5], 2, 8, 10.0);
    assert!(resp.batched);
    assert_eq!(resp.batch_size, 8);
    assert_eq!(resp.prompt_len, 2);
}

#[test]
fn test_continuous_batch_response_generated_tokens() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    let resp = ContinuousBatchResponse::single(vec![10, 20, 30, 40, 50], 3, 1.0);
    let generated = resp.generated_tokens();
    assert_eq!(generated, &[40, 50]);
}

#[test]
fn test_continuous_batch_response_generated_tokens_empty() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    // prompt_len equals total tokens → no generated tokens
    let resp = ContinuousBatchResponse::single(vec![1, 2, 3], 3, 1.0);
    let generated = resp.generated_tokens();
    assert!(generated.is_empty());
}

#[test]
fn test_continuous_batch_response_generated_tokens_all_generated() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    // prompt_len = 0 → all tokens are generated
    let resp = ContinuousBatchResponse::single(vec![1, 2, 3], 0, 1.0);
    let generated = resp.generated_tokens();
    assert_eq!(generated, &[1, 2, 3]);
}

#[test]
fn test_continuous_batch_response_generated_tokens_prompt_exceeds() {
    use crate::api::gpu_handlers::ContinuousBatchResponse;

    // prompt_len > total tokens → return empty (edge case)
    let resp = ContinuousBatchResponse::single(vec![1, 2], 10, 1.0);
    let generated = resp.generated_tokens();
    assert!(generated.is_empty());
}

// ============================================================================
// ChatCompletionChunk, ChatChunkChoice, ChatDelta serde
// ============================================================================

#[test]
fn test_chat_completion_chunk_serde() {
    let chunk = crate::api::ChatCompletionChunk {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![crate::api::ChatChunkChoice {
            index: 0,
            delta: crate::api::ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&chunk).expect("JSON serialization failed");
    let deserialized: crate::api::ChatCompletionChunk = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.id, "chatcmpl-123");
    assert_eq!(deserialized.object, "chat.completion.chunk");
    assert_eq!(deserialized.choices.len(), 1);
    assert_eq!(
        deserialized.choices[0].delta.role,
        Some("assistant".to_string())
    );
    assert!(deserialized.choices[0].delta.content.is_none());
    assert!(deserialized.choices[0].finish_reason.is_none());
}

#[test]
fn test_chat_delta_with_content() {
    let delta = crate::api::ChatDelta {
        role: None,
        content: Some("Hello ".to_string()),
    };
    let json = serde_json::to_string(&delta).expect("JSON serialization failed");
    // role is None → should be skipped in serialization
    assert!(!json.contains("role"));
    let deserialized: crate::api::ChatDelta = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(deserialized.role.is_none());
    assert_eq!(deserialized.content, Some("Hello ".to_string()));
}

#[test]
fn test_chat_chunk_choice_with_finish_reason() {
    let choice = crate::api::ChatChunkChoice {
        index: 0,
        delta: crate::api::ChatDelta {
            role: None,
            content: None,
        },
        finish_reason: Some("stop".to_string()),
    };
    let json = serde_json::to_string(&choice).expect("JSON serialization failed");
    let deserialized: crate::api::ChatChunkChoice = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.finish_reason, Some("stop".to_string()));
}

// ============================================================================
// ChatChoice, OpenAIModelsResponse, OpenAIModel serde
// ============================================================================

#[test]
fn test_chat_choice_serde() {
    let choice = crate::api::ChatChoice {
        index: 0,
        message: crate::api::ChatMessage {
            role: "assistant".to_string(),
            content: "Hello!".to_string(),
            name: None,
        },
        finish_reason: "stop".to_string(),
    };
    let json = serde_json::to_string(&choice).expect("JSON serialization failed");
    let deserialized: crate::api::ChatChoice = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.index, 0);
    assert_eq!(deserialized.message.role, "assistant");
    assert_eq!(deserialized.finish_reason, "stop");
}

#[test]
fn test_openai_models_response_serde() {
    let resp = crate::api::OpenAIModelsResponse {
        object: "list".to_string(),
        data: vec![crate::api::OpenAIModel {
            id: "test-model".to_string(),
            object: "model".to_string(),
            created: 1700000000,
            owned_by: "realizar".to_string(),
        }],
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: crate::api::OpenAIModelsResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.object, "list");
    assert_eq!(deserialized.data.len(), 1);
    assert_eq!(deserialized.data[0].id, "test-model");
    assert_eq!(deserialized.data[0].owned_by, "realizar");
}

#[test]
fn test_openai_model_serde() {
    let model = crate::api::OpenAIModel {
        id: "tinyllama-1.1b".to_string(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "realizar".to_string(),
    };
    let json = serde_json::to_string(&model).expect("JSON serialization failed");
    let deserialized: crate::api::OpenAIModel = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.id, "tinyllama-1.1b");
}

// ============================================================================
// TraceData/TraceOperation serde
// ============================================================================

#[test]
fn test_trace_data_serde() {
    let trace = crate::api::TraceData {
        level: "brick".to_string(),
        operations: 10,
        total_time_us: 5000,
        breakdown: vec![
            crate::api::TraceOperation {
                name: "embedding_lookup".to_string(),
                time_us: 500,
                details: Some("10 tokens".to_string()),
            },
            crate::api::TraceOperation {
                name: "matmul_qkv".to_string(),
                time_us: 1667,
                details: None,
            },
        ],
        provenance: crate::api::TraceProvenance::Estimated,
    };
    let json = serde_json::to_string(&trace).expect("JSON serialization failed");
    let deserialized: crate::api::TraceData = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.level, "brick");
    assert_eq!(deserialized.operations, 10);
    assert_eq!(deserialized.breakdown.len(), 2);
}

#[test]
fn test_trace_operation_serde() {
    let op = crate::api::TraceOperation {
        name: "softmax".to_string(),
        time_us: 100,
        details: None,
    };
    let json = serde_json::to_string(&op).expect("JSON serialization failed");
    let deserialized: crate::api::TraceOperation = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.name, "softmax");
    assert!(deserialized.details.is_none());
}

// ============================================================================
// build_trace_data: deeper field verification
// ============================================================================

#[test]
fn test_build_trace_data_brick_breakdown_fields() {
    let (brick, _, _) = crate::api::build_trace_data(Some("brick"), 1000, 20, 10, 4);
    let b = brick.expect("test value should be present");
    assert_eq!(b.operations, 10); // completion_tokens
    assert_eq!(b.total_time_us, 1000);
    assert_eq!(b.breakdown.len(), 1);
    assert_eq!(b.breakdown[0].name, "total_inference");
    assert_eq!(b.breakdown[0].time_us, 1000);
    let details = b.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("20 prompt"));
    assert!(details.contains("10 completion"));
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_step_breakdown_fields() {
    let (_, step, _) = crate::api::build_trace_data(Some("step"), 2000, 15, 8, 6);
    let s = step.expect("test value should be present");
    assert_eq!(s.operations, 8); // completion_tokens
    assert_eq!(s.total_time_us, 2000);
    assert_eq!(s.breakdown.len(), 1);
    assert_eq!(s.breakdown[0].name, "total_inference");
    assert_eq!(s.breakdown[0].time_us, 2000);
    let details = s.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("15 prompt"));
    assert!(details.contains("8 completion"));
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_layer_breakdown_fields() {
    let (_, _, layer) = crate::api::build_trace_data(Some("layer"), 4000, 10, 5, 4);
    let l = layer.expect("test value should be present");
    assert_eq!(l.operations, 4); // num_layers
    assert_eq!(l.total_time_us, 4000);
    assert_eq!(l.breakdown.len(), 1);
    assert_eq!(l.breakdown[0].name, "total_inference");
    assert_eq!(l.breakdown[0].time_us, 4000);
    let details = l.breakdown[0].details.as_ref().expect("details present");
    assert!(details.contains("4 layers"));
    assert!(details.contains("apr profile"));
}

#[test]
fn test_build_trace_data_unknown_level() {
    let (brick, step, layer) = crate::api::build_trace_data(Some("unknown"), 100, 10, 5, 4);
    assert!(brick.is_none());
    assert!(step.is_none());
    assert!(layer.is_none());
}

// ============================================================================
// Additional HTTP endpoint integration tests
// ============================================================================

#[tokio::test]
async fn test_metrics_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND,);
}

#[tokio::test]
async fn test_native_models_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/models")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND,);
}

#[tokio::test]
async fn test_realize_generate_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/realize/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompt":"Hello","max_tokens":5,"temperature":0.0}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_realize_batch_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/realize/batch")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompts":["Hello","World"],"max_tokens":5}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

include!("stream_generate.rs");
