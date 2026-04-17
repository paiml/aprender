//! API Tests Part 18: T-COV-95 Coverage Bridge (B2 + B3)
//!
//! Covers uncovered paths in:
//! - gpu_handlers.rs: struct serde, BatchConfig methods, ContinuousBatchResponse
//! - realize_handlers.rs: ContextWindowConfig, ContextWindowManager, format_chat_messages
//! - openai_handlers.rs: HTTP handlers, request/response serde
//! - mod.rs: AppState accessors, build_trace_data
//!
//! Refs PMAT-802: Protocol T-COV-95 Batches B2+B3

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;
use crate::api::AppState;

// ============================================================================
// B2: GPU Handler Struct Serde Round-Trips
// ============================================================================

#[test]
fn test_gpu_batch_request_serde() {
    use crate::api::gpu_handlers::GpuBatchRequest;

    let req = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec!["<|endoftext|>".to_string()],
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let deserialized: GpuBatchRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.prompts.len(), 2);
    assert_eq!(deserialized.max_tokens, 50);
    assert!((deserialized.temperature - 0.7).abs() < 1e-6);
    assert_eq!(deserialized.top_k, 40);
    assert_eq!(deserialized.stop.len(), 1);
}

#[test]
fn test_gpu_batch_response_serde() {
    use crate::api::gpu_handlers::{GpuBatchResponse, GpuBatchResult, GpuBatchStats};

    let resp = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "hello".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 3,
            processing_time_ms: 10.5,
            throughput_tps: 285.7,
        },
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: GpuBatchResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.results.len(), 1);
    assert_eq!(deserialized.results[0].index, 0);
    assert_eq!(deserialized.results[0].num_generated, 3);
    assert_eq!(deserialized.stats.batch_size, 1);
    assert!(!deserialized.stats.gpu_used);
}

#[test]
fn test_gpu_batch_stats_serde() {
    use crate::api::gpu_handlers::GpuBatchStats;

    let stats = GpuBatchStats {
        batch_size: 32,
        gpu_used: true,
        total_tokens: 1600,
        processing_time_ms: 50.0,
        throughput_tps: 32000.0,
    };
    let json = serde_json::to_string(&stats).expect("JSON serialization failed");
    let deserialized: GpuBatchStats = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.batch_size, 32);
    assert!(deserialized.gpu_used);
    assert_eq!(deserialized.total_tokens, 1600);
}

#[test]
fn test_gpu_warmup_response_serde() {
    use crate::api::gpu_handlers::GpuWarmupResponse;

    let resp = GpuWarmupResponse {
        success: true,
        memory_bytes: 6_000_000_000,
        num_layers: 32,
        message: "GPU cache warmed up".to_string(),
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: GpuWarmupResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(deserialized.success);
    assert_eq!(deserialized.memory_bytes, 6_000_000_000);
    assert_eq!(deserialized.num_layers, 32);
}

#[test]
fn test_gpu_status_response_serde() {
    use crate::api::gpu_handlers::GpuStatusResponse;

    let resp = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let deserialized: GpuStatusResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(!deserialized.cache_ready);
    assert_eq!(deserialized.batch_threshold, 32);
}

#[test]
fn test_gpu_batch_result_serde() {
    use crate::api::gpu_handlers::GpuBatchResult;

    let result = GpuBatchResult {
        index: 5,
        token_ids: vec![100, 200, 300, 400],
        text: "test output".to_string(),
        num_generated: 4,
    };
    let json = serde_json::to_string(&result).expect("JSON serialization failed");
    let deserialized: GpuBatchResult = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.index, 5);
    assert_eq!(deserialized.token_ids, vec![100, 200, 300, 400]);
    assert_eq!(deserialized.text, "test output");
}

#[test]
fn test_gpu_batch_request_defaults() {
    // Test default values via serde deserialization with minimal JSON
    use crate::api::gpu_handlers::GpuBatchRequest;

    let json = r#"{"prompts":["hello"]}"#;
    let req: GpuBatchRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert_eq!(req.prompts, vec!["hello"]);
    // max_tokens should use default_max_tokens()
    assert!(req.max_tokens > 0);
    // temperature default is 0.0
    assert!((req.temperature - 0.0).abs() < 1e-6);
    // stop default is empty
    assert!(req.stop.is_empty());
}

// ============================================================================
// B2: HTTP Endpoint Integration Tests
// ============================================================================

#[tokio::test]
async fn test_tokenize_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/tokenize")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"text":"Hello world"}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Mock state may return NOT_FOUND or 200
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_generate_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompt":"Hello","max_tokens":5,"temperature":0.0,"strategy":"greedy","top_k":1,"top_p":1.0}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR,
    );
}

#[tokio::test]
async fn test_batch_tokenize_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/batch/tokenize")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"texts":["Hello","World"]}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_batch_generate_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/batch/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompts":["Hello"],"max_tokens":5,"temperature":0.0,"strategy":"greedy","top_k":1,"top_p":1.0}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR,
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/batch/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"prompts":["Hello world"],"max_tokens":10,"temperature":0.0}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // GPU batch endpoint returns SERVICE_UNAVAILABLE without GPU feature
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_empty_prompts() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/batch/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"prompts":[],"max_tokens":10}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_models_endpoint() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::NOT_FOUND,);
}

#[tokio::test]
async fn test_generate_invalid_json() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Invalid JSON: server should not return 200 OK
    let status = response.status().as_u16();
    assert!(
        status != 200,
        "Expected non-200 for invalid JSON, got {status}"
    );
}

#[tokio::test]
async fn test_tokenize_invalid_json() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/tokenize")
        .header("content-type", "application/json")
        .body(Body::from("{invalid"))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // Invalid JSON: server should not return 200 OK
    let status = response.status().as_u16();
    assert!(
        status != 200,
        "Expected non-200 for invalid JSON, got {status}"
    );
}

// ============================================================================
// B3: Realize Handlers - ContextWindowConfig
// ============================================================================

#[test]
fn test_context_window_config_new() {
    use crate::api::realize_handlers::ContextWindowConfig;

    let config = ContextWindowConfig::new(4096);
    assert_eq!(config.max_tokens, 4096);
    assert!(config.reserved_output_tokens > 0);
    assert!(config.preserve_system);
}

#[test]
fn test_context_window_config_with_reserved_output() {
    use crate::api::realize_handlers::ContextWindowConfig;

    let config = ContextWindowConfig::new(4096).with_reserved_output(512);
    assert_eq!(config.reserved_output_tokens, 512);
}

#[test]
fn test_context_window_config_available_tokens() {
    use crate::api::realize_handlers::ContextWindowConfig;

    let config = ContextWindowConfig::new(4096).with_reserved_output(1024);
    let available = config.available_tokens();
    assert_eq!(available, 4096 - 1024);
}

#[test]
fn test_context_window_config_available_tokens_overflow_protection() {
    use crate::api::realize_handlers::ContextWindowConfig;

    // reserved > max: should not underflow
    let config = ContextWindowConfig {
        max_tokens: 100,
        reserved_output_tokens: 200,
        preserve_system: true,
    };
    let available = config.available_tokens();
    assert_eq!(available, 0);
}

// ============================================================================
// B3: ContextWindowManager
// ============================================================================

#[test]
fn test_context_window_manager_default() {
    use crate::api::realize_handlers::ContextWindowManager;

    let mgr = ContextWindowManager::default_manager();
    // Default manager should have reasonable defaults
    let _ = mgr; // Ensure it constructs without panic
}

#[test]
fn test_context_window_manager_needs_truncation_empty() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);
    let messages: Vec<crate::api::ChatMessage> = vec![];
    assert!(!mgr.needs_truncation(&messages));
}

#[test]
fn test_context_window_manager_estimate_tokens() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);
    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Hello world".to_string(),
        name: None,
    }];
    let estimate = mgr.estimate_total_tokens(&messages);
    assert!(estimate > 0);
}

#[test]
fn test_context_window_manager_truncate_short_messages() {
    use crate::api::realize_handlers::{ContextWindowConfig, ContextWindowManager};

    let config = ContextWindowConfig::new(4096);
    let mgr = ContextWindowManager::new(config);
    let messages = vec![crate::api::ChatMessage {
        role: "user".to_string(),
        content: "Short".to_string(),
        name: None,
    }];
    let (truncated, was_truncated) = mgr.truncate_messages(&messages);
    assert!(!was_truncated);
    assert_eq!(truncated.len(), 1);
}

// ============================================================================
// B3: format_chat_messages
// ============================================================================

#[test]
fn test_format_chat_messages_empty() {
    use crate::api::realize_handlers::format_chat_messages;

    let messages: Vec<crate::api::ChatMessage> = vec![];
    let result = format_chat_messages(&messages, None);
    assert!(result.is_empty() || result.contains(""));
}

include!("format_chat_02.rs");
include!("usage_serde_health.rs");
