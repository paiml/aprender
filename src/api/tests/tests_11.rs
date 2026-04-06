//! API Tests Part 11: GPU Handlers Coverage
//!
//! Tests for gpu_handlers.rs to improve coverage.
//! Focus: GPU batch requests, warmup, status, request validation, error paths.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;
use crate::api::{
    BatchGenerateResponse, BatchTokenizeResponse, GenerateResponse, GpuBatchRequest,
    GpuBatchResponse, GpuBatchResult, GpuBatchStats, GpuStatusResponse, GpuWarmupResponse,
    ModelsResponse, TokenizeResponse,
};

// =============================================================================
// GpuBatchRequest Tests
// =============================================================================

#[test]
fn test_gpu_batch_request_serialization() {
    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec!["</s>".to_string()],
    };

    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("World"));
    assert!(json.contains("50"));
    assert!(json.contains("0.7"));
    assert!(json.contains("40"));
    assert!(json.contains("</s>"));
}

#[test]
fn test_gpu_batch_request_deserialization_minimal() {
    let json = r#"{"prompts": ["Test prompt"]}"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(request.prompts.len(), 1);
    assert_eq!(request.prompts[0], "Test prompt");
    // Check defaults
    assert_eq!(request.max_tokens, 50); // default_max_tokens
    assert_eq!(request.temperature, 0.0); // default
    assert_eq!(request.top_k, 50); // default_top_k
    assert!(request.stop.is_empty()); // default
}

#[test]
fn test_gpu_batch_request_deserialization_full() {
    let json = r#"{
        "prompts": ["prompt1", "prompt2", "prompt3"],
        "max_tokens": 100,
        "temperature": 0.9,
        "top_k": 20,
        "stop": ["stop1", "stop2"]
    }"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("deserialize");

    assert_eq!(request.prompts.len(), 3);
    assert_eq!(request.max_tokens, 100);
    assert!((request.temperature - 0.9).abs() < 0.01);
    assert_eq!(request.top_k, 20);
    assert_eq!(request.stop.len(), 2);
}

#[test]
fn test_gpu_batch_request_clone() {
    let request = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 25,
        temperature: 0.5,
        top_k: 30,
        stop: vec![],
    };

    let cloned = request.clone();
    assert_eq!(cloned.prompts, request.prompts);
    assert_eq!(cloned.max_tokens, request.max_tokens);
    assert_eq!(cloned.temperature, request.temperature);
    assert_eq!(cloned.top_k, request.top_k);
}

#[test]
fn test_gpu_batch_request_debug() {
    let request = GpuBatchRequest {
        prompts: vec!["debug test".to_string()],
        max_tokens: 10,
        temperature: 0.1,
        top_k: 5,
        stop: vec![],
    };

    let debug = format!("{:?}", request);
    assert!(debug.contains("GpuBatchRequest"));
    assert!(debug.contains("debug test"));
}

// =============================================================================
// GpuBatchResponse Tests
// =============================================================================

#[test]
fn test_gpu_batch_response_serialization() {
    let response = GpuBatchResponse {
        results: vec![
            GpuBatchResult {
                index: 0,
                token_ids: vec![1, 2, 3],
                text: "Hello".to_string(),
                num_generated: 3,
            },
            GpuBatchResult {
                index: 1,
                token_ids: vec![4, 5, 6, 7],
                text: "World".to_string(),
                num_generated: 4,
            },
        ],
        stats: GpuBatchStats {
            batch_size: 2,
            gpu_used: true,
            total_tokens: 7,
            processing_time_ms: 15.5,
            throughput_tps: 451.6,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("World"));
    assert!(json.contains("batch_size"));
    assert!(json.contains("gpu_used"));
    assert!(json.contains("throughput_tps"));
}

#[test]
fn test_gpu_batch_response_deserialization() {
    let json = r#"{
        "results": [
            {"index": 0, "token_ids": [1, 2], "text": "test", "num_generated": 2}
        ],
        "stats": {
            "batch_size": 1,
            "gpu_used": false,
            "total_tokens": 2,
            "processing_time_ms": 10.0,
            "throughput_tps": 200.0
        }
    }"#;
    let response: GpuBatchResponse = serde_json::from_str(json).expect("deserialize");

    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].index, 0);
    assert_eq!(response.stats.batch_size, 1);
    assert!(!response.stats.gpu_used);
}

#[test]
fn test_gpu_batch_response_clone() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1],
            text: "x".to_string(),
            num_generated: 1,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 1,
            processing_time_ms: 5.0,
            throughput_tps: 200.0,
        },
    };

    let cloned = response.clone();
    assert_eq!(cloned.results.len(), response.results.len());
    assert_eq!(cloned.stats.batch_size, response.stats.batch_size);
}

#[test]
fn test_gpu_batch_response_debug() {
    let response = GpuBatchResponse {
        results: vec![],
        stats: GpuBatchStats {
            batch_size: 0,
            gpu_used: false,
            total_tokens: 0,
            processing_time_ms: 0.0,
            throughput_tps: 0.0,
        },
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuBatchResponse"));
}

// =============================================================================
// GpuBatchResult Tests
// =============================================================================

#[test]
fn test_gpu_batch_result_serialization() {
    let result = GpuBatchResult {
        index: 5,
        token_ids: vec![10, 20, 30, 40, 50],
        text: "Generated text here".to_string(),
        num_generated: 5,
    };

    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("\"index\":5"));
    assert!(json.contains("Generated text here"));
    assert!(json.contains("num_generated"));
}

#[test]
fn test_gpu_batch_result_deserialization() {
    let json = r#"{"index": 3, "token_ids": [100, 200], "text": "hi", "num_generated": 2}"#;
    let result: GpuBatchResult = serde_json::from_str(json).expect("deserialize");

    assert_eq!(result.index, 3);
    assert_eq!(result.token_ids, vec![100, 200]);
    assert_eq!(result.text, "hi");
    assert_eq!(result.num_generated, 2);
}

#[test]
fn test_gpu_batch_result_clone() {
    let result = GpuBatchResult {
        index: 1,
        token_ids: vec![1, 2, 3],
        text: "clone test".to_string(),
        num_generated: 3,
    };

    let cloned = result.clone();
    assert_eq!(cloned.index, result.index);
    assert_eq!(cloned.token_ids, result.token_ids);
    assert_eq!(cloned.text, result.text);
}

#[test]
fn test_gpu_batch_result_debug() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![],
        text: "debug".to_string(),
        num_generated: 0,
    };

    let debug = format!("{:?}", result);
    assert!(debug.contains("GpuBatchResult"));
}

// =============================================================================
// GpuBatchStats Tests
// =============================================================================

#[test]
fn test_gpu_batch_stats_serialization() {
    let stats = GpuBatchStats {
        batch_size: 32,
        gpu_used: true,
        total_tokens: 1024,
        processing_time_ms: 100.5,
        throughput_tps: 10189.05,
    };

    let json = serde_json::to_string(&stats).expect("serialize");
    assert!(json.contains("32"));
    assert!(json.contains("true"));
    assert!(json.contains("1024"));
    assert!(json.contains("100.5"));
    assert!(json.contains("10189.05"));
}

#[test]
fn test_gpu_batch_stats_deserialization() {
    let json = r#"{
        "batch_size": 64,
        "gpu_used": true,
        "total_tokens": 512,
        "processing_time_ms": 50.0,
        "throughput_tps": 10240.0
    }"#;
    let stats: GpuBatchStats = serde_json::from_str(json).expect("deserialize");

    assert_eq!(stats.batch_size, 64);
    assert!(stats.gpu_used);
    assert_eq!(stats.total_tokens, 512);
    assert!((stats.processing_time_ms - 50.0).abs() < 0.01);
}

#[test]
fn test_gpu_batch_stats_clone() {
    let stats = GpuBatchStats {
        batch_size: 16,
        gpu_used: false,
        total_tokens: 256,
        processing_time_ms: 25.0,
        throughput_tps: 10240.0,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.batch_size, stats.batch_size);
    assert_eq!(cloned.gpu_used, stats.gpu_used);
    assert_eq!(cloned.total_tokens, stats.total_tokens);
}

#[test]
fn test_gpu_batch_stats_debug() {
    let stats = GpuBatchStats {
        batch_size: 8,
        gpu_used: true,
        total_tokens: 128,
        processing_time_ms: 12.5,
        throughput_tps: 10240.0,
    };

    let debug = format!("{:?}", stats);
    assert!(debug.contains("GpuBatchStats"));
    assert!(debug.contains("batch_size"));
}

// =============================================================================
// GpuWarmupResponse Tests
// =============================================================================

#[test]
fn test_gpu_warmup_response_serialization() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 1_073_741_824, // 1GB
        num_layers: 32,
        message: "GPU cache warmed up successfully".to_string(),
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("1073741824"));
    assert!(json.contains("32"));
    assert!(json.contains("GPU cache warmed up successfully"));
}

#[test]
fn test_gpu_warmup_response_deserialization() {
    let json = r#"{
        "success": false,
        "memory_bytes": 0,
        "num_layers": 0,
        "message": "Warmup failed"
    }"#;
    let response: GpuWarmupResponse = serde_json::from_str(json).expect("deserialize");

    assert!(!response.success);
    assert_eq!(response.memory_bytes, 0);
    assert_eq!(response.num_layers, 0);
    assert_eq!(response.message, "Warmup failed");
}

#[test]
fn test_gpu_warmup_response_clone() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 500_000_000,
        num_layers: 24,
        message: "OK".to_string(),
    };

    let cloned = response.clone();
    assert_eq!(cloned.success, response.success);
    assert_eq!(cloned.memory_bytes, response.memory_bytes);
    assert_eq!(cloned.num_layers, response.num_layers);
    assert_eq!(cloned.message, response.message);
}

#[test]
fn test_gpu_warmup_response_debug() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 100,
        num_layers: 1,
        message: "test".to_string(),
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuWarmupResponse"));
}

// =============================================================================
// GpuStatusResponse Tests
// =============================================================================

#[test]
fn test_gpu_status_response_serialization() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 2_147_483_648, // 2GB
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("cache_ready"));
    assert!(json.contains("true"));
    assert!(json.contains("2147483648"));
    assert!(json.contains("batch_threshold"));
    assert!(json.contains("recommended_min_batch"));
}

#[test]
fn test_gpu_status_response_deserialization() {
    let json = r#"{
        "cache_ready": false,
        "cache_memory_bytes": 0,
        "batch_threshold": 32,
        "recommended_min_batch": 32
    }"#;
    let response: GpuStatusResponse = serde_json::from_str(json).expect("deserialize");

    assert!(!response.cache_ready);
    assert_eq!(response.cache_memory_bytes, 0);
    assert_eq!(response.batch_threshold, 32);
    assert_eq!(response.recommended_min_batch, 32);
}

#[test]
fn test_gpu_status_response_clone() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1_000_000,
        batch_threshold: 16,
        recommended_min_batch: 16,
    };

    let cloned = response.clone();
    assert_eq!(cloned.cache_ready, response.cache_ready);
    assert_eq!(cloned.cache_memory_bytes, response.cache_memory_bytes);
    assert_eq!(cloned.batch_threshold, response.batch_threshold);
}

include!("gpu_status.rs");
include!("gpu_batch.rs");
include!("batch_generate_02.rs");
