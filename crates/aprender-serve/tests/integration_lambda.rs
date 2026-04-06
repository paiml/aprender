//! Integration Tests for Lambda Serving Infrastructure
//!
//! Per `docs/specifications/serve-deploy-apr.md` Section 11.3:
//! End-to-end integration tests covering the full inference pipeline.
//!
//! ## Test Coverage
//!
//! - Lambda handler + Target detection
//! - Batch inference pipeline
//! - Metrics collection and export
//! - Error handling flow

#![cfg(feature = "lambda")]

use realizar::{
    lambda::{BatchLambdaRequest, LambdaError, LambdaHandler, LambdaMetrics, LambdaRequest},
    target::{DeployTarget, DockerConfig, TargetFeature, WasmConfig},
};

// =============================================================================
// Integration Test: Lambda Handler + Target Detection
// =============================================================================

/// Test that Lambda handler respects target capabilities
#[test]
#[ignore = "Mock APR format"]
fn test_lambda_handler_with_target_detection() {
    // Detect current target
    let target = DeployTarget::detect();
    let caps = target.capabilities();

    // Create handler with valid .apr model
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00integration_test_model";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Verify handler works regardless of target
    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0],
        model_id: None,
    };

    let response = handler.handle(&request).expect("inference failed");

    // Verify response
    assert!((response.prediction - 6.0).abs() < 0.001); // Sum of features
    assert!(response.latency_ms >= 0.0);

    // Log target info for debugging
    println!(
        "Target: {}, SIMD: {}, Threads: {}",
        target.name(),
        caps.supports_simd,
        caps.supports_threads
    );
}

/// Test Lambda handler behaves correctly based on target capabilities
#[test]
#[ignore = "Mock APR format"]
fn test_lambda_capabilities_affect_behavior() {
    let target = DeployTarget::detect();

    // Lambda and Native both support SIMD
    if target.supports(TargetFeature::Simd) {
        // SIMD-capable targets should have faster inference (verified by benchmark)
        let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00simd_test";
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        // Run multiple inferences to warm up
        let request = LambdaRequest {
            features: vec![1.0; 100], // Larger feature vector
            model_id: None,
        };

        for _ in 0..10 {
            let _ = handler.handle(&request);
        }

        // Final inference should be fast
        let response = handler.handle(&request).expect("test");
        assert!(
            response.latency_ms < 100.0,
            "SIMD inference should be fast: {} ms",
            response.latency_ms
        );
    }
}

// =============================================================================
// Integration Test: Batch Inference Pipeline
// =============================================================================

/// Test full batch inference pipeline
#[test]
#[ignore = "Mock APR format"]
fn test_batch_inference_pipeline() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00batch_test_model";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Create batch with varying inputs
    let batch_request = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0, 2.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![3.0, 4.0, 5.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![10.0],
                model_id: None,
            },
        ],
        max_parallelism: Some(2),
    };

    let response = handler
        .handle_batch(&batch_request)
        .expect("batch inference failed");

    // Verify all predictions
    assert_eq!(response.predictions.len(), 3);
    assert_eq!(response.success_count, 3);
    assert_eq!(response.error_count, 0);

    // Verify individual predictions (mock returns sum)
    assert!((response.predictions[0].prediction - 3.0).abs() < 0.001);
    assert!((response.predictions[1].prediction - 12.0).abs() < 0.001);
    assert!((response.predictions[2].prediction - 10.0).abs() < 0.001);

    // Verify latency is reasonable
    assert!(response.total_latency_ms >= 0.0);
    assert!(
        response.total_latency_ms < 1000.0,
        "Batch should complete in <1s"
    );
}

/// Test batch inference with mixed success/failure
#[test]
#[ignore = "Mock APR format"]
fn test_batch_inference_error_handling() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00error_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

    let batch_request = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![], // Will fail - empty features
                model_id: None,
            },
            LambdaRequest {
                features: vec![2.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![], // Will fail
                model_id: None,
            },
            LambdaRequest {
                features: vec![3.0],
                model_id: None,
            },
        ],
        max_parallelism: None,
    };

    let response = handler.handle_batch(&batch_request).expect("test");

    // Verify counts
    assert_eq!(response.predictions.len(), 5);
    assert_eq!(response.success_count, 3);
    assert_eq!(response.error_count, 2);

    // Verify successful predictions
    assert!((response.predictions[0].prediction - 1.0).abs() < 0.001);
    assert!(response.predictions[1].prediction.is_nan()); // Error placeholder
    assert!((response.predictions[2].prediction - 2.0).abs() < 0.001);
    assert!(response.predictions[3].prediction.is_nan()); // Error placeholder
    assert!((response.predictions[4].prediction - 3.0).abs() < 0.001);
}

// =============================================================================
// Integration Test: Metrics Collection
// =============================================================================

/// Test metrics collection across multiple requests
#[test]
#[ignore = "Mock APR format"]
fn test_metrics_collection_integration() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00metrics_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("test");
    let mut metrics = LambdaMetrics::new();

    // Simulate request flow
    let request = LambdaRequest {
        features: vec![1.0, 2.0],
        model_id: None,
    };

    // First request (cold start)
    let response1 = handler.handle(&request).expect("test");
    metrics.record_success(response1.latency_ms, response1.cold_start);

    assert_eq!(metrics.requests_total, 1);
    assert_eq!(metrics.cold_starts, 1);

    // Subsequent requests (warm)
    for _ in 0..9 {
        let response = handler.handle(&request).expect("test");
        metrics.record_success(response.latency_ms, response.cold_start);
    }

    assert_eq!(metrics.requests_total, 10);
    assert_eq!(metrics.requests_success, 10);
    assert_eq!(metrics.cold_starts, 1); // Only first was cold
    assert!(metrics.avg_latency_ms() > 0.0);

    // Verify Prometheus export
    let prom = metrics.to_prometheus();
    assert!(prom.contains("lambda_requests_total 10"));
    assert!(prom.contains("lambda_cold_starts 1"));
}

/// Test metrics with batch requests
#[test]
#[ignore = "Mock APR format"]
fn test_metrics_batch_integration() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00batch_metrics";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("test");
    let mut metrics = LambdaMetrics::new();

    let batch = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![],
                model_id: None,
            }, // Will fail
            LambdaRequest {
                features: vec![2.0],
                model_id: None,
            },
        ],
        max_parallelism: None,
    };

    let response = handler.handle_batch(&batch).expect("test");
    metrics.record_batch(
        response.success_count,
        response.error_count,
        response.total_latency_ms,
    );

    assert_eq!(metrics.batch_requests, 1);
    assert_eq!(metrics.requests_total, 3);
    assert_eq!(metrics.requests_success, 2);
    assert_eq!(metrics.requests_failed, 1);

    let prom = metrics.to_prometheus();
    assert!(prom.contains("lambda_batch_requests 1"));
}

// =============================================================================
// Integration Test: Target Configuration
// =============================================================================

/// Test Docker config generates valid Dockerfile
#[test]
fn test_docker_config_generation() {
    let config = DockerConfig::default();
    let dockerfile = config.generate_dockerfile();

    // Verify Dockerfile structure
    assert!(dockerfile.contains("FROM rust:1.83 AS builder"));
    assert!(dockerfile.contains("cargo build --release"));
    assert!(dockerfile.contains("distroless"));
    assert!(dockerfile.contains("EXPOSE 8080"));
    assert!(dockerfile.contains("ENTRYPOINT"));

    // Verify ARM64 variant
    let arm_config = DockerConfig::arm64();
    let arm_dockerfile = arm_config.generate_dockerfile();
    assert!(arm_dockerfile.contains("aarch64"));
}

/// Test WASM config generates valid build commands
#[test]
fn test_wasm_config_generation() {
    let config = WasmConfig::default();

    // Verify build command
    let cmd = config.build_command();
    assert!(cmd.contains("wasm-pack build"));
    assert!(cmd.contains("--target web"));
    assert!(cmd.contains("--release"));

    // Verify Cloudflare Worker template
    let worker = config.cloudflare_worker_template();
    assert!(worker.contains("import init"));
    assert!(worker.contains("async fetch"));
    assert!(worker.contains("predict"));
}

// =============================================================================
// Integration Test: Error Flow
// =============================================================================

/// Test error handling across the pipeline
#[test]
fn test_error_flow_integration() {
    // Test invalid model
    let invalid_model: &'static [u8] = b"INVALID_MAGIC";
    let result = LambdaHandler::from_bytes(invalid_model);
    assert!(matches!(result, Err(LambdaError::InvalidMagic { .. })));

    // Test empty model
    let empty_model: &'static [u8] = b"";
    let result = LambdaHandler::from_bytes(empty_model);
    assert!(matches!(result, Err(LambdaError::EmptyModel)));

    // Test empty features
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00error_flow";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

    let empty_request = LambdaRequest {
        features: vec![],
        model_id: None,
    };
    let result = handler.handle(&empty_request);
    assert!(matches!(result, Err(LambdaError::EmptyFeatures)));

    // Test empty batch
    let empty_batch = BatchLambdaRequest {
        instances: vec![],
        max_parallelism: None,
    };
    let result = handler.handle_batch(&empty_batch);
    assert!(matches!(result, Err(LambdaError::EmptyBatch)));
}

// =============================================================================
// Integration Test: Cold Start Benchmark
// =============================================================================

/// Test cold start metrics are captured correctly
#[test]
#[ignore = "Mock APR format"]
fn test_cold_start_metrics_integration() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00cold_start_test";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

    // Before first invocation
    assert!(handler.is_cold_start());
    assert!(handler.cold_start_metrics().is_none());

    // First invocation
    let request = LambdaRequest {
        features: vec![1.0],
        model_id: None,
    };
    let response = handler.handle(&request).expect("test");
    assert!(response.cold_start);

    // After first invocation
    assert!(!handler.is_cold_start());
    let metrics = handler.cold_start_metrics().expect("metrics should exist");
    assert!(metrics.total_ms >= 0.0);
    assert!(metrics.first_inference_ms >= 0.0);

    // Subsequent invocations are not cold
    let response2 = handler.handle(&request).expect("test");
    assert!(!response2.cold_start);
}

// =============================================================================
// Integration Test: Performance Comparison (Basic)
// =============================================================================

/// Basic performance comparison across configurations
#[test]
#[ignore = "Mock APR format"]
fn test_performance_comparison() {
    let model_bytes: &'static [u8] = b"APR\0\x01\x00\x00\x00perf_comparison";
    let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

    let small_request = LambdaRequest {
        features: vec![1.0; 10],
        model_id: None,
    };

    let large_request = LambdaRequest {
        features: vec![1.0; 1000],
        model_id: None,
    };

    // Warm up
    for _ in 0..5 {
        let _ = handler.handle(&small_request);
        let _ = handler.handle(&large_request);
    }

    // Measure small request
    let small_response = handler.handle(&small_request).expect("test");

    // Measure large request
    let large_response = handler.handle(&large_request).expect("test");

    // Both should complete quickly (mock inference)
    assert!(small_response.latency_ms < 10.0);
    assert!(large_response.latency_ms < 10.0);

    // Verify predictions
    assert!((small_response.prediction - 10.0).abs() < 0.001);
    assert!((large_response.prediction - 1000.0).abs() < 0.001);
}
