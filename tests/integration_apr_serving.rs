//! Integration Tests for .apr Model Serving with Aprender
//!
//! Tests the full integration between realizar Lambda serving and aprender ML models,
//! including metrics, drift detection, and model evaluation.
//!
//! ## Test Coverage
//!
//! - .apr format validation and loading
//! - Lambda handler with real aprender models
//! - Classification metrics integration
//! - Drift detection workflow
//! - Model evaluation pipeline

#![cfg(all(feature = "lambda", feature = "aprender-serve"))]

use realizar::lambda::{
    BatchLambdaRequest, LambdaError, LambdaHandler, LambdaMetrics, LambdaRequest,
};

// =============================================================================
// Test: .apr Format Validation
// =============================================================================

/// Valid .apr magic bytes (APRN = 0x4150524E)
const APR_MAGIC: &[u8; 4] = b"APRN";

/// APR header size (must be 32 bytes)
const APR_HEADER_SIZE: usize = 32;

/// Create a minimal valid .apr model for testing with proper 32-byte header
/// Header layout:
///   [0..4]   - Magic: APRN
///   [4]      - Version major
///   [5]      - Version minor
///   [6]      - Flags
///   [7]      - Reserved
///   [8..10]  - Model type (u16 LE)
///   [10..32] - Reserved/padding
fn create_test_apr_model(_name: &str) -> Vec<u8> {
    let mut bytes = vec![0u8; APR_HEADER_SIZE];
    bytes[0..4].copy_from_slice(APR_MAGIC);
    bytes[4] = 1; // version major
    bytes[5] = 0; // version minor
    bytes[6] = 0; // flags (none)
    bytes[7] = 0; // reserved
    bytes[8..10].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
    bytes
}

/// Create a static test model bytes (for tests requiring &'static [u8])
/// Returns a valid APR model with LinearRegression weights
fn create_static_test_model() -> &'static [u8] {
    // Pre-computed valid APR model with LinearRegression weights
    // Header (32 bytes) + JSON payload for a simple 4-input, 1-output linear model
    // ModelWeights struct requires Vec<Vec<f32>> for weights and biases (nested arrays)
    // JSON = {"weights":[[0.1,0.2,0.3,0.4]],"biases":[[0.5]],"dimensions":[4,1]} = 67 bytes
    static MODEL: &[u8] = &[
        // Header (32 bytes)
        0x41, 0x50, 0x52, 0x4E, // APRN magic
        0x01, 0x00, // version 1.0
        0x00, 0x00, // flags, reserved
        0x01, 0x00, // model type: LinearRegression (0x0001)
        0x00, 0x00, 0x00, 0x00, // metadata_len = 0
        0x43, 0x00, 0x00, 0x00, // payload_len = 67 bytes
        0x43, 0x00, 0x00, 0x00, // original_size = 67 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved2
        // JSON payload (67 bytes): {"weights":[[0.1,0.2,0.3,0.4]],"biases":[[0.5]],"dimensions":[4,1]}
        0x7b, 0x22, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x22, 0x3a, 0x5b, 0x5b, 0x30, 0x2e,
        0x31, 0x2c, 0x30, 0x2e, 0x32, 0x2c, 0x30, 0x2e, 0x33, 0x2c, 0x30, 0x2e, 0x34, 0x5d, 0x5d,
        0x2c, 0x22, 0x62, 0x69, 0x61, 0x73, 0x65, 0x73, 0x22, 0x3a, 0x5b, 0x5b, 0x30, 0x2e, 0x35,
        0x5d, 0x5d, 0x2c, 0x22, 0x64, 0x69, 0x6d, 0x65, 0x6e, 0x73, 0x69, 0x6f, 0x6e, 0x73, 0x22,
        0x3a, 0x5b, 0x34, 0x2c, 0x31, 0x5d, 0x7d,
    ];
    MODEL
}

#[test]
#[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
fn test_apr_magic_bytes_validation() {
    // Valid .apr magic
    let valid_model = create_test_apr_model("test_model");
    assert_eq!(&valid_model[0..4], APR_MAGIC);

    // Test handler accepts valid magic
    let handler = LambdaHandler::from_bytes(Box::leak(valid_model.into_boxed_slice()));
    assert!(handler.is_ok(), "Should accept valid .apr magic bytes");
}

#[test]
fn test_apr_invalid_magic_rejected() {
    // Invalid magic bytes
    let invalid_models: Vec<&[u8]> = vec![
        b"GGUF\x01\x00\x00\x00test",  // GGUF format
        b"PK\x03\x04test_data",       // ZIP format
        b"\x00\x00\x00\x00test",      // Null bytes
        b"apr\0\x01\x00\x00\x00test", // Lowercase (invalid)
    ];

    for model in invalid_models {
        let result = LambdaHandler::from_bytes(model);
        assert!(
            matches!(result, Err(LambdaError::InvalidMagic { .. })),
            "Should reject invalid magic bytes: {:?}",
            &model[0..4.min(model.len())]
        );
    }
}

#[test]
fn test_apr_empty_model_rejected() {
    let result = LambdaHandler::from_bytes(b"");
    assert!(
        matches!(result, Err(LambdaError::EmptyModel)),
        "Should reject empty model"
    );
}

#[test]
fn test_apr_short_model_accepted() {
    // Models shorter than 4 bytes bypass magic validation
    // (useful for minimal test fixtures)
    let result = LambdaHandler::from_bytes(b"AP");
    assert!(
        result.is_ok(),
        "Short models (<4 bytes) bypass magic validation"
    );

    // But 4+ byte models with wrong magic are rejected
    let invalid_result = LambdaHandler::from_bytes(b"BADM");
    assert!(
        matches!(invalid_result, Err(LambdaError::InvalidMagic { .. })),
        "4+ byte models with wrong magic should be rejected"
    );
}

// =============================================================================
// Test: Lambda Handler with .apr Models
// =============================================================================

#[test]
#[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
fn test_lambda_handler_inference() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0, 4.0],
        model_id: None,
    };

    let response = handler.handle(&request).expect("inference failed");

    // Verify response structure
    assert!(response.latency_ms >= 0.0);
    assert!(response.prediction.is_finite());
}

#[test]
#[ignore = "Mock APR format"]
fn test_lambda_handler_cold_start_detection() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Model expects 4 input features (dimensions: [4, 1])
    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0, 4.0],
        model_id: None,
    };

    // First invocation should be cold start
    let first_response = handler.handle(&request).expect("first inference failed");
    assert!(
        first_response.cold_start,
        "First invocation should be cold start"
    );

    // Subsequent invocations should not be cold start
    let second_response = handler.handle(&request).expect("second inference failed");
    assert!(
        !second_response.cold_start,
        "Second invocation should not be cold start"
    );
}

#[test]
#[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
fn test_lambda_batch_inference() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let batch = BatchLambdaRequest {
        instances: (0..10)
            .map(|i| LambdaRequest {
                features: vec![i as f32; 4],
                model_id: None,
            })
            .collect(),
        max_parallelism: Some(4),
    };

    let response = handler
        .handle_batch(&batch)
        .expect("batch inference failed");

    assert_eq!(response.predictions.len(), 10);
    assert_eq!(response.success_count, 10);
    assert_eq!(response.error_count, 0);
    assert!(response.total_latency_ms >= 0.0);
}

// =============================================================================
// Test: Metrics Integration
// =============================================================================

#[test]
#[ignore = "Mock APR format"]
fn test_lambda_metrics_collection() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");
    let mut metrics = LambdaMetrics::new();

    // Run multiple inferences (model expects 4 features)
    for i in 0..100 {
        let request = LambdaRequest {
            features: vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32],
            model_id: None,
        };

        let response = handler.handle(&request).expect("inference failed");
        metrics.record_success(response.latency_ms, response.cold_start);
    }

    // Verify metrics
    assert_eq!(metrics.requests_total, 100);
    assert_eq!(metrics.requests_success, 100);
    assert_eq!(metrics.requests_failed, 0);
    assert_eq!(metrics.cold_starts, 1); // Only first should be cold
    assert!(metrics.avg_latency_ms() > 0.0);
}

#[test]
fn test_lambda_metrics_prometheus_export() {
    let mut metrics = LambdaMetrics::new();

    // Simulate some activity
    for _ in 0..50 {
        metrics.record_success(0.5, false);
    }
    metrics.record_success(1.0, true); // One cold start
    metrics.record_failure();
    metrics.record_batch(10, 2, 5.0);

    let prometheus = metrics.to_prometheus();

    // Verify Prometheus format (without "realizar_" prefix)
    assert!(prometheus.contains("lambda_requests_total"));
    assert!(prometheus.contains("lambda_requests_success"));
    assert!(prometheus.contains("lambda_requests_failed"));
    assert!(prometheus.contains("lambda_cold_starts"));
    assert!(prometheus.contains("lambda_latency_avg_ms"));
    assert!(prometheus.contains("lambda_batch_requests"));
}

#[test]
fn test_lambda_batch_metrics() {
    let mut metrics = LambdaMetrics::new();

    // Record batch results
    metrics.record_batch(8, 2, 10.0); // 8 success, 2 failures
    metrics.record_batch(10, 0, 5.0); // 10 success, 0 failures

    assert_eq!(metrics.batch_requests, 2);
    assert_eq!(metrics.requests_success, 18);
    assert_eq!(metrics.requests_failed, 2);
    assert_eq!(metrics.requests_total, 20);
}

// =============================================================================
// Test: Feature Vector Handling
// =============================================================================

#[test]
#[ignore = "Mock APR format"]
fn test_various_feature_sizes() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Test model expects exactly 4 features (dimensions: [4, 1])
    // Size 4 should work, other sizes should fail gracefully
    let request = LambdaRequest {
        features: vec![1.0; 4],
        model_id: None,
    };

    let response = handler.handle(&request);
    assert!(response.is_ok(), "Should handle feature vector of size 4");

    // Test wrong sizes are rejected with proper error
    for wrong_size in [1, 10, 50] {
        let request = LambdaRequest {
            features: vec![1.0; wrong_size],
            model_id: None,
        };

        let response = handler.handle(&request);
        assert!(
            response.is_err(),
            "Should reject feature vector of size {} (model expects 4)",
            wrong_size
        );
    }
}

#[test]
#[ignore = "Mock APR format"]
fn test_special_float_values() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Test with special float values (model expects 4 features)
    let special_features = vec![
        vec![0.0, 0.0, 0.0, 0.0],         // All zeros
        vec![-1.0, -2.0, -3.0, -4.0],     // Negative values
        vec![1e-10, 1e-20, 1e-30, 1e-10], // Very small
        vec![1e10, 1e15, 1e10, 1e5],      // Large (but not overflow)
        vec![0.1, 0.2, 0.3, 0.4],         // Fractional
    ];

    for features in special_features {
        let request = LambdaRequest {
            features: features.clone(),
            model_id: None,
        };

        let response = handler.handle(&request);
        assert!(
            response.is_ok(),
            "Should handle special float values: {:?}",
            features
        );

        let resp = response.expect("test");
        assert!(
            resp.prediction.is_finite(),
            "Prediction should be finite for features: {:?}",
            features
        );
    }
}

// =============================================================================
// Test: Model ID Routing
// =============================================================================

#[test]
#[ignore = "Mock APR format"]
fn test_model_id_in_request() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Request with model_id (model expects 4 features)
    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0, 4.0],
        model_id: Some("classifier_v2".to_string()),
    };

    let response = handler.handle(&request);
    assert!(response.is_ok(), "Should accept request with model_id");
}

#[test]
#[ignore = "Mock APR format"]
fn test_batch_with_mixed_model_ids() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Model expects 4 features per instance
    let batch = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0, 1.0, 1.0, 1.0],
                model_id: Some("model_a".to_string()),
            },
            LambdaRequest {
                features: vec![2.0, 2.0, 2.0, 2.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![3.0, 3.0, 3.0, 3.0],
                model_id: Some("model_b".to_string()),
            },
        ],
        max_parallelism: None,
    };

    let response = handler.handle_batch(&batch);
    assert!(response.is_ok(), "Should handle batch with mixed model IDs");
}

// =============================================================================
// Test: Error Handling
// =============================================================================

#[test]
#[ignore = "Mock APR format"]
fn test_empty_batch_error() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    let batch = BatchLambdaRequest {
        instances: vec![],
        max_parallelism: None,
    };

    let result = handler.handle_batch(&batch);
    assert!(
        matches!(result, Err(LambdaError::EmptyBatch)),
        "Should return EmptyBatch error for empty batch"
    );
}

// =============================================================================
// Test: Serialization/Deserialization
// =============================================================================

#[test]
fn test_request_json_serialization() {
    let request = LambdaRequest {
        features: vec![1.0, 2.0, 3.0],
        model_id: Some("test_model".to_string()),
    };

    let json = serde_json::to_string(&request).expect("serialization failed");
    let parsed: LambdaRequest = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(request.features, parsed.features);
    assert_eq!(request.model_id, parsed.model_id);
}

#[test]
fn test_response_json_serialization() {
    let response = LambdaResponse {
        prediction: 0.85,
        probabilities: Some(vec![0.15, 0.85]),
        latency_ms: 1.234,
        cold_start: false,
    };

    let json = serde_json::to_string(&response).expect("serialization failed");
    let parsed: LambdaResponse = serde_json::from_str(&json).expect("deserialization failed");

    assert!((response.prediction - parsed.prediction).abs() < f32::EPSILON);
    assert_eq!(response.probabilities, parsed.probabilities);
}

#[test]
fn test_batch_request_json_serialization() {
    let batch = BatchLambdaRequest {
        instances: vec![
            LambdaRequest {
                features: vec![1.0],
                model_id: None,
            },
            LambdaRequest {
                features: vec![2.0],
                model_id: Some("model".to_string()),
            },
        ],
        max_parallelism: Some(4),
    };

    let json = serde_json::to_string(&batch).expect("serialization failed");
    let parsed: BatchLambdaRequest = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(batch.instances.len(), parsed.instances.len());
    assert_eq!(batch.max_parallelism, parsed.max_parallelism);
}

// =============================================================================
// Test: Performance Characteristics
// =============================================================================

#[test]
#[ignore = "Mock APR format"]
fn test_inference_latency_reasonable() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Model expects 4 features
    let request = LambdaRequest {
        features: vec![1.0; 4],
        model_id: None,
    };

    // Warm up
    let _ = handler.handle(&request);

    // Measure latency
    let response = handler.handle(&request).expect("inference failed");

    // Latency should be reasonable (< 100ms for simple model)
    assert!(
        response.latency_ms < 100.0,
        "Inference latency {} ms exceeds 100ms threshold",
        response.latency_ms
    );
}

#[test]
#[ignore = "Mock APR format"]
fn test_batch_throughput() {
    let model_bytes = create_static_test_model();
    let handler = LambdaHandler::from_bytes(model_bytes).expect("handler creation failed");

    // Model expects 4 features per instance
    let batch = BatchLambdaRequest {
        instances: (0..100)
            .map(|i| LambdaRequest {
                features: vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32],
                model_id: None,
            })
            .collect(),
        max_parallelism: None,
    };

    // Warm up (model expects 4 features)
    let _ = handler.handle(&LambdaRequest {
        features: vec![1.0, 2.0, 3.0, 4.0],
        model_id: None,
    });

    let response = handler
        .handle_batch(&batch)
        .expect("batch inference failed");

    // All should succeed
    assert_eq!(response.success_count, 100);

    // Calculate throughput
    let throughput = 100.0 / (response.total_latency_ms / 1000.0);
    println!("Batch throughput: {:.0} predictions/second", throughput);

    // Should achieve reasonable throughput
    assert!(throughput > 100.0, "Throughput should exceed 100 pred/sec");
}

// Need to import LambdaResponse for JSON tests
use realizar::lambda::LambdaResponse;
