//! Property-Based Tests for Lambda Serving Infrastructure
//!
//! Per EXTREME TDD methodology: Property-based tests verify invariants
//! that must hold for ALL valid inputs, not just specific test cases.
//!
//! ## Properties Tested
//!
//! - Handler creation invariants
//! - Batch inference size consistency
//! - Metrics arithmetic correctness
//! - Latency non-negativity
//! - Cold start detection correctness

#![cfg(feature = "lambda")]
#![allow(dead_code)] // Test helper functions may not all be used
#![allow(clippy::absurd_extreme_comparisons)] // Test bounds checks intentionally test edge cases
#![allow(unused_comparisons)] // Test bounds checks on unsigned types

use proptest::prelude::*;
use realizar::lambda::{BatchLambdaRequest, LambdaHandler, LambdaMetrics, LambdaRequest};

// =============================================================================
// Strategies for generating test data
// =============================================================================

/// Generate valid .apr model bytes
fn valid_apr_model() -> impl Strategy<Value = &'static [u8]> {
    // Use a set of pre-defined valid models (static lifetime required)
    prop_oneof![
        Just(&b"APR\0\x01\x00\x00\x00model_a"[..]),
        Just(&b"APR\0\x02\x00\x00\x00model_b_longer"[..]),
        Just(&b"APR\0\x03\x00\x00\x00m"[..]),
    ]
}

/// Generate valid feature vectors (non-empty)
fn valid_features() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        any::<f32>().prop_filter("finite", |f| f.is_finite()),
        1..100,
    )
}

/// Generate valid Lambda requests
fn valid_lambda_request() -> impl Strategy<Value = LambdaRequest> {
    (valid_features(), prop::option::of(any::<String>())).prop_map(|(features, model_id)| {
        LambdaRequest {
            features,
            model_id: model_id.map(|s| s.chars().take(50).collect()), // Limit string length
        }
    })
}

/// Generate batch sizes
fn batch_size() -> impl Strategy<Value = usize> {
    1..50usize
}

// =============================================================================
// Property Tests: Lambda Handler
// =============================================================================

proptest! {
    /// Property: Valid APR models should always load successfully
    #[test]
    fn prop_valid_apr_model_loads(model in valid_apr_model()) {
        let handler = LambdaHandler::from_bytes(model);
        prop_assert!(handler.is_ok(), "Valid APR model should load");
    }

    /// Property: Handler should report correct model size
    #[test]
    fn prop_model_size_matches(model in valid_apr_model()) {
        let handler = LambdaHandler::from_bytes(model).expect("test");
        prop_assert_eq!(handler.model_size_bytes(), model.len());
    }

    /// Property: First invocation is always cold start
    #[test]
    fn prop_first_invocation_is_cold(
        model in valid_apr_model(),
        features in valid_features()
    ) {
        let handler = LambdaHandler::from_bytes(model).expect("test");
        let request = LambdaRequest { features, model_id: None };

        // First invocation
        let response = handler.handle(&request).expect("test");
        prop_assert!(response.cold_start, "First invocation must be cold start");

        // Second invocation
        let response2 = handler.handle(&request).expect("test");
        prop_assert!(!response2.cold_start, "Second invocation must not be cold start");
    }

    /// Property: Latency is always non-negative
    #[test]
    fn prop_latency_non_negative(
        model in valid_apr_model(),
        features in valid_features()
    ) {
        let handler = LambdaHandler::from_bytes(model).expect("test");
        let request = LambdaRequest { features, model_id: None };

        let response = handler.handle(&request).expect("test");
        prop_assert!(response.latency_ms >= 0.0, "Latency must be non-negative");
    }

    /// Property: Prediction for sum of features equals sum (mock behavior)
    #[test]
    fn prop_mock_prediction_is_sum(
        model in valid_apr_model(),
        features in valid_features()
    ) {
        let handler = LambdaHandler::from_bytes(model).expect("test");
        let expected_sum: f32 = features.iter().sum();
        let request = LambdaRequest { features, model_id: None };

        let response = handler.handle(&request).expect("test");

        // Handle special cases: infinity, NaN
        if expected_sum.is_infinite() {
            prop_assert!(
                response.prediction.is_infinite(),
                "If sum is infinite, prediction should be infinite"
            );
        } else if expected_sum.is_nan() {
            prop_assert!(
                response.prediction.is_nan(),
                "If sum is NaN, prediction should be NaN"
            );
        } else {
            // Allow for floating point precision issues
            let diff = (response.prediction - expected_sum).abs();
            prop_assert!(
                diff < 0.01 || diff / expected_sum.abs().max(1.0) < 0.001,
                "Prediction {} should equal sum {} (diff: {})",
                response.prediction, expected_sum, diff
            );
        }
    }
}

// =============================================================================
// Property Tests: Batch Inference
// =============================================================================

proptest! {
    /// Property: Batch response has same length as request
    #[test]
    fn prop_batch_response_length_matches(
        model in valid_apr_model(),
        n in batch_size()
    ) {
        let handler = LambdaHandler::from_bytes(model).expect("test");

        let instances: Vec<LambdaRequest> = (0..n)
            .map(|i| LambdaRequest {
                features: vec![i as f32],
                model_id: None,
            })
            .collect();

        let batch = BatchLambdaRequest {
            instances: instances.clone(),
            max_parallelism: None,
        };

        let response = handler.handle_batch(&batch).expect("test");
        prop_assert_eq!(
            response.predictions.len(),
            instances.len(),
            "Predictions count must match instances count"
        );
    }

    /// Property: success_count + error_count = total predictions
    #[test]
    fn prop_batch_counts_sum_to_total(
        model in valid_apr_model(),
        n in batch_size()
    ) {
        let handler = LambdaHandler::from_bytes(model).expect("test");

        // Mix of valid and invalid requests
        let instances: Vec<LambdaRequest> = (0..n)
            .map(|i| LambdaRequest {
                features: if i % 3 == 0 { vec![] } else { vec![i as f32] },
                model_id: None,
            })
            .collect();

        let batch = BatchLambdaRequest {
            instances,
            max_parallelism: None,
        };

        let response = handler.handle_batch(&batch).expect("test");
        prop_assert_eq!(
            response.success_count + response.error_count,
            response.predictions.len(),
            "success + error must equal total predictions"
        );
    }

    /// Property: Batch latency is non-negative
    #[test]
    fn prop_batch_latency_non_negative(
        model in valid_apr_model(),
        n in batch_size()
    ) {
        let handler = LambdaHandler::from_bytes(model).expect("test");

        let instances: Vec<LambdaRequest> = (0..n)
            .map(|i| LambdaRequest {
                features: vec![i as f32],
                model_id: None,
            })
            .collect();

        let batch = BatchLambdaRequest {
            instances,
            max_parallelism: None,
        };

        let response = handler.handle_batch(&batch).expect("test");
        prop_assert!(
            response.total_latency_ms >= 0.0,
            "Batch latency must be non-negative"
        );
    }
}

// =============================================================================
// Property Tests: Metrics
// =============================================================================

proptest! {
    /// Property: Metrics counts are consistent after operations
    #[test]
    fn prop_metrics_counts_consistent(
        success_count in 0..100u64,
        failure_count in 0..100u64,
    ) {
        let mut metrics = LambdaMetrics::new();

        for _ in 0..success_count {
            metrics.record_success(1.0, false);
        }

        for _ in 0..failure_count {
            metrics.record_failure();
        }

        prop_assert_eq!(
            metrics.requests_total,
            success_count + failure_count,
            "Total must equal success + failure"
        );
        prop_assert_eq!(metrics.requests_success, success_count);
        prop_assert_eq!(metrics.requests_failed, failure_count);
    }

    /// Property: Average latency is non-negative when success > 0
    #[test]
    fn prop_avg_latency_non_negative(
        latencies in prop::collection::vec(0.0f64..1000.0, 1..50)
    ) {
        let mut metrics = LambdaMetrics::new();

        for latency in &latencies {
            metrics.record_success(*latency, false);
        }

        let avg = metrics.avg_latency_ms();
        prop_assert!(avg >= 0.0, "Average latency must be non-negative");

        // Verify average is in reasonable range
        let expected_avg: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let diff = (avg - expected_avg).abs();
        prop_assert!(
            diff < 0.001,
            "Average {} should match expected {} (diff: {})",
            avg, expected_avg, diff
        );
    }

    /// Property: Cold starts <= total requests
    #[test]
    fn prop_cold_starts_bounded(
        cold_count in 0..50u64,
        warm_count in 0..50u64,
    ) {
        let mut metrics = LambdaMetrics::new();

        for _ in 0..cold_count {
            metrics.record_success(1.0, true);
        }

        for _ in 0..warm_count {
            metrics.record_success(1.0, false);
        }

        prop_assert!(
            metrics.cold_starts <= metrics.requests_total,
            "Cold starts {} must be <= total {}",
            metrics.cold_starts, metrics.requests_total
        );
        prop_assert_eq!(metrics.cold_starts, cold_count);
    }

    /// Property: Batch metrics add correctly
    #[test]
    fn prop_batch_metrics_arithmetic(
        batches in prop::collection::vec((1..20usize, 0..5usize, 0.1f64..100.0), 1..10)
    ) {
        let mut metrics = LambdaMetrics::new();

        let mut expected_success = 0u64;
        let mut expected_failed = 0u64;
        let mut expected_batches = 0u64;

        for (success, error, latency) in &batches {
            metrics.record_batch(*success, *error, *latency);
            expected_success += *success as u64;
            expected_failed += *error as u64;
            expected_batches += 1;
        }

        prop_assert_eq!(metrics.requests_success, expected_success);
        prop_assert_eq!(metrics.requests_failed, expected_failed);
        prop_assert_eq!(metrics.batch_requests, expected_batches);
        prop_assert_eq!(
            metrics.requests_total,
            expected_success + expected_failed
        );
    }

    /// Property: Prometheus output contains required metrics
    #[test]
    fn prop_prometheus_contains_metrics(
        success in 0..100u64,
        failed in 0..100u64,
    ) {
        let mut metrics = LambdaMetrics::new();

        for _ in 0..success {
            metrics.record_success(1.0, false);
        }
        for _ in 0..failed {
            metrics.record_failure();
        }

        let prom = metrics.to_prometheus();

        prop_assert!(prom.contains("lambda_requests_total"));
        prop_assert!(prom.contains("lambda_requests_success"));
        prop_assert!(prom.contains("lambda_requests_failed"));
        prop_assert!(prom.contains("lambda_latency_avg_ms"));
        prop_assert!(prom.contains("lambda_cold_starts"));
        prop_assert!(prom.contains("lambda_batch_requests"));
    }
}

// =============================================================================
// Property Tests: Target Detection
// =============================================================================

proptest! {
    /// Property: Target name is never empty
    #[test]
    fn prop_target_name_not_empty(_dummy in 0..1i32) {
        use realizar::target::DeployTarget;

        for target in [
            DeployTarget::Native,
            DeployTarget::Lambda,
            DeployTarget::Docker,
            DeployTarget::Wasm,
        ] {
            prop_assert!(!target.name().is_empty());
        }
    }

    /// Property: Capabilities memory limit is sensible
    #[test]
    fn prop_capabilities_memory_sensible(_dummy in 0..1i32) {
        use realizar::target::DeployTarget;

        for target in [
            DeployTarget::Native,
            DeployTarget::Lambda,
            DeployTarget::Docker,
            DeployTarget::Wasm,
        ] {
            let caps = target.capabilities();
            // Memory limit is either 0 (unlimited) or > 0
            prop_assert!(caps.max_memory_mb >= 0);
        }
    }
}

// =============================================================================
// Property Tests: Error Handling
// =============================================================================

proptest! {
    /// Property: Empty features always produce EmptyFeatures error
    #[test]
    fn prop_empty_features_error(model in valid_apr_model()) {
        use realizar::lambda::LambdaError;

        let handler = LambdaHandler::from_bytes(model).expect("test");
        let request = LambdaRequest {
            features: vec![],
            model_id: None,
        };

        let result = handler.handle(&request);
        prop_assert!(matches!(result, Err(LambdaError::EmptyFeatures)));
    }

    /// Property: Empty batch always produces EmptyBatch error
    #[test]
    fn prop_empty_batch_error(model in valid_apr_model()) {
        use realizar::lambda::LambdaError;

        let handler = LambdaHandler::from_bytes(model).expect("test");
        let batch = BatchLambdaRequest {
            instances: vec![],
            max_parallelism: None,
        };

        let result = handler.handle_batch(&batch);
        prop_assert!(matches!(result, Err(LambdaError::EmptyBatch)));
    }
}
