//! API Tests Part 03
//!
//! Chat completion tests, OpenAI compatibility

#[allow(unused_imports)]
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
#[allow(unused_imports)]
use tower::util::ServiceExt;

#[allow(unused_imports)]
use crate::api::test_helpers::create_test_app_shared;
use crate::api::*;

#[test]
fn test_imp_134c_json_response_includes_min_max() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(500));

    // Build response (would be done by handler)
    let response = DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        // IMP-134: New min/max fields
        cpu_latency_min_us: metrics.cpu_latency_min_us(),
        cpu_latency_max_us: metrics.cpu_latency_max_us(),
        gpu_latency_min_us: metrics.gpu_latency_min_us(),
        gpu_latency_max_us: metrics.gpu_latency_max_us(),
        // IMP-135: Variance/stddev fields
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        // IMP-136: Histogram bucket configuration
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // IMP-134c: Response should have min/max fields with correct values
    assert_eq!(
        response.cpu_latency_min_us, 100,
        "IMP-134c: Response CPU min should be 100µs"
    );
    assert_eq!(
        response.cpu_latency_max_us, 500,
        "IMP-134c: Response CPU max should be 500µs"
    );
}

// IMP-134d: Single sample should set both min and max to same value
#[test]
fn test_imp_134d_min_max_single_sample() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Single sample
    metrics.record_cpu_latency(Duration::from_micros(42));

    // IMP-134d: Min and max of single sample should both equal that sample
    assert_eq!(
        metrics.cpu_latency_min_us(),
        42,
        "IMP-134d: Min of single sample should be 42µs"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        42,
        "IMP-134d: Max of single sample should be 42µs"
    );
}

// ============================================================
// IMP-135: Add latency variance/stddev tracking
// RED phase: Tests written first, implementation to follow
// ============================================================

// IMP-135a: DispatchMetrics should have variance and stddev methods
#[test]
fn test_imp_135a_dispatch_metrics_has_variance_stddev_methods() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record latencies: 100, 200, 300 (mean=200, variance=6666.67, stddev=81.65)
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    // For population variance: sum((x - mean)^2) / n
    // = ((100-200)^2 + (200-200)^2 + (300-200)^2) / 3
    // = (10000 + 0 + 10000) / 3 = 6666.67
    let cpu_var = metrics.cpu_latency_variance_us();
    let cpu_std = metrics.cpu_latency_stddev_us();

    assert!(
        (cpu_var - 6666.67).abs() < 1.0,
        "IMP-135a: CPU variance should be ~6666.67, got {}",
        cpu_var
    );
    assert!(
        (cpu_std - 81.65).abs() < 1.0,
        "IMP-135a: CPU stddev should be ~81.65, got {}",
        cpu_std
    );
}

// IMP-135b: Variance/stddev should be 0 when no samples or single sample
#[test]
fn test_imp_135b_variance_zero_edge_cases() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // No samples
    assert_eq!(
        metrics.cpu_latency_variance_us(),
        0.0,
        "IMP-135b: CPU variance should be 0 when empty"
    );
    assert_eq!(
        metrics.cpu_latency_stddev_us(),
        0.0,
        "IMP-135b: CPU stddev should be 0 when empty"
    );

    // Single sample - variance is 0
    metrics.record_cpu_latency(Duration::from_micros(100));
    assert_eq!(
        metrics.cpu_latency_variance_us(),
        0.0,
        "IMP-135b: CPU variance should be 0 for single sample"
    );
    assert_eq!(
        metrics.cpu_latency_stddev_us(),
        0.0,
        "IMP-135b: CPU stddev should be 0 for single sample"
    );
}

// IMP-135c: JSON response should include variance and stddev fields
#[test]
fn test_imp_135c_json_response_includes_variance_stddev() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    // Build response (would be done by handler)
    let response = DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        // IMP-135: New variance/stddev fields
        cpu_latency_variance_us: metrics.cpu_latency_variance_us(),
        cpu_latency_stddev_us: metrics.cpu_latency_stddev_us(),
        gpu_latency_variance_us: metrics.gpu_latency_variance_us(),
        gpu_latency_stddev_us: metrics.gpu_latency_stddev_us(),
        // IMP-136: Histogram bucket configuration
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // IMP-135c: Response should have variance/stddev fields
    assert!(
        (response.cpu_latency_variance_us - 6666.67).abs() < 1.0,
        "IMP-135c: Response CPU variance should be ~6666.67"
    );
    assert!(
        response.cpu_latency_stddev_us > 80.0,
        "IMP-135c: Response CPU stddev should be > 80"
    );
}

// IMP-135d: GPU variance/stddev should also work
#[test]
fn test_imp_135d_gpu_variance_stddev() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record GPU latencies: 500, 1000, 1500 (mean=1000, variance=166666.67)
    metrics.record_gpu_latency(Duration::from_micros(500));
    metrics.record_gpu_latency(Duration::from_micros(1000));
    metrics.record_gpu_latency(Duration::from_micros(1500));

    let gpu_var = metrics.gpu_latency_variance_us();
    let gpu_std = metrics.gpu_latency_stddev_us();

    // variance = ((500-1000)^2 + (1000-1000)^2 + (1500-1000)^2) / 3
    // = (250000 + 0 + 250000) / 3 = 166666.67
    assert!(
        (gpu_var - 166666.67).abs() < 1.0,
        "IMP-135d: GPU variance should be ~166666.67, got {}",
        gpu_var
    );
    assert!(
        (gpu_std - 408.25).abs() < 1.0,
        "IMP-135d: GPU stddev should be ~408.25, got {}",
        gpu_std
    );
}

// =============================================================================
// IMP-136: Histogram Bucket Configuration (RED PHASE - FAILING TESTS)
// =============================================================================
//
// Per spec: Expose histogram bucket boundaries for transparency.
// Users should be able to query what bucket ranges are used.
//
// Test TDD Anchors:
// - IMP-136a: DispatchMetrics should expose bucket boundaries as constant
// - IMP-136b: bucket_boundaries() should return the 5 bucket upper bounds
// - IMP-136c: JSON response should include bucket_boundaries field
// - IMP-136d: Prometheus output should include bucket boundary labels

// IMP-136a: DispatchMetrics should expose bucket boundaries as constant
#[test]
fn test_imp_136a_dispatch_metrics_exposes_bucket_boundaries() {
    use crate::gguf::DispatchMetrics;

    // IMP-136a: BUCKET_BOUNDARIES should be publicly accessible
    let boundaries = DispatchMetrics::BUCKET_BOUNDARIES;

    // Should have 4 boundaries (for 5 buckets)
    assert_eq!(
        boundaries.len(),
        4,
        "IMP-136a: Should have 4 bucket boundaries for 5 buckets"
    );

    // Verify standard Prometheus-style boundaries
    assert_eq!(
        boundaries[0], 100,
        "IMP-136a: Bucket 0 upper bound should be 100µs"
    );
    assert_eq!(
        boundaries[1], 500,
        "IMP-136a: Bucket 1 upper bound should be 500µs"
    );
    assert_eq!(
        boundaries[2], 1000,
        "IMP-136a: Bucket 2 upper bound should be 1000µs"
    );
    assert_eq!(
        boundaries[3], 5000,
        "IMP-136a: Bucket 3 upper bound should be 5000µs"
    );
}

// IMP-136b: bucket_boundaries() method should return all boundaries with +Inf
#[test]
fn test_imp_136b_bucket_boundaries_method() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-136b: bucket_boundaries() should return human-readable boundaries
    let boundaries = metrics.bucket_boundaries_us();

    // Should return 5 strings for 5 buckets
    assert_eq!(
        boundaries.len(),
        5,
        "IMP-136b: Should have 5 bucket boundary strings"
    );

    // Verify format: "0-100", "100-500", etc.
    assert_eq!(boundaries[0], "0-100", "IMP-136b: Bucket 0 range");
    assert_eq!(boundaries[1], "100-500", "IMP-136b: Bucket 1 range");
    assert_eq!(boundaries[2], "500-1000", "IMP-136b: Bucket 2 range");
    assert_eq!(boundaries[3], "1000-5000", "IMP-136b: Bucket 3 range");
    assert_eq!(
        boundaries[4], "5000+",
        "IMP-136b: Bucket 4 range (unbounded)"
    );
}

// IMP-136c: JSON response should include bucket_boundaries field
#[test]
fn test_imp_136c_json_response_includes_bucket_boundaries() {
    // IMP-136c: DispatchMetricsResponse should have bucket_boundaries field
    let response = DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        // IMP-136: New fields
        bucket_boundaries_us: vec![
            "0-100".to_string(),
            "100-500".to_string(),
            "500-1000".to_string(),
            "1000-5000".to_string(),
            "5000+".to_string(),
        ],
        cpu_latency_bucket_counts: vec![0, 0, 0, 0, 0],
        gpu_latency_bucket_counts: vec![0, 0, 0, 0, 0],
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // Serialize to JSON and verify field exists
    let json = serde_json::to_string(&response).expect("IMP-136c: Should serialize");
    assert!(
        json.contains("bucket_boundaries_us"),
        "IMP-136c: JSON should contain bucket_boundaries_us field"
    );
    assert!(
        json.contains("0-100"),
        "IMP-136c: JSON should contain bucket range '0-100'"
    );
}

// IMP-136d: Bucket data should be included in response
#[test]
fn test_imp_136d_response_includes_bucket_counts() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some latencies in different buckets
    metrics.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    metrics.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    metrics.record_cpu_latency(Duration::from_micros(750)); // bucket 2

    // IMP-136d: Response should include bucket counts
    let response = DispatchMetricsResponse {
        cpu_dispatches: 0,
        gpu_dispatches: 0,
        total_dispatches: 0,
        gpu_ratio: 0.0,
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        bucket_boundaries_us: metrics.bucket_boundaries_us(),
        // IMP-136d: New field for bucket counts
        cpu_latency_bucket_counts: metrics.cpu_latency_buckets().to_vec(),
        gpu_latency_bucket_counts: metrics.gpu_latency_buckets().to_vec(),
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // Verify bucket counts
    assert_eq!(
        response.cpu_latency_bucket_counts[0], 1,
        "IMP-136d: Bucket 0 should have 1 sample"
    );
    assert_eq!(
        response.cpu_latency_bucket_counts[1], 1,
        "IMP-136d: Bucket 1 should have 1 sample"
    );
    assert_eq!(
        response.cpu_latency_bucket_counts[2], 1,
        "IMP-136d: Bucket 2 should have 1 sample"
    );
}

include!("imp_137a.rs");
include!("imp_140c.rs");
include!("chat_completion_03.rs");
include!("chat_completion_02_02.rs");
