
/// IMP-130d: Prometheus latency histogram should have HELP and TYPE annotations
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130d_prometheus_latency_has_help_and_type() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-130d: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-130d: Should have HELP and TYPE annotations for histograms
    assert!(
        body_str.contains("# HELP realizar_dispatch_cpu_latency"),
        "IMP-130d: Should have HELP for CPU latency histogram"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_cpu_latency histogram"),
        "IMP-130d: Should have TYPE histogram for CPU latency"
    );
    assert!(
        body_str.contains("# HELP realizar_dispatch_gpu_latency"),
        "IMP-130d: Should have HELP for GPU latency histogram"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_gpu_latency histogram"),
        "IMP-130d: Should have TYPE histogram for GPU latency"
    );
}

// ========================================================================
// IMP-141: Add Throughput Metrics to Prometheus Export (RED PHASE)
// ========================================================================

/// IMP-141a: Prometheus export should include throughput_rps gauge
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141a_prometheus_includes_throughput_rps() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};
    use std::thread;
    use std::time::Duration;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141a: Should create AppState");

    // Record some dispatches to get non-zero throughput
    if let Some(metrics) = state.dispatch_metrics() {
        thread::sleep(Duration::from_millis(2));
        for _ in 0..10 {
            metrics.record_cpu_dispatch();
        }
    }

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141a: Should include throughput_rps metric
    assert!(
        body_str.contains("realizar_dispatch_throughput_rps"),
        "IMP-141a: Prometheus should include throughput_rps metric. Got: {}",
        body_str
    );
}

/// IMP-141b: Prometheus export should include elapsed_seconds gauge
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141b_prometheus_includes_elapsed_seconds() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141b: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141b: Should include elapsed_seconds metric
    assert!(
        body_str.contains("realizar_dispatch_elapsed_seconds"),
        "IMP-141b: Prometheus should include elapsed_seconds metric. Got: {}",
        body_str
    );
}

/// IMP-141c: throughput_rps should have correct HELP and TYPE annotations
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141c_throughput_rps_has_help_and_type() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141c: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141c: Should have HELP and TYPE for throughput_rps
    assert!(
        body_str.contains("# HELP realizar_dispatch_throughput_rps"),
        "IMP-141c: Should have HELP for throughput_rps"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_throughput_rps gauge"),
        "IMP-141c: Should have TYPE gauge for throughput_rps"
    );
}

/// IMP-141d: elapsed_seconds should have correct HELP and TYPE annotations
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_141d_elapsed_seconds_has_help_and_type() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-141d: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=prometheus")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-141d: Should have HELP and TYPE for elapsed_seconds
    assert!(
        body_str.contains("# HELP realizar_dispatch_elapsed_seconds"),
        "IMP-141d: Should have HELP for elapsed_seconds"
    );
    assert!(
        body_str.contains("# TYPE realizar_dispatch_elapsed_seconds gauge"),
        "IMP-141d: Should have TYPE gauge for elapsed_seconds"
    );
}

// ===== IMP-131: Latency percentiles in JSON response =====

/// IMP-131a: DispatchMetrics should have percentile calculation methods
#[cfg(feature = "gpu")]
#[test]
fn test_imp_131a_dispatch_metrics_has_percentile_methods() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // Record some latencies to test percentile calculation
    metrics.record_cpu_latency(std::time::Duration::from_micros(50));
    metrics.record_cpu_latency(std::time::Duration::from_micros(150));
    metrics.record_cpu_latency(std::time::Duration::from_micros(600));
    metrics.record_gpu_latency(std::time::Duration::from_micros(80));
    metrics.record_gpu_latency(std::time::Duration::from_micros(300));

    // IMP-131a: Should have percentile methods
    let _cpu_p50 = metrics.cpu_latency_p50_us();
    let _cpu_p95 = metrics.cpu_latency_p95_us();
    let _cpu_p99 = metrics.cpu_latency_p99_us();
    let _gpu_p50 = metrics.gpu_latency_p50_us();
    let _gpu_p95 = metrics.gpu_latency_p95_us();
    let _gpu_p99 = metrics.gpu_latency_p99_us();
}

/// IMP-131b: Percentile estimation from histogram buckets
#[cfg(feature = "gpu")]
#[test]
fn test_imp_131b_percentile_estimation_from_histogram() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // Record 100 samples: 50 in first bucket, 30 in second, 20 in third
    // This creates a known distribution for testing
    for _ in 0..50 {
        metrics.record_cpu_latency(std::time::Duration::from_micros(50)); // bucket 0: 0-100µs
    }
    for _ in 0..30 {
        metrics.record_cpu_latency(std::time::Duration::from_micros(200)); // bucket 1: 100-500µs
    }
    for _ in 0..20 {
        metrics.record_cpu_latency(std::time::Duration::from_micros(700)); // bucket 2: 500-1000µs
    }

    // p50 should be in first bucket (50th sample out of 100)
    let p50 = metrics.cpu_latency_p50_us();
    assert!(
        p50 <= 100.0,
        "IMP-131b: p50 should be in first bucket (<=100µs), got {:.1}µs",
        p50
    );

    // p95 should be in third bucket (95th sample)
    // First 50 in bucket 0, next 30 in bucket 1 (total 80), next 20 in bucket 2
    // 95th percentile is in bucket 2 (500-1000µs)
    let p95 = metrics.cpu_latency_p95_us();
    assert!(
        p95 >= 500.0 && p95 <= 1000.0,
        "IMP-131b: p95 should be in bucket 2 (500-1000µs), got {:.1}µs",
        p95
    );
}

/// IMP-131c: JSON response should include latency percentiles
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_131c_json_response_includes_percentiles() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state =
        AppState::with_cached_model(cached_model).expect("IMP-131c: Should create AppState");

    // Record some latencies
    if let Some(metrics) = state.dispatch_metrics() {
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        metrics.record_gpu_latency(std::time::Duration::from_micros(200));
    }

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);

    // IMP-131c: JSON should include percentile fields
    assert!(
        body_str.contains("cpu_latency_p50_us"),
        "IMP-131c: JSON should include cpu_latency_p50_us. Got: {}",
        body_str
    );
    assert!(
        body_str.contains("cpu_latency_p95_us"),
        "IMP-131c: JSON should include cpu_latency_p95_us"
    );
    assert!(
        body_str.contains("gpu_latency_p50_us"),
        "IMP-131c: JSON should include gpu_latency_p50_us"
    );
}

/// IMP-131d: Percentiles return 0 when no samples recorded
#[cfg(feature = "gpu")]
#[test]
fn test_imp_131d_percentiles_zero_when_empty() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-131d: Empty histogram should return 0 for all percentiles
    assert_eq!(
        metrics.cpu_latency_p50_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for p50"
    );
    assert_eq!(
        metrics.cpu_latency_p95_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for p95"
    );
    assert_eq!(
        metrics.cpu_latency_p99_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for p99"
    );
    assert_eq!(
        metrics.gpu_latency_p50_us(),
        0.0,
        "IMP-131d: Empty histogram should return 0 for GPU p50"
    );
}
