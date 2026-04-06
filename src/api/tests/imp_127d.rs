
/// IMP-127d: /metrics/dispatch returns 503 when no GPU model configured
#[tokio::test]
async fn test_imp_127d_dispatch_metrics_no_gpu_model() {
    // Use demo() which creates AppState without cached model / dispatch metrics
    let state = AppState::demo().expect("Should create demo AppState");
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

    // Should return 503 Service Unavailable when no dispatch metrics available
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "IMP-127d: /metrics/dispatch should return 503 when no GPU model configured"
    );
}

// ========================================================================
// IMP-128: Prometheus Format Export Tests
// ========================================================================

/// IMP-128a: /metrics/dispatch?format=prometheus returns Prometheus format
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128a_prometheus_format_endpoint() {
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
        AppState::with_cached_model(cached_model).expect("IMP-128a: Should create AppState");

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

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "IMP-128a: Prometheus format should return 200 OK"
    );

    // Verify text/plain content type for Prometheus
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("text/plain")),
        "IMP-128a: Prometheus response should be text/plain"
    );
}

/// IMP-128b: Prometheus format contains correct metric names
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128b_prometheus_format_structure() {
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
        AppState::with_cached_model(cached_model).expect("IMP-128b: Should create AppState");

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
    let text = String::from_utf8_lossy(&body);

    // Verify Prometheus metric format
    assert!(
        text.contains("realizar_dispatch_cpu_total"),
        "IMP-128b: Should have CPU dispatch counter"
    );
    assert!(
        text.contains("realizar_dispatch_gpu_total"),
        "IMP-128b: Should have GPU dispatch counter"
    );
    assert!(
        text.contains("realizar_dispatch_gpu_ratio"),
        "IMP-128b: Should have GPU ratio gauge"
    );
}

/// IMP-128c: Default format (no query param) returns JSON
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128c_default_format_is_json() {
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
        AppState::with_cached_model(cached_model).expect("IMP-128c: Should create AppState");

    let app = create_router(state);

    // Request without format parameter
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("application/json")),
        "IMP-128c: Default format should be JSON"
    );
}

/// IMP-128d: format=json explicitly returns JSON
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_128d_explicit_json_format() {
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
        AppState::with_cached_model(cached_model).expect("IMP-128d: Should create AppState");

    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics/dispatch?format=json")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert!(
        content_type.is_some_and(|s| s.contains("application/json")),
        "IMP-128d: format=json should return JSON"
    );
}

// ===== IMP-130: Latency histogram in Prometheus export =====

/// IMP-130a: Prometheus export should include CPU latency histogram
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130a_prometheus_includes_cpu_latency_histogram() {
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
        AppState::with_cached_model(cached_model).expect("IMP-130a: Should create AppState");

    // Record some CPU latency samples
    if let Some(metrics) = state.dispatch_metrics() {
        metrics.record_cpu_latency(std::time::Duration::from_micros(50));
        metrics.record_cpu_latency(std::time::Duration::from_micros(200));
        metrics.record_cpu_latency(std::time::Duration::from_micros(800));
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

    // IMP-130a: Should include CPU latency histogram buckets
    assert!(
        body_str.contains("realizar_dispatch_cpu_latency_bucket"),
        "IMP-130a: Prometheus should include CPU latency histogram buckets. Got: {}",
        body_str
    );
    assert!(
        body_str.contains("realizar_dispatch_cpu_latency_sum"),
        "IMP-130a: Prometheus should include CPU latency sum"
    );
    assert!(
        body_str.contains("realizar_dispatch_cpu_latency_count"),
        "IMP-130a: Prometheus should include CPU latency count"
    );
}

/// IMP-130b: Prometheus export should include GPU latency histogram
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130b_prometheus_includes_gpu_latency_histogram() {
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
        AppState::with_cached_model(cached_model).expect("IMP-130b: Should create AppState");

    // Record some GPU latency samples
    if let Some(metrics) = state.dispatch_metrics() {
        metrics.record_gpu_latency(std::time::Duration::from_micros(150));
        metrics.record_gpu_latency(std::time::Duration::from_micros(600));
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

    // IMP-130b: Should include GPU latency histogram buckets
    assert!(
        body_str.contains("realizar_dispatch_gpu_latency_bucket"),
        "IMP-130b: Prometheus should include GPU latency histogram buckets. Got: {}",
        body_str
    );
    assert!(
        body_str.contains("realizar_dispatch_gpu_latency_sum"),
        "IMP-130b: Prometheus should include GPU latency sum"
    );
    assert!(
        body_str.contains("realizar_dispatch_gpu_latency_count"),
        "IMP-130b: Prometheus should include GPU latency count"
    );
}

/// IMP-130c: Prometheus latency histogram should have correct bucket labels
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_imp_130c_prometheus_latency_buckets_have_correct_labels() {
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
        AppState::with_cached_model(cached_model).expect("IMP-130c: Should create AppState");

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

    // IMP-130c: Should have Prometheus histogram bucket labels (le="X")
    // Bucket boundaries: 100µs, 500µs, 1000µs, 5000µs, +Inf
    assert!(
        body_str.contains(r#"le="100""#),
        "IMP-130c: Should have 100µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="500""#),
        "IMP-130c: Should have 500µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="1000""#),
        "IMP-130c: Should have 1000µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="5000""#),
        "IMP-130c: Should have 5000µs bucket label"
    );
    assert!(
        body_str.contains(r#"le="+Inf""#),
        "IMP-130c: Should have +Inf bucket label"
    );
}
