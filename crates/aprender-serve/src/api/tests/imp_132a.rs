
// ===== IMP-132: Wire latency recording into adaptive attention path =====

/// IMP-132a: Adaptive attention should record latency for CPU dispatches
#[cfg(feature = "gpu")]
#[test]
fn test_imp_132a_adaptive_attention_records_cpu_latency() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
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
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    // Generate tokens to trigger CPU dispatches (cache < 64 tokens)
    let _ = cached_model.generate_with_cache_adaptive(&[1, 2, 3], &gen_config, &metrics);

    // IMP-132a: After CPU dispatches, latency should be recorded
    assert!(
        metrics.cpu_latency_count() > 0,
        "IMP-132a: CPU latency count should be > 0 after adaptive generation. Got: {}",
        metrics.cpu_latency_count()
    );
}

/// IMP-132b: Latency values should be reasonable (not zero for executed paths)
#[cfg(feature = "gpu")]
#[test]
#[ignore = "GPU latency test - timing-sensitive and may fail under coverage instrumentation"]
fn test_imp_132b_latency_values_are_reasonable() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
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
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    // Generate tokens
    let _ = cached_model.generate_with_cache_adaptive(&[1, 2, 3], &gen_config, &metrics);

    // IMP-132b: Mean latency should be > 0 (actual time was measured)
    let mean_latency = metrics.cpu_latency_mean_us();
    assert!(
        mean_latency > 0.0,
        "IMP-132b: Mean CPU latency should be > 0µs after attention. Got: {:.1}µs",
        mean_latency
    );
}

/// IMP-132c: Latency count should match dispatch count
#[cfg(feature = "gpu")]
#[test]
fn test_imp_132c_latency_count_matches_dispatch_count() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 2, // 2 layers for more dispatches
        num_heads: 2,
        num_kv_heads: 2,
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
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    // Generate tokens
    let _ = cached_model.generate_with_cache_adaptive(&[1, 2, 3, 4, 5], &gen_config, &metrics);

    // IMP-132c: Every CPU dispatch should record latency
    let cpu_dispatches = metrics.cpu_dispatches();
    let cpu_latency_count = metrics.cpu_latency_count();

    assert_eq!(
        cpu_dispatches, cpu_latency_count,
        "IMP-132c: CPU latency count ({}) should match dispatch count ({})",
        cpu_latency_count, cpu_dispatches
    );
}

/// IMP-132d: GPU dispatches should also record latency (when cache >= 64)
#[cfg(feature = "gpu")]
#[test]
fn test_imp_132d_gpu_dispatches_record_latency() {
    use crate::gguf::{
        DispatchMetrics, GGUFConfig, OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let metrics = Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 80, // Generate enough to trigger GPU dispatch
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    // Generate enough tokens to trigger GPU dispatch (cache >= 64 tokens)
    let _ = cached_model.generate_with_cache_adaptive(&[1], &gen_config, &metrics);

    // IMP-132d: After many tokens, should have GPU dispatches with latency recorded
    let gpu_dispatches = metrics.gpu_dispatches();
    let gpu_latency_count = metrics.gpu_latency_count();

    if gpu_dispatches > 0 {
        assert_eq!(
            gpu_dispatches, gpu_latency_count,
            "IMP-132d: GPU latency count ({}) should match dispatch count ({})",
            gpu_latency_count, gpu_dispatches
        );
    }
}

// ============================================================
// IMP-133: Add latency mean to JSON response
// RED phase: Tests written first, implementation to follow
// ============================================================

// IMP-133a: DispatchMetrics should have cpu_latency_mean_us method
#[test]
fn test_imp_133a_dispatch_metrics_has_mean_methods() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some latencies
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    metrics.record_gpu_latency(Duration::from_micros(500));
    metrics.record_gpu_latency(Duration::from_micros(700));

    // IMP-133a: Mean methods should exist and return correct values
    let cpu_mean = metrics.cpu_latency_mean_us();
    let gpu_mean = metrics.gpu_latency_mean_us();

    assert!(
        (cpu_mean - 200.0).abs() < 1.0,
        "IMP-133a: CPU mean should be ~200µs, got {}",
        cpu_mean
    );
    assert!(
        (gpu_mean - 600.0).abs() < 1.0,
        "IMP-133a: GPU mean should be ~600µs, got {}",
        gpu_mean
    );
}

// IMP-133b: Mean should be 0 when no samples recorded
#[test]
fn test_imp_133b_mean_zero_when_empty() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-133b: Mean should be 0.0 when no samples recorded
    assert_eq!(
        metrics.cpu_latency_mean_us(),
        0.0,
        "IMP-133b: CPU mean should be 0 when empty"
    );
    assert_eq!(
        metrics.gpu_latency_mean_us(),
        0.0,
        "IMP-133b: GPU mean should be 0 when empty"
    );
}

// IMP-133c: JSON response should include mean latency fields
#[test]
fn test_imp_133c_json_response_includes_mean() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());

    // Record some latencies
    metrics.record_cpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_dispatch();
    metrics.record_cpu_latency(Duration::from_micros(300));

    // Build response (would be done by handler)
    let response = DispatchMetricsResponse {
        cpu_dispatches: metrics.cpu_dispatches(),
        gpu_dispatches: metrics.gpu_dispatches(),
        total_dispatches: metrics.total_dispatches(),
        gpu_ratio: metrics.gpu_ratio(),
        cpu_latency_p50_us: metrics.cpu_latency_p50_us(),
        cpu_latency_p95_us: metrics.cpu_latency_p95_us(),
        cpu_latency_p99_us: metrics.cpu_latency_p99_us(),
        gpu_latency_p50_us: metrics.gpu_latency_p50_us(),
        gpu_latency_p95_us: metrics.gpu_latency_p95_us(),
        gpu_latency_p99_us: metrics.gpu_latency_p99_us(),
        // IMP-133: New mean fields
        cpu_latency_mean_us: metrics.cpu_latency_mean_us(),
        gpu_latency_mean_us: metrics.gpu_latency_mean_us(),
        // IMP-134: New min/max fields
        cpu_latency_min_us: metrics.cpu_latency_min_us(),
        cpu_latency_max_us: metrics.cpu_latency_max_us(),
        gpu_latency_min_us: metrics.gpu_latency_min_us(),
        gpu_latency_max_us: metrics.gpu_latency_max_us(),
        // IMP-135: Variance/stddev fields
        cpu_latency_variance_us: metrics.cpu_latency_variance_us(),
        cpu_latency_stddev_us: metrics.cpu_latency_stddev_us(),
        gpu_latency_variance_us: metrics.gpu_latency_variance_us(),
        gpu_latency_stddev_us: metrics.gpu_latency_stddev_us(),
        // IMP-136: Histogram bucket configuration
        bucket_boundaries_us: metrics.bucket_boundaries_us(),
        cpu_latency_bucket_counts: metrics.cpu_latency_buckets().to_vec(),
        gpu_latency_bucket_counts: metrics.gpu_latency_buckets().to_vec(),
        // IMP-140: Throughput metrics
        throughput_rps: 0.0,
        elapsed_seconds: 0.0,
    };

    // IMP-133c: Response should have mean fields with correct values
    assert!(
        (response.cpu_latency_mean_us - 200.0).abs() < 1.0,
        "IMP-133c: Response CPU mean should be ~200µs, got {}",
        response.cpu_latency_mean_us
    );
    assert_eq!(
        response.gpu_latency_mean_us, 0.0,
        "IMP-133c: Response GPU mean should be 0 (no GPU samples)"
    );
}

// IMP-133d: Mean should handle single sample correctly
#[test]
fn test_imp_133d_mean_single_sample() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Single sample
    metrics.record_cpu_latency(Duration::from_micros(42));

    // IMP-133d: Mean of single sample should equal that sample
    assert!(
        (metrics.cpu_latency_mean_us() - 42.0).abs() < 0.1,
        "IMP-133d: Mean of single sample should be 42µs, got {}",
        metrics.cpu_latency_mean_us()
    );
}

// ============================================================
// IMP-134: Add min/max latency tracking
// RED phase: Tests written first, implementation to follow
// ============================================================

// IMP-134a: DispatchMetrics should have min/max methods
#[test]
fn test_imp_134a_dispatch_metrics_has_min_max_methods() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some latencies with varying values
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(300));

    metrics.record_gpu_latency(Duration::from_micros(200));
    metrics.record_gpu_latency(Duration::from_micros(800));

    // IMP-134a: Min/max methods should exist and return correct values
    assert_eq!(
        metrics.cpu_latency_min_us(),
        50,
        "IMP-134a: CPU min should be 50µs"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        300,
        "IMP-134a: CPU max should be 300µs"
    );
    assert_eq!(
        metrics.gpu_latency_min_us(),
        200,
        "IMP-134a: GPU min should be 200µs"
    );
    assert_eq!(
        metrics.gpu_latency_max_us(),
        800,
        "IMP-134a: GPU max should be 800µs"
    );
}

// IMP-134b: Min/max should be 0 when no samples recorded
#[test]
fn test_imp_134b_min_max_zero_when_empty() {
    use crate::gguf::DispatchMetrics;

    let metrics = DispatchMetrics::new();

    // IMP-134b: Min/max should be 0 when no samples recorded
    assert_eq!(
        metrics.cpu_latency_min_us(),
        0,
        "IMP-134b: CPU min should be 0 when empty"
    );
    assert_eq!(
        metrics.cpu_latency_max_us(),
        0,
        "IMP-134b: CPU max should be 0 when empty"
    );
    assert_eq!(
        metrics.gpu_latency_min_us(),
        0,
        "IMP-134b: GPU min should be 0 when empty"
    );
    assert_eq!(
        metrics.gpu_latency_max_us(),
        0,
        "IMP-134b: GPU max should be 0 when empty"
    );
}

// IMP-134c: JSON response should include min/max latency fields
