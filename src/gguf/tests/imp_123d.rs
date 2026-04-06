
#[test]
#[cfg(feature = "gpu")]
fn test_imp_123d_thread_safe_metrics() {
    // IMP-123d: Metrics should be thread-safe
    use std::sync::Arc;
    use std::thread;

    let metrics = Arc::new(DispatchMetrics::new());
    let num_threads = 4;
    let dispatches_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let m = Arc::clone(&metrics);
            thread::spawn(move || {
                for _ in 0..dispatches_per_thread {
                    if i % 2 == 0 {
                        m.record_cpu_dispatch();
                    } else {
                        m.record_gpu_dispatch();
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // 2 threads did CPU, 2 did GPU
    assert_eq!(
        metrics.total_dispatches(),
        num_threads * dispatches_per_thread,
        "IMP-123d: Should have all dispatches recorded"
    );
    assert_eq!(
        metrics.cpu_dispatches(),
        2 * dispatches_per_thread,
        "IMP-123d: Should have correct CPU count"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        2 * dispatches_per_thread,
        "IMP-123d: Should have correct GPU count"
    );
}

// ========================================================================
// IMP-129: Dispatch Latency Histogram
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129a_latency_histogram_struct() {
    // IMP-129a: DispatchMetrics should track latency with histogram buckets
    let metrics = DispatchMetrics::new();

    // Should have latency tracking methods
    assert_eq!(metrics.cpu_latency_count(), 0);
    assert_eq!(metrics.gpu_latency_count(), 0);
    assert!(metrics.cpu_latency_mean_us() == 0.0 || metrics.cpu_latency_mean_us().is_nan());
    assert!(metrics.gpu_latency_mean_us() == 0.0 || metrics.gpu_latency_mean_us().is_nan());
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129b_record_latency() {
    // IMP-129b: Should record latency for CPU and GPU dispatches
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record CPU latency
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));

    // Record GPU latency
    metrics.record_gpu_latency(Duration::from_micros(1000));

    assert_eq!(metrics.cpu_latency_count(), 2);
    assert_eq!(metrics.gpu_latency_count(), 1);

    // Mean should be calculated correctly
    let cpu_mean = metrics.cpu_latency_mean_us();
    assert!(
        (cpu_mean - 150.0).abs() < 1.0,
        "IMP-129b: CPU mean should be ~150us, got {}",
        cpu_mean
    );

    let gpu_mean = metrics.gpu_latency_mean_us();
    assert!(
        (gpu_mean - 1000.0).abs() < 1.0,
        "IMP-129b: GPU mean should be ~1000us, got {}",
        gpu_mean
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129c_histogram_buckets() {
    // IMP-129c: Should have histogram bucket counts
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record various latencies to populate buckets
    // Buckets: 0-100us, 100-500us, 500-1000us, 1000-5000us, 5000+us
    metrics.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    metrics.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    metrics.record_cpu_latency(Duration::from_micros(600)); // bucket 2
    metrics.record_cpu_latency(Duration::from_micros(2000)); // bucket 3
    metrics.record_cpu_latency(Duration::from_micros(10000)); // bucket 4

    let buckets = metrics.cpu_latency_buckets();
    assert_eq!(buckets.len(), 5, "IMP-129c: Should have 5 buckets");
    assert_eq!(buckets[0], 1, "IMP-129c: Bucket 0 (0-100us) should have 1");
    assert_eq!(
        buckets[1], 1,
        "IMP-129c: Bucket 1 (100-500us) should have 1"
    );
    assert_eq!(
        buckets[2], 1,
        "IMP-129c: Bucket 2 (500-1000us) should have 1"
    );
    assert_eq!(
        buckets[3], 1,
        "IMP-129c: Bucket 3 (1000-5000us) should have 1"
    );
    assert_eq!(buckets[4], 1, "IMP-129c: Bucket 4 (5000+us) should have 1");
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_129d_thread_safe_latency() {
    // IMP-129d: Latency recording should be thread-safe
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    let metrics = Arc::new(DispatchMetrics::new());
    let num_threads = 4;
    let recordings_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let m = Arc::clone(&metrics);
            thread::spawn(move || {
                for j in 0..recordings_per_thread {
                    let latency = Duration::from_micros((i * 100 + j) as u64);
                    if i % 2 == 0 {
                        m.record_cpu_latency(latency);
                    } else {
                        m.record_gpu_latency(latency);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // 2 threads did CPU, 2 did GPU
    assert_eq!(
        metrics.cpu_latency_count(),
        2 * recordings_per_thread,
        "IMP-129d: Should have all CPU latencies recorded"
    );
    assert_eq!(
        metrics.gpu_latency_count(),
        2 * recordings_per_thread,
        "IMP-129d: Should have all GPU latencies recorded"
    );
}

// ============================================================
// IMP-124: Wire adaptive attention into forward_single_with_cache
// RED phase: Tests written first, implementation to follow
// ============================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124a_forward_single_with_cache_adaptive() {
    // IMP-124a: forward_single_with_cache_adaptive should exist and produce valid output
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(
        config.num_layers,
        config.hidden_dim,
        128, // max_seq_len
    );
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process first token (position 0) - cache is empty, no dispatch recorded
    let result = model.forward_single_with_cache_adaptive(0, &mut cache, 0, &metrics);
    assert!(result.is_ok(), "IMP-124a: Should produce valid output");

    let logits = result.expect("Should have logits");
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "IMP-124a: Should output vocab_size logits"
    );

    // Process second token (position 1) - cache now has entries, dispatch will be recorded
    let result2 = model.forward_single_with_cache_adaptive(1, &mut cache, 1, &metrics);
    assert!(result2.is_ok(), "IMP-124a: Second token should work");

    // Metrics should now record dispatches (from non-empty cache attention)
    assert!(
        metrics.total_dispatches() > 0,
        "IMP-124a: Should record dispatch decisions after second token"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124b_adaptive_matches_standard() {
    // IMP-124b: Adaptive forward should match standard forward
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let mut cache1 = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let mut cache2 = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Generate 10 tokens with both methods
    for i in 0..10 {
        let token = (i % 10) as u32;
        let standard = model
            .forward_single_with_cache(token, &mut cache1, i)
            .expect("Standard forward should work");
        let adaptive = model
            .forward_single_with_cache_adaptive(token, &mut cache2, i, &metrics)
            .expect("Adaptive forward should work");

        // Outputs should match (within floating point tolerance)
        for (j, (&s, &a)) in standard.iter().zip(adaptive.iter()).enumerate() {
            assert!(
                (s - a).abs() < 1e-4,
                "IMP-124b: Output mismatch at position {} token {}: {} vs {}",
                j,
                i,
                s,
                a
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124c_tracks_metrics_per_layer() {
    // IMP-124c: Each layer should record a dispatch decision
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process 5 tokens
    for i in 0..5 {
        let _ = model.forward_single_with_cache_adaptive(i as u32, &mut cache, i, &metrics);
    }

    // With short sequences (< 64 tokens), should use CPU path exclusively
    // First token (position 0) has empty cache, no dispatch recorded
    // Tokens at positions 1-4 should each record at least one dispatch
    // Note: actual count depends on layer count in test model
    let expected_min_dispatches = 4; // At least 1 dispatch per non-first token
    assert!(
        metrics.total_dispatches() >= expected_min_dispatches,
        "IMP-124c: Should record at least {} dispatches, got {}",
        expected_min_dispatches,
        metrics.total_dispatches()
    );

    // All dispatches should be CPU (cache_len < 64)
    assert_eq!(
        metrics.cpu_dispatches(),
        metrics.total_dispatches(),
        "IMP-124c: All dispatches should be CPU for short sequences"
    );
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "IMP-124c: No GPU dispatches for short sequences"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_124d_long_cache_uses_gpu() {
    // IMP-124d: Long cache (>= 64 tokens) should trigger GPU dispatch
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let mut cache = OwnedQuantizedKVCache::new(
        config.num_layers,
        config.hidden_dim,
        256, // Enough for 65+ tokens
    );
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process 70 tokens to exceed GPU threshold (64)
    for i in 0..70 {
        let _ = model.forward_single_with_cache_adaptive(i as u32, &mut cache, i, &metrics);
    }

    // After 64 tokens, GPU should start being used
    assert!(
        metrics.gpu_dispatches() > 0,
        "IMP-124d: Should have GPU dispatches for long sequences, got cpu={} gpu={}",
        metrics.cpu_dispatches(),
        metrics.gpu_dispatches()
    );

    // GPU ratio should be positive
    assert!(
        metrics.gpu_ratio() > 0.0,
        "IMP-124d: GPU ratio should be > 0 for long sequences"
    );
}

// ============================================================
// IMP-125: Generate with cache adaptive for full generation loop
// RED phase: Tests written first, implementation to follow
// ============================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125a_generate_with_cache_adaptive() {
    // IMP-125a: generate_with_cache_adaptive should exist and produce valid tokens
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy for determinism
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompt = vec![1u32, 2, 3]; // 3-token prompt
    let result = model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    assert!(result.is_ok(), "IMP-125a: Should produce valid output");
    let tokens = result.expect("Should have tokens");

    // Should have prompt + generated tokens
    assert!(
        tokens.len() >= prompt.len(),
        "IMP-125a: Output should include at least prompt tokens"
    );
    assert!(
        tokens.len() <= prompt.len() + gen_config.max_tokens,
        "IMP-125a: Output should not exceed max length"
    );
}
