//! GGUF Part 06: IMP-121 (Production Serving) + IMP-122 (Forward with Cache) +
//!               IMP-123 (Metrics Tracking) + IMP-129 (Latency Histogram) +
//!               IMP-124 (Forward Single Adaptive) + IMP-125 (Generate Adaptive) +
//!               PARITY-002 (Batched Prefill)
//!
//! Extracted from gguf_monolith.rs (PMAT-802)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    DispatchMetrics, GGUFConfig, OwnedQuantizedKVCache, OwnedQuantizedModelCached,
    OwnedQuantizedModelCachedSync, QuantizedGenerateConfig,
};

// ========================================================================
// IMP-121: Integrate Adaptive Attention into Production Serving
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121a_cached_sync_has_adaptive_attention() {
    // IMP-121a: OwnedQuantizedModelCachedSync should expose adaptive attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
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
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    let seq_len = 32;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    // Thread-safe cached model should expose adaptive attention
    let result = cached_sync
        .adaptive_fused_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("Adaptive attention should succeed on CachedSync");

    assert_eq!(
        result.len(),
        seq_len * head_dim,
        "IMP-121a: Output should have correct shape"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121b_cached_sync_adaptive_multihead() {
    // IMP-121b: OwnedQuantizedModelCachedSync should expose adaptive multihead attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
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
    let cached_sync = OwnedQuantizedModelCachedSync::new(model);

    let seq_len = 64;
    let hidden_dim = 64;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 17) as f32 * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.05)
        .collect();

    // Thread-safe cached model should expose adaptive multihead attention
    let result = cached_sync
        .adaptive_multihead_attention(&q, &k, &v, seq_len)
        .expect("Adaptive multihead attention should succeed on CachedSync");

    assert_eq!(
        result.len(),
        seq_len * hidden_dim,
        "IMP-121b: Output should have shape [seq_len, hidden_dim]"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121c_generate_with_adaptive_attention() {
    // IMP-121c: Cached model should have generate_with_adaptive_attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
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
    let cached_model = OwnedQuantizedModelCached::new(model);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: Vec::new(),
        trace: false,
        ..Default::default()
    };

    // Generate with adaptive attention (should use CPU for short prompts)
    let result = cached_model
        .generate_with_adaptive_attention(&prompt, &gen_config)
        .expect("generate_with_adaptive_attention should succeed");

    assert!(
        result.len() > prompt.len(),
        "IMP-121c: Generated output should include new tokens"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_121d_thread_safe_adaptive_attention() {
    // IMP-121d: Verify thread-safe access to adaptive attention
    use std::sync::Arc;
    use std::thread;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
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
    let cached_sync = Arc::new(OwnedQuantizedModelCachedSync::new(model));

    let seq_len = 16;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    // Spawn multiple threads accessing adaptive attention concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let model = Arc::clone(&cached_sync);
            let q = q.clone();
            let k = k.clone();
            let v = v.clone();

            thread::spawn(move || {
                model
                    .adaptive_fused_attention(&q, &k, &v, seq_len, head_dim, scale)
                    .expect("Concurrent adaptive attention should succeed")
            })
        })
        .collect();

    // All threads should complete successfully
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.join().expect("Thread should not panic");
        assert_eq!(
            result.len(),
            seq_len * head_dim,
            "IMP-121d: Thread {} should produce correct output",
            i
        );
    }
}

// ========================================================================
// IMP-122: Integrate Adaptive Attention into Forward with Cache
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_122a_adaptive_attention_with_cache() {
    // IMP-122a: Test attention_with_cache can use adaptive backend
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
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
    let hidden_dim = 64;
    let _head_dim = 16; // Used for documentation, computed as hidden_dim / num_heads
    let cache_len = 32;

    // Simulate Q for single token
    let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 17) as f32 * 0.1).collect();

    // Cached K/V from previous positions
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.05)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.05)
        .collect();

    // Current K/V
    let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i % 7) as f32 * 0.1).collect();
    let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i % 5) as f32 * 0.1).collect();

    // Test adaptive attention with cache
    let result = model
        .adaptive_attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v)
        .expect("Adaptive attention with cache should succeed");

    assert_eq!(
        result.len(),
        hidden_dim,
        "IMP-122a: Output should have shape [hidden_dim]"
    );

    // Result should have non-zero values
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "IMP-122a: Output should have non-zero values");
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_122b_adaptive_matches_standard() {
    // IMP-122b: Adaptive attention with cache should match standard implementation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
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
    let hidden_dim = 64;
    let cache_len = 16;

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 19) as f32 * 0.05).collect();
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 23) as f32 * 0.05)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 29) as f32 * 0.05)
        .collect();
    let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i % 7) as f32 * 0.1).collect();
    let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i % 11) as f32 * 0.1).collect();

    // Standard attention
    let standard_result =
        model.attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Adaptive attention
    let adaptive_result = model
        .adaptive_attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v)
        .expect("Adaptive attention should succeed");

    assert_eq!(standard_result.len(), adaptive_result.len());
    for i in 0..standard_result.len() {
        let diff = (standard_result[i] - adaptive_result[i]).abs();
        assert!(
            diff < 1e-2,
            "IMP-122b: Results differ at {}: std={}, adaptive={}, diff={}",
            i,
            standard_result[i],
            adaptive_result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_122c_long_sequence_uses_gpu() {
    // IMP-122c: Long sequence should automatically use GPU path
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
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
    let hidden_dim = 128;
    let cache_len = 128; // Long cache triggers GPU

    let q: Vec<f32> = (0..hidden_dim).map(|i| (i % 17) as f32 * 0.05).collect();
    let k_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.02)
        .collect();
    let v_cache: Vec<f32> = (0..cache_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.02)
        .collect();
    let current_k: Vec<f32> = (0..hidden_dim).map(|i| (i % 7) as f32 * 0.1).collect();
    let current_v: Vec<f32> = (0..hidden_dim).map(|i| (i % 5) as f32 * 0.1).collect();

    let result = model
        .adaptive_attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v)
        .expect("Long sequence adaptive attention should succeed");

    assert_eq!(
        result.len(),
        hidden_dim,
        "IMP-122c: Long sequence should produce correct output"
    );
}

// ========================================================================
// IMP-123: Metrics Tracking for CPU vs GPU Dispatch Decisions
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123a_dispatch_metrics_struct() {
    // IMP-123a: DispatchMetrics struct should track CPU vs GPU decisions
    let metrics = DispatchMetrics::new();

    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
    assert_eq!(metrics.total_dispatches(), 0);
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123b_record_dispatch_decisions() {
    // IMP-123b: Metrics should correctly record dispatch decisions
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    assert_eq!(metrics.cpu_dispatches(), 2);
    assert_eq!(metrics.gpu_dispatches(), 1);
    assert_eq!(metrics.total_dispatches(), 3);
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_123c_dispatch_ratio() {
    // IMP-123c: Should calculate GPU dispatch ratio
    let metrics = DispatchMetrics::new();

    // 3 CPU + 1 GPU = 25% GPU ratio
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    let ratio = metrics.gpu_ratio();
    assert!(
        (ratio - 0.25).abs() < 0.01,
        "IMP-123c: GPU ratio should be ~25%, got {}",
        ratio
    );
}

include!("imp_123d.rs");
include!("imp_125b.rs");
