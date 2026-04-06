
#[test]
#[cfg(feature = "gpu")]
fn test_imp_125b_adaptive_matches_standard() {
    // IMP-125b: Adaptive generate should match standard generate
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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
        max_tokens: 10,
        temperature: 0.0, // Greedy for determinism
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompt = vec![1u32, 2, 3];

    // Generate with both methods
    let standard = model
        .generate_with_cache(&prompt, &gen_config)
        .expect("Standard should work");
    let adaptive = model
        .generate_with_cache_adaptive(&prompt, &gen_config, &metrics)
        .expect("Adaptive should work");

    // Token sequences should match (same sampling with temp=0)
    assert_eq!(
        standard, adaptive,
        "IMP-125b: Adaptive should produce same tokens as standard"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125c_tracks_metrics_during_generation() {
    // IMP-125c: Should track metrics during full generation
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompt = vec![1u32, 2, 3, 4, 5];
    let _ = model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    // Should have recorded dispatches
    assert!(
        metrics.total_dispatches() > 0,
        "IMP-125c: Should track dispatch decisions during generation"
    );

    // With short context, all should be CPU
    assert_eq!(
        metrics.cpu_dispatches(),
        metrics.total_dispatches(),
        "IMP-125c: Short generation should use CPU"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_125d_long_generation_uses_gpu() {
    // IMP-125d: Long generation should eventually use GPU
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 100, // Long enough to trigger GPU
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompt = vec![1u32, 2, 3, 4, 5];
    let _ = model.generate_with_cache_adaptive(&prompt, &gen_config, &metrics);

    // After 64+ tokens, GPU should be used
    assert!(
        metrics.gpu_dispatches() > 0,
        "IMP-125d: Long generation should use GPU, got cpu={} gpu={}",
        metrics.cpu_dispatches(),
        metrics.gpu_dispatches()
    );

    // GPU ratio should be positive
    assert!(
        metrics.gpu_ratio() > 0.0,
        "IMP-125d: GPU ratio should be > 0 for long generation"
    );
}

// ============================================================
// PARITY-002: Batched Prompt Prefill Tests
// RED phase: Tests written first, implementation to follow
// ============================================================

/// PARITY-002a: forward_batch_with_cache should exist and process multiple tokens
#[test]
#[cfg(feature = "gpu")]
fn test_parity002a_forward_batch_with_cache_exists() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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

    // Process a batch of 4 tokens at once
    let prompt = vec![1u32, 2, 3, 4];
    let result = model.forward_batch_with_cache(&prompt, &mut cache, &metrics);

    assert!(
        result.is_ok(),
        "PARITY-002a: forward_batch_with_cache should exist and produce valid output"
    );

    let logits = result.expect("Should have logits");
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "PARITY-002a: Should output vocab_size logits for last token"
    );
}

/// PARITY-002b: Batched prefill should process all tokens and populate cache
#[test]
#[cfg(feature = "gpu")]
fn test_parity002b_batched_prefill_populates_cache() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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

    // Process 8 tokens at once
    let prompt = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let _ = model.forward_batch_with_cache(&prompt, &mut cache, &metrics);

    // Cache should have all 8 positions filled
    assert_eq!(
        cache.len(),
        8,
        "PARITY-002b: Cache should have all 8 prompt tokens after batched prefill"
    );
}

/// PARITY-002c: Batched prefill uses CPU (GPU is intentionally disabled)
///
/// FINDING (PARITY-002): GPU matmul is 6.6x SLOWER than CPU for attention
/// because attention = per-head MATVEC, and GPU overhead dominates for small matrices.
/// See IMP-600: GPU 2.7x slower for MATVEC, 57x faster for GEMM.
///
/// This test verifies CPU path is used (correct behavior for single-request inference).
#[test]
#[cfg(feature = "gpu")]
fn test_parity002c_batched_prefill_triggers_gpu() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_layers: 2,
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
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 512);
    let metrics = std::sync::Arc::new(DispatchMetrics::new());

    // Process batch
    let prompt: Vec<u32> = (0..64).map(|i| i as u32 % 100).collect();
    let _ = model.forward_batch_with_cache(&prompt, &mut cache, &metrics);

    // PARITY-002 FINDING: CPU path is used (GPU disabled because it's slower)
    // GPU dispatches = 0 is CORRECT behavior for attention MATVEC
    // CPU dispatches should be > 0
    assert!(
        metrics.cpu_dispatches() > 0,
        "PARITY-002c: CPU should be used for attention (dispatches: {}, expected > 0)",
        metrics.cpu_dispatches()
    );
    // GPU is intentionally disabled for attention (per-head MATVEC is slower on GPU)
    assert_eq!(
        metrics.gpu_dispatches(),
        0,
        "PARITY-002c: GPU should NOT be used for attention (MATVEC is slower on GPU)"
    );
}

/// PARITY-002d: Batched prefill should produce same result as sequential
#[test]
#[cfg(feature = "gpu")]
fn test_parity002d_batched_matches_sequential() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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
    let prompt = vec![1u32, 2, 3, 4];

    // Sequential processing (baseline)
    let mut cache_seq = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics_seq = std::sync::Arc::new(DispatchMetrics::new());
    let mut logits_seq = vec![];
    for (pos, &token) in prompt.iter().enumerate() {
        logits_seq = model
            .forward_single_with_cache_adaptive(token, &mut cache_seq, pos, &metrics_seq)
            .expect("Sequential should work");
    }

    // Batched processing
    let mut cache_batch = OwnedQuantizedKVCache::new(config.num_layers, config.hidden_dim, 128);
    let metrics_batch = std::sync::Arc::new(DispatchMetrics::new());
    let logits_batch = model
        .forward_batch_with_cache(&prompt, &mut cache_batch, &metrics_batch)
        .expect("Batched should work");

    // Results should match (within floating point tolerance)
    assert_eq!(
        logits_seq.len(),
        logits_batch.len(),
        "PARITY-002d: Logits length should match"
    );

    let max_diff: f32 = logits_seq
        .iter()
        .zip(logits_batch.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // GH-278: True LayerNorm (with mean subtraction) produces slightly more numerical
    // divergence between batched and sequential paths than old RMSNorm implementation.
    // Relaxed from 1e-4 to 1e-3 which still validates correctness.
    assert!(
        max_diff < 1e-3,
        "PARITY-002d: Batched and sequential should produce same logits (max_diff: {})",
        max_diff
    );
}

/// PARITY-002e: generate_with_batched_prefill should use batched prefill then sequential gen
#[test]
#[cfg(feature = "gpu")]
fn test_parity002e_generate_with_batched_prefill() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
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
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompt = vec![1u32, 2, 3, 4];
    let result = model.generate_with_batched_prefill(&prompt, &gen_config, &metrics);

    assert!(
        result.is_ok(),
        "PARITY-002e: generate_with_batched_prefill should exist and work"
    );

    let tokens = result.expect("Should have tokens");

    // Should have prompt + generated tokens
    assert!(
        tokens.len() >= prompt.len(),
        "PARITY-002e: Output should include at least prompt tokens"
    );
    assert!(
        tokens.len() <= prompt.len() + gen_config.max_tokens,
        "PARITY-002e: Output should not exceed prompt + max_tokens"
    );
}
