
/// IMP-029: Full generation loop produces coherent output (M15)
/// Target: Generate tokens without crash, deterministic output
#[test]
#[cfg(feature = "gpu")]
fn test_imp_029_text_generation() {
    use crate::gpu::{GpuGenerateConfig, GpuModel, GpuModelConfig};

    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut model = GpuModel::from_gguf_config(config).expect("IMP-029: Should create model");

    // Test 1: Generate multiple tokens
    let prompt = vec![1, 2, 3];
    let gen_config = GpuGenerateConfig::deterministic(20);
    let tokens = model
        .generate(&prompt, &gen_config)
        .expect("IMP-029: Generation should succeed");

    assert!(
        tokens.len() > prompt.len(),
        "IMP-029: Should generate at least one token"
    );
    assert!(
        tokens.len() <= prompt.len() + 20,
        "IMP-029: Should respect max_tokens"
    );

    // Test 2: Deterministic generation produces same output
    let mut model2 = GpuModel::from_gguf_config(GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    })
    .expect("IMP-029: Should create second model");

    let tokens2 = model2
        .generate(&prompt, &gen_config)
        .expect("IMP-029: Second generation should succeed");

    assert_eq!(
        tokens, tokens2,
        "IMP-029: Deterministic generation should be reproducible"
    );

    // Test 3: All generated tokens are valid
    for &token in &tokens {
        assert!(
            token < 256,
            "IMP-029: Token {} should be within vocab size",
            token
        );
    }

    // Test 4: Generation with stop token
    let stop_token = tokens[prompt.len()]; // First generated token
    let gen_config_stop = GpuGenerateConfig::deterministic(50).with_stop_tokens(vec![stop_token]);
    let tokens_stopped = model
        .generate(&prompt, &gen_config_stop)
        .expect("IMP-029: Generation with stop should succeed");

    assert_eq!(
        tokens_stopped.len(),
        prompt.len(),
        "IMP-029: Should stop before adding stop token"
    );

    // Test 5: Long generation (100 tokens) completes without crash
    let long_config = GpuGenerateConfig::deterministic(100);
    let long_tokens = model
        .generate(&prompt, &long_config)
        .expect("IMP-029: Long generation should complete");

    assert!(
        long_tokens.len() >= prompt.len(),
        "IMP-029: Long generation should produce output"
    );
}

/// IMP-030: Benchmark harness for apples-to-apples comparison (M15)
/// Target: Reproducible measurements with < 5% variance
/// Perf test: CV < 1.0 assertion is flaky under CPU contention and coverage
/// instrumentation. Run via `cargo test -- --ignored` for targeted perf validation.
#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn test_imp_030_benchmark_harness() {
    use crate::gpu::{GpuGenerateConfig, GpuModel, GpuModelConfig};
    use std::time::Instant;

    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut model = GpuModel::from_gguf_config(config).expect("IMP-030: Should create model");

    // Warmup runs (per Mytkowicz et al. [4])
    let prompt = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);
    for _ in 0..5 {
        let _ = model.generate(&prompt, &gen_config);
    }

    // Measure multiple runs
    let num_runs = 5;
    let mut throughputs = Vec::with_capacity(num_runs);

    for _ in 0..num_runs {
        let start = Instant::now();
        let tokens = model
            .generate(&prompt, &gen_config)
            .expect("IMP-030: Generation should succeed");
        let elapsed = start.elapsed();

        let generated = tokens.len() - prompt.len();
        let throughput = generated as f64 / elapsed.as_secs_f64();
        throughputs.push(throughput);
    }

    // Calculate statistics
    let mean: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let variance: f64 =
        throughputs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / throughputs.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean; // Coefficient of variation

    // Test 1: Mean throughput is positive
    assert!(
        mean > 0.0,
        "IMP-030: Mean throughput should be positive (got {})",
        mean
    );

    // Test 2: CV should be reasonable (< 100% for test environment)
    // Production target is < 5%, but test environment has more variance
    assert!(
        cv < 1.0,
        "IMP-030: CV ({:.2}) should be < 1.0 for reasonable reproducibility",
        cv
    );

    // Test 3: All runs produced consistent token counts
    let mut model2 = GpuModel::from_gguf_config(GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    })
    .expect("IMP-030: Should create model");

    let tokens1 = model.generate(&prompt, &gen_config).expect("test");
    let tokens2 = model2.generate(&prompt, &gen_config).expect("test");

    assert_eq!(
        tokens1.len(),
        tokens2.len(),
        "IMP-030: Deterministic runs should produce same token count"
    );

    // Test 4: Benchmark struct captures required metrics
    #[allow(clippy::items_after_statements)]
    #[derive(Debug)]
    struct BenchmarkResult {
        model_name: String,
        prompt_tokens: usize,
        generated_tokens: usize,
        total_time_ms: f64,
        throughput_tok_s: f64,
    }

    let start = Instant::now();
    let tokens = model.generate(&prompt, &gen_config).expect("test");
    let elapsed = start.elapsed();

    let result = BenchmarkResult {
        model_name: "test-model".to_string(),
        prompt_tokens: prompt.len(),
        generated_tokens: tokens.len() - prompt.len(),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_tok_s: (tokens.len() - prompt.len()) as f64 / elapsed.as_secs_f64(),
    };

    assert!(
        !result.model_name.is_empty(),
        "IMP-030: Model name should be set"
    );
    assert!(
        result.prompt_tokens > 0,
        "IMP-030: Prompt tokens should be tracked"
    );
    assert!(
        result.generated_tokens > 0,
        "IMP-030: Generated tokens should be tracked"
    );
    assert!(
        result.total_time_ms > 0.0,
        "IMP-030: Time should be measured"
    );
    assert!(
        result.throughput_tok_s > 0.0,
        "IMP-030: Throughput should be calculated"
    );
}

// ============================================================================
// Phase 7: KV Cache Optimization (M16) - EXTREME TDD
// ============================================================================

/// IMP-031: forward_gpu_with_cache() for initial prompt processing (M16)
/// Target: Process prompt and populate KV cache
#[test]
#[cfg(feature = "gpu")]
fn test_imp_031_forward_with_cache() {
    use crate::gpu::{GpuModel, GpuModelConfig, StreamingKVCache};

    // Test config: small model for fast testing
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut model =
        GpuModel::from_gguf_config(config.clone()).expect("IMP-031: Should create model");

    // Create KV cache for the model
    let max_seq_len = 512;
    let head_dim = config.hidden_dim / config.num_heads;
    let mut kv_cache =
        StreamingKVCache::new(config.num_layers, max_seq_len, config.num_heads, head_dim);

    // Test 1: Process prompt with cache
    let prompt = vec![1, 2, 3, 4, 5];
    let logits = model
        .forward_gpu_with_cache(&prompt, &mut kv_cache)
        .expect("IMP-031: forward_with_cache should succeed");

    // Test 2: Logits should be for final position only (vocab_size elements)
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "IMP-031: Should return logits for final position only (got {}, expected {})",
        logits.len(),
        config.vocab_size
    );

    // Test 3: KV cache should have entries for prompt length
    assert_eq!(
        kv_cache.len(),
        prompt.len(),
        "IMP-031: KV cache should contain {} positions (got {})",
        prompt.len(),
        kv_cache.len()
    );

    // Test 4: Cache values should be non-zero (actually computed)
    // Get layer 0's cached KV
    let (keys, values) = kv_cache.get_range(0, 0, prompt.len());

    let key_sum: f32 = keys.iter().map(|x| x.abs()).sum();
    let value_sum: f32 = values.iter().map(|x| x.abs()).sum();

    assert!(key_sum > 0.0, "IMP-031: Cached keys should be non-zero");
    assert!(value_sum > 0.0, "IMP-031: Cached values should be non-zero");
}

/// IMP-032: forward_gpu_incremental() for single-token decode (M16)
/// Target: Process single token using cached KV
#[test]
#[cfg(feature = "gpu")]
fn test_imp_032_forward_incremental() {
    use crate::gpu::{GpuModel, GpuModelConfig, StreamingKVCache};

    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut model =
        GpuModel::from_gguf_config(config.clone()).expect("IMP-032: Should create model");

    let max_seq_len = 512;
    let head_dim = config.hidden_dim / config.num_heads;
    let mut kv_cache =
        StreamingKVCache::new(config.num_layers, max_seq_len, config.num_heads, head_dim);

    // First, process prompt to populate cache
    let prompt = vec![1, 2, 3, 4, 5];
    let _ = model
        .forward_gpu_with_cache(&prompt, &mut kv_cache)
        .expect("IMP-032: Initial forward should succeed");

    let cache_len_after_prompt = kv_cache.len();

    // Test 1: Process single token incrementally
    let new_token = 42usize;
    let logits = model
        .forward_gpu_incremental(new_token, &mut kv_cache)
        .expect("IMP-032: Incremental forward should succeed");

    // Test 2: Should return vocab_size logits
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "IMP-032: Incremental should return vocab_size logits"
    );

    // Test 3: Cache should grow by 1
    assert_eq!(
        kv_cache.len(),
        cache_len_after_prompt + 1,
        "IMP-032: Cache should grow by 1 position"
    );

    // Test 4: Multiple incremental steps should work
    for token in [10, 20, 30] {
        let prev_len = kv_cache.len();
        let logits = model
            .forward_gpu_incremental(token, &mut kv_cache)
            .expect("IMP-032: Repeated incremental should succeed");

        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(kv_cache.len(), prev_len + 1);
    }

    // Test 5: Final cache length should be prompt + all incremental tokens
    assert_eq!(
        kv_cache.len(),
        prompt.len() + 4, // 1 + 3 incremental tokens
        "IMP-032: Final cache length should match all tokens"
    );
}
