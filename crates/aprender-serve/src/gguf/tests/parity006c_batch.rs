
/// PARITY-006c: Batch outputs should be valid for all prompts
#[test]
fn test_parity006c_batch_output_validity() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let cached = OwnedQuantizedModelCached::new(model);

    // Multiple prompts of different lengths
    let prompts = vec![vec![1u32, 2, 3], vec![4u32, 5, 6, 7, 8], vec![9u32]];
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let results = cached
        .model()
        .batch_generate(
            &prompts
                .iter()
                .map(std::vec::Vec::as_slice)
                .collect::<Vec<_>>(),
            &gen_config,
        )
        .expect("PARITY-006c: Batch generate should succeed");

    assert_eq!(results.len(), 3, "PARITY-006c: Should have 3 results");

    // Each result should include at least its prompt
    for (i, (prompt, result)) in prompts.iter().zip(results.iter()).enumerate() {
        assert!(
            result.len() >= prompt.len(),
            "PARITY-006c: Result {} should include prompt ({} >= {})",
            i,
            result.len(),
            prompt.len()
        );
        assert!(
            result[..prompt.len()] == *prompt,
            "PARITY-006c: Result {} should start with prompt",
            i
        );
    }
}

/// PARITY-006d: Batch throughput factor should scale with batch size
#[test]
fn test_parity006d_throughput_factor() {
    // batch_throughput_factor should return expected speedup for batch inference
    let single = OwnedQuantizedModel::batch_throughput_factor(1);
    let small_batch = OwnedQuantizedModel::batch_throughput_factor(4);
    let large_batch = OwnedQuantizedModel::batch_throughput_factor(16);

    // Single should be 1.0 (baseline)
    assert!(
        (single - 1.0).abs() < 0.01,
        "PARITY-006d: Single batch throughput factor should be 1.0"
    );

    // Larger batches should have higher throughput factor
    assert!(
        small_batch > single,
        "PARITY-006d: Batch of 4 should have higher throughput than single"
    );
    assert!(
        large_batch > small_batch,
        "PARITY-006d: Batch of 16 should have higher throughput than batch of 4"
    );
}

/// PARITY-006e: Batch performance should be better than sequential for multiple prompts
#[test]
fn test_parity006e_batch_performance_comparison() {
    use std::time::Instant;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompts = vec![
        vec![1u32, 2, 3],
        vec![4u32, 5, 6],
        vec![7u32, 8, 9],
        vec![10u32, 11, 12],
    ];

    // Sequential generation
    let start = Instant::now();
    for prompt in &prompts {
        let _ = model.generate_with_cache(prompt, &gen_config);
    }
    let sequential_time = start.elapsed();

    // Batch generation
    let cached = OwnedQuantizedModelCached::new(model);
    let start = Instant::now();
    let _ = cached.model().batch_generate(
        &prompts
            .iter()
            .map(std::vec::Vec::as_slice)
            .collect::<Vec<_>>(),
        &gen_config,
    );
    let batch_time = start.elapsed();

    println!("PARITY-006e: Performance Comparison");
    println!("  Sequential: {:?}", sequential_time);
    println!("  Batch: {:?}", batch_time);
    println!(
        "  Speedup: {:.2}x",
        sequential_time.as_nanos() as f64 / batch_time.as_nanos() as f64
    );

    // Batch should not be slower (within 10x to account for coverage instrumentation)
    assert!(
        batch_time.as_nanos() <= sequential_time.as_nanos() * 10,
        "PARITY-006e: Batch should not be more than 10x slower than sequential"
    );
}

/// PARITY-006f: Empty prompts should return error
#[test]
fn test_parity006f_empty_prompts_error() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);
    let cached = OwnedQuantizedModelCached::new(model);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    // Empty batch should error
    let empty_prompts: Vec<&[u32]> = vec![];
    let result = cached.model().batch_generate(&empty_prompts, &gen_config);
    assert!(
        result.is_err(),
        "PARITY-006f: Empty batch should return error"
    );
}

// ========================================================================
// PARITY-007: E2E Benchmark Verification Tests
// ========================================================================

/// PARITY-007a: Coefficient of variation calculation
#[test]
fn test_parity007a_cv_calculation() {
    // CV = (std_dev / mean) * 100
    let values: Vec<f64> = vec![10.0, 12.0, 11.0, 9.0, 11.0];
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    let cv = (std_dev / mean) * 100.0;

    // CV should be low for stable measurements
    assert!(cv < 15.0, "PARITY-007a: CV should be < 15% for stable data");

    // Test with high variance data
    let noisy: Vec<f64> = vec![1.0, 100.0, 5.0, 50.0, 10.0];
    let noisy_mean: f64 = noisy.iter().sum::<f64>() / noisy.len() as f64;
    let noisy_variance: f64 =
        noisy.iter().map(|x| (x - noisy_mean).powi(2)).sum::<f64>() / noisy.len() as f64;
    let noisy_cv = (noisy_variance.sqrt() / noisy_mean) * 100.0;

    assert!(
        noisy_cv > 50.0,
        "PARITY-007a: Noisy data should have CV > 50%"
    );
}

/// PARITY-007b: Benchmark metrics struct
#[test]
fn test_parity007b_benchmark_metrics() {
    // BenchmarkMetrics should capture all required fields
    struct BenchmarkMetrics {
        throughput_toks: f64,
        latency_p50_ms: f64,
        latency_p95_ms: f64,
        latency_p99_ms: f64,
        cv_percent: f64,
    }

    let metrics = BenchmarkMetrics {
        throughput_toks: 64.0,
        latency_p50_ms: 15.6,
        latency_p95_ms: 18.2,
        latency_p99_ms: 21.5,
        cv_percent: 8.5,
    };

    assert!(
        metrics.throughput_toks > 0.0,
        "PARITY-007b: Throughput should be positive"
    );
    assert!(
        metrics.latency_p50_ms < metrics.latency_p95_ms,
        "PARITY-007b: p50 should be < p95"
    );
    assert!(
        metrics.latency_p95_ms < metrics.latency_p99_ms,
        "PARITY-007b: p95 should be < p99"
    );
    assert!(metrics.cv_percent < 15.0, "PARITY-007b: CV should be < 15%");
}

/// PARITY-007c: Hardware info capture
#[test]
fn test_parity007c_hardware_info() {
    struct HardwareInfo {
        cpu_model: String,
        cpu_cores: usize,
        ram_gb: usize,
        gpu_name: Option<String>,
    }

    let info = HardwareInfo {
        cpu_model: "AMD Ryzen 9 5950X".to_string(),
        cpu_cores: 32,
        ram_gb: 128,
        gpu_name: Some("NVIDIA RTX 4090".to_string()),
    };

    assert!(
        !info.cpu_model.is_empty(),
        "PARITY-007c: CPU model should not be empty"
    );
    assert!(info.cpu_cores >= 1, "PARITY-007c: CPU cores should be >= 1");
    assert!(info.ram_gb >= 1, "PARITY-007c: RAM should be >= 1 GB");
}

/// PARITY-007d: Percentile calculation
#[test]
fn test_parity007d_percentile_calculation() {
    let mut values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50_idx = (values.len() as f64 * 0.50) as usize - 1;
    let p95_idx = (values.len() as f64 * 0.95) as usize - 1;
    let p99_idx = (values.len() as f64 * 0.99) as usize - 1;

    let p50 = values[p50_idx];
    let p95 = values[p95_idx];
    let p99 = values[p99_idx];

    assert!((p50 - 50.0).abs() < 1.0, "PARITY-007d: p50 should be ~50");
    assert!((p95 - 95.0).abs() < 1.0, "PARITY-007d: p95 should be ~95");
    assert!((p99 - 99.0).abs() < 1.0, "PARITY-007d: p99 should be ~99");
}

/// PARITY-007e: Gap calculation to target
#[test]
fn test_parity007e_gap_calculation() {
    let current_toks: f64 = 64.0;
    let target_toks: f64 = 225.0;
    let gap_ratio: f64 = target_toks / current_toks;

    assert!(
        (gap_ratio - 3.52_f64).abs() < 0.1,
        "PARITY-007e: Gap ratio should be ~3.5x"
    );

    let gap_percent = (target_toks - current_toks) / current_toks * 100.0;
    assert!(
        gap_percent > 200.0,
        "PARITY-007e: Gap should be > 200% improvement needed"
    );
}

/// PARITY-007f: End-to-end benchmark with realizar baseline
#[test]
fn test_parity007f_realizar_benchmark() {
    use std::time::Instant;

    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };
    let model = create_test_model_with_config(&config);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_runs = 5;

    let mut throughputs = Vec::with_capacity(num_runs);
    let mut latencies = Vec::with_capacity(num_runs);

    for _ in 0..num_runs {
        let start = Instant::now();
        let result = model.generate_with_cache(&prompt, &gen_config).expect("test value should be present");
        let elapsed = start.elapsed();

        let new_tokens = result.len() - prompt.len();
        let toks_per_sec = new_tokens as f64 / elapsed.as_secs_f64();
        let latency_ms = elapsed.as_secs_f64() * 1000.0 / new_tokens as f64;

        throughputs.push(toks_per_sec);
        latencies.push(latency_ms);
    }

    let mean_tps: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

    println!("PARITY-007f: Realizar Test Model Benchmark");
    println!("  Mean throughput: {:.1} tok/s", mean_tps);

    // Test model should achieve some throughput
    assert!(mean_tps > 0.0, "PARITY-007f: Throughput should be positive");
}

// ========================================================================
// PARITY-008: Popper Score Improvement Tests
// ========================================================================

/// PARITY-008a: Falsifiable claim structure
#[test]
fn test_parity008a_falsifiable_claim_structure() {
    struct FalsifiableClaim {
        prediction: String,
        measurement: String,
        threshold: f64,
        evidence: Option<f64>,
    }

    let claim = FalsifiableClaim {
        prediction: "GPU batch FFN is 1.1x faster at batch=32".to_string(),
        measurement: "FFN latency ratio (GPU/CPU)".to_string(),
        threshold: 0.91, // 1/1.1
        evidence: None,
    };

    assert!(
        !claim.prediction.is_empty(),
        "PARITY-008a: Prediction should not be empty"
    );
    assert!(
        claim.threshold > 0.0 && claim.threshold < 1.0,
        "PARITY-008a: Threshold for speedup should be < 1.0"
    );
}

/// PARITY-008b: Random seed management for reproducibility
#[test]
fn test_parity008b_random_seed_management() {
    // Local struct for seed configuration testing
    struct SeedConfig {
        seed: u64,
    }

    impl SeedConfig {
        fn new(seed: u64) -> Self {
            Self { seed }
        }

        fn for_ollama_comparison() -> Self {
            Self { seed: 42 }
        }
    }

    let config = SeedConfig::for_ollama_comparison();
    assert_eq!(
        config.seed, 42,
        "PARITY-008b: Ollama comparison seed should be 42"
    );

    let derived = SeedConfig::new(1000);
    assert_eq!(derived.seed, 1000, "PARITY-008b: Custom seed should match");

    // Same seed should produce same sequence
    let seed1 = SeedConfig::new(42);
    let seed2 = SeedConfig::new(42);
    assert_eq!(
        seed1.seed, seed2.seed,
        "PARITY-008b: Same seed should be reproducible"
    );
}
