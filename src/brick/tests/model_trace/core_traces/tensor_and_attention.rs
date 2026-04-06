use super::super::super::super::*;

// ========================================================================
// F250-F270: Model-Level Inference Tracing Tests (Phase 13)
// ========================================================================

/// F250: TensorStats computes correctly with known input
#[test]
fn test_f250_tensor_stats_correct() {
    // Known input: [1, 2, 3, 4, 5]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = TensorStats::from_slice(&data);

    assert_eq!(stats.count, 5);
    assert!((stats.min - 1.0).abs() < 1e-6);
    assert!((stats.max - 5.0).abs() < 1e-6);
    assert!((stats.mean - 3.0).abs() < 1e-6);

    // Standard deviation of [1,2,3,4,5] = sqrt(2.5) ~ 1.5811
    assert!((stats.std - 1.5811).abs() < 0.001);

    // L2 norm = sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55) ~ 7.416
    assert!((stats.l2_norm - 7.416).abs() < 0.01);

    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert!(!stats.has_anomaly());
}

/// F251: NaN detection has 100% recall
#[test]
fn test_f251_nan_detection() {
    // Inject NaN values
    let data = vec![1.0, 2.0, f32::NAN, 4.0, f32::NAN, 6.0];
    let stats = TensorStats::from_slice(&data);

    // Must detect both NaN values
    assert_eq!(stats.nan_count, 2);
    assert!(stats.has_anomaly());
    assert!(stats.anomaly_description().unwrap().contains("NaN"));
}

/// F252: Explosion detection triggers on large values
#[test]
fn test_f252_explosion_detection() {
    // Inject explosion: value > 1e6
    let data = vec![1.0, 2.0, 1.5e6, 4.0, 5.0];
    let stats = TensorStats::from_slice(&data);

    assert!(stats.has_anomaly());
    assert!(stats.anomaly_description().unwrap().contains("Explosion"));

    // Also test min explosion
    let data2 = vec![-2e6, 1.0, 2.0];
    let stats2 = TensorStats::from_slice(&data2);
    assert!(stats2.has_anomaly());
}

/// F253: Attention top-k is sorted in descending order
#[test]
fn test_f253_attention_topk_sorted() {
    let weights = vec![0.1, 0.3, 0.05, 0.4, 0.15];
    let trace = AttentionWeightTrace::from_weights(0, 0, 4, &weights, 3);

    // Top-k weights should be descending
    assert_eq!(trace.top_k_positions.len(), 3);
    assert!(trace.top_k_weights.windows(2).all(|w| w[0] >= w[1]));

    // Highest weight is 0.4 at position 3
    assert_eq!(trace.top_k_positions[0], 3);
    assert!((trace.top_k_weights[0] - 0.4).abs() < 1e-6);
}

/// F254: Attention weights sum to approximately 1
#[test]
fn test_f254_attention_weights_sum() {
    // Create normalized attention weights
    let weights = vec![0.2, 0.3, 0.15, 0.25, 0.1];
    let total: f32 = weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-5);

    let trace = AttentionWeightTrace::from_weights(0, 0, 4, &weights, 5);
    let recovered: f32 = trace.top_k_weights.iter().sum::<f32>() + trace.tail_mass;
    assert!((recovered - 1.0).abs() < 1e-5);
}

/// F255: Entropy computation is correct
#[test]
fn test_f255_entropy_computation() {
    // Uniform distribution: entropy = ln(n)
    let n = 4;
    let uniform_weights: Vec<f32> = vec![0.25; n];
    let trace = AttentionWeightTrace::from_weights(0, 0, 3, &uniform_weights, n);

    // Entropy of uniform distribution = ln(4) ~ 1.386
    let expected_entropy = (n as f32).ln();
    assert!((trace.entropy - expected_entropy).abs() < 0.01);

    // Concentrated distribution: lower entropy
    let concentrated = vec![0.9, 0.05, 0.03, 0.02];
    let trace2 = AttentionWeightTrace::from_weights(0, 0, 3, &concentrated, n);
    assert!(trace2.entropy < trace.entropy);
}

/// F256: Logit tracking is accurate with deterministic model
#[test]
fn test_f256_logit_tracking() {
    let mut trace = LogitEvolutionTrace::new(0, 1.0, 1.0);

    // Track token 42
    let token = trace.track_token(42, "test".to_string());
    token.record_layer(1.5, 10);
    token.record_layer(2.0, 5);
    token.record_layer(3.0, 1);

    assert_eq!(token.per_layer_logit.len(), 3);
    assert_eq!(token.per_layer_rank.len(), 3);
    assert!((token.per_layer_logit[2] - 3.0).abs() < 1e-6);
    assert_eq!(token.per_layer_rank[2], 1);
}

/// F257: Rank computation is correct vs argsort
#[test]
fn test_f257_rank_computation() {
    let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];

    // Token 3 (value 5.0) should be rank 0 (highest)
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 3), 0);

    // Token 4 (value 4.0) should be rank 1
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 4), 1);

    // Token 1 (value 3.0) should be rank 2
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 1), 2);

    // Token 0 (value 1.0) should be rank 4 (lowest)
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 0), 4);
}

/// F258: Cosine similarity is in range [-1, 1]
#[test]
fn test_f258_cosine_similarity_range() {
    // Identical vectors: cosine = 1
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let trace = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &b, QuantType::F32);
    assert!((trace.cosine_similarity - 1.0).abs() < 1e-5);

    // Opposite vectors: cosine = -1
    let c = vec![-1.0, -2.0, -3.0];
    let trace2 = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &c, QuantType::F32);
    assert!((trace2.cosine_similarity - (-1.0)).abs() < 1e-5);

    // Orthogonal vectors: cosine = 0
    let d = vec![1.0, 0.0, 0.0];
    let e = vec![0.0, 1.0, 0.0];
    let trace3 = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &d, &e, QuantType::F32);
    assert!(trace3.cosine_similarity.abs() < 1e-5);

    // All results must be in [-1, 1]
    assert!(trace.cosine_similarity >= -1.0 && trace.cosine_similarity <= 1.0);
    assert!(trace2.cosine_similarity >= -1.0 && trace2.cosine_similarity <= 1.0);
    assert!(trace3.cosine_similarity >= -1.0 && trace3.cosine_similarity <= 1.0);
}

/// F259: SNR dB computation is correct
#[test]
fn test_f259_snr_db_computation() {
    // Identical signals: infinite SNR
    let a = vec![1.0, 2.0, 3.0];
    let trace = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &a, QuantType::F32);
    assert!(trace.snr_db.is_infinite() && trace.snr_db > 0.0);

    // Known SNR: signal [1,1,1], noise [0.1, 0.1, 0.1]
    // Signal power = 1, Noise power = 0.01, SNR = 100 = 20 dB
    let signal = vec![1.0, 1.0, 1.0];
    let noisy = vec![1.1, 1.1, 1.1];
    let trace2 =
        QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &noisy, &signal, QuantType::F32);
    // SNR should be around 20 dB
    assert!(trace2.snr_db > 15.0 && trace2.snr_db < 25.0);
}

/// F260: KV cache size tracking is exact
#[test]
fn test_f260_kv_cache_size_tracking() {
    let mut trace = KvCacheStateTrace::new(0, 2048);
    trace.cache_size_bytes = 1024 * 1024; // 1 MB
    trace.valid_positions = 512;

    assert_eq!(trace.cache_size_bytes, 1024 * 1024);
    assert_eq!(trace.valid_positions, 512);
    assert_eq!(trace.max_positions, 2048);

    let utilization = trace.utilization();
    assert!((utilization - 0.25).abs() < 1e-6); // 512/2048 = 0.25
}

/// F261: Eviction counting is exact
#[test]
fn test_f261_eviction_counting() {
    let mut session = KvCacheSessionTrace::default();

    // Add steps with evictions
    let mut step1 = KvCacheStateTrace::new(0, 100);
    step1.evictions_this_step = 5;
    step1.cache_hit_rate = 0.8;
    session.add_step(step1);

    let mut step2 = KvCacheStateTrace::new(1, 100);
    step2.evictions_this_step = 3;
    step2.cache_hit_rate = 0.7;
    session.add_step(step2);

    assert_eq!(session.total_evictions, 8); // 5 + 3 exact
    assert_eq!(session.steps.len(), 2);
}

/// F262: Hit rate is always in [0, 1]
#[test]
fn test_f262_hit_rate_bounded() {
    let mut session = KvCacheSessionTrace::default();

    for i in 0..10 {
        let mut step = KvCacheStateTrace::new(i, 100);
        step.cache_hit_rate = (i as f32) / 10.0; // 0.0 to 0.9
        session.add_step(step);
    }

    // Average hit rate should be bounded
    assert!(session.avg_hit_rate >= 0.0);
    assert!(session.avg_hit_rate <= 1.0);

    // Verify average: (0 + 0.1 + ... + 0.9) / 10 = 4.5 / 10 = 0.45
    assert!((session.avg_hit_rate - 0.45).abs() < 0.01);
}

/// Additional: QuantType bits and compression ratio
#[test]
fn test_quant_type_bits() {
    assert_eq!(QuantType::F32.bits_per_element(), 32.0);
    assert_eq!(QuantType::F16.bits_per_element(), 16.0);
    assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);

    // Compression ratios
    assert!((QuantType::F32.compression_ratio() - 1.0).abs() < 0.01);
    assert!((QuantType::F16.compression_ratio() - 2.0).abs() < 0.01);
    assert!((QuantType::Q4_K.compression_ratio() - 7.11).abs() < 0.1);
}

/// Additional: AttentionWeightTrace diagnostic patterns
#[test]
fn test_attention_diagnostics() {
    // Attention sink pattern
    let sink_weights = vec![0.9, 0.05, 0.03, 0.02];
    let trace = AttentionWeightTrace::from_weights(0, 0, 3, &sink_weights, 4);
    assert!(trace.is_attention_sink(0.5));

    // Recency bias pattern
    let recency_weights = vec![0.05, 0.05, 0.1, 0.8]; // High weight on recent position
    let trace2 = AttentionWeightTrace::from_weights(0, 0, 3, &recency_weights, 4);
    assert!(trace2.has_recency_bias(2, 0.7));
}

/// Additional: TokenLogitEvolution decisive layer detection
#[test]
fn test_token_decisive_layer() {
    let mut token = TokenLogitEvolution::new(42, "test".to_string());

    // Gradual change: decisive layer should be where biggest jump occurs
    token.record_layer(1.0, 100); // Layer 0
    token.record_layer(1.5, 50); // Layer 1: rank dropped 50
    token.record_layer(2.0, 48); // Layer 2: rank dropped 2
    token.record_layer(3.0, 1); // Layer 3: rank dropped 47

    let decisive = token.decisive_layer();
    assert_eq!(decisive, Some(1)); // Biggest jump was 100->50 at layer 1
}

/// Additional: KvCacheSessionTrace thrashing detection
#[test]
fn test_kv_cache_thrashing() {
    let mut session = KvCacheSessionTrace::default();

    // Simulate thrashing: high evictions, low hit rate
    for i in 0..10 {
        let mut step = KvCacheStateTrace::new(i, 100);
        step.evictions_this_step = 10;
        step.cache_hit_rate = 0.3;
        session.add_step(step);
    }

    assert!(session.has_thrashing(50, 0.5)); // 100 evictions, 0.3 hit rate

    // Non-thrashing scenario
    let mut healthy = KvCacheSessionTrace::default();
    for i in 0..10 {
        let mut step = KvCacheStateTrace::new(i, 100);
        step.evictions_this_step = 1;
        step.cache_hit_rate = 0.95;
        healthy.add_step(step);
    }

    assert!(!healthy.has_thrashing(50, 0.5));
}
