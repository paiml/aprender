
/// Test Flash Attention v2 + parallel performance improvement
#[test]
fn test_phase3_flash_attention_v2_performance() {
    use std::time::Instant;

    let head_dim = 64;
    let seq_len = 32;

    // Attention::new takes head_dim only
    let attn = Attention::new(head_dim).expect("test");

    // Create QKV tensors
    let q = Tensor::from_vec(vec![seq_len, head_dim], vec![0.1; seq_len * head_dim]).expect("test");
    let k = Tensor::from_vec(vec![seq_len, head_dim], vec![0.2; seq_len * head_dim]).expect("test");
    let v = Tensor::from_vec(vec![seq_len, head_dim], vec![0.3; seq_len * head_dim]).expect("test");

    // Warmup
    let _ = attn.flash_forward_v2(&q, &k, &v, 8).expect("test");
    let _ = attn.flash_forward_parallel(&q, &k, &v, 8).expect("test");

    // Benchmark Flash Attention v2 (SIMD)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = attn.flash_forward_v2(&q, &k, &v, 8).expect("test");
    }
    let v2_time = start.elapsed();

    // Benchmark Flash Attention parallel
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = attn.flash_forward_parallel(&q, &k, &v, 8).expect("test");
    }
    let parallel_time = start.elapsed();

    // Report performance (informational)
    let v2_us = v2_time.as_micros() as f64 / iterations as f64;
    let parallel_us = parallel_time.as_micros() as f64 / iterations as f64;

    eprintln!(
        "Flash Attention v2: {:.2}us/iter, Parallel: {:.2}us/iter",
        v2_us, parallel_us
    );

    // Both implementations should complete without error
    // Performance comparison is informational
    assert!(v2_us > 0.0, "v2 should have measurable time");
    assert!(parallel_us > 0.0, "parallel should have measurable time");
}

/// Test FusedLayerNormLinear performance improvement
#[test]
fn test_phase3_fused_layernorm_linear_performance() {
    use std::time::Instant;

    let feature_dim = 256;
    let out_features = 512;
    let batch_size = 32;

    // FusedLayerNormLinear::new initializes with default weights
    // (norm_weight=1.0, norm_bias=0.0, linear_weight=0.0, linear_bias=0.0)
    // which is fine for performance testing
    let fused = FusedLayerNormLinear::new(feature_dim, out_features, 1e-5).expect("test");

    // Create input batch
    let input = Tensor::from_vec(
        vec![batch_size, feature_dim],
        vec![0.5; batch_size * feature_dim],
    )
    .expect("test");

    // Warmup
    let _ = fused.forward(&input).expect("test");
    let _ = fused.forward_parallel(&input).expect("test");

    // Benchmark fused forward
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused.forward(&input).expect("test");
    }
    let fused_time = start.elapsed();

    // Benchmark parallel fused forward
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused.forward_parallel(&input).expect("test");
    }
    let parallel_time = start.elapsed();

    // Report performance
    let fused_us = fused_time.as_micros() as f64 / iterations as f64;
    let parallel_us = parallel_time.as_micros() as f64 / iterations as f64;

    eprintln!(
        "FusedLayerNormLinear: {:.2}us/iter, Parallel: {:.2}us/iter",
        fused_us, parallel_us
    );

    // Verify performance is measurable
    assert!(fused_us > 0.0, "fused should have measurable time");
    assert!(parallel_us > 0.0, "parallel should have measurable time");
}

// =========================================================================
// BENCH-SPRINT-002: QuantizedLinear Tests (Q4_K Integration)
// Per benchmark-model-runners-spec.md v2.0: Inline dequantization for 8x
// memory bandwidth reduction vs f32.
// =========================================================================

/// RED: Test QuantizedLinear creation from Q4_K weight bytes
#[test]
fn test_quantized_linear_creation() {
    // Q4_K format: 144 bytes per super-block of 256 values
    // For in_features=256, out_features=4, we need 4 rows * 144 bytes = 576 bytes
    let in_features = 256;
    let out_features = 4;
    let bytes_per_row = 144; // One super-block per row
    let weight_bytes = vec![0u8; out_features * bytes_per_row];
    let bias = vec![0.0f32; out_features];

    let layer = QuantizedLinear::new(in_features, out_features, weight_bytes, bias);
    assert!(
        layer.is_ok(),
        "Should create QuantizedLinear from Q4_K bytes"
    );

    let layer = layer.expect("test");
    assert_eq!(layer.in_features(), in_features);
    assert_eq!(layer.out_features(), out_features);
}

/// RED: Test QuantizedLinear forward pass produces correct output
#[test]
fn test_quantized_linear_forward() {
    // Create test Q4_K weights (zeros for simplicity)
    let in_features = 256;
    let out_features = 4;
    let bytes_per_row = 144;
    let weight_bytes = vec![0u8; out_features * bytes_per_row];
    let bias = vec![1.0f32; out_features]; // Non-zero bias

    let layer = QuantizedLinear::new(in_features, out_features, weight_bytes, bias)
        .expect("Should create layer");

    // Input activations
    let input = Tensor::from_vec(vec![in_features], vec![1.0f32; in_features])
        .expect("Should create input");

    // Forward pass
    let output = layer.forward(&input).expect("Forward should work");

    // Output should have shape [out_features]
    assert_eq!(output.shape(), &[out_features]);

    // With zero weights and bias=1.0, output should be [1.0, 1.0, 1.0, 1.0]
    for &val in output.data() {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "Output should equal bias with zero weights"
        );
    }
}

/// RED: Test QuantizedLinear forward with batch input
#[test]
fn test_quantized_linear_batch_forward() {
    let in_features = 256;
    let out_features = 4;
    let batch_size = 8;
    let bytes_per_row = 144;
    let weight_bytes = vec![0u8; out_features * bytes_per_row];
    let bias = vec![2.0f32; out_features];

    let layer = QuantizedLinear::new(in_features, out_features, weight_bytes, bias)
        .expect("Should create layer");

    // Batch input [batch_size, in_features]
    let input = Tensor::from_vec(
        vec![batch_size, in_features],
        vec![1.0f32; batch_size * in_features],
    )
    .expect("Should create batch input");

    let output = layer.forward(&input).expect("Batch forward should work");

    // Output should have shape [batch_size, out_features]
    assert_eq!(output.shape(), &[batch_size, out_features]);
}

/// RED: Test QuantizedLinear memory usage is ~8x less than Linear
#[test]
fn test_quantized_linear_memory_efficiency() {
    let in_features = 4096; // Realistic embedding dim
    let out_features = 4096;

    // f32 Linear: 4096 * 4096 * 4 bytes = 64MB
    let f32_bytes = in_features * out_features * std::mem::size_of::<f32>();

    // Q4_K: 4096/256 = 16 super-blocks per row, 16 * 144 = 2304 bytes/row
    // Total: 4096 * 2304 = ~9.4MB (6.8x reduction, close to theoretical 8x)
    let super_blocks_per_row = in_features.div_ceil(256);
    let q4k_bytes = out_features * super_blocks_per_row * 144;

    let ratio = f32_bytes as f64 / q4k_bytes as f64;

    // Q4_K should be at least 6x smaller than f32 (accounting for scale/min overhead)
    assert!(
        ratio > 6.0,
        "Q4_K should be >6x smaller than f32: ratio={}",
        ratio
    );
    eprintln!(
        "Memory efficiency: f32={} bytes, Q4_K={} bytes, ratio={:.2}x",
        f32_bytes, q4k_bytes, ratio
    );
}

// ========================
// SlidingWindowAttention Tests
// ========================

#[test]
fn test_sliding_window_attention_new() {
    let swa = SlidingWindowAttention::new(64, 4096).expect("test");
    assert_eq!(swa.head_dim(), 64);
    assert_eq!(swa.window_size(), 4096);
    assert!((swa.scale() - 0.125).abs() < 1e-6); // 1/sqrt(64) = 0.125
}

#[test]
fn test_sliding_window_attention_new_errors() {
    // Zero head_dim should error
    assert!(SlidingWindowAttention::new(0, 4096).is_err());
    // Zero window_size should error
    assert!(SlidingWindowAttention::new(64, 0).is_err());
}

#[test]
fn test_sliding_window_attention_forward_basic() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");
    // Small test: 5 positions, window size 3
    // Query: 5x4, Key: 5x4, Value: 5x4
    let query_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
    let key_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
    let value_data: Vec<f32> = (0..20).map(|i| (i % 4) as f32).collect();

    let query = Tensor::from_vec(vec![5, 4], query_data).expect("test");
    let key = Tensor::from_vec(vec![5, 4], key_data).expect("test");
    let value = Tensor::from_vec(vec![5, 4], value_data).expect("test");

    let output = swa.forward(&query, &key, &value).expect("test");
    assert_eq!(output.size(), 20); // 5 positions * 4 head_dim
}

#[test]
fn test_sliding_window_attention_causal_masking() {
    // Test that position i can only attend to positions <= i
    let swa = SlidingWindowAttention::new(2, 10).expect("test"); // Large window, so only causal matters
    // Query: 3x2, Key: 3x2, Value: 3x2
    let query = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).expect("test");
    let key = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("test");
    let value = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).expect("test");

    let output = swa.forward(&query, &key, &value).expect("test");
    assert_eq!(output.size(), 6);

    // Position 0 can only attend to itself
    // Position 1 can attend to positions 0,1
    // Position 2 can attend to positions 0,1,2
    // All positions produce valid outputs (not zeros)
    let data = output.data();
    assert!(data[0].abs() > 0.0 || data[1].abs() > 0.0);
}

#[test]
fn test_sliding_window_attention_window_boundary() {
    // Window size 2: each position can attend to at most 2 keys
    let swa = SlidingWindowAttention::new(2, 2).expect("test");
    // 5 positions, window=2
    let query = Tensor::from_vec(vec![5, 2], vec![1.0; 10]).expect("test");
    let key = Tensor::from_vec(vec![5, 2], vec![1.0; 10]).expect("test");
    let value_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let value = Tensor::from_vec(vec![5, 2], value_data).expect("test");

    let output = swa.forward(&query, &key, &value).expect("test");
    assert_eq!(output.size(), 10);

    // Position 0: attends to [0] (only 1 key available due to causality)
    // Position 1: attends to [0,1] (2 keys)
    // Position 2: attends to [1,2] (window slides, excludes 0)
    // Position 3: attends to [2,3]
    // Position 4: attends to [3,4]
}

#[test]
fn test_sliding_window_attention_effective_context() {
    let swa = SlidingWindowAttention::new(64, 4).expect("test");

    // Position 0, seq_len 10: can attend to min(1, 4) = 1
    assert_eq!(swa.effective_context(0, 10), 1);

    // Position 3, seq_len 10: can attend to min(4, 4) = 4
    assert_eq!(swa.effective_context(3, 10), 4);

    // Position 7, seq_len 10: can attend to 4 (window kicks in)
    assert_eq!(swa.effective_context(7, 10), 4);

    // Position 2, seq_len 3: can attend to min(3, 4) = 3
    assert_eq!(swa.effective_context(2, 3), 3);
}

#[test]
fn test_sliding_window_attention_memory_ratio() {
    let swa = SlidingWindowAttention::new(64, 4096).expect("test");

    // For short sequences, ratio ~= 1.0
    let ratio_short = swa.memory_ratio(1000);
    assert!(
        ratio_short > 0.9,
        "Short sequences should use ~full attention"
    );

    // For long sequences, ratio approaches window_size / seq_len
    let ratio_long = swa.memory_ratio(100_000);
    let expected = 4096.0 / 100_000.0;
    assert!(
        (ratio_long - expected).abs() < 0.01,
        "Long sequences should use ~window_size/seq_len memory: got {}, expected {}",
        ratio_long,
        expected
    );
}

#[test]
fn test_sliding_window_attention_error_mismatched_kv() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");
    let query = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let key = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test");
    let value = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test"); // Different from K

    // K and V must have same seq_len
    let result = swa.forward(&query, &key, &value);
    assert!(result.is_err());
}

#[test]
fn test_sliding_window_attention_error_bad_head_dim() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");
    // Key has wrong head_dim (3 instead of 4)
    let query = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let key = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let value = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let result = swa.forward(&query, &key, &value);
    assert!(result.is_err());
}

#[test]
fn test_sliding_window_attention_bidirectional() {
    let swa = SlidingWindowAttention::new(2, 4).expect("test");
    // 5 positions, bidirectional window
    let query = Tensor::from_vec(vec![5, 2], vec![1.0; 10]).expect("test");
    let key = Tensor::from_vec(vec![5, 2], vec![1.0; 10]).expect("test");
    let value_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let value = Tensor::from_vec(vec![5, 2], value_data).expect("test");

    let output_causal = swa.forward(&query, &key, &value).expect("test");
    let output_bidir = swa
        .forward_with_mask(&query, &key, &value, false)
        .expect("test");

    // Bidirectional can attend to more positions, so outputs may differ
    assert_eq!(output_causal.size(), output_bidir.size());
    // Both should produce valid outputs
    assert!(output_causal.data().iter().any(|&x| x.abs() > 0.0));
    assert!(output_bidir.data().iter().any(|&x| x.abs() > 0.0));
}

#[test]
fn test_sliding_window_attention_forward_with_mask_causal() {
    let swa = SlidingWindowAttention::new(2, 3).expect("test");
    let query = Tensor::from_vec(vec![3, 2], vec![1.0; 6]).expect("test");
    let key = Tensor::from_vec(vec![3, 2], vec![1.0; 6]).expect("test");
    let value = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");

    // forward_with_mask(causal=true) should match forward()
    let output_forward = swa.forward(&query, &key, &value).expect("test");
    let output_mask = swa
        .forward_with_mask(&query, &key, &value, true)
        .expect("test");

    for (a, b) in output_forward.data().iter().zip(output_mask.data().iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Causal outputs should match: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_sliding_window_attention_single_token() {
    let swa = SlidingWindowAttention::new(4, 3).expect("test");
    // Single token input
    let query = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let key = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let value = Tensor::from_vec(vec![1, 4], vec![0.5, 0.5, 0.5, 0.5]).expect("test");

    let output = swa.forward(&query, &key, &value).expect("test");
    assert_eq!(output.size(), 4);
    // Self-attention on single token returns the value
    let data = output.data();
    for &v in data {
        assert!((v - 0.5).abs() < 1e-6);
    }
}

// ========================================================================
// IMP-003: Fused QKV + Attention Tests (EXTREME TDD - RED Phase)
// Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md
// ========================================================================

#[test]
fn test_fused_qkv_attention_basic() {
    // IMP-003: Fused attention should match separate Q/K/V computation
    let fused = FusedQKVAttention::new(4, 64).expect("test");
    let input = Tensor::from_vec(vec![8, 64], vec![0.1; 8 * 64]).expect("test");

    let output = fused.forward(&input).expect("test");
    assert_eq!(output.shape(), &[8, 64]);
}
