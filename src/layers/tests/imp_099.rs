
/// IMP-099: Benchmark fused Q4_K matvec vs f32 matvec
///
/// Compares memory bandwidth and compute performance of:
/// - f32 matvec: 4 bytes per weight, SIMD accumulation
/// - Q4_K matvec: ~0.56 bytes per weight, fused dequant+dot
#[test]
#[ignore] // Run manually: cargo test --release test_imp_099_q4k_vs_f32_benchmark -- --nocapture --ignored
fn test_imp_099_q4k_vs_f32_benchmark() {
    use crate::quantize::{fused_q4k_parallel_matvec, QK_K};
    use std::time::Instant;

    println!("\n=== IMP-099: Q4_K vs f32 Matmul Benchmark ===\n");

    // Realistic dimensions for transformer layer
    // Qwen 2.5 1.5B: hidden=1536, intermediate=8960
    let in_dim: usize = 1536; // Must be multiple of 256 for Q4_K
    let out_dim: usize = 8960;
    let iterations = 100;

    // Create test data
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();

    // Q4_K weights: 144 bytes per 256 values
    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * 144;
    let q4k_weight_size = out_dim * bytes_per_row;
    let q4k_weights: Vec<u8> = (0..q4k_weight_size).map(|i| (i % 256) as u8).collect();

    // f32 weights: 4 bytes per value
    let f32_weight_size = in_dim * out_dim;
    let f32_weights: Vec<f32> = (0..f32_weight_size)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();

    println!("Dimensions: {} x {}", in_dim, out_dim);
    println!("Q4_K weight size: {:.2} MB", q4k_weight_size as f64 / 1e6);
    println!(
        "f32 weight size: {:.2} MB",
        (f32_weight_size * 4) as f64 / 1e6
    );
    println!(
        "Compression ratio: {:.1}x\n",
        (f32_weight_size * 4) as f64 / q4k_weight_size as f64
    );

    // Warmup
    let _ = fused_q4k_parallel_matvec(&q4k_weights, &activations, in_dim, out_dim);
    let _ = crate::gpu::cpu_matmul(&activations, &f32_weights, 1, in_dim, out_dim);

    // Benchmark Q4_K fused matvec
    let q4k_start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_parallel_matvec(&q4k_weights, &activations, in_dim, out_dim);
    }
    let q4k_elapsed = q4k_start.elapsed();
    let q4k_per_op = q4k_elapsed.as_secs_f64() / iterations as f64;

    // Benchmark f32 matvec (using cpu_matmul which calls cpu_vector_matmul for m=1)
    let f32_start = Instant::now();
    for _ in 0..iterations {
        let _ = crate::gpu::cpu_matmul(&activations, &f32_weights, 1, in_dim, out_dim);
    }
    let f32_elapsed = f32_start.elapsed();
    let f32_per_op = f32_elapsed.as_secs_f64() / iterations as f64;

    // Calculate metrics
    let q4k_gops = (in_dim * out_dim) as f64 / q4k_per_op / 1e9;
    let f32_gops = (in_dim * out_dim) as f64 / f32_per_op / 1e9;
    let q4k_bw = q4k_weight_size as f64 / q4k_per_op / 1e9;
    let f32_bw = (f32_weight_size * 4) as f64 / f32_per_op / 1e9;

    println!("=== Results ({} iterations) ===", iterations);
    println!("Q4_K fused:");
    println!("  Time: {:.3} ms/op", q4k_per_op * 1000.0);
    println!("  Throughput: {:.2} GOPS", q4k_gops);
    println!("  Bandwidth: {:.2} GB/s", q4k_bw);
    println!();
    println!("f32 matvec:");
    println!("  Time: {:.3} ms/op", f32_per_op * 1000.0);
    println!("  Throughput: {:.2} GOPS", f32_gops);
    println!("  Bandwidth: {:.2} GB/s", f32_bw);
    println!();
    println!("Speedup (Q4_K vs f32): {:.2}x", f32_per_op / q4k_per_op);
    println!("Effective bandwidth amplification: {:.2}x", f32_bw / q4k_bw);
}

// =========================================================================
// Coverage Tests: Getter Methods
// =========================================================================

/// Test LayerNorm getter methods
#[test]
fn test_layer_norm_getters() {
    let ln = LayerNorm::new(64, 1e-5).expect("test");
    assert_eq!(ln.normalized_shape(), 64);
    assert!((ln.eps() - 1e-5).abs() < 1e-10);
}

/// Test Linear getter methods
#[test]
fn test_linear_getters() {
    let linear = Linear::new(32, 64).expect("test");
    assert_eq!(linear.in_features(), 32);
    assert_eq!(linear.out_features(), 64);
}

/// Test Linear mutable accessors
#[test]
fn test_linear_mutable_accessors() {
    let mut linear = Linear::new(4, 2).expect("test");

    // Modify weights
    let weights = linear.weight_mut();
    assert_eq!(weights.len(), 4 * 2);
    weights[0] = 1.0;
    assert_eq!(linear.weight_mut()[0], 1.0);

    // Modify bias
    let bias = linear.bias_mut();
    assert_eq!(bias.len(), 2);
    bias[0] = 0.5;
    assert_eq!(linear.bias_mut()[0], 0.5);
}

/// Test QuantizedLinear getter methods
#[test]
fn test_quantized_linear_getters() {
    // Create minimal Q4_K weight (144 bytes per 256 values)
    let weight_bytes = vec![0u8; 144 * 2]; // 512 values = 2 super-blocks
    let bias = vec![0.0f32; 2];
    let ql = QuantizedLinear::new(256, 2, weight_bytes, bias).expect("test");

    assert_eq!(ql.in_features(), 256);
    assert_eq!(ql.out_features(), 2);
    assert_eq!(ql.weight_bytes().len(), 144 * 2);
    assert_eq!(ql.bias().len(), 2);
    assert!(ql.memory_bytes() > 0);
}

/// Test FusedLayerNormLinear getter methods
#[test]
fn test_fused_layer_norm_linear_getters() {
    let fused = FusedLayerNormLinear::new(8, 4, 1e-5).expect("test");
    assert_eq!(fused.feature_dim(), 8);
    assert_eq!(fused.out_features(), 4);
}

/// Test FusedLayerNormLinear mutable accessors
#[test]
fn test_fused_layer_norm_linear_mutable_accessors() {
    let mut fused = FusedLayerNormLinear::new(4, 2, 1e-5).expect("test");

    // Modify norm weights
    let norm_w = fused.norm_weight_mut();
    assert_eq!(norm_w.len(), 4);
    norm_w[0] = 2.0;
    assert_eq!(fused.norm_weight_mut()[0], 2.0);

    // Modify norm bias
    let norm_b = fused.norm_bias_mut();
    assert_eq!(norm_b.len(), 4);
    norm_b[0] = 0.1;
    assert_eq!(fused.norm_bias_mut()[0], 0.1);

    // Modify linear weights
    let lin_w = fused.linear_weight_mut();
    assert_eq!(lin_w.len(), 4 * 2);
    lin_w[0] = 3.0;
    assert_eq!(fused.linear_weight_mut()[0], 3.0);

    // Modify linear bias
    let lin_b = fused.linear_bias_mut();
    assert_eq!(lin_b.len(), 2);
    lin_b[0] = 0.2;
    assert_eq!(fused.linear_bias_mut()[0], 0.2);
}

/// Test FeedForward getter methods
#[test]
fn test_ffn_getters() {
    let ffn = FeedForward::new(8, 32).expect("test");
    assert_eq!(ffn.hidden_dim(), 8);
    assert_eq!(ffn.intermediate_dim(), 32);
}

/// Test FeedForward mutable accessors
#[test]
fn test_ffn_mutable_accessors() {
    let mut ffn = FeedForward::new(4, 8).expect("test");

    // Get mutable references to fc1 and fc2
    let fc1 = ffn.fc1_mut();
    assert_eq!(fc1.in_features(), 4);
    assert_eq!(fc1.out_features(), 8);

    let fc2 = ffn.fc2_mut();
    assert_eq!(fc2.in_features(), 8);
    assert_eq!(fc2.out_features(), 4);
}

/// Test Attention getter methods
#[test]
fn test_attention_getters() {
    let attn = Attention::new(64).expect("test");
    assert_eq!(attn.head_dim(), 64);
    assert!((attn.scale() - 1.0 / 8.0).abs() < 1e-5); // 1/sqrt(64) = 0.125
}

/// Test Attention scale calculation
#[test]
fn test_attention_scale_various_dims() {
    // head_dim=16 -> scale = 1/4 = 0.25
    let attn16 = Attention::new(16).expect("test");
    assert!((attn16.scale() - 0.25).abs() < 1e-5);

    // head_dim=128 -> scale = 1/sqrt(128) ≈ 0.0884
    let attn128 = Attention::new(128).expect("test");
    assert!((attn128.scale() - 1.0 / (128.0f32).sqrt()).abs() < 1e-5);
}

/// Test gelu with single element
#[test]
fn test_gelu_single_cov() {
    let single = Tensor::from_vec(vec![1], vec![0.0f32]).expect("test");
    let result = gelu(&single).expect("test");
    // GELU(0) = 0
    assert!(result.data()[0].abs() < 1e-5);
}

/// Test softmax with single element (should return 1.0)
#[test]
fn test_softmax_single_element_cov() {
    let single = Tensor::from_vec(vec![1], vec![5.0f32]).expect("test");
    let result = softmax(&single).expect("test");
    assert!((result.data()[0] - 1.0).abs() < 1e-5);
}

/// Test softmax probabilities sum to 1
#[test]
fn test_softmax_sum_to_one_cov() {
    let t = Tensor::from_vec(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]).expect("test");
    let result = softmax(&t).expect("test");
    let sum: f32 = result.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: Debug/Clone implementations
// =========================================================================

#[test]
fn test_layer_norm_debug_clone() {
    let layer_norm = LayerNorm::new(64, 1e-5).expect("test");
    let debug = format!("{:?}", layer_norm);
    assert!(debug.contains("LayerNorm"));

    let cloned = layer_norm.clone();
    assert_eq!(cloned.normalized_shape(), layer_norm.normalized_shape());
}

#[test]
fn test_linear_debug_clone() {
    let linear = Linear::new(32, 64).expect("test");
    let debug = format!("{:?}", linear);
    assert!(debug.contains("Linear"));

    let cloned = linear.clone();
    assert_eq!(cloned.in_features(), linear.in_features());
    assert_eq!(cloned.out_features(), linear.out_features());
}

#[test]
fn test_rope_debug_clone() {
    let rope = RoPE::new(64, 10000.0).expect("test");
    let debug = format!("{:?}", rope);
    assert!(debug.contains("RoPE"));

    let cloned = rope.clone();
    assert_eq!(cloned.dim(), rope.dim());
}

#[test]
fn test_rope_scaling_type_debug_clone_copy() {
    // Test None variant
    let none = RopeScalingType::None;
    let debug_none = format!("{:?}", none);
    assert!(debug_none.contains("None"));
    let cloned_none = none;
    assert_eq!(cloned_none, RopeScalingType::None);

    // Test Linear variant
    let linear = RopeScalingType::Linear { scale: 2.0 };
    let debug_linear = format!("{:?}", linear);
    assert!(debug_linear.contains("Linear"));
    assert!(debug_linear.contains("2.0"));
    let cloned_linear = linear;
    assert_eq!(cloned_linear, linear);

    // Test Ntk variant
    let ntk = RopeScalingType::Ntk { scale: 1.5 };
    let debug_ntk = format!("{:?}", ntk);
    assert!(debug_ntk.contains("Ntk"));
    assert_eq!(ntk, RopeScalingType::Ntk { scale: 1.5 });

    // Test DynamicNtk variant
    let dynamic = RopeScalingType::DynamicNtk {
        original_max_len: 2048,
        target_max_len: 4096,
    };
    let debug_dynamic = format!("{:?}", dynamic);
    assert!(debug_dynamic.contains("DynamicNtk"));
    assert!(debug_dynamic.contains("2048"));

    // Test Yarn variant
    let yarn = RopeScalingType::Yarn {
        original_max_len: 2048,
        target_max_len: 8192,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let debug_yarn = format!("{:?}", yarn);
    assert!(debug_yarn.contains("Yarn"));
    assert!(debug_yarn.contains("8192"));

    // Test Default
    let default = RopeScalingType::default();
    assert_eq!(default, RopeScalingType::None);
}

#[test]
fn test_scaled_rope_debug_clone() {
    let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("test");
    let debug = format!("{:?}", scaled);
    assert!(debug.contains("ScaledRoPE"));

    let cloned = scaled.clone();
    assert_eq!(cloned.dim(), scaled.dim());
}

#[test]
fn test_alibi_debug_clone() {
    let alibi = ALiBi::new(8).expect("test");
    let debug = format!("{:?}", alibi);
    assert!(debug.contains("ALiBi"));

    let cloned = alibi.clone();
    assert_eq!(cloned.num_heads(), alibi.num_heads());
}

#[test]
fn test_kv_cache_debug_clone() {
    let cache = KVCache::new(2, 512, 64).expect("test");
    let debug = format!("{:?}", cache);
    assert!(debug.contains("KVCache"));

    let cloned = cache.clone();
    assert_eq!(cloned.num_layers(), cache.num_layers());
}

#[test]
fn test_attention_debug_clone() {
    let attn = Attention::new(64).expect("test");
    let debug = format!("{:?}", attn);
    assert!(debug.contains("Attention"));

    let cloned = attn.clone();
    assert!((cloned.scale() - attn.scale()).abs() < 1e-6);
}

#[test]
fn test_feed_forward_debug_clone() {
    let ffn = FeedForward::new(64, 256).expect("test");
    let debug = format!("{:?}", ffn);
    assert!(debug.contains("FeedForward"));

    let cloned = ffn.clone();
    assert_eq!(cloned.hidden_dim(), ffn.hidden_dim());
}

#[test]
fn test_multi_head_attention_debug_clone() {
    let mha = MultiHeadAttention::new(256, 4, 4).expect("test");
    let debug = format!("{:?}", mha);
    assert!(debug.contains("MultiHeadAttention"));

    let cloned = mha.clone();
    assert_eq!(cloned.num_heads(), mha.num_heads());
}

#[test]
fn test_embedding_debug_clone() {
    let emb = Embedding::new(1000, 256).expect("test");
    let debug = format!("{:?}", emb);
    assert!(debug.contains("Embedding"));

    let cloned = emb.clone();
    assert_eq!(cloned.vocab_size(), emb.vocab_size());
    assert_eq!(cloned.embed_dim(), emb.embed_dim());
}

#[test]
fn test_model_config_debug_clone() {
    let config = ModelConfig {
        vocab_size: 50000,
        hidden_dim: 1024,
        num_layers: 12,
        num_heads: 8,
        intermediate_dim: 4096,
        eps: 1e-5,
    };
    let debug = format!("{:?}", config);
    assert!(debug.contains("ModelConfig"));
    assert!(debug.contains("50000"));

    let cloned = config.clone();
    assert_eq!(cloned.vocab_size, config.vocab_size);
    assert_eq!(cloned.hidden_dim, config.hidden_dim);
    assert_eq!(cloned.num_layers, config.num_layers);
}

#[test]
fn test_model_debug_clone() {
    let config = ModelConfig {
        vocab_size: 1000,
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 2,
        intermediate_dim: 256,
        eps: 1e-5,
    };
    let model = Model::new(config.clone()).expect("test");
    let debug = format!("{:?}", model);
    assert!(debug.contains("Model"));

    let cloned = model.clone();
    // Verify the cloned model has the same config
    assert_eq!(cloned.config().num_layers, model.config().num_layers);
}

#[test]
fn test_transformer_block_debug_clone() {
    let block = TransformerBlock::new(256, 4, 1024, 1e-5).expect("test");
    let debug = format!("{:?}", block);
    assert!(debug.contains("TransformerBlock"));

    let cloned = block.clone();
    assert_eq!(cloned.hidden_dim(), block.hidden_dim());
}
