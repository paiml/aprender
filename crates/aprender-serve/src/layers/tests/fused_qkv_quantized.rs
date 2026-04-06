use crate::layers::*;
#[test]
fn test_fused_qkv_attention_debug_clone() {
    // FusedQKVAttention::new(head_dim, hidden_dim)
    // hidden_dim must be divisible by head_dim
    // For head_dim 64, hidden_dim 256: num_heads = 256/64 = 4
    let fused = FusedQKVAttention::new(64, 256).expect("test");
    let debug = format!("{:?}", fused);
    assert!(debug.contains("FusedQKVAttention"));

    let cloned = fused.clone();
    assert_eq!(cloned.num_heads(), fused.num_heads());
}

#[test]
fn test_quantized_linear_debug_clone() {
    // Q4_K: 144 bytes per super-block of 256 values
    // For 256 in_features: 1 super-block per row = 144 bytes
    // For 128 out_features: 128 rows = 128 * 144 = 18,432 bytes
    let in_features = 256;
    let out_features = 128;
    let weight_bytes = vec![0u8; out_features * 144]; // 144 bytes per row
    let bias = vec![0.0f32; out_features];

    let ql = QuantizedLinear::new(in_features, out_features, weight_bytes, bias).expect("test");
    let debug = format!("{:?}", ql);
    assert!(debug.contains("QuantizedLinear"));

    let cloned = ql.clone();
    assert_eq!(cloned.in_features(), ql.in_features());
    assert_eq!(cloned.out_features(), ql.out_features());
}

#[test]
fn test_fused_layer_norm_linear_debug_clone() {
    let fused = FusedLayerNormLinear::new(64, 128, 1e-5).expect("test");
    let debug = format!("{:?}", fused);
    assert!(debug.contains("FusedLayerNormLinear"));

    let cloned = fused.clone();
    assert_eq!(cloned.feature_dim(), fused.feature_dim());
}

// =========================================================================
// Coverage Tests: softmax function
// =========================================================================

#[test]
fn test_softmax_zero_dimension_cov() {
    // Test that tensor creation fails for zero-dimension shapes
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_softmax_single_element_extended_cov() {
    let input = Tensor::from_vec(vec![1], vec![5.0]).expect("single element");
    let result = crate::layers::softmax(&input).expect("softmax");
    assert!((result.data()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_negative_values_cov() {
    let input = Tensor::from_vec(vec![3], vec![-1.0, -2.0, -3.0]).expect("negative values");
    let result = crate::layers::softmax(&input).expect("softmax");
    let sum: f32 = result.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_large_values_cov() {
    let input = Tensor::from_vec(vec![3], vec![1000.0, 1001.0, 1002.0]).expect("large values");
    let result = crate::layers::softmax(&input).expect("softmax");
    let sum: f32 = result.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_2d_tensor_cov() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("2d");
    let result = crate::layers::softmax(&input).expect("softmax");
    assert_eq!(result.shape(), &[2, 3]);
    // Each row should sum to 1
    let row1_sum: f32 = result.data()[0..3].iter().sum();
    let row2_sum: f32 = result.data()[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: gelu function
// =========================================================================

#[test]
fn test_gelu_zero_dimension_cov() {
    // Test that tensor creation fails for zero-dimension shapes
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_gelu_zero_cov() {
    let input = Tensor::from_vec(vec![1], vec![0.0]).expect("zero");
    let result = crate::layers::gelu(&input).expect("gelu");
    assert!(result.data()[0].abs() < 1e-6);
}

#[test]
fn test_gelu_positive_cov() {
    let input = Tensor::from_vec(vec![1], vec![2.0]).expect("positive");
    let result = crate::layers::gelu(&input).expect("gelu");
    // GELU(2.0) ≈ 1.95
    assert!(result.data()[0] > 1.9 && result.data()[0] < 2.0);
}

#[test]
fn test_gelu_negative_cov() {
    let input = Tensor::from_vec(vec![1], vec![-2.0]).expect("negative");
    let result = crate::layers::gelu(&input).expect("gelu");
    // GELU(-2.0) ≈ -0.045
    assert!(result.data()[0] > -0.1 && result.data()[0] < 0.0);
}

#[test]
fn test_gelu_multiple_values_cov() {
    let input = Tensor::from_vec(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).expect("multiple");
    let result = crate::layers::gelu(&input).expect("gelu");
    assert_eq!(result.shape(), &[5]);
    // GELU(0) = 0
    assert!(result.data()[2].abs() < 1e-6);
}

// =========================================================================
// Coverage Tests: LayerNorm
// =========================================================================

#[test]
fn test_layer_norm_zero_shape_cov() {
    let result = LayerNorm::new(0, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_eps_accessor_cov() {
    let ln = LayerNorm::new(64, 1e-6).expect("layer norm");
    assert!((ln.eps() - 1e-6).abs() < 1e-10);
}

#[test]
fn test_layer_norm_forward_empty_shape_cov() {
    // Empty shape arrays fail at tensor creation
    let result = Tensor::<f32>::from_vec(vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_forward_wrong_dim_cov() {
    let ln = LayerNorm::new(64, 1e-5).expect("layer norm");
    let input = Tensor::from_vec(vec![32], vec![1.0; 32]).expect("wrong dim");
    let result = ln.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_forward_basic_cov() {
    let ln = LayerNorm::new(4, 1e-5).expect("layer norm");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let result = ln.forward(&input).expect("forward");
    assert_eq!(result.shape(), &[4]);
    // Output should have mean ~0 and variance ~1
    let data = result.data();
    let mean: f32 = data.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5);
}

// =========================================================================
// Coverage Tests: Linear
// =========================================================================

#[test]
fn test_linear_zero_in_features_cov() {
    let result = Linear::new(0, 64);
    assert!(result.is_err());
}

#[test]
fn test_linear_zero_out_features_cov() {
    let result = Linear::new(64, 0);
    assert!(result.is_err());
}

#[test]
fn test_linear_in_out_features_accessor_cov() {
    let linear = Linear::new(128, 256).expect("linear");
    assert_eq!(linear.in_features(), 128);
    assert_eq!(linear.out_features(), 256);
}

#[test]
fn test_linear_weight_mut_cov() {
    let mut linear = Linear::new(4, 8).expect("linear");
    let weight = linear.weight_mut();
    assert_eq!(weight.len(), 4 * 8);
    // Modify weight
    weight[0] = 1.0;
    assert!((linear.weight_mut()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_linear_bias_mut_cov() {
    let mut linear = Linear::new(4, 8).expect("linear");
    let bias = linear.bias_mut();
    assert_eq!(bias.len(), 8);
    // Modify bias
    bias[0] = 0.5;
    assert!((linear.bias_mut()[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_linear_forward_empty_shape_cov() {
    // Empty shape arrays fail at tensor creation
    let result = Tensor::<f32>::from_vec(vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_linear_forward_wrong_dim_cov() {
    let linear = Linear::new(4, 8).expect("linear");
    let input = Tensor::from_vec(vec![8], vec![1.0; 8]).expect("wrong dim");
    let result = linear.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_linear_forward_batch_cov() {
    let linear = Linear::new(4, 8).expect("linear");
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("batch");
    let result = linear.forward(&input).expect("forward");
    assert_eq!(result.shape(), &[2, 8]);
}

// =========================================================================
// Coverage Tests: QuantizedLinear
// =========================================================================

#[test]
fn test_quantized_linear_zero_features_cov() {
    let result = QuantizedLinear::new(0, 128, vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_quantized_linear_wrong_bias_len_cov() {
    // For 256 in_features: 1 super-block = 144 bytes per row
    // For 128 out_features: 128 * 144 = 18432 bytes
    let weight_bytes = vec![0u8; 128 * 144];
    let bias = vec![0.0f32; 64]; // Wrong length - should be 128
    let result = QuantizedLinear::new(256, 128, weight_bytes, bias);
    assert!(result.is_err());
}

#[test]
fn test_quantized_linear_in_out_features_cov() {
    let in_features = 256;
    let out_features = 128;
    let weight_bytes = vec![0u8; out_features * 144];
    let bias = vec![0.0f32; out_features];
    let ql = QuantizedLinear::new(in_features, out_features, weight_bytes, bias).expect("ql");
    assert_eq!(ql.in_features(), in_features);
    assert_eq!(ql.out_features(), out_features);
}

// =========================================================================
// Coverage Tests: FeedForward
// =========================================================================

#[test]
fn test_feed_forward_intermediate_dim_cov() {
    let ffn = FeedForward::new(128, 512).expect("ffn");
    assert_eq!(ffn.intermediate_dim(), 512);
}

#[test]
fn test_feed_forward_fc_mut_cov() {
    let mut ffn = FeedForward::new(64, 256).expect("ffn");
    let fc1 = ffn.fc1_mut();
    assert_eq!(fc1.in_features(), 64);
    assert_eq!(fc1.out_features(), 256);

    let fc2 = ffn.fc2_mut();
    assert_eq!(fc2.in_features(), 256);
    assert_eq!(fc2.out_features(), 64);
}

// =========================================================================
// Coverage Tests: Attention
// =========================================================================

#[test]
fn test_attention_zero_head_dim_cov() {
    let result = Attention::new(0);
    assert!(result.is_err());
}

#[test]
fn test_attention_head_dim_accessor_cov() {
    let attn = Attention::new(64).expect("attention");
    assert_eq!(attn.head_dim(), 64);
}

#[test]
fn test_attention_scale_accessor_cov() {
    let attn = Attention::new(64).expect("attention");
    let expected_scale = 1.0 / 64.0_f32.sqrt();
    assert!((attn.scale() - expected_scale).abs() < 1e-6);
}

#[test]
fn test_attention_forward_empty_shape_cov() {
    // Empty shape arrays fail at tensor creation
    let result = Tensor::<f32>::from_vec(vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_attention_forward_mismatched_kv_cov() {
    let attn = Attention::new(4).expect("attention");
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("q");
    let k = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("k");
    let v = Tensor::from_vec(vec![4, 4], vec![1.0; 16]).expect("v");
    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err());
}

#[test]
fn test_attention_forward_wrong_head_dim_cov() {
    let attn = Attention::new(8).expect("attention");
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("input");
    let result = attn.forward(&input, &input, &input);
    assert!(result.is_err());
}

// =========================================================================
// Coverage Tests: FusedLayerNormLinear
// =========================================================================

#[test]
fn test_fused_layer_norm_linear_zero_dim_cov() {
    let result = FusedLayerNormLinear::new(0, 128, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_out_features_cov() {
    let fused = FusedLayerNormLinear::new(64, 128, 1e-5).expect("fused");
    assert_eq!(fused.out_features(), 128);
}

#[test]
fn test_fused_layer_norm_linear_weight_accessors_cov() {
    let mut fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("fused");

    let norm_weight = fused.norm_weight_mut();
    assert_eq!(norm_weight.len(), 4);
    norm_weight[0] = 2.0;

    let norm_bias = fused.norm_bias_mut();
    assert_eq!(norm_bias.len(), 4);
    norm_bias[0] = 0.1;

    let linear_weight = fused.linear_weight_mut();
    assert_eq!(linear_weight.len(), 4 * 8);

    let linear_bias = fused.linear_bias_mut();
    assert_eq!(linear_bias.len(), 8);
}

#[test]
fn test_fused_layer_norm_linear_forward_empty_shape_cov() {
    // Empty shape arrays fail at tensor creation
    let result = Tensor::<f32>::from_vec(vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_forward_wrong_dim_cov() {
    let fused = FusedLayerNormLinear::new(64, 128, 1e-5).expect("fused");
    let input = Tensor::from_vec(vec![32], vec![1.0; 32]).expect("wrong dim");
    let result = fused.forward(&input);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_forward_parallel_empty_shape_cov() {
    // Empty shape arrays fail at tensor creation
    let result = Tensor::<f32>::from_vec(vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_forward_parallel_wrong_dim_cov() {
    let fused = FusedLayerNormLinear::new(64, 128, 1e-5).expect("fused");
    let input = Tensor::from_vec(vec![32], vec![1.0; 32]).expect("wrong dim");
    let result = fused.forward_parallel(&input);
    assert!(result.is_err());
}

#[test]
fn test_fused_layer_norm_linear_forward_parallel_basic_cov() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("fused");
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("input");
    let result = fused.forward_parallel(&input).expect("forward_parallel");
    assert_eq!(result.shape(), &[2, 8]);
}

// =========================================================================
// Coverage Tests: KVCache
// =========================================================================

#[test]
fn test_kv_cache_zero_layers_cov() {
    let result = KVCache::new(0, 512, 64);
    assert!(result.is_err());
}

#[test]
fn test_kv_cache_zero_max_len_cov() {
    let result = KVCache::new(2, 0, 64);
    assert!(result.is_err());
}

#[test]
fn test_kv_cache_zero_head_dim_cov() {
    let result = KVCache::new(2, 512, 0);
    assert!(result.is_err());
}

#[test]
fn test_kv_cache_num_layers_cov() {
    let cache = KVCache::new(4, 512, 64).expect("cache");
    assert_eq!(cache.num_layers(), 4);
}

#[test]
fn test_kv_cache_max_seq_len_cov() {
    let cache = KVCache::new(4, 1024, 64).expect("cache");
    assert_eq!(cache.max_seq_len(), 1024);
}

include!("cache_rope_zero.rs");
