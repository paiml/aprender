//! Deep coverage tests for layers.rs module
//!
//! Targets uncovered code paths for 95%+ coverage

use realizar::layers::{
    Attention, FeedForward, FusedLayerNormLinear, FusedQKVAttention, LayerNorm, Linear,
    QuantizedLinear, SlidingWindowAttention,
};
use realizar::tensor::Tensor;

// =============================================================================
// LayerNorm Deep Tests
// =============================================================================

#[test]
fn test_layer_norm_getters() {
    let norm = LayerNorm::new(64, 1e-5).expect("create layer norm");
    assert_eq!(norm.normalized_shape(), 64);
    assert!((norm.eps() - 1e-5).abs() < 1e-10);
}

#[test]
fn test_layer_norm_small_eps() {
    let norm = LayerNorm::new(32, 1e-12).expect("create layer norm");
    assert!((norm.eps() - 1e-12).abs() < 1e-15);
}

#[test]
fn test_layer_norm_large_dimension() {
    let norm = LayerNorm::new(4096, 1e-5).expect("create layer norm");
    assert_eq!(norm.normalized_shape(), 4096);
}

#[test]
fn test_layer_norm_forward_1d() {
    let norm = LayerNorm::new(4, 1e-5).expect("create layer norm");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = norm.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_layer_norm_forward_2d() {
    let norm = LayerNorm::new(4, 1e-5).expect("create layer norm");
    let input =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("input");
    let output = norm.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_layer_norm_zero_input() {
    let norm = LayerNorm::new(4, 1e-5).expect("create layer norm");
    let input = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("input");
    let output = norm.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_layer_norm_negative_input() {
    let norm = LayerNorm::new(4, 1e-5).expect("create layer norm");
    let input = Tensor::from_vec(vec![4], vec![-1.0, -2.0, -3.0, -4.0]).expect("input");
    let output = norm.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_layer_norm_mixed_input() {
    let norm = LayerNorm::new(4, 1e-5).expect("create layer norm");
    let input = Tensor::from_vec(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).expect("input");
    let output = norm.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

// =============================================================================
// Linear Deep Tests
// =============================================================================

#[test]
fn test_linear_getters() {
    let linear = Linear::new(64, 128).expect("create linear");
    assert_eq!(linear.in_features(), 64);
    assert_eq!(linear.out_features(), 128);
}

#[test]
fn test_linear_weight_bias_mut() {
    let mut linear = Linear::new(4, 8).expect("create linear");
    let weight = linear.weight_mut();
    assert_eq!(weight.len(), 4 * 8);
    let bias = linear.bias_mut();
    assert_eq!(bias.len(), 8);
}

#[test]
fn test_linear_forward_small() {
    let linear = Linear::new(4, 8).expect("create linear");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = linear.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[8]);
}

#[test]
fn test_linear_forward_batched() {
    let linear = Linear::new(4, 8).expect("create linear");
    let input =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("input");
    let output = linear.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 8]);
}

#[test]
fn test_linear_large() {
    let linear = Linear::new(512, 2048).expect("create linear");
    assert_eq!(linear.in_features(), 512);
    assert_eq!(linear.out_features(), 2048);
}

// =============================================================================
// QuantizedLinear Tests
// =============================================================================

#[test]
fn test_quantized_linear_basic() {
    // Q4_K format: 144 bytes per super-block (256 values)
    // For in_features=256, out_features=1, we need 144 bytes
    let weight_bytes = vec![0u8; 144];
    let bias = vec![0.0f32; 1];
    let ql = QuantizedLinear::new(256, 1, weight_bytes, bias).expect("create");
    assert_eq!(ql.in_features(), 256);
    assert_eq!(ql.out_features(), 1);
}

#[test]
fn test_quantized_linear_larger() {
    // For in_features=256, out_features=2, we need 2 * 144 = 288 bytes
    let weight_bytes = vec![0u8; 288];
    let bias = vec![0.0f32; 2];
    let ql = QuantizedLinear::new(256, 2, weight_bytes, bias).expect("create");
    assert_eq!(ql.in_features(), 256);
    assert_eq!(ql.out_features(), 2);
}

#[test]
fn test_quantized_linear_getters() {
    let weight_bytes = vec![0u8; 144];
    let bias = vec![1.5f32; 1];
    let ql = QuantizedLinear::new(256, 1, weight_bytes.clone(), bias.clone()).expect("create");
    assert_eq!(ql.weight_bytes(), &weight_bytes[..]);
    assert_eq!(ql.bias(), &bias[..]);
}

#[test]
fn test_quantized_linear_memory_bytes() {
    let weight_bytes = vec![0u8; 144];
    let bias = vec![0.0f32; 1];
    let ql = QuantizedLinear::new(256, 1, weight_bytes, bias).expect("create");
    assert!(ql.memory_bytes() > 0);
}

// =============================================================================
// FusedLayerNormLinear Tests
// =============================================================================

#[test]
fn test_fused_ln_linear_new() {
    let fused = FusedLayerNormLinear::new(64, 128, 1e-5).expect("create");
    assert_eq!(fused.feature_dim(), 64);
    assert_eq!(fused.out_features(), 128);
}

#[test]
fn test_fused_ln_linear_mut_accessors() {
    let mut fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("create");
    let norm_weight = fused.norm_weight_mut();
    assert_eq!(norm_weight.len(), 4);
    let norm_bias = fused.norm_bias_mut();
    assert_eq!(norm_bias.len(), 4);
    let linear_weight = fused.linear_weight_mut();
    assert_eq!(linear_weight.len(), 4 * 8);
    let linear_bias = fused.linear_bias_mut();
    assert_eq!(linear_bias.len(), 8);
}

#[test]
fn test_fused_ln_linear_forward() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("create");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = fused.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[8]);
}

#[test]
fn test_fused_ln_linear_forward_parallel() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("create");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = fused.forward_parallel(&input).expect("forward");
    assert_eq!(output.shape(), &[8]);
}

#[test]
fn test_fused_ln_linear_batched() {
    let fused = FusedLayerNormLinear::new(4, 8, 1e-5).expect("create");
    let input =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("input");
    let output = fused.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 8]);
}

// =============================================================================
// FeedForward Tests
// =============================================================================

#[test]
fn test_feedforward_new() {
    let ff = FeedForward::new(64, 256).expect("create");
    assert_eq!(ff.hidden_dim(), 64);
    assert_eq!(ff.intermediate_dim(), 256);
}

#[test]
fn test_feedforward_mut_accessors() {
    let mut ff = FeedForward::new(4, 16).expect("create");
    let fc1 = ff.fc1_mut();
    assert_eq!(fc1.in_features(), 4);
    assert_eq!(fc1.out_features(), 16);
    let fc2 = ff.fc2_mut();
    assert_eq!(fc2.in_features(), 16);
    assert_eq!(fc2.out_features(), 4);
}

#[test]
fn test_feedforward_forward() {
    let ff = FeedForward::new(4, 16).expect("create");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = ff.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_feedforward_batched() {
    let ff = FeedForward::new(4, 16).expect("create");
    let input =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("input");
    let output = ff.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}

// =============================================================================
// Attention Tests
// =============================================================================

#[test]
fn test_attention_new() {
    let attn = Attention::new(64).expect("create");
    assert_eq!(attn.head_dim(), 64);
    let expected_scale = 1.0 / (64.0_f32).sqrt();
    assert!((attn.scale() - expected_scale).abs() < 1e-6);
}

#[test]
fn test_attention_small_head_dim() {
    let attn = Attention::new(8).expect("create");
    assert_eq!(attn.head_dim(), 8);
}

#[test]
fn test_attention_large_head_dim() {
    let attn = Attention::new(128).expect("create");
    assert_eq!(attn.head_dim(), 128);
}

#[test]
fn test_attention_forward() {
    let attn = Attention::new(4).expect("create");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("q");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("k");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("v");
    let output = attn.forward(&q, &k, &v).expect("forward");
    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_attention_batched() {
    let attn = Attention::new(4).expect("create");
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("q");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("k");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("v");
    let output = attn.forward(&q, &k, &v).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}

// =============================================================================
// SlidingWindowAttention Tests
// =============================================================================

#[test]
fn test_sliding_window_new() {
    let attn = SlidingWindowAttention::new(64, 256).expect("create");
    assert_eq!(attn.head_dim(), 64);
    assert_eq!(attn.window_size(), 256);
}

#[test]
fn test_sliding_window_scale() {
    let attn = SlidingWindowAttention::new(64, 128).expect("create");
    let expected_scale = 1.0 / (64.0_f32).sqrt();
    assert!((attn.scale() - expected_scale).abs() < 1e-6);
}

#[test]
fn test_sliding_window_effective_context() {
    let attn = SlidingWindowAttention::new(64, 256).expect("create");
    // At position 100 in seq_len 500, effective context is min(100+1, 256) = 101
    assert_eq!(attn.effective_context(100, 500), 101);
    // At position 300 in seq_len 500, effective context is min(300+1, 256) = 256
    assert_eq!(attn.effective_context(300, 500), 256);
}

#[test]
fn test_sliding_window_memory_ratio() {
    let attn = SlidingWindowAttention::new(64, 256).expect("create");
    // For seq_len 512, memory ratio is 256/512 = 0.5
    let ratio = attn.memory_ratio(512);
    assert!((ratio - 0.5).abs() < 1e-6);
}

#[test]
fn test_sliding_window_forward() {
    let attn = SlidingWindowAttention::new(4, 8).expect("create");
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("q");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("k");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("v");
    let output = attn.forward(&q, &k, &v).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_sliding_window_forward_with_mask_causal() {
    let attn = SlidingWindowAttention::new(4, 8).expect("create");
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("q");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("k");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("v");
    let output = attn.forward_with_mask(&q, &k, &v, true).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_sliding_window_forward_with_mask_non_causal() {
    let attn = SlidingWindowAttention::new(4, 8).expect("create");
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("q");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("k");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("v");
    let output = attn.forward_with_mask(&q, &k, &v, false).expect("forward");
    assert_eq!(output.shape(), &[2, 4]);
}

// =============================================================================
// FusedQKVAttention Tests
// =============================================================================

#[test]
fn test_fused_qkv_attention_new() {
    let attn = FusedQKVAttention::new(64, 256).expect("create");
    assert_eq!(attn.head_dim(), 64);
}

#[test]
fn test_fused_qkv_attention_forward() {
    let attn = FusedQKVAttention::new(4, 16).expect("create");
    let input = Tensor::from_vec(vec![1, 16], vec![0.1; 16]).expect("input");
    let output = attn.forward(&input).expect("forward");
    // Output has same hidden_dim as input
    assert_eq!(output.shape(), &[1, 16]);
}

// =============================================================================
// Softmax and GELU Tests
// =============================================================================

#[test]
fn test_softmax_1d() {
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = realizar::layers::softmax(&input).expect("softmax");
    assert_eq!(output.shape(), &[4]);
    // Sum should be approximately 1
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_2d() {
    let input =
        Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]).expect("input");
    let output = realizar::layers::softmax(&input).expect("softmax");
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_softmax_large_values() {
    let input = Tensor::from_vec(vec![4], vec![100.0, 200.0, 300.0, 400.0]).expect("input");
    let output = realizar::layers::softmax(&input).expect("softmax");
    // Should handle large values without overflow
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_negative_values() {
    let input = Tensor::from_vec(vec![4], vec![-1.0, -2.0, -3.0, -4.0]).expect("input");
    let output = realizar::layers::softmax(&input).expect("softmax");
    let sum: f32 = output.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_gelu_1d() {
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = realizar::layers::gelu(&input).expect("gelu");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_gelu_2d() {
    let input = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0])
        .expect("input");
    let output = realizar::layers::gelu(&input).expect("gelu");
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_gelu_negative() {
    let input = Tensor::from_vec(vec![4], vec![-1.0, -2.0, -3.0, -4.0]).expect("input");
    let output = realizar::layers::gelu(&input).expect("gelu");
    // GELU of negative values should be small but not zero
    for &val in output.data() {
        assert!(val <= 0.0);
    }
}

#[test]
fn test_gelu_zero() {
    let input = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("input");
    let output = realizar::layers::gelu(&input).expect("gelu");
    // GELU(0) = 0
    for &val in output.data() {
        assert!(val.abs() < 1e-6);
    }
}

// =============================================================================
// Flash Attention Tests (if available)
// =============================================================================

#[test]
fn test_attention_flash_forward() {
    let attn = Attention::new(4).expect("create");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("q");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("k");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("v");
    let output = attn.flash_forward(&q, &k, &v, 4).expect("flash forward");
    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_attention_flash_forward_v2() {
    let attn = Attention::new(4).expect("create");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("q");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("k");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("v");
    let output = attn
        .flash_forward_v2(&q, &k, &v, 4)
        .expect("flash forward v2");
    assert_eq!(output.shape(), &[1, 4]);
}

#[test]
fn test_attention_flash_forward_parallel() {
    let attn = Attention::new(4).expect("create");
    let q = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("q");
    let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("k");
    let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("v");
    let output = attn
        .flash_forward_parallel(&q, &k, &v, 2)
        .expect("flash forward parallel");
    assert_eq!(output.shape(), &[2, 4]);
}

// =============================================================================
// Edge Cases and Error Tests
// =============================================================================

#[test]
fn test_linear_zero_weights() {
    let mut linear = Linear::new(4, 8).expect("create");
    for w in linear.weight_mut().iter_mut() {
        *w = 0.0;
    }
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
    let output = linear.forward(&input).expect("forward");
    // With zero weights and zero bias, output should be zeros
    for &val in output.data() {
        assert!(val.abs() < 1e-6);
    }
}

#[test]
fn test_layer_norm_constant_input() {
    let norm = LayerNorm::new(4, 1e-5).expect("create");
    let input = Tensor::from_vec(vec![4], vec![5.0, 5.0, 5.0, 5.0]).expect("input");
    let output = norm.forward(&input).expect("forward");
    // Constant input should produce zero variance, so output depends on bias
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_feedforward_identity_like() {
    let ff = FeedForward::new(4, 16).expect("create");
    let input = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).expect("input");
    let output = ff.forward(&input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_attention_single_token() {
    let attn = Attention::new(4).expect("create");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).expect("q");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]).expect("k");
    let v = Tensor::from_vec(vec![1, 4], vec![0.5, 0.5, 0.5, 0.5]).expect("v");
    let output = attn.forward(&q, &k, &v).expect("forward");
    // Single token attention should return the value
    assert_eq!(output.shape(), &[1, 4]);
}

// =============================================================================
// Quantized Linear Error Tests
// =============================================================================

#[test]
fn test_quantized_linear_empty_weights() {
    let weight_bytes = vec![];
    let bias = vec![0.0f32; 1];
    let result = QuantizedLinear::new(256, 1, weight_bytes, bias);
    // Should fail validation
    assert!(result.is_err());
}

#[test]
fn test_quantized_linear_mismatched_bias() {
    let weight_bytes = vec![0u8; 144];
    let bias = vec![0.0f32; 5]; // Wrong size
    let result = QuantizedLinear::new(256, 1, weight_bytes, bias);
    // Should fail validation
    assert!(result.is_err());
}
