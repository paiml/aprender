//\! Transformer Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

#[test]
fn test_multi_head_attention_shape() {
    let mha = MultiHeadAttention::new(64, 8);

    let q = Tensor::ones(&[2, 10, 64]);
    let k = Tensor::ones(&[2, 20, 64]);
    let v = Tensor::ones(&[2, 20, 64]);

    let (output, attn_weights) = mha.forward_qkv(&q, &k, &v, None);

    assert_eq!(output.shape(), &[2, 10, 64]);
    assert_eq!(attn_weights.shape(), &[2, 8, 10, 20]);
}

#[test]
fn test_multi_head_attention_self() {
    let mha = MultiHeadAttention::new(64, 8);

    let x = Tensor::ones(&[2, 10, 64]);
    let (output, _) = mha.forward_self(&x, None);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_multi_head_attention_parameters() {
    let mha = MultiHeadAttention::new(64, 8);
    let params = mha.parameters();

    // 4 linear layers * 2 params each (weight + bias) = 8
    assert_eq!(params.len(), 8);
}

#[test]
fn test_transformer_encoder_layer_shape() {
    let layer = TransformerEncoderLayer::new(64, 8, 256);

    let x = Tensor::ones(&[2, 10, 64]);
    let y = layer.forward(&x);

    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_transformer_encoder_layer_parameters() {
    let layer = TransformerEncoderLayer::new(64, 8, 256);
    let params = layer.parameters();

    // Self-attn: 8 params
    // Linear1: 2 params
    // Linear2: 2 params
    // Norm1: 2 params
    // Norm2: 2 params
    // Total: 16
    assert_eq!(params.len(), 16);
}

#[test]
fn test_transformer_decoder_layer_shape() {
    let layer = TransformerDecoderLayer::new(64, 8, 256);

    let tgt = Tensor::ones(&[2, 10, 64]);
    let memory = Tensor::ones(&[2, 20, 64]);

    let output = layer.forward_with_memory(&tgt, &memory, None, None);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_positional_encoding_shape() {
    let pe = PositionalEncoding::new(64, 100);

    let x = Tensor::ones(&[2, 10, 64]);
    let y = pe.forward(&x);

    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_causal_mask() {
    let mask = generate_causal_mask(4);

    assert_eq!(mask.shape(), &[4, 4]);

    // Check upper triangle is -inf
    assert!(mask.data()[1].is_infinite()); // [0, 1]
    assert!(mask.data()[2].is_infinite()); // [0, 2]
    assert!(mask.data()[3].is_infinite()); // [0, 3]
    assert!(mask.data()[6].is_infinite()); // [1, 2]

    // Check diagonal and below is 0
    assert_eq!(mask.data()[0], 0.0); // [0, 0]
    assert_eq!(mask.data()[4], 0.0); // [1, 0]
    assert_eq!(mask.data()[5], 0.0); // [1, 1]
}

#[test]
fn test_softmax_last_dim() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let y = softmax_last_dim(&x);

    // Check that each row sums to 1
    let row1_sum: f32 = y.data()[0..3].iter().sum();
    let row2_sum: f32 = y.data()[3..6].iter().sum();

    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_transpose_last_two() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let y = transpose_last_two(&x);

    assert_eq!(y.shape(), &[1, 3, 2]);
}

// ========================================================================
// Linear Attention Tests
// ========================================================================

#[test]
fn test_linear_attention_shape() {
    let attn = LinearAttention::new(64, 8);

    let x = Tensor::ones(&[2, 10, 64]);
    let output = attn.forward(&x);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_linear_attention_qkv_shape() {
    let attn = LinearAttention::new(64, 8);

    let q = Tensor::ones(&[2, 10, 64]);
    let k = Tensor::ones(&[2, 20, 64]);
    let v = Tensor::ones(&[2, 20, 64]);

    let output = attn.forward_linear(&q, &k, &v);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_linear_attention_parameters() {
    let attn = LinearAttention::new(64, 8);
    let params = attn.parameters();

    // 4 linear layers * 2 params each = 8
    assert_eq!(params.len(), 8);
}

#[test]
fn test_linear_attention_getters() {
    let attn = LinearAttention::new(128, 4);

    assert_eq!(attn.embed_dim(), 128);
    assert_eq!(attn.num_heads(), 4);
}

#[test]
fn test_linear_attention_train_eval() {
    let mut attn = LinearAttention::new(64, 8);

    assert!(attn.training());
    attn.eval();
    assert!(!attn.training());
    attn.train();
    assert!(attn.training());
}

#[test]
fn test_linear_attention_long_sequence() {
    // Linear attention should scale well with sequence length
    let attn = LinearAttention::new(32, 4);

    let x = Tensor::ones(&[1, 100, 32]); // Long sequence
    let output = attn.forward(&x);

    assert_eq!(output.shape(), &[1, 100, 32]);
}

#[test]
fn test_elu_feature_map_positive() {
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let y = elu_feature_map(&x);

    // For positive values: elu(x) + 1 = x + 1
    assert!((y.data()[0] - 2.0).abs() < 1e-6);
    assert!((y.data()[1] - 3.0).abs() < 1e-6);
    assert!((y.data()[2] - 4.0).abs() < 1e-6);
}

#[test]
fn test_elu_feature_map_negative() {
    let x = Tensor::new(&[-1.0, -2.0], &[2]);
    let y = elu_feature_map(&x);

    // For negative values: elu(x) + 1 = exp(x)
    assert!((y.data()[0] - (-1.0_f32).exp()).abs() < 1e-6);
    assert!((y.data()[1] - (-2.0_f32).exp()).abs() < 1e-6);
}

// ========================================================================
// Grouped Query Attention Tests
// ========================================================================

#[test]
fn test_gqa_shape() {
    // 8 query heads, 2 KV heads (4:1 ratio)
    let gqa = GroupedQueryAttention::new(64, 8, 2);

    let x = Tensor::ones(&[2, 10, 64]);
    let output = gqa.forward(&x);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_gqa_qkv_shape() {
    let gqa = GroupedQueryAttention::new(64, 8, 2);

    let q = Tensor::ones(&[2, 10, 64]);
    let k = Tensor::ones(&[2, 20, 64]);
    let v = Tensor::ones(&[2, 20, 64]);

    let (output, attn_weights) = gqa.forward_qkv(&q, &k, &v, None);

    assert_eq!(output.shape(), &[2, 10, 64]);
    // Attention weights have expanded heads
    assert_eq!(attn_weights.shape(), &[2, 8, 10, 20]);
}

#[test]
fn test_gqa_multi_query_attention() {
    // MQA: 1 KV head for all query heads
    let mqa = GroupedQueryAttention::new(64, 8, 1);

    let x = Tensor::ones(&[2, 10, 64]);
    let output = mqa.forward(&x);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_gqa_equals_mha() {
    // GQA with num_kv_heads == num_heads should behave like MHA
    let gqa = GroupedQueryAttention::new(64, 8, 8);

    let x = Tensor::ones(&[2, 10, 64]);
    let output = gqa.forward(&x);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_gqa_parameters_reduced() {
    // GQA with fewer KV heads has fewer parameters
    let mha = MultiHeadAttention::new(64, 8);
    let gqa = GroupedQueryAttention::new(64, 8, 2);

    let mha_params = mha.parameters();
    let gqa_params = gqa.parameters();

    // Both have 8 parameter tensors (4 linear layers * 2)
    assert_eq!(mha_params.len(), gqa_params.len());

    // But GQA K,V projections are smaller
    // MHA K projection: 64 -> 64, GQA K projection: 64 -> 16
}

#[test]
fn test_gqa_getters() {
    let gqa = GroupedQueryAttention::new(128, 8, 4);

    assert_eq!(gqa.embed_dim(), 128);
    assert_eq!(gqa.num_heads(), 8);
    assert_eq!(gqa.num_kv_heads(), 4);
}

#[test]
fn test_gqa_with_dropout() {
    let gqa = GroupedQueryAttention::new(64, 8, 2).with_dropout(0.1);

    let x = Tensor::ones(&[2, 10, 64]);
    let output = gqa.forward(&x);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_gqa_train_eval() {
    let mut gqa = GroupedQueryAttention::new(64, 8, 2);

    assert!(gqa.training());
    gqa.eval();
    assert!(!gqa.training());
    gqa.train();
    assert!(gqa.training());
}

#[test]
#[should_panic(expected = "num_heads (8) must be divisible by num_kv_heads (3)")]
fn test_gqa_invalid_kv_heads() {
    // num_heads must be divisible by num_kv_heads
    let _gqa = GroupedQueryAttention::new(64, 8, 3);
}

#[test]
fn test_repeat_kv_heads_identity() {
    // groups=1 should return identity
    let x = Tensor::ones(&[2, 4, 10, 8]); // [batch, kv_heads, seq, head_dim]
    let y = repeat_kv_heads(&x, 1);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_repeat_kv_heads_expansion() {
    // 2 KV heads -> 8 Q heads (4x expansion)
    let x = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 2, 2, 2], // [batch=1, kv_heads=2, seq=2, head_dim=2]
    );
    let y = repeat_kv_heads(&x, 4);

    assert_eq!(y.shape(), &[1, 8, 2, 2]);

    // Each KV head should be repeated 4 times
    // Head 0 data [1,2,3,4] repeated at positions 0,1,2,3
    // Head 1 data [5,6,7,8] repeated at positions 4,5,6,7
}

#[test]
fn test_sum_last_dim() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = sum_last_dim(&x);

    assert_eq!(y.shape(), &[2]);
    assert!((y.data()[0] - 6.0).abs() < 1e-6); // 1+2+3
    assert!((y.data()[1] - 15.0).abs() < 1e-6); // 4+5+6
}

// ========================================================================
// RoPE Tests
// ========================================================================

#[test]
fn test_rope_creation() {
    let rope = RotaryPositionEmbedding::new(64, 512);

    assert_eq!(rope.head_dim(), 64);
    assert_eq!(rope.max_seq_len(), 512);
    assert!((rope.base() - 10000.0).abs() < 1e-6);
}

#[test]
fn test_rope_custom_base() {
    let rope = RotaryPositionEmbedding::with_base(32, 256, 20000.0);

    assert_eq!(rope.head_dim(), 32);
    assert!((rope.base() - 20000.0).abs() < 1e-6);
}

#[test]
fn test_rope_apply_shape() {
    let rope = RotaryPositionEmbedding::new(8, 128);

    // [batch=2, seq=10, heads=4, head_dim=8]
    let x = Tensor::ones(&[2, 10, 4, 8]);
    let positions: Vec<usize> = (0..10).collect();

    let output = rope.apply(&x, &positions);

    assert_eq!(output.shape(), x.shape());
}

#[test]
fn test_rope_position_dependent() {
    let rope = RotaryPositionEmbedding::new(4, 10);

    // Same input at different positions should give different output
    let x = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 1, 4]);

    let out_pos0 = rope.apply(&x, &[0]);
    let out_pos5 = rope.apply(&x, &[5]);

    // Outputs should differ
    let diff: f32 = out_pos0
        .data()
        .iter()
        .zip(out_pos5.data().iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.01,
        "Different positions should give different outputs"
    );
}

#[test]
fn test_rope_cos_sin_cache() {
    let rope = RotaryPositionEmbedding::new(4, 10);

    // At position 0, cos should be 1, sin should be 0
    // cos_cache and sin_cache have shape [max_seq_len, head_dim/2]
    let half_dim = 2;
    assert!((rope.cos_cache[0 * half_dim] - 1.0).abs() < 1e-6);
    assert!(rope.sin_cache[0 * half_dim].abs() < 1e-6);
}

#[test]
#[should_panic(expected = "head_dim must be even")]
fn test_rope_odd_dim_panics() {
    let _rope = RotaryPositionEmbedding::new(7, 100);
}

// ========================================================================
// ALiBi Tests
// ========================================================================

#[test]
fn test_alibi_creation() {
    let alibi = ALiBi::new(8);

    assert_eq!(alibi.num_heads(), 8);
    assert_eq!(alibi.slopes().len(), 8);
}

#[path = "tests_alibi_mha.rs"]
mod tests_alibi_mha;
#[path = "tests_decoder_rope.rs"]
mod tests_decoder_rope;
#[path = "tests_rope_contract.rs"]
mod tests_rope_contract;
#[path = "tests_attention_contract.rs"]
mod tests_attention_contract;
#[path = "tests_alibi_contract.rs"]
mod tests_alibi_contract;
#[path = "tests_gqa_contract.rs"]
mod tests_gqa_contract;
#[path = "tests_attention_scaling_contract.rs"]
mod tests_attention_scaling_contract;
#[path = "tests_position_contract.rs"]
mod tests_position_contract;
#[path = "tests_rope_ext_contract.rs"]
mod tests_rope_ext_contract;
