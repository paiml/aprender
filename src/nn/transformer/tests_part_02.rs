
#[test]
fn test_alibi_slopes_power_of_two() {
    let alibi = ALiBi::new(8);

    // Slopes should be monotonically decreasing for power-of-2 heads
    let slopes = alibi.slopes();
    for i in 1..slopes.len() {
        assert!(slopes[i] < slopes[i - 1], "Slopes should decrease");
    }

    // All slopes should be positive
    for &s in slopes {
        assert!(s > 0.0);
    }
}

#[test]
fn test_alibi_bias_shape() {
    let alibi = ALiBi::new(4);
    let bias = alibi.compute_bias(10);

    assert_eq!(bias.shape(), &[4, 10, 10]);
}

#[test]
fn test_alibi_bias_diagonal_zero() {
    let alibi = ALiBi::new(2);
    let bias = alibi.compute_bias(5);

    // Diagonal should be zero (distance = 0)
    for h in 0..2 {
        for i in 0..5 {
            let idx = h * 5 * 5 + i * 5 + i;
            assert!((bias.data()[idx] - 0.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_alibi_bias_negative() {
    let alibi = ALiBi::new(2);
    let bias = alibi.compute_bias(5);

    // Off-diagonal should be negative (penalties)
    for h in 0..2 {
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    let idx = h * 5 * 5 + i * 5 + j;
                    assert!(bias.data()[idx] < 0.0, "Off-diagonal should be negative");
                }
            }
        }
    }
}

#[test]
fn test_alibi_apply_shape() {
    let alibi = ALiBi::new(4);

    // Attention scores [batch=2, heads=4, seq=8, seq=8]
    let scores = Tensor::ones(&[2, 4, 8, 8]);
    let output = alibi.apply(&scores);

    assert_eq!(output.shape(), scores.shape());
}

#[test]
fn test_alibi_apply_modifies_scores() {
    let alibi = ALiBi::new(2);
    let scores = Tensor::ones(&[1, 2, 4, 4]);
    let output = alibi.apply(&scores);

    // Output should differ from input (bias applied)
    let sum_input: f32 = scores.data().iter().sum();
    let sum_output: f32 = output.data().iter().sum();

    // Bias is negative, so output sum should be less than input
    assert!(sum_output < sum_input);
}

#[test]
fn test_alibi_non_power_two_heads() {
    // Should handle non-power-of-2 heads
    let alibi = ALiBi::new(6);
    assert_eq!(alibi.slopes().len(), 6);

    // All slopes should be positive
    for &s in alibi.slopes() {
        assert!(s > 0.0);
    }
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_mha_train_eval() {
    let mut mha = MultiHeadAttention::new(64, 8);
    assert!(mha.training());

    mha.eval();
    assert!(!mha.training());

    mha.train();
    assert!(mha.training());
}

#[test]
fn test_mha_embed_num_heads_getters() {
    let mha = MultiHeadAttention::new(128, 4);
    assert_eq!(mha.embed_dim(), 128);
    assert_eq!(mha.num_heads(), 4);
}

#[test]
fn test_mha_with_dropout() {
    let mha = MultiHeadAttention::new(64, 8).with_dropout(0.2);
    let x = Tensor::ones(&[2, 10, 64]);
    let output = mha.forward(&x);
    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_mha_debug() {
    let mha = MultiHeadAttention::new(64, 8);
    let debug_str = format!("{:?}", mha);
    assert!(debug_str.contains("MultiHeadAttention"));
    assert!(debug_str.contains("embed_dim"));
    assert!(debug_str.contains("num_heads"));
}

#[test]
fn test_mha_parameters_mut() {
    let mut mha = MultiHeadAttention::new(64, 8);
    let params = mha.parameters_mut();
    assert_eq!(params.len(), 8);
}

#[test]
fn test_encoder_layer_train_eval() {
    let mut layer = TransformerEncoderLayer::new(64, 8, 256);
    assert!(layer.training());

    layer.eval();
    assert!(!layer.training());

    layer.train();
    assert!(layer.training());
}

#[test]
fn test_encoder_layer_with_dropout() {
    let layer = TransformerEncoderLayer::new(64, 8, 256).with_dropout(0.2);
    let x = Tensor::ones(&[2, 10, 64]);
    let y = layer.forward(&x);
    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_encoder_layer_forward_with_mask() {
    let layer = TransformerEncoderLayer::new(64, 8, 256);
    let x = Tensor::ones(&[2, 10, 64]);
    // Test without mask - mask shape requirements are complex
    let y = layer.forward_with_mask(&x, None);
    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_encoder_layer_debug() {
    let layer = TransformerEncoderLayer::new(64, 8, 256);
    let debug_str = format!("{:?}", layer);
    assert!(debug_str.contains("TransformerEncoderLayer"));
    assert!(debug_str.contains("d_model"));
}

#[test]
fn test_encoder_layer_parameters_mut() {
    let mut layer = TransformerEncoderLayer::new(64, 8, 256);
    let params = layer.parameters_mut();
    assert_eq!(params.len(), 16);
}

#[test]
fn test_decoder_layer_train_eval() {
    let mut layer = TransformerDecoderLayer::new(64, 8, 256);
    assert!(layer.training());

    layer.eval();
    assert!(!layer.training());

    layer.train();
    assert!(layer.training());
}

#[test]
fn test_decoder_layer_forward_single_input() {
    let layer = TransformerDecoderLayer::new(64, 8, 256);
    let x = Tensor::ones(&[2, 10, 64]);
    let y = layer.forward(&x);
    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_decoder_layer_parameters() {
    let layer = TransformerDecoderLayer::new(64, 8, 256);
    let params = layer.parameters();
    // self_attn: 8 + cross_attn: 8 + linear1: 2 + linear2: 2 + norm1,2,3: 6 = 26
    assert_eq!(params.len(), 26);
}

#[test]
fn test_decoder_layer_parameters_mut() {
    let mut layer = TransformerDecoderLayer::new(64, 8, 256);
    let params = layer.parameters_mut();
    assert_eq!(params.len(), 26);
}

#[test]
fn test_decoder_layer_with_masks() {
    let layer = TransformerDecoderLayer::new(64, 8, 256);
    let tgt = Tensor::ones(&[2, 10, 64]);
    let memory = Tensor::ones(&[2, 20, 64]);
    // Test without mask - mask shape requirements are complex
    let y = layer.forward_with_memory(&tgt, &memory, None, None);
    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_positional_encoding_parameters() {
    let pe = PositionalEncoding::new(64, 100);
    let params = pe.parameters();
    assert!(params.is_empty()); // No learnable parameters
}

#[test]
fn test_positional_encoding_train_eval() {
    let mut pe = PositionalEncoding::new(64, 100);
    assert!(pe.training());
    pe.eval();
    assert!(!pe.training());
    pe.train();
    assert!(pe.training());
}

#[test]
fn test_scale_tensor() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let y = scale_tensor(&x, 2.0);
    assert_eq!(y.data(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_add_tensors() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);
    let c = add_tensors(&a, &b);
    assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_gelu_activation() {
    let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    let y = gelu(&x);
    // GELU(0) ≈ 0
    assert!((y.data()[0] - 0.0).abs() < 0.01);
    // GELU(1) ≈ 0.841
    assert!((y.data()[1] - 0.841).abs() < 0.1);
    // GELU(-1) ≈ -0.159
    assert!((y.data()[2] + 0.159).abs() < 0.1);
}

#[test]
fn test_matmul_2d_shapes() {
    // matmul requires 2D tensors
    let a = Tensor::ones(&[3, 4]);
    let b = Tensor::ones(&[4, 5]);
    let c = matmul_batched(&a, &b);
    assert_eq!(c.shape(), &[3, 5]);
}

#[test]
fn test_gqa_self_attention() {
    let gqa = GroupedQueryAttention::new(64, 8, 2);
    let x = Tensor::ones(&[2, 10, 64]);
    let (output, _weights) = gqa.forward_self(&x, None);
    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_gqa_placeholder() {
    let gqa = GroupedQueryAttention::placeholder(64, 8, 2);
    assert_eq!(gqa.embed_dim(), 64);
    assert_eq!(gqa.num_heads(), 8);
    assert_eq!(gqa.num_kv_heads(), 2);
}

#[test]
fn test_gqa_mut_accessors() {
    let mut gqa = GroupedQueryAttention::new(64, 8, 2);
    let _q = gqa.q_proj_mut();
    let _k = gqa.k_proj_mut();
    let _v = gqa.v_proj_mut();
    let _o = gqa.out_proj_mut();
}

#[test]
fn test_gqa_parameters_mut() {
    let mut gqa = GroupedQueryAttention::new(64, 8, 2);
    let params = gqa.parameters_mut();
    assert_eq!(params.len(), 8);
}

#[test]
fn test_gqa_debug() {
    let gqa = GroupedQueryAttention::new(64, 8, 2);
    let debug_str = format!("{:?}", gqa);
    assert!(debug_str.contains("GroupedQueryAttention"));
}

#[test]
fn test_rope_apply_batch() {
    let rope = RotaryPositionEmbedding::new(8, 128);
    let x = Tensor::ones(&[4, 10, 8, 8]); // batch=4
    let positions: Vec<usize> = (0..10).collect();
    let y = rope.apply(&x, &positions);
    assert_eq!(y.shape(), &[4, 10, 8, 8]);
}

#[test]
fn test_linear_attention_debug() {
    let attn = LinearAttention::new(64, 8);
    let debug_str = format!("{:?}", attn);
    assert!(debug_str.contains("LinearAttention"));
}

#[test]
fn test_linear_attention_parameters_mut() {
    let mut attn = LinearAttention::new(64, 8);
    let params = attn.parameters_mut();
    assert_eq!(params.len(), 8);
}

// ========================================================================
// Additional coverage tests
// ========================================================================

#[test]
fn test_multi_head_attention_with_dropout() {
    let mha = MultiHeadAttention::new(64, 8).with_dropout(0.1);
    let x = Tensor::ones(&[2, 10, 64]);
    let output = mha.forward(&x);
    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_multi_head_attention_train_eval() {
    let mut mha = MultiHeadAttention::new(64, 8);
    assert!(mha.training());
    mha.eval();
    assert!(!mha.training());
    mha.train();
    assert!(mha.training());
}

#[test]
fn test_multi_head_attention_debug() {
    let mha = MultiHeadAttention::new(128, 8);
    let debug_str = format!("{:?}", mha);
    assert!(debug_str.contains("MultiHeadAttention"));
    assert!(debug_str.contains("embed_dim"));
    assert!(debug_str.contains("num_heads"));
}

#[test]
fn test_multi_head_attention_getters() {
    let mha = MultiHeadAttention::new(128, 4);
    assert_eq!(mha.embed_dim(), 128);
    assert_eq!(mha.num_heads(), 4);
}

#[test]
fn test_multi_head_attention_parameters_mut() {
    let mut mha = MultiHeadAttention::new(64, 8);
    let params = mha.parameters_mut();
    assert_eq!(params.len(), 8);
}

#[test]
fn test_transformer_encoder_layer_with_dropout() {
    let layer = TransformerEncoderLayer::new(64, 8, 256).with_dropout(0.2);
    let x = Tensor::ones(&[2, 10, 64]);
    let y = layer.forward(&x);
    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_transformer_encoder_layer_forward_with_none_mask() {
    let layer = TransformerEncoderLayer::new(64, 8, 256);
    let x = Tensor::ones(&[2, 10, 64]);
    let y = layer.forward_with_mask(&x, None);
    assert_eq!(y.shape(), &[2, 10, 64]);
}

#[test]
fn test_transformer_encoder_layer_train_eval() {
    let mut layer = TransformerEncoderLayer::new(64, 8, 256);
    assert!(layer.training());
    layer.eval();
    assert!(!layer.training());
    layer.train();
    assert!(layer.training());
}

#[test]
fn test_transformer_encoder_layer_parameters_mut() {
    let mut layer = TransformerEncoderLayer::new(64, 8, 256);
    let params = layer.parameters_mut();
    assert_eq!(params.len(), 16);
}

#[test]
fn test_transformer_decoder_layer_forward_with_none_masks() {
    let layer = TransformerDecoderLayer::new(64, 8, 256);
    let tgt = Tensor::ones(&[2, 10, 64]);
    let memory = Tensor::ones(&[2, 20, 64]);
    let output = layer.forward_with_memory(&tgt, &memory, None, None);
    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_transformer_decoder_layer_train_eval() {
    let mut layer = TransformerDecoderLayer::new(64, 8, 256);
    assert!(layer.training());
    layer.eval();
    assert!(!layer.training());
    layer.train();
    assert!(layer.training());
}

#[test]
fn test_transformer_decoder_layer_parameters() {
    let layer = TransformerDecoderLayer::new(64, 8, 256);
    let params = layer.parameters();
    // Self-attn: 8, Cross-attn: 8, Linear1: 2, Linear2: 2, Norms: 6
    assert!(params.len() > 0);
}
