
#[test]
fn test_owned_quantized_tensor_clone() {
    // Verify Clone implementation for OwnedQuantizedTensor
    let original = OwnedQuantizedTensor {
        data: vec![1, 2, 3, 4],
        in_dim: 2,
        out_dim: 2,
        qtype: GGUF_TYPE_Q8_0,
    };

    let cloned = original.clone();

    assert_eq!(cloned.data, original.data);
    assert_eq!(cloned.in_dim, original.in_dim);
    assert_eq!(cloned.out_dim, original.out_dim);
    assert_eq!(cloned.qtype, original.qtype);
}

#[test]
fn test_owned_qkv_weights_clone() {
    // Verify Clone implementation for OwnedQKVWeights
    let tensor = OwnedQuantizedTensor {
        data: vec![1, 2, 3],
        in_dim: 1,
        out_dim: 3,
        qtype: GGUF_TYPE_Q4_K,
    };

    let original = OwnedQKVWeights::Fused(tensor);
    let cloned = original.clone();

    assert_eq!(cloned.out_dim(), original.out_dim());
}

#[test]
fn test_owned_quantized_layer_clone() {
    // Verify Clone implementation for OwnedQuantizedLayer
    let original = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0, 2.0],
        attn_norm_bias: Some(vec![0.1, 0.2]),
        qkv_weight: OwnedQKVWeights::Fused(OwnedQuantizedTensor {
            data: vec![1, 2, 3],
            in_dim: 1,
            out_dim: 3,
            qtype: GGUF_TYPE_Q4_K,
        }),
        qkv_bias: None,
        attn_output_weight: OwnedQuantizedTensor {
            data: vec![4, 5],
            in_dim: 1,
            out_dim: 2,
            qtype: GGUF_TYPE_Q4_K,
        },
        attn_output_bias: None,
        ffn_up_weight: OwnedQuantizedTensor {
            data: vec![6, 7],
            in_dim: 1,
            out_dim: 2,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_up_bias: None,
        ffn_down_weight: OwnedQuantizedTensor {
            data: vec![8, 9],
            in_dim: 2,
            out_dim: 1,
            qtype: GGUF_TYPE_Q4_K,
        },
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let cloned = original.clone();

    assert_eq!(cloned.attn_norm_weight, original.attn_norm_weight);
    assert_eq!(cloned.attn_norm_bias, original.attn_norm_bias);
    assert_eq!(cloned.qkv_weight.out_dim(), original.qkv_weight.out_dim());
}
