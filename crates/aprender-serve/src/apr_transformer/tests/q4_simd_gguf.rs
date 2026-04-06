
#[test]
fn test_from_gguf_separate_qkv() {
    use crate::gguf::test_helpers::create_q4k_test_data;
    use crate::gguf::{GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let hidden_dim = 64;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        intermediate_dim: 128,
        num_heads,
        num_kv_heads,
        num_layers: 1,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let q_weight = create_q4k_test_data(hidden_dim, hidden_dim);
    let k_weight = create_q4k_test_data(hidden_dim, kv_dim);
    let v_weight = create_q4k_test_data(hidden_dim, kv_dim);

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        },
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, 128),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(128, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, 128)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; 100 * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, 100),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let q4_model = QuantizedAprTransformerQ4::from_gguf(&model);

    // Separate QKV should be concatenated into fused format
    assert_eq!(q4_model.layers.len(), 1);
    let layer = &q4_model.layers[0];
    // Concatenated QKV: data should be q_data + k_data + v_data
    let q_data_len = create_q4k_test_data(hidden_dim, hidden_dim).data.len();
    let k_data_len = create_q4k_test_data(hidden_dim, kv_dim).data.len();
    let v_data_len = create_q4k_test_data(hidden_dim, kv_dim).data.len();
    assert_eq!(
        layer.qkv_weight.data.len(),
        q_data_len + k_data_len + v_data_len
    );
    assert_eq!(layer.qkv_weight.in_dim, hidden_dim);
    assert_eq!(layer.qkv_weight.out_dim, hidden_dim + kv_dim + kv_dim);
}

#[test]
fn test_from_gguf_gqa_config() {
    use crate::gguf::test_helpers::create_test_model_with_config;
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_heads: 8,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 200,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let gguf_model = create_test_model_with_config(&config);
    let q4_model = QuantizedAprTransformerQ4::from_gguf(&gguf_model);

    assert_eq!(q4_model.config.num_heads, 8);
    assert_eq!(q4_model.config.num_kv_heads, 2);
    // create_test_model_with_config creates 1 layer regardless of config.num_layers
    assert_eq!(q4_model.layers.len(), 1);
    assert_eq!(q4_model.config.context_length, 512);
    assert_eq!(q4_model.config.rope_theta, 10000.0);
}

#[test]
fn test_from_gguf_with_ffn_gate() {
    use crate::gguf::test_helpers::create_q4k_test_data;
    use crate::gguf::{GGUFConfig, OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel};

    let hidden_dim = 64;
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let kv_dim = 4 * (hidden_dim / 4);
    let qkv_out = hidden_dim + 2 * kv_dim;

    let layer = OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_q4k_test_data(hidden_dim, qkv_out)),
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, 128),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(128, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, 128)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let model = OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; 100 * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, 100),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    let q4_model = QuantizedAprTransformerQ4::from_gguf(&model);
    let layer = &q4_model.layers[0];

    // Gate weight should be present
    assert!(layer.ffn_gate_weight.is_some());
    let gate = layer.ffn_gate_weight.as_ref().expect("test value should be present");
    assert_eq!(gate.in_dim, hidden_dim);
    assert_eq!(gate.out_dim, 128);

    // Other weights should be present
    assert_eq!(layer.attn_output_weight.in_dim, hidden_dim);
    assert_eq!(layer.attn_output_weight.out_dim, hidden_dim);
    assert_eq!(layer.ffn_up_weight.in_dim, hidden_dim);
    assert_eq!(layer.ffn_up_weight.out_dim, 128);
    assert_eq!(layer.ffn_down_weight.in_dim, 128);
    assert_eq!(layer.ffn_down_weight.out_dim, hidden_dim);
}
