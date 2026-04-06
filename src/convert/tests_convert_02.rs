
#[test]
fn test_convert_with_q6_k_tensors() {
    let vocab_size = 32;
    let hidden_dim = 64;
    let intermediate_dim = 128;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q6k_data = create_q6_k_data(hidden_dim * hidden_dim);
    let ffn_q6k = create_q6_k_data(hidden_dim * intermediate_dim);

    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q6_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6k_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q6_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_q6k,
        )
        .add_q6_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_q6k,
        )
        .add_q6_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_q6k,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();
    assert_eq!(apr.config.hidden_dim, hidden_dim);
}

// ============================================================================
// AprTransformer::from_apr_bytes direct tests
// ============================================================================

#[test]
fn test_apr_transformer_from_apr_bytes_small_file() {
    let result = AprTransformer::from_apr_bytes(&[0; 32]);
    assert!(result.is_err());
}

#[test]
fn test_apr_transformer_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"NOAP");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_transformer_from_apr_bytes_version_1() {
    // APR v1 should also be accepted
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR1");
    data[4] = 1;
    // Still likely fails due to missing data but tests version check
    let _ = AprTransformer::from_apr_bytes(&data);
}

#[test]
fn test_apr_transformer_from_apr_bytes_version_2() {
    // APR v2 format
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR2");
    data[4] = 2;
    // Still likely fails due to missing data but tests version check
    let _ = AprTransformer::from_apr_bytes(&data);
}

// ============================================================================
// Edge cases and boundary conditions
// ============================================================================

#[test]
fn test_convert_preserves_rope_theta() {
    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 64)
        .rope_freq_base("llama", 50000.0) // Non-default rope_theta
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 128)
        .add_f32_tensor(
            "token_embd.weight",
            &[32, 64],
            &create_f32_embedding_data(32, 64),
        )
        .add_f32_tensor(
            "blk.0.attn_norm.weight",
            &[64],
            &create_f32_norm_weights(64),
        )
        .add_q4_k_tensor("blk.0.attn_q.weight", &[64, 64], &create_q4_k_data(64 * 64))
        .add_q4_k_tensor("blk.0.attn_k.weight", &[64, 64], &create_q4_k_data(64 * 64))
        .add_q4_k_tensor("blk.0.attn_v.weight", &[64, 64], &create_q4_k_data(64 * 64))
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[64, 64],
            &create_q4_k_data(64 * 64),
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[64], &create_f32_norm_weights(64))
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[64, 128],
            &create_q4_k_data(64 * 128),
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[128, 64],
            &create_q4_k_data(128 * 64),
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[64, 128],
            &create_q4_k_data(64 * 128),
        )
        .add_f32_tensor("output_norm.weight", &[64], &create_f32_norm_weights(64))
        .build();

    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();
    assert!((apr.config.rope_theta - 50000.0).abs() < 0.1);
}

#[test]
fn test_convert_preserves_epsilon() {
    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-6) // Different epsilon
        .ffn_hidden_dim("llama", 128)
        .add_f32_tensor(
            "token_embd.weight",
            &[32, 64],
            &create_f32_embedding_data(32, 64),
        )
        .add_f32_tensor(
            "blk.0.attn_norm.weight",
            &[64],
            &create_f32_norm_weights(64),
        )
        .add_q4_k_tensor("blk.0.attn_q.weight", &[64, 64], &create_q4_k_data(64 * 64))
        .add_q4_k_tensor("blk.0.attn_k.weight", &[64, 64], &create_q4_k_data(64 * 64))
        .add_q4_k_tensor("blk.0.attn_v.weight", &[64, 64], &create_q4_k_data(64 * 64))
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[64, 64],
            &create_q4_k_data(64 * 64),
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[64], &create_f32_norm_weights(64))
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[64, 128],
            &create_q4_k_data(64 * 128),
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[128, 64],
            &create_q4_k_data(128 * 64),
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[64, 128],
            &create_q4_k_data(64 * 128),
        )
        .add_f32_tensor("output_norm.weight", &[64], &create_f32_norm_weights(64))
        .build();

    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();
    assert!((apr.config.eps - 1e-6).abs() < 1e-8);
}

#[test]
fn test_convert_multi_layer() {
    // Create a model with 2 layers
    let hidden_dim = 64;
    let intermediate_dim = 128;
    let vocab_size = 32;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let mut builder = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 2)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        );

    // Add both layers
    for layer in 0..2 {
        builder = builder
            .add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", layer),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_q.weight", layer),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_k.weight", layer),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_v.weight", layer),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_output.weight", layer),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", layer),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_up.weight", layer),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_down.weight", layer),
                &[intermediate_dim as u64, hidden_dim as u64],
                &ffn_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_gate.weight", layer),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_data,
            );
    }

    let gguf_data = builder
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let apr = GgufToAprConverter::convert(&gguf_data).unwrap();
    assert_eq!(apr.config.num_layers, 2);
    assert_eq!(apr.layers.len(), 2);
}
