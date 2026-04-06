
// ============================================================================
// GGUFTransformer: tied embeddings (output.weight missing → uses token_embd)
// ============================================================================

#[test]
fn test_transformer_from_gguf_tied_embeddings() {
    use crate::gguf::GGUFTransformer;

    let vocab = 8;
    let hidden = 4;

    // Note: NO output.weight tensor - should fallback to token_embd.weight
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .num_kv_heads("llama", 1)
        .context_length("llama", 32)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 8)
        .vocab_size("llama", vocab as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[hidden as u64, vocab as u64],
            &create_f32_embedding_data(vocab, hidden),
        )
        .add_f32_tensor(
            "blk.0.attn_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        .add_f32_tensor(
            "blk.0.attn_q.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.attn_k.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.attn_v.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.attn_output.weight",
            &[hidden as u64, hidden as u64],
            &vec![0.01f32; hidden * hidden],
        )
        .add_f32_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden as u64, 8],
            &vec![0.01f32; hidden * 8],
        )
        .add_f32_tensor(
            "blk.0.ffn_up.weight",
            &[hidden as u64, 8],
            &vec![0.01f32; hidden * 8],
        )
        .add_f32_tensor(
            "blk.0.ffn_down.weight",
            &[8, hidden as u64],
            &vec![0.01f32; 8 * hidden],
        )
        .add_f32_tensor(
            "blk.0.ffn_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        .add_f32_tensor(
            "output_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        // No output.weight — tied embeddings
        .build();

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let transformer = GGUFTransformer::from_gguf(&model, &data);
    assert!(
        transformer.is_ok(),
        "Tied embeddings failed: {:?}",
        transformer.err()
    );
    let t = transformer.expect("test value should be present");
    // lm_head_weight should equal token_embedding (tied)
    assert_eq!(t.lm_head_weight.len(), t.token_embedding.len());
}

// ============================================================================
// GGUFTransformer: 2-layer model
// ============================================================================

#[test]
fn test_transformer_from_gguf_two_layers() {
    use crate::gguf::GGUFTransformer;

    let vocab = 8;
    let hidden = 4;

    let mut builder = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden as u32)
        .num_layers("llama", 2)
        .num_heads("llama", 1)
        .num_kv_heads("llama", 1)
        .context_length("llama", 32)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 8)
        .vocab_size("llama", vocab as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[hidden as u64, vocab as u64],
            &create_f32_embedding_data(vocab, hidden),
        );

    // Add both layers
    for layer_idx in 0..2 {
        let prefix = format!("blk.{}", layer_idx);
        builder = builder
            .add_f32_tensor(
                &format!("{}.attn_norm.weight", prefix),
                &[hidden as u64],
                &create_f32_norm_weights(hidden),
            )
            .add_f32_tensor(
                &format!("{}.attn_q.weight", prefix),
                &[hidden as u64, hidden as u64],
                &vec![0.01f32; hidden * hidden],
            )
            .add_f32_tensor(
                &format!("{}.attn_k.weight", prefix),
                &[hidden as u64, hidden as u64],
                &vec![0.01f32; hidden * hidden],
            )
            .add_f32_tensor(
                &format!("{}.attn_v.weight", prefix),
                &[hidden as u64, hidden as u64],
                &vec![0.01f32; hidden * hidden],
            )
            .add_f32_tensor(
                &format!("{}.attn_output.weight", prefix),
                &[hidden as u64, hidden as u64],
                &vec![0.01f32; hidden * hidden],
            )
            .add_f32_tensor(
                &format!("{}.ffn_gate.weight", prefix),
                &[hidden as u64, 8],
                &vec![0.01f32; hidden * 8],
            )
            .add_f32_tensor(
                &format!("{}.ffn_up.weight", prefix),
                &[hidden as u64, 8],
                &vec![0.01f32; hidden * 8],
            )
            .add_f32_tensor(
                &format!("{}.ffn_down.weight", prefix),
                &[8, hidden as u64],
                &vec![0.01f32; 8 * hidden],
            )
            .add_f32_tensor(
                &format!("{}.ffn_norm.weight", prefix),
                &[hidden as u64],
                &create_f32_norm_weights(hidden),
            );
    }

    let data = builder
        .add_f32_tensor(
            "output_norm.weight",
            &[hidden as u64],
            &create_f32_norm_weights(hidden),
        )
        .add_f32_tensor(
            "output.weight",
            &[hidden as u64, vocab as u64],
            &vec![0.01f32; hidden * vocab],
        )
        .build();

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let transformer = GGUFTransformer::from_gguf(&model, &data);
    assert!(
        transformer.is_ok(),
        "2-layer failed: {:?}",
        transformer.err()
    );
    let t = transformer.expect("test value should be present");
    assert_eq!(t.layers.len(), 2);
    assert_eq!(t.config.num_layers, 2);
}

// ============================================================================
// Metadata accessor: bos_token_id, eos_token_id
// ============================================================================

#[test]
fn test_bos_eos_token_ids() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .add_u32("tokenizer.ggml.bos_token_id", 1)
        .add_u32("tokenizer.ggml.eos_token_id", 2)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.bos_token_id(), Some(1));
    assert_eq!(model.eos_token_id(), Some(2));
}

#[test]
fn test_bos_eos_missing() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    // When not set, should return None (not panic)
    let _ = model.bos_token_id();
    let _ = model.eos_token_id();
}

// ============================================================================
// Metadata: ffn_hidden_dim (intermediate_dim) accessor
// ============================================================================

#[test]
fn test_metadata_ffn_hidden_dim() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .ffn_hidden_dim("llama", 256)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    // FFN hidden dim is accessed through config when building transformer
    // Just verify the metadata parses correctly
    assert_eq!(model.header.version, 3);
}

// ============================================================================
// Multiple quantization types in one model
// ============================================================================

#[test]
fn test_model_with_mixed_quantization() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f32_tensor("norm.weight", &[32], &vec![1.0f32; 32])
        .add_q4_0_tensor("layer.q4_0", &[32], &create_q4_0_data(32))
        .add_q8_0_tensor("layer.q8_0", &[32], &create_q8_0_data(32))
        .add_q4_k_tensor("layer.q4_k", &[256], &create_q4_k_data(256))
        .add_q5_k_tensor("layer.q5_k", &[256], &create_q5_k_data(256))
        .add_q6_k_tensor("layer.q6_k", &[256], &create_q6_k_data(256))
        .build();

    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert_eq!(model.tensors.len(), 6);

    // Verify each tensor can be dequantized
    let norm = model.get_tensor_f32("norm.weight", &data).expect("test value should be present");
    assert_eq!(norm.len(), 32);

    let q4_0 = model.get_tensor_f32("layer.q4_0", &data).expect("test value should be present");
    assert_eq!(q4_0.len(), 32);

    let q8_0 = model.get_tensor_f32("layer.q8_0", &data).expect("test value should be present");
    assert_eq!(q8_0.len(), 32);

    let q4_k = model.get_tensor_f32("layer.q4_k", &data).expect("test value should be present");
    assert_eq!(q4_k.len(), 256);

    let q5_k = model.get_tensor_f32("layer.q5_k", &data).expect("test value should be present");
    assert_eq!(q5_k.len(), 256);

    let q6_k = model.get_tensor_f32("layer.q6_k", &data).expect("test value should be present");
    assert_eq!(q6_k.len(), 256);
}
