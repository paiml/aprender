
#[test]
fn test_phase35_transformer_from_minimal_llama() {
    // Build a minimal LLaMA-style GGUF
    let data = build_minimal_llama_gguf(
        100, // vocab_size
        64,  // hidden_dim (must be divisible by num_heads)
        128, // intermediate_dim
        4,   // num_heads
        4,   // num_kv_heads
    );

    // Parse the GGUF model
    let model = GGUFModel::from_bytes(&data).expect("Should parse minimal LLaMA GGUF");

    // Load as quantized transformer
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load transformer");

    // Verify config was loaded correctly
    assert_eq!(transformer.config.architecture, "llama");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);
    assert_eq!(transformer.config.num_kv_heads, 4);

    // Verify token embedding was loaded
    assert_eq!(transformer.token_embedding.len(), 100 * 64); // vocab * hidden

    // Verify output norm was loaded
    assert_eq!(transformer.output_norm_weight.len(), 64);

    // Verify layer was loaded with correct structure
    assert_eq!(transformer.layers.len(), 1);

    let layer = &transformer.layers[0];
    // Attention norm should be f32
    assert_eq!(layer.attn_norm_weight.len(), 64);

    // QKV should be separate (LLaMA style)
    match &layer.qkv_weight {
        QKVWeights::Separate { q, k, v } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(k.qtype, GGUF_TYPE_Q4_K);
            assert_eq!(v.qtype, GGUF_TYPE_Q4_K);
            // Q: hidden -> hidden (64*64)
            assert_eq!(q.num_elements, 64 * 64);
            // K, V: hidden -> kv_dim (64*64 since num_kv_heads = num_heads)
            assert_eq!(k.num_elements, 64 * 64);
            assert_eq!(v.num_elements, 64 * 64);
        },
        QKVWeights::Fused(_) => panic!("Expected Separate QKV for LLaMA style"),
    }

    // FFN weights should be quantized
    assert_eq!(layer.ffn_up_weight.qtype, GGUF_TYPE_Q4_K);
    assert_eq!(layer.ffn_down_weight.qtype, GGUF_TYPE_Q4_K);
    assert!(layer.ffn_gate_weight.is_some(), "LLaMA should have gate");
}

#[test]
fn test_phase35_transformer_from_minimal_phi2() {
    // Build a minimal Phi-2 style GGUF (fused QKV)
    let data = build_minimal_phi2_gguf(
        100, // vocab_size
        64,  // hidden_dim
        128, // intermediate_dim
        4,   // num_heads
    );

    let model = GGUFModel::from_bytes(&data).expect("Should parse minimal Phi-2 GGUF");

    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load transformer");

    // Verify config
    assert_eq!(transformer.config.architecture, "phi2");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
    assert_eq!(transformer.config.num_heads, 4);

    // Verify layer has fused QKV
    let layer = &transformer.layers[0];
    match &layer.qkv_weight {
        QKVWeights::Fused(fused) => {
            assert_eq!(fused.qtype, GGUF_TYPE_Q4_K);
            // Fused: hidden -> 3 * hidden
            assert_eq!(fused.num_elements, 64 * (3 * 64));
        },
        QKVWeights::Separate { .. } => panic!("Expected Fused QKV for Phi-2 style"),
    }

    // Phi-2 style has no gate
    assert!(layer.ffn_gate_weight.is_none());
}

#[test]
fn test_phase35_transformer_with_gqa() {
    // Test Grouped Query Attention (GQA) - fewer KV heads than Q heads
    let hidden_dim = 64usize;
    let num_heads = 8usize;
    let num_kv_heads = 2usize; // GQA: 4 Q heads per KV head
    let head_dim = hidden_dim / num_heads; // 8
    let kv_dim = num_kv_heads * head_dim; // 16
    let vocab_size = 100usize;
    let intermediate_dim = 128usize;

    // Create tensor data
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", num_heads as u32)
        .num_kv_heads("llama", num_kv_heads as u32)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse GQA model");
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load GQA transformer");

    // Verify GQA config
    assert_eq!(transformer.config.num_heads, 8);
    assert_eq!(transformer.config.num_kv_heads, 2);

    // Verify K, V have smaller dimensions
    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, k, v } => {
            assert_eq!(q.num_elements, 64 * 64); // full
            assert_eq!(k.num_elements, 64 * 16); // reduced
            assert_eq!(v.num_elements, 64 * 16); // reduced
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_transformer_multiple_layers() {
    // Test with 2 layers to verify layer indexing
    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * hidden_dim);
    let v_data = create_q4_k_data(hidden_dim * hidden_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let mut builder = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 2) // 2 layers
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        );

    // Add tensors for both layers
    for layer_idx in 0..2 {
        builder = builder
            .add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", layer_idx),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_q.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &q_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_k.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &k_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_v.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &v_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.attn_output.weight", layer_idx),
                &[hidden_dim as u64, hidden_dim as u64],
                &attn_out_data,
            )
            .add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", layer_idx),
                &[hidden_dim as u64],
                &norm_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_up.weight", layer_idx),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_up_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_down.weight", layer_idx),
                &[intermediate_dim as u64, hidden_dim as u64],
                &ffn_down_data,
            )
            .add_q4_k_tensor(
                &format!("blk.{}.ffn_gate.weight", layer_idx),
                &[hidden_dim as u64, intermediate_dim as u64],
                &ffn_gate_data,
            );
    }

    let data = builder
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse multi-layer model");
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load multi-layer");

    // Verify both layers loaded
    assert_eq!(transformer.config.num_layers, 2);
    assert_eq!(transformer.layers.len(), 2);

    // Both layers should have same structure
    for (idx, layer) in transformer.layers.iter().enumerate() {
        assert_eq!(layer.attn_norm_weight.len(), 64, "Layer {} norm", idx);
        assert!(layer.ffn_gate_weight.is_some(), "Layer {} gate", idx);
    }
}

#[test]
fn test_phase35_get_tensor_ref_q4_0() {
    // Test Q4_0 byte size calculation through actual loading
    let hidden_dim = 64usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q4_0_data = create_q4_0_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 128)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        // Use Q4_0 for attention weights
        .add_q4_0_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_q4_0_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q4_0_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_0_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 128],
            &create_q4_0_data(hidden_dim * 128),
        )
        .add_q4_0_tensor(
            "blk.0.ffn_down.weight",
            &[128, hidden_dim as u64],
            &create_q4_0_data(128 * hidden_dim),
        )
        .add_q4_0_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 128],
            &create_q4_0_data(hidden_dim * 128),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q4_0 model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q4_0");

    // Verify Q4_0 type was preserved
    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, k, v } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q4_0);
            assert_eq!(k.qtype, GGUF_TYPE_Q4_0);
            assert_eq!(v.qtype, GGUF_TYPE_Q4_0);

            // Q4_0 byte size: 18 bytes per 32 elements
            // 64*64 = 4096 elements => 128 blocks => 2304 bytes
            assert_eq!(q.byte_size, 128 * 18);
        },
        _ => panic!("Expected Separate"),
    }
}
