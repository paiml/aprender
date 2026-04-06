
#[test]
fn test_phase35_get_tensor_ref_q8_0() {
    // Test Q8_0 byte size calculation
    let hidden_dim = 64usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q8_0_data = create_q8_0_data(hidden_dim * hidden_dim);

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
        .add_q8_0_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_q8_0_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q8_0_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q8_0_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 128],
            &create_q8_0_data(hidden_dim * 128),
        )
        .add_q8_0_tensor(
            "blk.0.ffn_down.weight",
            &[128, hidden_dim as u64],
            &create_q8_0_data(128 * hidden_dim),
        )
        .add_q8_0_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 128],
            &create_q8_0_data(hidden_dim * 128),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q8_0 model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q8_0");

    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, .. } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q8_0);
            // Q8_0: 34 bytes per 32 elements
            // 4096 elements => 128 blocks => 4352 bytes
            assert_eq!(q.byte_size, 128 * 34);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_get_tensor_ref_q5_k() {
    // Test Q5_K byte size calculation
    let hidden_dim = 256usize; // Must be multiple of 256 for K-quants
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q5_k_data = create_q5_k_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 512)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q5_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_q5_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_q5_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_q5_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q5_k_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q5_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 512],
            &create_q5_k_data(hidden_dim * 512),
        )
        .add_q5_k_tensor(
            "blk.0.ffn_down.weight",
            &[512, hidden_dim as u64],
            &create_q5_k_data(512 * hidden_dim),
        )
        .add_q5_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 512],
            &create_q5_k_data(hidden_dim * 512),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q5_K model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q5_K");

    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, .. } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q5_K);
            // Q5_K: 176 bytes per 256 elements
            // 256*256 = 65536 elements => 256 super-blocks => 45056 bytes
            assert_eq!(q.byte_size, 256 * 176);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_get_tensor_ref_q6_k() {
    // Test Q6_K byte size calculation
    let hidden_dim = 256usize;
    let vocab_size = 100usize;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q6_k_data = create_q6_k_data(hidden_dim * hidden_dim);

    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", 512)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q6_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_q6_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q6_k_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q6_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, 512],
            &create_q6_k_data(hidden_dim * 512),
        )
        .add_q6_k_tensor(
            "blk.0.ffn_down.weight",
            &[512, hidden_dim as u64],
            &create_q6_k_data(512 * hidden_dim),
        )
        .add_q6_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, 512],
            &create_q6_k_data(hidden_dim * 512),
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse Q6_K model");
    let transformer = QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load Q6_K");

    match &transformer.layers[0].qkv_weight {
        QKVWeights::Separate { q, .. } => {
            assert_eq!(q.qtype, GGUF_TYPE_Q6_K);
            // Q6_K: 210 bytes per 256 elements
            // 65536 elements => 256 super-blocks => 53760 bytes
            assert_eq!(q.byte_size, 256 * 210);
        },
        _ => panic!("Expected Separate"),
    }
}

#[test]
fn test_phase35_transformer_missing_tensor_error() {
    // Test error handling when required tensor is missing
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 64)
        .num_layers("llama", 1)
        .num_heads("llama", 4)
        .num_kv_heads("llama", 4)
        .context_length("llama", 256)
        // Missing token_embd.weight!
        .build();

    let model = GGUFModel::from_bytes(&data).expect("Should parse empty model");
    let result = QuantizedGGUFTransformer::from_gguf(&model, &data);

    assert!(
        result.is_err(),
        "Should fail when token_embd.weight is missing"
    );

    // Extract error message
    // Phase 2: ValidatedModelConfig catches vocab_size=0 (inferred from missing
    // token_embd.weight) before tensor loading. Both error types are valid.
    match result {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("token_embd.weight")
                    || err.contains("Tensor")
                    || err.contains("not found")
                    || err.contains("vocab_size"),
                "Error should mention missing tensor or invalid vocab_size: {}",
                err
            );
        },
        Ok(_) => panic!("Expected error for missing tensor"),
    }
}

#[test]
fn test_phase35_transformer_lm_head_fallback() {
    // Test that lm_head falls back to token_embd when output.weight is missing
    let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);

    let model = GGUFModel::from_bytes(&data).expect("Should parse");
    let transformer =
        QuantizedGGUFTransformer::from_gguf(&model, &data).expect("Should load with fallback");

    // lm_head should exist (fallback to token_embd)
    assert!(transformer.lm_head_weight.byte_size > 0);
    assert_eq!(transformer.lm_head_weight.num_elements, 100 * 64); // vocab * hidden
}

#[test]
fn test_phase35_data_factory_helpers() {
    // Test the data factory helper functions directly
    let q4_0 = create_q4_0_data(64);
    assert_eq!(q4_0.len(), 2 * 18); // 64/32 = 2 blocks * 18 bytes

    let q8_0 = create_q8_0_data(64);
    assert_eq!(q8_0.len(), 2 * 34); // 64/32 = 2 blocks * 34 bytes

    let q4_k = create_q4_k_data(256);
    assert_eq!(q4_k.len(), 144); // 256/256 = 1 super-block * 144 bytes

    let q5_k = create_q5_k_data(512);
    assert_eq!(q5_k.len(), 2 * 176); // 512/256 = 2 super-blocks * 176 bytes

    let q6_k = create_q6_k_data(512);
    assert_eq!(q6_k.len(), 2 * 210); // 512/256 = 2 super-blocks * 210 bytes

    let embed = create_f32_embedding_data(10, 8);
    assert_eq!(embed.len(), 80); // 10 * 8

    let norm = create_f32_norm_weights(32);
    assert_eq!(norm.len(), 32);
    assert!(norm.iter().all(|&v| (v - 1.0).abs() < 1e-6)); // All ones
}
