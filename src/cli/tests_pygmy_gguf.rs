
// ============================================================================
// Verify synthetic model is actually valid GGUF
// ============================================================================

#[test]
fn test_pygmy_gguf_parses_correctly() {
    use crate::gguf::GGUFModel;

    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);

    let model = GGUFModel::from_bytes(&gguf_data);
    assert!(model.is_ok(), "Pygmy GGUF should parse: {:?}", model.err());

    let model = model.unwrap();
    assert_eq!(model.architecture(), Some("llama"));
    assert!(model.tensors.len() > 5, "Should have multiple tensors");
}

#[test]
fn test_micro_gguf_parses_correctly() {
    use crate::gguf::test_factory::{
        create_f32_embedding_data, create_f32_norm_weights, create_q4_k_data,
    };
    use crate::gguf::GGUFModel;

    let vocab_size = 16;
    let hidden_dim = 32;
    let intermediate_dim = 64;
    let kv_dim = 32;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 2)
        .num_kv_heads("llama", 2)
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

    let model = GGUFModel::from_bytes(&gguf_data);
    assert!(model.is_ok(), "Micro GGUF should parse: {:?}", model.err());
}
