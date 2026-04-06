
/// Build a minimal Phi-2 style GGUF model (fused QKV)
#[must_use]
pub fn build_minimal_phi2_gguf(
    vocab_size: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
) -> Vec<u8> {
    // Create tensor data
    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);

    // Fused QKV: hidden -> 3 * hidden - use row-padded layout for 2D tensors
    let qkv_out_dim = 3 * hidden_dim;
    let qkv_data = create_q4_k_data_2d(hidden_dim, qkv_out_dim);
    let attn_out_data = create_q4_k_data_2d(hidden_dim, hidden_dim);
    let ffn_up_data = create_q4_k_data_2d(hidden_dim, intermediate_dim);
    let ffn_down_data = create_q4_k_data_2d(intermediate_dim, hidden_dim);

    GGUFBuilder::new()
        // Metadata
        .architecture("phi2")
        .hidden_dim("phi2", hidden_dim as u32)
        .num_layers("phi2", 1)
        .num_heads("phi2", num_heads as u32)
        .num_kv_heads("phi2", num_heads as u32) // MHA, not GQA
        .context_length("phi2", 256)
        .rope_freq_base("phi2", 10000.0)
        .rms_epsilon("phi2", 1e-5)
        // Token embedding
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        // Layer 0 attention (fused QKV)
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_qkv.weight",
            &[hidden_dim as u64, qkv_out_dim as u64],
            &qkv_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        // Layer 0 FFN (no gate for Phi-2 style GELU)
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
        // Output
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build()
}
