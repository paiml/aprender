pub(crate) use super::*;
pub(crate) use std::collections::BTreeMap;

// ========================================================================
// hf_to_gguf_name: Attention projection patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_q_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.q_proj.weight"),
        "blk.0.attn_q.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_q_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.5.self_attn.q_proj.bias"),
        "blk.5.attn_q.bias"
    );
}

#[test]
fn test_hf_to_gguf_name_k_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.3.self_attn.k_proj.weight"),
        "blk.3.attn_k.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_k_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.11.self_attn.k_proj.bias"),
        "blk.11.attn_k.bias"
    );
}

#[test]
fn test_hf_to_gguf_name_v_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.7.self_attn.v_proj.weight"),
        "blk.7.attn_v.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_v_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.v_proj.bias"),
        "blk.0.attn_v.bias"
    );
}

#[test]
fn test_hf_to_gguf_name_o_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.2.self_attn.o_proj.weight"),
        "blk.2.attn_output.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_o_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.31.self_attn.o_proj.bias"),
        "blk.31.attn_output.bias"
    );
}

// ========================================================================
// hf_to_gguf_name: Fused QKV patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_qkv_proj_weight() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.qkv_proj.weight"),
        "blk.0.attn_qkv.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_qkv_proj_bias() {
    assert_eq!(
        hf_to_gguf_name("model.layers.4.self_attn.qkv_proj.bias"),
        "blk.4.attn_qkv.bias"
    );
}

// ========================================================================
// hf_to_gguf_name: MLP / FFN patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_gate_proj() {
    assert_eq!(
        hf_to_gguf_name("model.layers.1.mlp.gate_proj.weight"),
        "blk.1.ffn_gate.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_up_proj() {
    assert_eq!(
        hf_to_gguf_name("model.layers.10.mlp.up_proj.weight"),
        "blk.10.ffn_up.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_down_proj() {
    assert_eq!(
        hf_to_gguf_name("model.layers.23.mlp.down_proj.weight"),
        "blk.23.ffn_down.weight"
    );
}

// ========================================================================
// hf_to_gguf_name: Layer norm patterns
// ========================================================================

#[test]
fn test_hf_to_gguf_name_input_layernorm() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.input_layernorm.weight"),
        "blk.0.attn_norm.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_post_attention_layernorm() {
    assert_eq!(
        hf_to_gguf_name("model.layers.15.post_attention_layernorm.weight"),
        "blk.15.ffn_norm.weight"
    );
}

// ========================================================================
// hf_to_gguf_name: Non-layer tensors (embedding, lm_head, output norm)
// ========================================================================

#[test]
fn test_hf_to_gguf_name_embed_tokens() {
    assert_eq!(
        hf_to_gguf_name("model.embed_tokens.weight"),
        "token_embd.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_lm_head() {
    assert_eq!(hf_to_gguf_name("lm_head.weight"), "output.weight");
}

#[test]
fn test_hf_to_gguf_name_output_norm() {
    assert_eq!(hf_to_gguf_name("model.norm.weight"), "output_norm.weight");
}

// ========================================================================
// hf_to_gguf_name: Unknown / passthrough
// ========================================================================

#[test]
fn test_hf_to_gguf_name_unknown_passthrough() {
    // Completely unknown names should pass through unchanged
    assert_eq!(hf_to_gguf_name("some.custom.tensor"), "some.custom.tensor");
}

#[test]
fn test_hf_to_gguf_name_unknown_layer_suffix_passthrough() {
    // GH-277: Unknown layer suffixes now pass through with the APR name unchanged
    // (the legacy mapper doesn't have a mapping, so it falls through to the
    // global check which also fails, resulting in passthrough with warning)
    assert_eq!(
        hf_to_gguf_name("model.layers.0.some_unknown_suffix.weight"),
        "model.layers.0.some_unknown_suffix.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_empty_string() {
    assert_eq!(hf_to_gguf_name(""), "");
}

// ========================================================================
// hf_to_gguf_name: Multi-digit layer indices (regression guard)
//
// Bug class: off-by-one in layer number parsing. Models with 100+ layers
// (e.g. Llama-70B has 80 layers) must produce correct multi-digit indices.
// ========================================================================

#[test]
fn test_hf_to_gguf_name_high_layer_index() {
    assert_eq!(
        hf_to_gguf_name("model.layers.79.self_attn.q_proj.weight"),
        "blk.79.attn_q.weight"
    );
}

#[test]
fn test_hf_to_gguf_name_three_digit_layer_index() {
    // Some very deep models exceed 100 layers
    assert_eq!(
        hf_to_gguf_name("model.layers.127.mlp.gate_proj.weight"),
        "blk.127.ffn_gate.weight"
    );
}

// ========================================================================
// hf_to_gguf_name: Consistency — roundtrip with Architecture::qwen2_map_name
//
// Bug class: asymmetric mapping. If HF->GGUF and GGUF->HF aren't inverses,
// round-trip export/import corrupts tensor names silently.
// ========================================================================

#[test]
fn test_hf_to_gguf_name_all_layer_suffixes_mapped() {
    // Verify every known suffix produces a different GGUF name (no collisions)
    let suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ];

    let mut gguf_names: Vec<String> = suffixes
        .iter()
        .map(|s| hf_to_gguf_name(&format!("model.layers.0.{s}")))
        .collect();

    let count_before = gguf_names.len();
    gguf_names.sort();
    gguf_names.dedup();
    assert_eq!(
        gguf_names.len(),
        count_before,
        "Name collision detected in hf_to_gguf_name mapping"
    );
}

// ========================================================================
// GH-279: Qwen3 QK norm tensor export mappings
// ========================================================================

#[test]
fn test_hf_to_gguf_name_qk_norm_gh279() {
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.q_norm.weight"),
        "blk.0.attn_q_norm.weight"
    );
    assert_eq!(
        hf_to_gguf_name("model.layers.0.self_attn.k_norm.weight"),
        "blk.0.attn_k_norm.weight"
    );
    assert_eq!(
        hf_to_gguf_name("model.layers.35.self_attn.q_norm.weight"),
        "blk.35.attn_q_norm.weight"
    );
    assert_eq!(
        hf_to_gguf_name("model.layers.35.self_attn.k_norm.weight"),
        "blk.35.attn_k_norm.weight"
    );
}

#[test]
fn test_qk_norm_roundtrip_import_export_gh279() {
    // GGUF import → APR → GGUF export must roundtrip
    let gguf_q = "blk.5.attn_q_norm.weight";
    let gguf_k = "blk.5.attn_k_norm.weight";

    let apr_q = Architecture::Qwen3.map_name(gguf_q);
    let apr_k = Architecture::Qwen3.map_name(gguf_k);

    assert_eq!(apr_q, "model.layers.5.self_attn.q_norm.weight");
    assert_eq!(apr_k, "model.layers.5.self_attn.k_norm.weight");

    assert_eq!(hf_to_gguf_name(&apr_q), gguf_q);
    assert_eq!(hf_to_gguf_name(&apr_k), gguf_k);
}

// ========================================================================
// infer_vocab_hidden: Embedding tensor present
// ========================================================================

#[test]
fn test_infer_vocab_hidden_from_embed_tokens() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Qwen2-0.5B: vocab=151936, hidden=896
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 151936 * 896], vec![151936, 896]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 151936);
    assert_eq!(hidden, 896);
}

#[test]
fn test_infer_vocab_hidden_from_token_embd() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF-style naming
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![0.0; 32000 * 4096], vec![32000, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 32000);
    assert_eq!(hidden, 4096);
}

// ========================================================================
// infer_vocab_hidden: Fallback to lm_head
// ========================================================================

#[test]
fn test_infer_vocab_hidden_fallback_to_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // No embedding tensor, but lm_head present
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.0; 32000 * 4096], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 4096 * 4096], vec![4096, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 32000);
    assert_eq!(hidden, 4096);
}

#[test]
fn test_infer_vocab_hidden_fallback_to_output_weight() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF-style output.weight as lm_head equivalent
    tensors.insert(
        "output.weight".to_string(),
        (vec![0.0; 128256 * 4096], vec![128256, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 128256);
    assert_eq!(hidden, 4096);
}

// ========================================================================
// infer_vocab_hidden: Fallback to q_proj for hidden_dim only
// ========================================================================

#[test]
fn test_infer_vocab_hidden_hidden_from_q_proj() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // No embedding or lm_head — only layer weights
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 4096 * 4096], vec![4096, 4096]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![0.0; 11008 * 4096], vec![11008, 4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    // vocab_size cannot be inferred without embedding/lm_head
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 4096);
}

// ========================================================================
// infer_vocab_hidden: Empty tensor map
// ========================================================================

#[test]
fn test_infer_vocab_hidden_empty_tensors() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 0);
}

// ========================================================================
// infer_vocab_hidden: 1D tensors (should NOT match — requires 2D)
//
// Bug class: 1D norm weights like input_layernorm.weight with shape [4096]
// could be misinterpreted as vocab_size=4096, hidden=0 if the dimension
// check is missing.
// ========================================================================

#[test]
fn test_infer_vocab_hidden_ignores_1d_embedding() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D tensor should NOT be treated as an embedding
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 4096], vec![4096]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    // 1D tensor doesn't satisfy the shape.len() == 2 check
    assert_eq!(vocab, 0);
    assert_eq!(hidden, 0);
}

#[test]
fn test_infer_vocab_hidden_ignores_1d_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D lm_head should NOT match
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.0; 32000], vec![32000]),
    );
    // But 2D q_proj should give hidden_dim
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 2048 * 2048], vec![2048, 2048]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    assert_eq!(vocab, 0); // 1D lm_head skipped
    assert_eq!(hidden, 2048); // from q_proj fallback
}

// ========================================================================
// infer_vocab_hidden: Embedding takes priority over lm_head
//
// Bug class: If lm_head is checked first and embed_tokens second, the
// wrong tensor could provide dimensions for tied-embedding models where
// lm_head and embed_tokens have different names but same data.
// ========================================================================

#[test]
fn test_infer_vocab_hidden_embedding_priority_over_lm_head() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Both present — embedding should win
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.0; 32000 * 4096], vec![32000, 4096]),
    );
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 151936 * 896], vec![151936, 896]),
    );
    let (vocab, hidden) = infer_vocab_hidden(&tensors);
    // Embedding shapes should be used, not lm_head
    assert_eq!(vocab, 151936);
    assert_eq!(hidden, 896);
}


include!("export_tests_include_01.rs");
