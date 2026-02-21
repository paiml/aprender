use super::*;

// ========================================================================
// infer_model_config: GQA detection (k_proj < q_proj)
// ========================================================================

#[test]
fn test_infer_model_config_gqa_detection() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Model with GQA: hidden=4096, 32 attention heads, 8 KV heads
    // head_dim = 4096/32 = 128
    // k_proj shape: [num_kv_heads * head_dim, hidden] = [1024, 4096]
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![], vec![1024, 4096]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    assert_eq!(v["num_attention_heads"], 32);
    // GQA: k_proj first dim = 1024, head_dim = 128
    // num_kv_heads = 1024 / 128 = 8
    assert_eq!(v["num_key_value_heads"], 8);
}

// ========================================================================
// infer_model_config: 1D embedding tensor fallback
// ========================================================================

#[test]
fn test_infer_model_config_1d_embedding_fallback() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 1D embedding tensor - uses last dim as-is
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // 1D tensor: shape.last() = 4096
    assert_eq!(v["hidden_size"], 4096);
}

// ========================================================================
// infer_model_config: num_attention_heads fallback for various hidden sizes
// ========================================================================

#[test]
fn test_infer_model_config_heads_fallback_hidden_896() {
    // Qwen2.5-0.5B: hidden=896 -> 14 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 896]),
    );
    // No q_proj -> fallback path for num_attention_heads
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 14);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_1536() {
    // Qwen2.5-1.5B: hidden=1536 -> 12 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 1536]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 12);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_2048() {
    // Llama-style: hidden=2048 -> 16 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 16);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_4096() {
    // Llama-7B: hidden=4096 -> 32 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 32);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_5120() {
    // Llama-13B: hidden=5120 -> 40 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 5120]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 40);
}

#[test]
fn test_infer_model_config_heads_fallback_hidden_8192() {
    // Llama-70B: hidden=8192 -> 64 heads
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 8192]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 64);
}

#[test]
fn test_infer_model_config_heads_fallback_unknown_hidden_size() {
    // Non-standard hidden size: 3072 -> default formula: (3072 / 128).max(1) = 24
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 3072]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // 3072 / 128 = 24
    assert_eq!(v["num_attention_heads"], 24);
}

// ========================================================================
// infer_model_config: num_attention_heads with q_proj present (non-fallback)
// ========================================================================

#[test]
fn test_infer_model_config_heads_from_q_proj_small_hidden() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // hidden < 4096 -> head_dim = 64
    // hidden = 2048, head_dim = 64, num_heads = 2048/64 = 32
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![2048, 2048]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // hidden=2048 < 4096 -> head_dim=64, num_heads = 2048/64 = 32
    assert_eq!(v["num_attention_heads"], 32);
}

#[test]
fn test_infer_model_config_heads_from_q_proj_large_hidden() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // hidden >= 4096 -> head_dim = 128
    // hidden = 4096, head_dim = 128, num_heads = 4096/128 = 32
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_attention_heads"], 32);
}

// ========================================================================
// infer_model_config: intermediate_size default when no MLP tensors
// ========================================================================

#[test]
fn test_infer_model_config_intermediate_size_default() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    // No gate_proj/up_proj/feed_forward.w1 -> default to hidden_size * 4
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // hidden=2048, default intermediate = 2048*4 = 8192
    assert_eq!(v["intermediate_size"], 8192);
}

// ========================================================================
// infer_model_config: intermediate_size from up_proj
// ========================================================================

#[test]
fn test_infer_model_config_intermediate_size_from_up_proj() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    // up_proj with specific intermediate size
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        (vec![], vec![14336, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["intermediate_size"], 14336);
}

// ========================================================================
// infer_model_config: intermediate_size from feed_forward.w1
// ========================================================================

#[test]
fn test_infer_model_config_intermediate_size_from_feed_forward_w1() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    tensors.insert(
        "model.layers.0.feed_forward.w1.weight".to_string(),
        (vec![], vec![11008, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["intermediate_size"], 11008);
}

// ========================================================================
// infer_model_config: num_key_value_heads defaults to num_attention_heads (MHA)
// ========================================================================

#[test]
fn test_infer_model_config_kv_heads_default_mha() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 4096]),
    );
    // q_proj present but NO k_proj -> num_kv_heads defaults to num_attention_heads
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![4096, 4096]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    assert_eq!(v["num_key_value_heads"], v["num_attention_heads"]);
}

// ========================================================================
// infer_model_config: Output is always valid JSON
// ========================================================================

#[test]
fn test_infer_model_config_output_is_valid_json() {
    // Test with various tensor configurations to ensure JSON is always valid
    let configs: Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>> = vec![
        BTreeMap::new(), // empty
        {
            let mut m = BTreeMap::new();
            m.insert(
                "model.embed_tokens.weight".to_string(),
                (vec![], vec![100, 64]),
            );
            m
        },
        {
            let mut m = BTreeMap::new();
            m.insert("token_embd.weight".to_string(), (vec![], vec![50000, 768]));
            m.insert("blk.0.attn_q.weight".to_string(), (vec![], vec![768, 768]));
            m.insert("blk.5.attn_q.weight".to_string(), (vec![], vec![768, 768]));
            m
        },
    ];

    for tensors in &configs {
        let config = infer_model_config(tensors);
        let result: std::result::Result<serde_json::Value, _> = serde_json::from_str(&config);
        assert!(
            result.is_ok(),
            "infer_model_config produced invalid JSON for {} tensors: {config}",
            tensors.len()
        );
    }
}

// ========================================================================
// infer_model_config: Config contains all required HuggingFace fields
// ========================================================================

#[test]
fn test_infer_model_config_contains_all_required_fields() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // All fields required by HuggingFace SafeTensors inference (GH-193)
    let required_fields = [
        "architectures",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "sliding_window",
        "tie_word_embeddings",
        "torch_dtype",
        "use_cache",
        "use_sliding_window",
        "vocab_size",
    ];

    for field in &required_fields {
        assert!(
            !v[field].is_null(),
            "Required field '{field}' is missing from config JSON"
        );
    }
}

// ========================================================================
// infer_model_config: head_dim calculation
// ========================================================================

#[test]
fn test_infer_model_config_head_dim_affects_kv_heads() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // hidden=8192, 64 heads -> head_dim = 128
    // k_proj shape [2048, 8192] -> kv_heads = 2048/128 = 16
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 8192]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![], vec![8192, 8192]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![], vec![2048, 8192]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // hidden=8192 >= 4096 -> head_dim=128
    // num_heads = 8192/128 = 64
    assert_eq!(v["num_attention_heads"], 64);
    // kv_heads = 2048/128 = 16
    assert_eq!(v["num_key_value_heads"], 16);
}

// ========================================================================
// infer_model_config: Embedding shape with GGUF transposed layout
// ========================================================================

#[test]
fn test_infer_model_config_transposed_embedding_gguf() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // GGUF sometimes stores embeddings as [hidden_size, vocab_size]
    // The function picks the smaller dim as hidden_size
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![], vec![4096, 151936]),
    );

    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));

    // min(4096, 151936) = 4096 -> hidden
    assert_eq!(v["hidden_size"], 4096);
    // max(4096, 151936) = 151936 -> vocab
    assert_eq!(v["vocab_size"], 151936);
}

// ========================================================================
// infer_model_config: Alternative q_proj name patterns
// ========================================================================

#[test]
fn test_infer_model_config_attn_q_proj_pattern() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![], vec![32000, 2048]),
    );
    // "attn.q_proj" pattern (without "self_attn" prefix)
    tensors.insert(
        "model.layers.0.attn.q_proj.weight".to_string(),
        (vec![], vec![2048, 2048]),
    );
    let config = infer_model_config(&tensors);
    let v: serde_json::Value = serde_json::from_str(&config)
        .unwrap_or_else(|e| panic!("Config is not valid JSON: {e}\n{config}"));
    // Should detect q_proj via "attn.q_proj" pattern
    // hidden=2048 < 4096 -> head_dim=64, num_heads=32
    assert_eq!(v["num_attention_heads"], 32);
}
