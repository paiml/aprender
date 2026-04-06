
// ============================================================================
// Q4_K perrow path (2D tensor with dims[1] % 256 != 0)
// ============================================================================

#[test]
fn test_from_apr_bytes_q4k_perrow_path() {
    // Use hidden=128 (not multiple of 256) so 2D Q4_K tensors hit perrow path
    let hidden = 128;
    let intermediate = 128;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q4_K,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q4_K perrow: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.layers.len(), 1);
    // 2D weights with dims[1]=128 hit perrow dequant path
    assert_eq!(
        apr.layers[0].qkv_weight.len(),
        hidden * (hidden + 2 * hidden)
    );
}

// ============================================================================
// Q6_K perrow path (2D tensor with dims[1] % 256 != 0)
// ============================================================================

#[test]
fn test_from_apr_bytes_q6k_perrow_path() {
    let hidden = 128;
    let intermediate = 128;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Override down_proj to Q6_K with 2D dims where cols % 256 != 0
    for t in &mut tensors {
        if t.name.contains("down_proj") {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q6_K perrow: {}", result.unwrap_err());
}

// ============================================================================
// Q5_K perrow path (2D tensor with dims[1] % 256 != 0)
// ============================================================================

#[test]
fn test_from_apr_bytes_q5k_perrow_path() {
    let hidden = 128;
    let intermediate = 128;
    let vocab = 4;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_Q5_K,
        DTYPE_F32,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q5_K perrow: {}", result.unwrap_err());
}

// ============================================================================
// Weight tying path (no lm_head.weight, uses embedding)
// ============================================================================

#[test]
fn test_from_apr_bytes_weight_tying_via_embed() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Remove lm_head.weight to trigger weight tying
    tensors.retain(|t| t.name != "lm_head.weight");
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "Weight tying should succeed: {}",
        result.unwrap_err()
    );

    let apr = result.expect("test value should be present");
    // lm_head should be same as embedding
    assert_eq!(apr.lm_head_weight.len(), vocab * hidden);
}

// ============================================================================
// GGUF naming convention (blk.X, token_embd, output_norm, output.weight)
// ============================================================================

#[test]
fn test_from_apr_bytes_gguf_naming() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);

    let tensors = vec![
        TensorDef {
            name: "token_embd.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.1),
        },
        TensorDef {
            name: "blk.0.attn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.attn_q.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.attn_k.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.attn_v.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.attn_output.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, hidden as u64],
            data: make_f32_data(hidden * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.ffn_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "blk.0.ffn_gate.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.ffn_up.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![intermediate as u64, hidden as u64],
            data: make_f32_data(intermediate * hidden, 0.01),
        },
        TensorDef {
            name: "blk.0.ffn_down.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64, intermediate as u64],
            data: make_f32_data(hidden * intermediate, 0.01),
        },
        TensorDef {
            name: "output_norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![hidden as u64],
            data: make_f32_data(hidden, 1.0),
        },
        TensorDef {
            name: "output.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![vocab as u64, hidden as u64],
            data: make_f32_data(vocab * hidden, 0.01),
        },
    ];
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "GGUF naming: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.layers.len(), 1);
    assert!(apr.layers[0].ffn_norm_weight.is_some());
    assert!(apr.layers[0].ffn_gate_weight.is_some());
}

// ============================================================================
// Metadata alias coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_metadata_aliases() {
    let hidden = 8;
    let intermediate = 32;
    let vocab = 16;
    // Use alternative metadata field names to cover .or_else() branches
    let meta = r#"{
        "model_type": "qwen2",
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 4,
        "vocab_size": 16,
        "intermediate_dim": 32,
        "rms_norm_eps": 1e-6,
        "max_seq_len": 1024
    }"#;
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    let data = build_apr_v2(meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Metadata aliases: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert_eq!(apr.config.architecture, "qwen2");
    assert_eq!(apr.config.hidden_dim, hidden);
    assert_eq!(apr.config.intermediate_dim, intermediate);
    assert_eq!(apr.config.context_length, 1024);
}

#[test]
fn test_from_apr_bytes_architecture_auto_filtered() {
    let meta = r#"{
        "architecture": "Auto",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 16,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-6
    }"#;
    let tensors = make_hf_tensors(8, 32, 4, 4, 16, DTYPE_F32, DTYPE_F32, DTYPE_F32);
    let data = build_apr_v2(meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok());
    // "Auto" should be filtered out and defaulted to "unknown"
    assert_eq!(result.expect("test value should be present").config.architecture, "unknown");
}

// ============================================================================
// Error paths
// ============================================================================

#[test]
fn test_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"NOPE");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Invalid APR magic"));
}

#[test]
fn test_from_apr_bytes_truncated_metadata() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    // Set metadata offset=64, size=9999 (beyond file)
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&9999u32.to_le_bytes());
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Metadata extends beyond file"));
}

#[test]
fn test_from_apr_bytes_no_embedding_tensor() {
    // Valid header + metadata but NO embedding tensor
    // GH-478: num_hidden_layers must be ≥1 to pass contract gate
    let meta = r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "intermediate_size": 8, "vocab_size": 4}"#;
    let tensors = vec![
        TensorDef {
            name: "model.norm.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![8],
            data: make_f32_data(8, 1.0),
        },
        TensorDef {
            name: "lm_head.weight".into(),
            dtype: DTYPE_F32,
            dims: vec![4, 8],
            data: make_f32_data(32, 0.01),
        },
    ];
    let data = build_apr_v2(meta, &tensors);
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("No embedding tensor found"));
}

#[test]
fn test_from_apr_bytes_no_lm_head_no_embed_for_tying() {
    // Has token_embd.weight but no lm_head.weight — exercises weight tying via token_embd
    // GH-478: num_hidden_layers must be ≥1 to pass contract gate
    let meta = r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1, "intermediate_size": 8, "vocab_size": 4}"#;
    let tensors = vec![TensorDef {
        name: "token_embd.weight".into(),
        dtype: DTYPE_F32,
        dims: vec![4, 8],
        data: make_f32_data(32, 0.1),
    }];
    let data = build_apr_v2(meta, &tensors);
    let result = AprTransformer::from_apr_bytes(&data);
    // token_embd.weight should be found as embedding AND as tied lm_head
    assert!(result.is_ok(), "token_embd tying: {}", result.unwrap_err());
}

// ============================================================================
// Q4K lm_head paths (lm_head_weight_q4k / lm_head_weight_q6k)
// ============================================================================

#[test]
fn test_from_apr_bytes_q4k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256; // multiple of 256 for flat Q4K
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_Q4_K,
    );
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q4K lm_head: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert!(apr.lm_head_weight_q4k.is_some());
}

#[test]
fn test_from_apr_bytes_q6k_lm_head() {
    let hidden = 256;
    let intermediate = 256;
    let vocab = 256;
    let meta = minimal_metadata(hidden, 1, 4, 4, vocab, intermediate);
    let mut tensors = make_hf_tensors(
        hidden,
        intermediate,
        4,
        4,
        vocab,
        DTYPE_F32,
        DTYPE_F32,
        DTYPE_F32,
    );
    // Set lm_head to Q6_K
    for t in &mut tensors {
        if t.name == "lm_head.weight" {
            let num_elements: usize = t.dims.iter().map(|d| *d as usize).product();
            t.dtype = DTYPE_Q6_K;
            t.data = make_q6k_data(num_elements);
        }
    }
    let data = build_apr_v2(&meta, &tensors);

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_ok(), "Q6K lm_head: {}", result.unwrap_err());

    let apr = result.expect("test value should be present");
    assert!(apr.lm_head_weight_q6k.is_some());
}
