
// ============================================================================
// from_apr_bytes() with embedded tensors (F32 dtype)
// ============================================================================

/// Build a minimal valid APR v2 binary with embedding + lm_head + layer tensors
fn build_minimal_apr_bytes() -> Vec<u8> {
    let hidden_dim: usize = 4;
    let vocab_size: usize = 4;
    let num_layers: usize = 1;
    let num_heads: usize = 2;
    let num_kv_heads: usize = 2;
    let intermediate_dim: usize = 8;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let _qkv_out_dim = hidden_dim + 2 * kv_dim;

    // Metadata JSON
    let metadata = format!(
        r#"{{"hidden_size":{hidden_dim},"num_hidden_layers":{num_layers},"num_attention_heads":{num_heads},"num_key_value_heads":{num_kv_heads},"vocab_size":{vocab_size},"intermediate_size":{intermediate_dim},"rope_theta":10000.0,"rms_norm_eps":0.00001,"context_length":64}}"#
    );
    let metadata_bytes = metadata.as_bytes();

    // Define tensors we need
    struct TensorDef {
        name: String,
        dims: Vec<usize>,
        dtype: u8,
    }

    let tensor_defs = vec![
        TensorDef {
            name: "model.embed_tokens.weight".to_string(),
            dims: vec![vocab_size, hidden_dim],
            dtype: 0, // F32
        },
        TensorDef {
            name: "lm_head.weight".to_string(),
            dims: vec![vocab_size, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.norm.weight".to_string(),
            dims: vec![hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            dims: vec![hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dims: vec![hidden_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.k_proj.weight".to_string(),
            dims: vec![kv_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.v_proj.weight".to_string(),
            dims: vec![kv_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.self_attn.o_proj.weight".to_string(),
            dims: vec![hidden_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.mlp.up_proj.weight".to_string(),
            dims: vec![intermediate_dim, hidden_dim],
            dtype: 0,
        },
        TensorDef {
            name: "model.layers.0.mlp.down_proj.weight".to_string(),
            dims: vec![hidden_dim, intermediate_dim],
            dtype: 0,
        },
    ];

    // Calculate tensor sizes
    let mut tensor_data_parts: Vec<Vec<u8>> = Vec::new();
    let mut running_offset: usize = 0;

    struct TensorEntry {
        name: String,
        dims: Vec<usize>,
        dtype: u8,
        offset: usize,
        size: usize,
    }

    let mut entries = Vec::new();
    for def in &tensor_defs {
        let num_elems: usize = def.dims.iter().product();
        let byte_size = num_elems * 4; // F32 = 4 bytes
        let data: Vec<u8> = (0..num_elems)
            .flat_map(|i| {
                let val = ((i % 7) as f32 - 3.0) * 0.01;
                val.to_le_bytes().to_vec()
            })
            .collect();
        entries.push(TensorEntry {
            name: def.name.clone(),
            dims: def.dims.clone(),
            dtype: def.dtype,
            offset: running_offset,
            size: byte_size,
        });
        tensor_data_parts.push(data);
        running_offset += byte_size;
    }

    // Build tensor index
    let mut tensor_index = Vec::new();
    for entry in &entries {
        let name_bytes = entry.name.as_bytes();
        tensor_index.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        tensor_index.extend_from_slice(name_bytes);
        tensor_index.push(entry.dtype);
        tensor_index.push(entry.dims.len() as u8);
        for &dim in &entry.dims {
            tensor_index.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        tensor_index.extend_from_slice(&(entry.offset as u64).to_le_bytes());
        tensor_index.extend_from_slice(&(entry.size as u64).to_le_bytes());
    }

    // Layout: Header (64) | Metadata | Tensor Index | Tensor Data
    let metadata_offset: usize = 64;
    let metadata_size = metadata_bytes.len();
    let tensor_index_offset = metadata_offset + metadata_size;
    let data_offset = tensor_index_offset + tensor_index.len();

    let mut bytes = vec![0u8; 64]; // header
    bytes[0..4].copy_from_slice(b"APR\0");
    // tensor_count
    bytes[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    // metadata_offset
    bytes[12..20].copy_from_slice(&(metadata_offset as u64).to_le_bytes());
    // metadata_size
    bytes[20..24].copy_from_slice(&(metadata_size as u32).to_le_bytes());
    // tensor_index_offset
    bytes[24..32].copy_from_slice(&(tensor_index_offset as u64).to_le_bytes());
    // data_offset
    bytes[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());

    // Metadata
    bytes.extend_from_slice(metadata_bytes);
    // Tensor index
    bytes.extend_from_slice(&tensor_index);
    // Tensor data
    for part in &tensor_data_parts {
        bytes.extend_from_slice(part);
    }

    bytes
}

#[test]
fn test_from_apr_bytes_valid_minimal() {
    let data = build_minimal_apr_bytes();
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(
        result.is_ok(),
        "Should parse valid APR bytes: {:?}",
        result.err()
    );

    let model = result.expect("parse should succeed");
    assert_eq!(model.config.hidden_dim, 4);
    assert_eq!(model.config.num_layers, 1);
    assert_eq!(model.config.num_heads, 2);
    assert_eq!(model.config.num_kv_heads, 2);
    assert_eq!(model.config.vocab_size, 4);
    assert_eq!(model.config.intermediate_dim, 8);
    assert_eq!(model.token_embedding.len(), 4 * 4); // vocab_size * hidden_dim
    assert_eq!(model.layers.len(), 1);
    assert_eq!(model.lm_head_weight.len(), 4 * 4); // vocab_size * hidden_dim
}

#[test]
fn test_from_apr_bytes_then_forward() {
    let data = build_minimal_apr_bytes();
    let model = AprTransformer::from_apr_bytes(&data).expect("parse should succeed");
    let logits = model.forward(&[0]).expect("forward should succeed");
    assert_eq!(logits.len(), 4); // vocab_size=4
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Multi-layer model tests (more coverage of layer iteration)
// ============================================================================

#[test]
fn test_forward_two_layer_model() {
    let hidden_dim = 8;
    let num_heads = 2;
    let num_kv_heads = 2;
    let vocab_size = 8;
    let intermediate_dim = 16;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    let config = AprTransformerConfig {
        architecture: "test-2layer".to_string(),
        hidden_dim,
        num_layers: 2,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let make_layer = || AprTransformerLayer {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; qkv_out_dim * hidden_dim],
        qkv_bias: None,
        attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.01; intermediate_dim * hidden_dim]),
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    let mut token_embedding = vec![0.0f32; vocab_size * hidden_dim];
    for tok in 0..vocab_size {
        for d in 0..hidden_dim {
            token_embedding[tok * hidden_dim + d] = ((tok * hidden_dim + d) as f32) * 0.001;
        }
    }

    let model = AprTransformer {
        config,
        token_embedding,
        layers: vec![make_layer(), make_layer()],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let logits = model.forward(&[0, 1]).expect("forward should succeed");
    assert_eq!(logits.len(), vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// forward_with_cache with GELU path and subsequent tokens
// ============================================================================

#[test]
fn test_forward_with_cache_three_tokens_sequential() {
    let model = make_pygmy_model();
    let mut cache = AprKVCache::new(&model.config);

    for pos in 0..3 {
        let logits = model
            .forward_with_cache(pos as u32, &mut cache, pos)
            .expect("forward_with_cache should succeed");
        assert_eq!(logits.len(), 16);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "Logits at pos={pos} should all be finite"
        );
    }
    assert_eq!(cache.len(), 3);
}

#[test]
fn test_forward_with_cache_gelu_multiple_positions() {
    let model = make_pygmy_model_gelu();
    let mut cache = AprKVCache::new(&model.config);

    let logits0 = model
        .forward_with_cache(0, &mut cache, 0)
        .expect("forward_with_cache pos=0 should succeed");
    let logits1 = model
        .forward_with_cache(1, &mut cache, 1)
        .expect("forward_with_cache pos=1 should succeed");

    assert_eq!(logits0.len(), 16);
    assert_eq!(logits1.len(), 16);
    // Different inputs should typically produce different outputs
    // (though not guaranteed with zero-ish weights)
}

// ============================================================================
// forward_with_cache without FFN norm (exercises else branch)
// ============================================================================

#[test]
fn test_forward_with_cache_no_ffn_norm() {
    let model = make_pygmy_model_gelu(); // This has ffn_norm_weight = None
    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(5, &mut cache, 0)
        .expect("should succeed without ffn_norm");
    assert_eq!(logits.len(), 16);
}

// ============================================================================
// Edge cases: forward parity between forward() and forward_traced()
// ============================================================================

#[test]
fn test_forward_and_forward_traced_logits_match() {
    let model = make_pygmy_model();
    let logits_forward = model.forward(&[1]).expect("forward should succeed");
    let trace = model
        .forward_traced(&[1])
        .expect("forward_traced should succeed");

    assert_eq!(logits_forward.len(), trace.logits.len());
    for (i, (a, b)) in logits_forward.iter().zip(trace.logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Logit mismatch at index {i}: forward={a}, traced={b}"
        );
    }
}

#[test]
fn test_forward_and_forward_traced_gelu_logits_match() {
    let model = make_pygmy_model_gelu();
    let logits_forward = model.forward(&[0]).expect("forward should succeed");
    let trace = model
        .forward_traced(&[0])
        .expect("forward_traced should succeed");

    assert_eq!(logits_forward.len(), trace.logits.len());
    for (i, (a, b)) in logits_forward.iter().zip(trace.logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Logit mismatch at index {i}: forward={a}, traced={b}"
        );
    }
}
