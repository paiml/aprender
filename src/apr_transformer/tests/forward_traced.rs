
#[test]
fn test_forward_traced_swiglu_multi_token() {
    let model = make_pygmy_model();
    let trace = model
        .forward_traced(&[0, 1])
        .expect("forward_traced should succeed");
    assert_eq!(trace.input_tokens.len(), 2);
    assert_eq!(trace.logits.len(), 16);
    // embed_stats covers seq_len*hidden_dim = 2*8 = 16 elements
    assert_eq!(trace.embed_stats.count, 16);
}

#[test]
fn test_forward_traced_empty_tokens_error() {
    let model = make_pygmy_model();
    let result = model.forward_traced(&[]);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer::forward_traced() - GELU path
// ============================================================================

#[test]
fn test_forward_traced_gelu_returns_trace() {
    let model = make_pygmy_model_gelu();
    let trace = model
        .forward_traced(&[1])
        .expect("forward_traced should succeed");
    assert_eq!(trace.logits.len(), 16);
    assert_eq!(trace.layer_activations.len(), 1);
    // All stats should be populated
    assert!(trace.embed_stats.count > 0);
    assert!(trace.final_norm_stats.count > 0);
    assert!(trace.logits_stats.count > 0);
}

// ============================================================================
// TracedForward trait impl
// ============================================================================

#[test]
fn test_traced_forward_trait_impl() {
    let mut model = make_pygmy_model();
    // Call through the trait
    let trace =
        TracedForward::forward_traced(&mut model, &[1]).expect("TracedForward should succeed");
    assert_eq!(trace.logits.len(), 16);
}

// ============================================================================
// AprTransformer::predict_next()
// ============================================================================

#[test]
fn test_predict_next_returns_token() {
    let model = make_pygmy_model();
    let next = model
        .predict_next(&[1])
        .expect("predict_next should succeed");
    assert!(next < 16, "Predicted token should be within vocab_size");
}

#[test]
fn test_predict_next_deterministic() {
    let model = make_pygmy_model();
    let next1 = model
        .predict_next(&[1])
        .expect("predict_next should succeed");
    let next2 = model
        .predict_next(&[1])
        .expect("predict_next should succeed");
    assert_eq!(
        next1, next2,
        "predict_next should be deterministic for same input"
    );
}

#[test]
fn test_predict_next_empty_error() {
    let model = make_pygmy_model();
    let result = model.predict_next(&[]);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer::generate()
// ============================================================================

#[test]
fn test_generate_produces_tokens() {
    let model = make_pygmy_model();
    let output = model.generate(&[1], 3).expect("generate should succeed");
    // Output includes prompt + generated tokens
    assert!(output.len() >= 2, "Should generate at least one token");
    assert!(
        output.len() <= 4,
        "Should generate at most 3 tokens + prompt"
    );
    assert_eq!(output[0], 1, "First token should be prompt");
}

#[test]
fn test_generate_stops_at_eos() {
    // Create model where token 2 (EOS) is always the highest logit
    let hidden_dim = 8;
    let vocab_size = 16;
    let config = AprTransformerConfig {
        architecture: "test-eos".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: Some(2),
    ..Default::default()
    };
    let mut model = AprTransformer::new(config);

    // Bias lm_head so token 2 always wins
    // Set lm_head_bias so index 2 has a very large value
    let mut bias = vec![0.0f32; vocab_size];
    bias[2] = 100.0; // EOS token = 2
    model.lm_head_bias = Some(bias);

    let output = model.generate(&[1], 10).expect("generate should succeed");
    // Should stop early due to EOS
    assert!(
        output.len() <= 3,
        "Should stop after generating EOS token 2. Got len={}",
        output.len()
    );
    // Last generated token should be EOS (2)
    assert_eq!(*output.last().expect("output should not be empty"), 2);
}

#[test]
fn test_generate_max_tokens_limit() {
    let model = make_pygmy_model();
    let output = model.generate(&[1], 5).expect("generate should succeed");
    // Output length = prompt(1) + generated(up to 5)
    assert!(output.len() <= 6);
}

// ============================================================================
// AprTransformer::forward_with_cache()
// ============================================================================

#[test]
fn test_forward_with_cache_first_token() {
    let model = make_pygmy_model();
    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(1, &mut cache, 0)
        .expect("forward_with_cache should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_with_cache_multiple_positions() {
    let model = make_pygmy_model();
    let mut cache = AprKVCache::new(&model.config);

    // Process first token
    let logits0 = model
        .forward_with_cache(1, &mut cache, 0)
        .expect("forward_with_cache pos=0 should succeed");
    assert_eq!(logits0.len(), 16);

    // Process second token (uses cached KV from first)
    let logits1 = model
        .forward_with_cache(2, &mut cache, 1)
        .expect("forward_with_cache pos=1 should succeed");
    assert_eq!(logits1.len(), 16);

    // Cache should have 2 positions
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_forward_with_cache_gelu_path() {
    let model = make_pygmy_model_gelu();
    let mut cache = AprKVCache::new(&model.config);
    let logits = model
        .forward_with_cache(1, &mut cache, 0)
        .expect("forward_with_cache gelu should succeed");
    assert_eq!(logits.len(), 16);
    assert!(logits.iter().all(|v| v.is_finite()));
}

// ============================================================================
// AprTransformer::generate_with_cache()
// ============================================================================

#[test]
fn test_generate_with_cache_delegation() {
    let model = make_pygmy_model();
    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let output = model
        .generate_with_cache(&[1], &gen_config)
        .expect("generate_with_cache should succeed");
    assert!(output.len() >= 2);
    assert_eq!(output[0], 1);
}

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let model = make_pygmy_model();
    let gen_config = GenerateConfig::default();
    let result = model.generate_with_cache(&[], &gen_config);
    assert!(result.is_err());
}

// ============================================================================
// AprTransformer serialization (Serialize/Deserialize)
// ============================================================================

#[test]
fn test_apr_transformer_serde_roundtrip() {
    let model = make_pygmy_model();
    let json = serde_json::to_vec(&model).expect("serialize should succeed");
    let deserialized: AprTransformer =
        serde_json::from_slice(&json).expect("deserialize should succeed");

    assert_eq!(deserialized.config.hidden_dim, model.config.hidden_dim);
    assert_eq!(deserialized.config.num_layers, model.config.num_layers);
    assert_eq!(deserialized.config.vocab_size, model.config.vocab_size);
    assert_eq!(
        deserialized.token_embedding.len(),
        model.token_embedding.len()
    );
    assert_eq!(deserialized.layers.len(), model.layers.len());
    assert_eq!(
        deserialized.lm_head_weight.len(),
        model.lm_head_weight.len()
    );
}

#[test]
fn test_apr_transformer_debug() {
    let model = make_pygmy_model();
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("AprTransformer"));
}

#[test]
fn test_apr_transformer_clone() {
    let model = make_pygmy_model();
    let cloned = model.clone();
    assert_eq!(cloned.config.hidden_dim, model.config.hidden_dim);
    assert_eq!(cloned.token_embedding.len(), model.token_embedding.len());
    assert_eq!(cloned.layers.len(), model.layers.len());
}

// ============================================================================
// from_apr_bytes() error cases
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let result = AprTransformer::from_apr_bytes(&[0; 32]);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("too small"),
        "Error should mention size: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_bad_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("magic") || err_msg.contains("Invalid"),
        "Error should mention magic: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_valid_magic_apr0() {
    // Test APR\0 magic (version 0) with minimal valid structure
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"APR\0");
    // tensor_count = 0
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    // metadata_offset = 64, metadata_size = 2 (valid JSON "{}")
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // tensor_index_offset = 66
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    // data_offset = 66
    data[32..40].copy_from_slice(&66u64.to_le_bytes());
    // Write metadata "{}" at offset 64
    data[64] = b'{';
    data[65] = b'}';

    // GH-478: Empty metadata defaults to zero dimensions, which the contract gate
    // now correctly rejects before weight loading. Previously fell through to
    // "missing embedding" error.
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("embedding") || err_msg.contains("FATAL")
            || err_msg.contains("contract_gate"),
        "Should fail with missing embedding or contract gate: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_metadata_beyond_file() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"APR\0");
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count=0
    // metadata_offset = 64, metadata_size = 9999 (beyond file)
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&9999u32.to_le_bytes());
    data[24..32].copy_from_slice(&64u64.to_le_bytes());
    data[32..40].copy_from_slice(&64u64.to_le_bytes());

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("beyond") || err_msg.contains("Metadata"),
        "Should fail with metadata beyond file: {err_msg}"
    );
}

#[test]
fn test_from_apr_bytes_valid_magic_apr2() {
    // Test APR2 magic (version 2) -- same error expected for no embedding
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(b"APR2");
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    data[32..40].copy_from_slice(&66u64.to_le_bytes());
    data[64] = b'{';
    data[65] = b'}';

    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err()); // No embedding tensor
}
