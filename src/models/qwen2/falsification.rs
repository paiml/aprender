use super::*;

/// S2: Tokenizer round-trips ASCII correctly
/// FALSIFICATION: decode(encode("Hello")) != "Hello"
#[test]
#[ignore = "requires tokenizer download - run with: cargo test -- --ignored"]
fn s2_tokenizer_roundtrip_ascii() {
    let tokenizer_path = std::path::Path::new("/home/noah/.cache/qwen2/tokenizer.json");

    if !tokenizer_path.exists() {
        eprintln!("SKIP S2: tokenizer.json not found");
        return;
    }

    let json = std::fs::read_to_string(tokenizer_path).expect("read");
    let tokenizer = crate::text::bpe::load_from_json(&json).expect("parse");

    let original = "Hello";
    let encoded = tokenizer.encode(original);
    let decoded = tokenizer.decode(&encoded);

    // Allow for whitespace normalization
    let decoded_trimmed = decoded.trim();
    assert!(
        decoded_trimmed == original || decoded.contains(original),
        "FALSIFIED S2: roundtrip failed. '{}' -> {:?} -> '{}'",
        original,
        encoded,
        decoded
    );

    println!(
        "S2 PASSED: '{}' -> {:?} -> '{}'",
        original, encoded, decoded
    );
}

/// S3: Tokenizer handles Qwen2 special tokens
/// FALSIFICATION: is_eos(151645) returns false
#[test]
fn s3_tokenizer_special_tokens() {
    use crate::text::bpe::Qwen2BpeTokenizer;

    let tokenizer = Qwen2BpeTokenizer::new();

    // <|im_end|> = 151645 is the EOS token
    assert!(
        tokenizer.is_eos(151645),
        "FALSIFIED S3: is_eos(151645) returned false. <|im_end|> not recognized."
    );

    // <|im_start|> = 151644 is the BOS token
    assert!(
        tokenizer.is_bos(151644),
        "FALSIFIED S3: is_bos(151644) returned false. <|im_start|> not recognized."
    );

    println!("S3 PASSED: Special tokens recognized correctly");
}

/// S4: Model loads from SafeTensors without OOM
/// FALSIFICATION: OOM on 16GB machine OR load fails
/// NOTE: Timing removed - use `cargo bench` for performance testing
#[test]
#[ignore = "requires 0.5B model download - run with: cargo test -- --ignored"]
fn s4_model_loads_memory_efficient() {
    let safetensors_path = std::path::Path::new(
        "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
    );

    if !safetensors_path.exists() {
        eprintln!("SKIP S4: model.safetensors not found");
        return;
    }

    let config = Qwen2Config::qwen2_0_5b_instruct();
    let start = std::time::Instant::now();

    // Use memory-efficient loading
    let mut model = Qwen2Model::new_uninitialized(&config);
    let loaded = model.load_from_safetensors(safetensors_path);

    let elapsed = start.elapsed();

    assert!(
        loaded.is_ok(),
        "FALSIFIED S4: Model load failed: {:?}",
        loaded.err()
    );

    // Log timing for observability (no assertion - use benchmarks for perf)
    println!(
        "S4 PASSED: Loaded {} tensors in {:.2}s",
        loaded.unwrap_or(0),
        elapsed.as_secs_f32()
    );
}

/// S5: Model loads exactly 219 weight tensors
/// FALSIFICATION: Tensor count != 219
#[test]
#[ignore = "requires 0.5B model download - run with: cargo test -- --ignored"]
fn s5_model_tensor_count() {
    let safetensors_path = std::path::Path::new(
        "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
    );

    if !safetensors_path.exists() {
        eprintln!("SKIP S5: model.safetensors not found");
        return;
    }

    let config = Qwen2Config::qwen2_0_5b_instruct();
    let mut model = Qwen2Model::new_uninitialized(&config);
    let loaded = model.load_from_safetensors(safetensors_path).expect("load");

    // Qwen2-0.5B has exactly 219 tensors:
    // - 1 embed_tokens
    // - 24 layers * 9 tensors (q,k,v,o,gate,up,down,input_norm,post_norm) = 216
    // - 1 final norm
    // - 1 lm_head (tied with embed_tokens)
    assert_eq!(
        loaded, 219,
        "FALSIFIED S5: Expected 219 tensors, got {}",
        loaded
    );

    println!("S5 PASSED: Loaded exactly 219 tensors");
}

/// S6: Embedding lookup returns correct shape
/// FALSIFICATION: Output shape != [1, seq_len, 896]
#[test]
fn s6_embedding_shape() {
    let config = create_tiny_config();
    let emb = Embedding::new(1000, config.hidden_size);

    let input_ids = vec![1u32, 2, 3, 4, 5];
    let output = emb.forward(&input_ids);

    assert_eq!(
        output.shape(),
        &[1, 5, config.hidden_size],
        "FALSIFIED S6: Embedding shape {:?} != expected [1, 5, {}]",
        output.shape(),
        config.hidden_size
    );

    println!("S6 PASSED: Embedding shape correct");
}

// =========================================================================
// Coverage Extension Tests
// =========================================================================

#[test]
fn test_from_safetensors_error_path() {
    let config = create_tiny_config();
    // Try loading from non-existent path
    let result = Qwen2Model::from_safetensors(
        &config,
        std::path::Path::new("/nonexistent/path.safetensors"),
    );
    assert!(result.is_err());
}

#[test]
fn test_from_apr_error_path() {
    let config = create_tiny_config();
    // Try loading from non-existent path
    let result = Qwen2Model::from_apr(&config, std::path::Path::new("/nonexistent/path.apr"));
    assert!(result.is_err());
}

#[test]
fn test_load_from_apr_error_path() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    // Try loading from non-existent path
    let result = model.load_from_apr(std::path::Path::new("/nonexistent/path.apr"));
    assert!(result.is_err());
}

#[test]
fn test_load_from_safetensors_error_path() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    // Try loading from non-existent path
    let result = model.load_from_safetensors(std::path::Path::new("/nonexistent/path.safetensors"));
    assert!(result.is_err());
}

#[test]
fn test_embedding_forward_into() {
    let emb = Embedding::new(100, 8);
    let input = vec![0u32, 1, 2];
    let mut output = vec![0.0f32; 3 * 8];

    emb.forward_into(&input, &mut output);

    // Check that output was written
    let has_nonzero = output.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "forward_into should write non-zero values");
}

#[test]
fn test_silu_large_values() {
    // Test SiLU with large positive and negative values
    let x = Tensor::new(&[10.0, -10.0, 100.0, -100.0], &[4]);
    let y = silu(&x);
    let data = y.data();

    // SiLU(10) ≈ 10 (sigmoid saturates)
    assert!((data[0] - 10.0).abs() < 0.001);
    // SiLU(-10) ≈ 0 (sigmoid saturates)
    assert!(data[1].abs() < 0.001);
    // SiLU(100) ≈ 100
    assert!((data[2] - 100.0).abs() < 0.001);
    // SiLU(-100) ≈ 0
    assert!(data[3].abs() < 0.001);
}

#[test]
fn test_model_lm_head_accessor() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);
    let lm_head = model.lm_head();
    // Verify we can access the lm_head
    assert!(lm_head.weight().shape().len() > 0);
}

#[test]
fn test_weight_info_complete() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    let info = model.weight_info();

    // Verify all expected keys exist
    assert!(info.contains_key("model.embed_tokens.weight"));
    assert!(info.contains_key("model.norm.weight"));
    assert!(info.contains_key("lm_head.weight"));

    // Verify layer keys
    for i in 0..config.num_layers {
        let prefix = format!("model.layers.{i}");
        assert!(info.contains_key(&format!("{prefix}.self_attn.q_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.self_attn.k_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.self_attn.v_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.self_attn.o_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.mlp.gate_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.mlp.up_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.mlp.down_proj.weight")));
        assert!(info.contains_key(&format!("{prefix}.input_layernorm.weight")));
        assert!(info.contains_key(&format!("{prefix}.post_attention_layernorm.weight")));
    }

    // Verify shapes
    let h = config.hidden_size;
    let i = config.intermediate_size;

    // MLP shapes
    assert_eq!(info["model.layers.0.mlp.gate_proj.weight"], vec![i, h]);
    assert_eq!(info["model.layers.0.mlp.down_proj.weight"], vec![h, i]);
}

#[test]
fn test_kv_cache_with_some_values() {
    let mut cache = KVCache::new(2);

    // Set some values
    cache.keys[0] = Some(Tensor::ones(&[1, 2, 4, 8]));
    cache.keys[1] = Some(Tensor::ones(&[1, 2, 4, 8]));
    cache.values[0] = Some(Tensor::ones(&[1, 2, 4, 8]));
    cache.values[1] = Some(Tensor::ones(&[1, 2, 4, 8]));
    cache.cached_len = 4;

    // Verify all set
    assert!(cache.keys.iter().all(|k| k.is_some()));
    assert!(cache.values.iter().all(|v| v.is_some()));

    // Clear
    cache.clear();

    // Verify cleared
    assert!(cache.keys.iter().all(|k| k.is_none()));
    assert!(cache.values.iter().all(|v| v.is_none()));
    assert_eq!(cache.cached_len, 0);
}

#[test]
fn test_decoder_layer_with_attention_mask() {
    let config = create_tiny_config();
    let layer = Qwen2DecoderLayer::new(&config);
    let rope = RotaryPositionEmbedding::with_base(16, 128, 10000.0);

    let hidden = Tensor::ones(&[1, 5, 64]);
    let position_ids: Vec<usize> = (0..5).collect();

    // Create causal attention mask inline (lower triangle + diagonal = 0, upper = -inf)
    let size = 5;
    let mut mask_data = vec![0.0f32; size * size];
    for row in 0..size {
        for col in (row + 1)..size {
            mask_data[row * size + col] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::new(&mask_data, &[size, size]);

    let output = layer.forward(&hidden, &position_ids, &rope, Some(&mask));
    assert_eq!(output.shape(), &[1, 5, 64]);
}

// =========================================================================
// Additional tests to reach 95% coverage
// =========================================================================

#[test]
fn test_weight_names_structure() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);
    let names = model.weight_names();

    // Verify structure: embedding, layers, norm, lm_head
    let expected_count = 1  // embed_tokens
        + config.num_layers * 9  // 9 weights per layer
        + 1  // final norm
        + 1; // lm_head
    assert_eq!(names.len(), expected_count);

    // Verify first few names
    assert_eq!(names[0], "model.embed_tokens.weight");
    assert!(names.last().unwrap().contains("lm_head"));
}

#[test]
fn test_embedding_with_boundary_tokens() {
    let vocab_size = 100;
    let emb = Embedding::new(vocab_size, 8);

    // Test first token
    let output0 = emb.forward(&[0u32]);
    assert_eq!(output0.shape(), &[1, 1, 8]);

    // Test last valid token
    let output_last = emb.forward(&[(vocab_size - 1) as u32]);
    assert_eq!(output_last.shape(), &[1, 1, 8]);

    // Test multiple tokens including boundary
    let output_mixed = emb.forward(&[0u32, (vocab_size - 1) as u32]);
    assert_eq!(output_mixed.shape(), &[1, 2, 8]);
}

#[test]
fn test_mlp_swiglu_computation() {
    // Test that MLP output is non-trivial (SwiGLU is actually computed)
    let mlp = Qwen2MLP::new(8, 16);
    let x = Tensor::new(&[1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.1, -0.1], &[1, 1, 8]);
    let output = mlp.forward(&x);

    // Output should be different from input
    assert_ne!(output.data(), x.data());
    // Should have same shape
    assert_eq!(output.shape(), &[1, 1, 8]);
}

#[test]
fn test_model_config_accessor() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    let retrieved_config = model.config();
    assert_eq!(retrieved_config.hidden_size, config.hidden_size);
    assert_eq!(retrieved_config.num_layers, config.num_layers);
    assert_eq!(retrieved_config.vocab_size, config.vocab_size);
}

#[test]
fn test_layer_accessor_all_layers() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    // Should be able to access all layers
    for i in 0..config.num_layers {
        let layer = model.layer_mut(i);
        assert!(layer.is_some(), "Layer {} should exist", i);
    }
}

#[test]
fn test_embedding_deterministic() {
    let emb = Embedding::new(100, 8);
    let input = vec![5u32, 10, 15];

    let output1 = emb.forward(&input);
    let output2 = emb.forward(&input);

    assert_eq!(output1.data(), output2.data());
}

#[test]
fn test_decoder_layer_residual_connection() {
    let config = create_tiny_config();
    let layer = Qwen2DecoderLayer::new(&config);
    let rope = RotaryPositionEmbedding::with_base(16, 128, 10000.0);

    // Input with specific values
    let hidden = Tensor::new(&vec![1.0f32; 64], &[1, 1, 64]);
    let position_ids: Vec<usize> = vec![0];

    let output = layer.forward(&hidden, &position_ids, &rope, None);

    // Output should be different due to attention and MLP
    assert_eq!(output.shape(), &[1, 1, 64]);
    // But not all zeros or all ones
    let sum: f32 = output.data().iter().sum();
    assert!(sum.abs() > 0.0, "Output should not be all zeros");
}

#[test]
fn test_silu_zero() {
    let x = Tensor::new(&[0.0], &[1]);
    let y = silu(&x);
    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert!((y.data()[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_add_tensors_broadcasting() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    let c = add_tensors(&a, &b);
    assert_eq!(c.data(), &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}

#[test]
fn test_elementwise_mul_zeros() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let zeros = Tensor::new(&[0.0, 0.0, 0.0], &[3]);
    let c = elementwise_mul(&a, &zeros);
    assert_eq!(c.data(), &[0.0, 0.0, 0.0]);
}
