//! Qwen2 Model Tests - Extreme TDD
//! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

fn create_tiny_config() -> Qwen2Config {
    Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 1000,
        max_seq_len: 128,
        intermediate_size: 128,
        rope_theta: 10000.0,
    }
}

#[test]
fn test_embedding_shape() {
    let emb = Embedding::new(1000, 64);
    let input = vec![1u32, 2, 3, 4, 5];
    let output = emb.forward(&input);

    assert_eq!(output.shape(), &[1, 5, 64]);
}

#[test]
fn test_embedding_lookup() {
    let emb = Embedding::new(100, 8);
    let input = vec![0u32, 1, 2];
    let output = emb.forward(&input);

    // Each token should produce different embeddings
    let data = output.data();
    let emb0 = &data[0..8];
    let emb1 = &data[8..16];
    let emb2 = &data[16..24];

    assert_ne!(emb0, emb1);
    assert_ne!(emb1, emb2);
}

#[test]
fn test_qwen2_mlp_shape() {
    let mlp = Qwen2MLP::new(64, 128);
    let x = Tensor::ones(&[1, 5, 64]);
    let output = mlp.forward(&x);

    assert_eq!(output.shape(), &[1, 5, 64]);
}

#[test]
fn test_qwen2_model_creation() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    assert_eq!(model.num_layers(), 2);
    assert_eq!(model.config().hidden_size, 64);
}

#[test]
fn test_silu_activation() {
    // SiLU(x) = x * sigmoid(x)
    // At x=0: SiLU(0) = 0 * 0.5 = 0
    let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    let y = silu(&x);

    let data = y.data();
    assert!((data[0] - 0.0).abs() < 1e-5); // SiLU(0) = 0
    assert!(data[1] > 0.5); // SiLU(1) ≈ 0.731
    assert!(data[2] < 0.0); // SiLU(-1) ≈ -0.269 (negative!)
}

// ========== Additional Coverage Tests ==========

#[test]
fn test_embedding_placeholder() {
    let emb = Embedding::placeholder(1000, 64);
    assert_eq!(emb.vocab_size, 1000);
    assert_eq!(emb.hidden_size, 64);
    // Placeholder has minimal weight
    assert_eq!(emb.weight.data().len(), 1);
}

#[test]
fn test_embedding_set_weight() {
    let mut emb = Embedding::placeholder(10, 4);
    let new_weight = Tensor::ones(&[10, 4]);
    emb.set_weight(new_weight);
    assert_eq!(emb.weight().data().len(), 40);
}

#[test]
fn test_embedding_weight_accessor() {
    let emb = Embedding::new(10, 4);
    let weight = emb.weight();
    assert_eq!(weight.shape(), &[10, 4]);
}

#[test]
fn test_embedding_out_of_vocab() {
    let emb = Embedding::new(10, 4);
    // Token 100 is out of vocabulary (vocab_size=10)
    let output = emb.forward(&[0, 100, 2]);
    // Should still produce output (OOV token gets zeros)
    assert_eq!(output.shape(), &[1, 3, 4]);
}

#[test]
fn test_qwen2_mlp_placeholder() {
    let _mlp = Qwen2MLP::placeholder(64, 128);
    // Placeholder MLPs should exist but have minimal weights
    // Note: Cannot do forward pass on placeholder (no weights set)
}

#[test]
fn test_qwen2_mlp_mut_accessors() {
    let mut mlp = Qwen2MLP::new(64, 128);
    let gate = mlp.gate_proj_mut();
    assert!(gate.weight().shape().len() > 0);
    let up = mlp.up_proj_mut();
    assert!(up.weight().shape().len() > 0);
    let down = mlp.down_proj_mut();
    assert!(down.weight().shape().len() > 0);
}

#[test]
fn test_qwen2_decoder_layer_placeholder() {
    let config = create_tiny_config();
    let _layer = Qwen2DecoderLayer::placeholder(&config);
    // Just verify placeholder can be created without panic
}

#[test]
fn test_qwen2_decoder_layer_mut_accessors() {
    let config = create_tiny_config();
    let mut layer = Qwen2DecoderLayer::new(&config);

    let _attn = layer.self_attn_mut();
    let _mlp = layer.mlp_mut();
    let _input_norm = layer.input_layernorm_mut();
    let _post_norm = layer.post_attention_layernorm_mut();
}

#[test]
fn test_kv_cache_new() {
    let cache = KVCache::new(4);
    assert_eq!(cache.keys.len(), 4);
    assert_eq!(cache.values.len(), 4);
    assert_eq!(cache.cached_len, 0);
    assert!(cache.keys.iter().all(|k| k.is_none()));
    assert!(cache.values.iter().all(|v| v.is_none()));
}

#[test]
fn test_kv_cache_clear() {
    let mut cache = KVCache::new(2);
    cache.cached_len = 10;
    cache.keys[0] = Some(Tensor::ones(&[1, 2, 3]));
    cache.values[0] = Some(Tensor::ones(&[1, 2, 3]));

    cache.clear();

    assert_eq!(cache.cached_len, 0);
    assert!(cache.keys.iter().all(|k| k.is_none()));
    assert!(cache.values.iter().all(|v| v.is_none()));
}

#[test]
fn test_qwen2_model_uninitialized() {
    let config = create_tiny_config();
    let model = Qwen2Model::new_uninitialized(&config);
    assert_eq!(model.num_layers(), 2);
    // Uninitialized model has placeholder weights
}

#[test]
fn test_qwen2_model_train_eval() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    model.train();
    assert!(model.training);

    model.eval();
    assert!(!model.training);
}

#[test]
fn test_qwen2_model_cache_operations() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    // Initially no cache
    assert!(model.kv_cache.is_none());

    // Enable cache
    model.enable_cache();
    assert!(model.kv_cache.is_some());

    // Clear cache (should not panic even if empty)
    model.clear_cache();

    // Disable cache
    model.disable_cache();
    assert!(model.kv_cache.is_none());

    // Clear on disabled cache should not panic
    model.clear_cache();
}

#[test]
fn test_qwen2_model_weight_names() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    let names = model.weight_names();
    assert!(names.contains(&"model.embed_tokens.weight".to_string()));
    assert!(names.contains(&"model.norm.weight".to_string()));
    assert!(names.contains(&"lm_head.weight".to_string()));
    // Should have layer-specific names
    assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
    assert!(names.contains(&"model.layers.1.mlp.gate_proj.weight".to_string()));
}

#[test]
fn test_qwen2_model_weight_info() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    let info = model.weight_info();
    assert!(info.contains_key("model.embed_tokens.weight"));
    assert_eq!(info["model.embed_tokens.weight"], vec![1000, 64]);
    assert!(info.contains_key("model.norm.weight"));
    assert_eq!(info["model.norm.weight"], vec![64]);
}

#[test]
fn test_qwen2_model_weights() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    let weights = model.weights();
    assert!(weights.contains_key("model.embed_tokens.weight"));
    assert_eq!(weights["model.embed_tokens.weight"].len(), 1000 * 64);
}

#[test]
fn test_qwen2_model_num_parameters() {
    let config = create_tiny_config();
    let model = Qwen2Model::new(&config);

    let num_params = model.num_parameters();
    // Should have embedding + layers + norm + lm_head
    assert!(num_params > 0);
    // Embedding alone is 1000 * 64 = 64000
    assert!(num_params >= 64000);
}

#[test]
fn test_qwen2_model_mut_accessors() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let _embed = model.embed_tokens_mut();
    let layer = model.layer_mut(0);
    assert!(layer.is_some());
    let bad_layer = model.layer_mut(100);
    assert!(bad_layer.is_none());
    let _norm = model.norm_mut();
    let _lm_head = model.lm_head_mut();
}

#[test]
fn test_elementwise_mul() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
    let c = elementwise_mul(&a, &b);
    assert_eq!(c.data(), &[2.0, 6.0, 12.0]);
}

#[test]
fn test_add_tensors() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);
    let c = add_tensors(&a, &b);
    assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_qwen2_decoder_layer_forward() {
    let config = create_tiny_config();
    let layer = Qwen2DecoderLayer::new(&config);
    let rope = RotaryPositionEmbedding::with_base(16, 128, 10000.0);

    let hidden = Tensor::ones(&[1, 5, 64]);
    let position_ids: Vec<usize> = (0..5).collect();

    let output = layer.forward(&hidden, &position_ids, &rope, None);
    assert_eq!(output.shape(), &[1, 5, 64]);
}

// =========================================================================
// Regression test: lm_head weight tying (GH-XXX)
// =========================================================================

#[test]
fn test_linear_placeholder_not_ready() {
    // GIVEN: a placeholder Linear layer (simulating uninitialized model)
    let linear = Linear::placeholder(64, 128);

    // THEN: it should NOT be ready for inference
    assert!(
        !linear.is_ready(),
        "Placeholder Linear should not be ready (weight_t is None)"
    );
}

#[test]
fn test_linear_after_set_weight_is_ready() {
    // GIVEN: a placeholder Linear layer
    let mut linear = Linear::placeholder(64, 128);
    assert!(!linear.is_ready(), "Precondition: placeholder not ready");

    // WHEN: set_weight is called
    let weight = Tensor::ones(&[128, 64]);
    linear.set_weight(weight);

    // THEN: it should be ready for inference
    assert!(
        linear.is_ready(),
        "Linear should be ready after set_weight (weight_t cached)"
    );
}

#[test]
fn test_uninitialized_model_lm_head_not_ready() {
    // GIVEN: an uninitialized model (using placeholder constructors)
    let config = create_tiny_config();
    let model = Qwen2Model::new_uninitialized(&config);

    // THEN: lm_head should NOT be ready (no weights loaded)
    assert!(
        !model.lm_head().is_ready(),
        "Uninitialized model's lm_head should not be ready"
    );
}

/// Regression test for weight tying bug in load_from_safetensors.
///
/// When SafeTensors file uses weight tying (lm_head shares weights with
/// embed_tokens), there is no "lm_head.weight" tensor. The loader must
/// fall back to using "model.embed_tokens.weight" for lm_head.
///
/// Without this fix, lm_head.weight_t remains None and forward() panics.
#[test]
#[ignore = "requires 0.5B model download - run with: cargo test -- --ignored"]
fn test_safetensors_weight_tying_lm_head_ready() {
    // This test verifies the INVARIANT: after load_from_safetensors,
    // ALL Linear layers must be ready (weight_t is Some).
    //
    // We test this by loading a real SafeTensors file if available,
    // or skip if not (integration test covers this).
    let safetensors_path = std::path::Path::new(
        "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
    );

    if !safetensors_path.exists() {
        // Skip if model not downloaded (CI may not have it)
        eprintln!("Skipping weight tying test: SafeTensors file not found");
        return;
    }

    // GIVEN: Qwen2 config matching the SafeTensors file
    let config = Qwen2Config::qwen2_0_5b_instruct();

    // WHEN: loading with new_uninitialized + load_from_safetensors
    let mut model = Qwen2Model::new_uninitialized(&config);
    let loaded = model
        .load_from_safetensors(safetensors_path)
        .expect("Should load SafeTensors");

    // THEN: lm_head must be ready (weight_t cached via weight tying)
    assert!(
        model.lm_head().is_ready(),
        "BUG: lm_head not ready after load_from_safetensors! \
         Weight tying fallback not implemented. Loaded {} tensors.",
        loaded
    );
}

// =========================================================================
// Section S: Popperian Falsification Tests for Qwen2 Native Inference
// =========================================================================
//
// These tests follow Karl Popper's criterion of demarcation: each test
// specifies conditions under which the claim would be PROVEN FALSE.
// A test that cannot fail is not scientific.
//
// Reference: Popper, K. (1959). The Logic of Scientific Discovery.
// =========================================================================

/// S1: Tokenizer loads from tokenizer.json
/// FALSIFICATION: Encoding "Hello" returns empty or panics
#[test]
#[ignore = "requires tokenizer download - run with: cargo test -- --ignored"]
fn s1_tokenizer_loads_from_json() {
    let tokenizer_path = std::path::Path::new("/home/noah/.cache/qwen2/tokenizer.json");

    if !tokenizer_path.exists() {
        eprintln!("SKIP S1: tokenizer.json not found at {:?}", tokenizer_path);
        eprintln!("Download: curl -L -o ~/.cache/qwen2/tokenizer.json \\");
        eprintln!("  https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer.json");
        return;
    }

    let json = std::fs::read_to_string(tokenizer_path).expect("read tokenizer.json");
    let tokenizer = crate::text::bpe::load_from_json(&json).expect("parse tokenizer.json");
    let tokens = tokenizer.encode("Hello");

    assert!(
        !tokens.is_empty(),
        "FALSIFIED S1: encode('Hello') returned empty. Tokenizer not functional."
    );

    println!("S1 PASSED: encode('Hello') -> {} tokens", tokens.len());
}

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

#[test]
fn test_load_from_apr_with_file() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = create_tiny_config();

    // Create a minimal APR file with just embed_tokens weight
    let mut writer = AprV2Writer::new(AprV2Metadata::default());

    // Add a small embed_tokens weight tensor
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    writer.add_f32_tensor(
        "embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Add norm weight
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f32_tensor("norm.weight", vec![config.hidden_size], &norm_data);

    // Add lm_head weight
    let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    writer.add_f32_tensor(
        "lm_head.weight",
        vec![config.vocab_size, config.hidden_size],
        &lm_head_data,
    );

    // Add layer 0 weights
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{layer_idx}");
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let head_dim = h / config.num_attention_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        // Attention weights
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );

        // MLP weights
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i],
            &vec![0.01; h * i],
        );

        // Layer norms
        writer.add_f32_tensor(
            &format!("{prefix}.input_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
    }

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("create temp file");
    let apr_bytes = writer.write().expect("serialize APR");
    temp_file.write_all(&apr_bytes).expect("write APR data");
    temp_file.flush().expect("flush");

    // Now test loading
    let mut model = Qwen2Model::new_uninitialized(&config);
    let result = model.load_from_apr(temp_file.path());

    assert!(result.is_ok(), "Should load APR file: {:?}", result.err());
    let loaded = result.unwrap();
    // Should load: embed + norm + lm_head + layers*(9 weights)
    let expected = 3 + config.num_layers * 9;
    assert_eq!(loaded, expected, "Should load {} tensors", expected);
}

#[test]
fn test_load_from_apr_weight_tying() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = create_tiny_config();

    // Create APR file WITHOUT lm_head.weight (triggers weight tying)
    let mut writer = AprV2Writer::new(AprV2Metadata::default());

    // Add embed_tokens weight
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    writer.add_f32_tensor(
        "embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Add norm weight
    writer.add_f32_tensor(
        "norm.weight",
        vec![config.hidden_size],
        &vec![1.0; config.hidden_size],
    );

    // NO lm_head.weight - should fall back to embed_tokens.weight

    // Add layer weights
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{layer_idx}");
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let head_dim = h / config.num_attention_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i],
            &vec![0.01; h * i],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.input_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
    }

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("create temp file");
    let apr_bytes = writer.write().expect("serialize APR");
    temp_file.write_all(&apr_bytes).expect("write APR data");
    temp_file.flush().expect("flush");

    // Test loading with weight tying
    let mut model = Qwen2Model::new_uninitialized(&config);
    let result = model.load_from_apr(temp_file.path());

    assert!(
        result.is_ok(),
        "Should load APR file with weight tying: {:?}",
        result.err()
    );
    // Should load embed + norm + lm_head(from embed) + layers*(9 weights)
    let loaded = result.unwrap();
    assert!(
        loaded >= 2 + config.num_layers * 9,
        "Should load tensors with weight tying"
    );
}

#[test]
fn test_from_apr_static_method() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = create_tiny_config();

    // Create minimal valid APR file
    let mut writer = AprV2Writer::new(AprV2Metadata::default());
    writer.add_f32_tensor(
        "embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &vec![0.01; config.vocab_size * config.hidden_size],
    );
    writer.add_f32_tensor(
        "norm.weight",
        vec![config.hidden_size],
        &vec![1.0; config.hidden_size],
    );

    // Add minimal layer weights
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layers.{layer_idx}");
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let head_dim = h / config.num_attention_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.q_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_dim, h],
            &vec![0.01; kv_dim * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.self_attn.o_proj.weight"),
            vec![h, h],
            &vec![0.01; h * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![i, h],
            &vec![0.01; i * h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![h, i],
            &vec![0.01; h * i],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.input_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
        writer.add_f32_tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![h],
            &vec![1.0; h],
        );
    }

    // Write to temp file
    let mut temp_file = NamedTempFile::new().expect("create temp file");
    let apr_bytes = writer.write().expect("serialize APR");
    temp_file.write_all(&apr_bytes).expect("write APR data");
    temp_file.flush().expect("flush");

    // Test from_apr static method
    let result = Qwen2Model::from_apr(&config, temp_file.path());
    assert!(result.is_ok(), "from_apr should work: {:?}", result.err());

    let model = result.unwrap();
    assert_eq!(model.num_layers(), config.num_layers);
}
