//! Qwen2 Model Tests - Extreme TDD
//! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

/// Generate causal attention mask as a Tensor (for tests).
fn generate_causal_mask(size: usize) -> Tensor {
    let mut data = vec![0.0f32; size * size];
    generate_causal_mask_into(size, &mut data);
    Tensor::new(&data, &[size, size])
}

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
fn test_qwen2_model_forward_shape() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3, 4, 5];
    let position_ids: Vec<usize> = (0..5).collect();
    let logits = model.forward(&input_ids, &position_ids);

    assert_eq!(logits.shape(), &[1, 5, config.vocab_size]);
}

#[test]
fn test_qwen2_model_deterministic() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    model.eval();

    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();

    let logits1 = model.forward(&input_ids, &position_ids);
    let logits2 = model.forward(&input_ids, &position_ids);

    assert_eq!(logits1.data(), logits2.data());
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

#[test]
fn test_causal_mask() {
    let mask = generate_causal_mask(4);

    assert_eq!(mask.shape(), &[4, 4]);

    // Check upper triangle is -inf
    assert!(mask.data()[1].is_infinite()); // [0, 1]
    assert!(mask.data()[2].is_infinite()); // [0, 2]
    assert!(mask.data()[3].is_infinite()); // [0, 3]

    // Check diagonal and below is 0
    assert_eq!(mask.data()[0], 0.0); // [0, 0]
    assert_eq!(mask.data()[4], 0.0); // [1, 0]
    assert_eq!(mask.data()[5], 0.0); // [1, 1]
}

#[test]
fn test_argmax() {
    let slice = [1.0_f32, 5.0, 2.0, 3.0];
    assert_eq!(argmax(&slice), 1);

    let slice2 = [0.0_f32, -1.0, -2.0];
    assert_eq!(argmax(&slice2), 0);
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
fn test_qwen2_model_generate_greedy() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    model.eval();

    let prompt = vec![1u32, 2, 3];
    // Generate with temperature=0 (greedy)
    let output = model.generate(&prompt, 2, 0.0, 1.0);

    // Should have prompt + new tokens
    assert!(output.len() >= prompt.len());
    assert!(output.len() <= prompt.len() + 2);
    // Prompt should be preserved
    assert_eq!(&output[..3], &[1, 2, 3]);
}

#[test]
fn test_qwen2_model_generate_with_temperature() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    model.eval();

    let prompt = vec![1u32, 2, 3];
    // Generate with temperature > 0
    let output = model.generate(&prompt, 2, 0.8, 1.0);

    assert!(output.len() >= prompt.len());
}

#[test]
fn test_sample_with_temperature() {
    let logits = vec![10.0f32, 1.0, 0.0, -1.0];

    // With low temperature, should mostly pick index 0
    let mut count_0 = 0;
    for _ in 0..10 {
        let sample = sample_with_temperature(&logits, 0.1);
        if sample == 0 {
            count_0 += 1;
        }
    }
    // With temperature 0.1, should heavily favor index 0
    assert!(count_0 >= 5, "Expected mostly 0s, got {count_0}/10");
}

#[test]
fn test_sample_with_high_temperature() {
    let logits = vec![1.0f32, 1.0, 1.0, 1.0];

    // With uniform logits, all indices should be possible
    let mut seen = [false; 4];
    for _ in 0..100 {
        let sample = sample_with_temperature(&logits, 1.0) as usize;
        if sample < 4 {
            seen[sample] = true;
        }
    }
    // Should see at least some variety
    let variety = seen.iter().filter(|&&x| x).count();
    assert!(
        variety >= 2,
        "Expected variety, but only saw {variety} different values"
    );
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
fn test_argmax_empty() {
    let slice: [f32; 0] = [];
    // Should return 0 for empty slice
    assert_eq!(argmax(&slice), 0);
}

#[test]
fn test_argmax_single() {
    let slice = [42.0f32];
    assert_eq!(argmax(&slice), 0);
}

#[test]
fn test_causal_mask_size_1() {
    let mask = generate_causal_mask(1);
    assert_eq!(mask.shape(), &[1, 1]);
    assert_eq!(mask.data()[0], 0.0);
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

/// S11: Logits shape matches vocab
/// FALSIFICATION: Output shape != [1, seq_len, vocab_size]
#[test]
fn s11_logits_shape() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();
    let logits = model.forward(&input_ids, &position_ids);

    assert_eq!(
        logits.shape(),
        &[1, 3, config.vocab_size],
        "FALSIFIED S11: Logits shape {:?} != expected [1, 3, {}]",
        logits.shape(),
        config.vocab_size
    );

    println!("S11 PASSED: Logits shape matches vocab");
}

/// S12: Logits are finite (no NaN/Inf)
/// FALSIFICATION: Any NaN or Inf in output
#[test]
fn s12_logits_finite() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();
    let logits = model.forward(&input_ids, &position_ids);

    let has_nan = logits.data().iter().any(|x| x.is_nan());
    let has_inf = logits.data().iter().any(|x| x.is_infinite());

    assert!(!has_nan, "FALSIFIED S12: Logits contain NaN values");
    assert!(!has_inf, "FALSIFIED S12: Logits contain Inf values");

    println!("S12 PASSED: All logits are finite");
}

/// S14: Top-1 token is deterministic (temp=0)
/// FALSIFICATION: Same input produces different outputs
#[test]
fn s14_deterministic_generation() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3];

    // Generate twice with temperature=0 (greedy)
    let output1 = model.generate(&input_ids, 5, 0.0, 0.9);
    let output2 = model.generate(&input_ids, 5, 0.0, 0.9);

    assert_eq!(
        output1, output2,
        "FALSIFIED S14: Different outputs for same input with temp=0.\n  Run 1: {:?}\n  Run 2: {:?}",
        output1, output2
    );

    println!("S14 PASSED: Generation is deterministic at temp=0");
}

/// S20: Response length <= max_new_tokens
/// FALSIFICATION: Output exceeds requested length
#[test]
fn s20_length_control() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3];
    let max_new_tokens = 10;

    let output = model.generate(&input_ids, max_new_tokens, 0.7, 0.9);
    let new_tokens = output.len() - input_ids.len();

    assert!(
        new_tokens <= max_new_tokens,
        "FALSIFIED S20: Generated {} tokens > max {}",
        new_tokens,
        max_new_tokens
    );

    println!(
        "S20 PASSED: Generated {} <= {} tokens",
        new_tokens, max_new_tokens
    );
}

// =========================================================================
// Coverage Extension Tests
// =========================================================================

#[test]
fn test_forward_profiled() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    model.eval();

    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();

    // forward_profiled prints to stderr but returns same shape as forward
    let logits = model.forward_profiled(&input_ids, &position_ids);
    assert_eq!(logits.shape(), &[1, 3, config.vocab_size]);
}

#[test]
fn test_generate_profiled() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    model.eval();

    let prompt = vec![1u32, 2, 3];
    // generate_profiled should work the same as generate but with profiling
    let output = model.generate_profiled(&prompt, 2, 0.5);

    assert!(output.len() >= prompt.len());
    assert_eq!(&output[..3], &[1, 2, 3]);
}

#[test]
fn test_decoder_layer_forward_profiled() {
    let config = create_tiny_config();
    let layer = Qwen2DecoderLayer::new(&config);
    let rope = RotaryPositionEmbedding::with_base(16, 128, 10000.0);

    let hidden = Tensor::ones(&[1, 5, 64]);
    let position_ids: Vec<usize> = (0..5).collect();

    let (output, attn_time, mlp_time) = layer.forward_profiled(&hidden, &position_ids, &rope, None);

    assert_eq!(output.shape(), &[1, 5, 64]);
    // Verify timing values are returned (they're always valid Durations)
    let _ = attn_time.as_secs_f64();
    let _ = mlp_time.as_secs_f64();
}

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
fn test_forward_with_larger_sequence() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);
    model.eval();

    // Test with longer sequence to cover buffer reallocation paths
    let input_ids: Vec<u32> = (0..20).map(|i| i as u32).collect();
    let position_ids: Vec<usize> = (0..20).collect();

    let logits = model.forward(&input_ids, &position_ids);
    assert_eq!(logits.shape(), &[1, 20, config.vocab_size]);

    // Run again with same size (should reuse cached buffers)
    let logits2 = model.forward(&input_ids, &position_ids);
    assert_eq!(logits2.shape(), &[1, 20, config.vocab_size]);
}

#[test]
fn test_forward_different_sequence_lengths() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    // First call with length 5
    let input1 = vec![1u32, 2, 3, 4, 5];
    let pos1: Vec<usize> = (0..5).collect();
    let logits1 = model.forward(&input1, &pos1);
    assert_eq!(logits1.shape(), &[1, 5, config.vocab_size]);

    // Second call with length 10 (should trigger buffer reallocation)
    let input2: Vec<u32> = (0..10).map(|i| i as u32).collect();
    let pos2: Vec<usize> = (0..10).collect();
    let logits2 = model.forward(&input2, &pos2);
    assert_eq!(logits2.shape(), &[1, 10, config.vocab_size]);

    // Third call with length 3 (smaller, reuses existing buffer)
    let input3 = vec![1u32, 2, 3];
    let pos3: Vec<usize> = (0..3).collect();
    let logits3 = model.forward(&input3, &pos3);
    assert_eq!(logits3.shape(), &[1, 3, config.vocab_size]);
}

#[test]
fn test_causal_mask_into() {
    let mut data = vec![0.0f32; 9];
    generate_causal_mask_into(3, &mut data);

    // Check pattern: lower triangle + diagonal = 0, upper triangle = -inf
    assert_eq!(data[0], 0.0); // [0,0]
    assert!(data[1].is_infinite() && data[1] < 0.0); // [0,1]
    assert!(data[2].is_infinite() && data[2] < 0.0); // [0,2]
    assert_eq!(data[3], 0.0); // [1,0]
    assert_eq!(data[4], 0.0); // [1,1]
    assert!(data[5].is_infinite() && data[5] < 0.0); // [1,2]
    assert_eq!(data[6], 0.0); // [2,0]
    assert_eq!(data[7], 0.0); // [2,1]
    assert_eq!(data[8], 0.0); // [2,2]
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
fn test_generate_eos_token() {
    // Test that generation stops at EOS token
    // Note: This is hard to test without controlling model output,
    // but we can verify the EOS check logic is executed
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let prompt = vec![1u32];
    // Generate with temperature to get various tokens
    let output = model.generate(&prompt, 100, 0.0, 0.9);

    // Should not exceed max_new_tokens
    assert!(output.len() <= 101);
}

#[test]
fn test_sample_temperature_edge_cases() {
    // Test with very high temperature (uniform distribution)
    let logits = vec![0.0f32; 10];
    let sample = sample_with_temperature(&logits, 100.0);
    assert!(sample < 10, "Sample should be valid index");

    // Test with single logit
    let single = vec![1.0f32];
    let sample = sample_with_temperature(&single, 1.0);
    assert_eq!(sample, 0);
}

#[test]
fn test_argmax_with_nan() {
    // Test argmax behavior with NaN values
    let slice = [1.0f32, f32::NAN, 0.5];
    let idx = argmax(&slice);
    // NaN comparisons are tricky - implementation specific
    // Just verify it returns a valid index
    assert!(idx < 3);
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
fn test_generate_single_token() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let prompt = vec![1u32];
    // Generate just 1 token
    let output = model.generate(&prompt, 1, 0.0, 0.9);

    assert!(output.len() >= 1);
    assert!(output.len() <= 2);
}

#[test]
fn test_forward_profiled_multiple_calls() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    // Multiple profiled calls to ensure timing accumulation works
    let input_ids = vec![1u32, 2, 3, 4, 5];
    let position_ids: Vec<usize> = (0..5).collect();

    let logits1 = model.forward_profiled(&input_ids, &position_ids);
    let logits2 = model.forward_profiled(&input_ids, &position_ids);

    // Results should be deterministic
    assert_eq!(logits1.data(), logits2.data());
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

    // Create attention mask
    let mask = generate_causal_mask(5);

    let output = layer.forward(&hidden, &position_ids, &rope, Some(&mask));
    assert_eq!(output.shape(), &[1, 5, 64]);
}

#[test]
fn test_decoder_layer_profiled_with_mask() {
    let config = create_tiny_config();
    let layer = Qwen2DecoderLayer::new(&config);
    let rope = RotaryPositionEmbedding::with_base(16, 128, 10000.0);

    let hidden = Tensor::ones(&[1, 3, 64]);
    let position_ids: Vec<usize> = (0..3).collect();
    let mask = generate_causal_mask(3);

    let (output, attn_time, mlp_time) =
        layer.forward_profiled(&hidden, &position_ids, &rope, Some(&mask));

    assert_eq!(output.shape(), &[1, 3, 64]);
    // Times should be positive
    assert!(attn_time.as_nanos() > 0);
    assert!(mlp_time.as_nanos() > 0);
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
fn test_generate_with_zero_max_tokens() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let prompt = vec![1u32, 2, 3];
    // Generate 0 new tokens
    let output = model.generate(&prompt, 0, 0.0, 0.9);

    // Should return just the prompt
    assert_eq!(output, prompt);
}

#[test]
fn test_forward_single_token() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32];
    let position_ids = vec![0usize];
    let logits = model.forward(&input_ids, &position_ids);

    assert_eq!(logits.shape(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_forward_profiled_prints_output() {
    // Verify forward_profiled produces output to stderr
    // We can't easily capture stderr in unit tests, but we can verify
    // the function completes without panic
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2];
    let position_ids = vec![0usize, 1];

    // This should print profiling info to stderr
    let logits = model.forward_profiled(&input_ids, &position_ids);
    assert_eq!(logits.shape(), &[1, 2, config.vocab_size]);
}

#[test]
fn test_sample_with_very_low_temperature() {
    // Very low temperature should behave like greedy
    let logits = vec![10.0f32, 1.0, 0.0, -1.0];

    let mut count_0 = 0;
    for _ in 0..20 {
        let sample = sample_with_temperature(&logits, 0.01);
        if sample == 0 {
            count_0 += 1;
        }
    }
    // With temp 0.01, should almost always pick index 0
    assert!(count_0 >= 18, "Expected mostly 0s, got {count_0}/20");
}

#[test]
fn test_generate_profiled_with_greedy() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let prompt = vec![1u32];
    // Generate with temperature=0 (greedy) + profiling
    // Note: generate_profiled doesn't take top_p, uses temperature only
    let output = model.generate_profiled(&prompt, 3, 0.0);

    assert!(output.len() >= prompt.len());
    assert!(output.len() <= prompt.len() + 3);
}

#[test]
fn test_causal_mask_large() {
    // Test larger mask to verify loop coverage
    let mask = generate_causal_mask(10);
    assert_eq!(mask.shape(), &[10, 10]);

    // Spot check some values
    assert_eq!(mask.data()[0], 0.0); // [0,0]
    assert!(mask.data()[9].is_infinite()); // [0,9]
    assert_eq!(mask.data()[99], 0.0); // [9,9]
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
fn test_argmax_negative_values() {
    let slice = [-5.0f32, -1.0, -10.0, -2.0];
    // argmax should find -1.0 at index 1
    assert_eq!(argmax(&slice), 1);
}

#[test]
fn test_argmax_all_same() {
    let slice = [3.0f32, 3.0, 3.0];
    // All equal - returns some valid index (max_by behavior)
    let idx = argmax(&slice);
    assert!(idx < 3, "Should return valid index");
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
fn test_multiple_generate_calls() {
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    // Multiple generate calls should work independently
    let prompt1 = vec![1u32];
    let prompt2 = vec![2u32, 3];

    let output1 = model.generate(&prompt1, 2, 0.0, 0.9);
    let output2 = model.generate(&prompt2, 2, 0.0, 0.9);

    assert!(output1.len() >= 1);
    assert!(output2.len() >= 2);
    // Different prompts should potentially give different outputs
    // (though with random init, might be same)
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
fn test_generate_internal_profile_branch() {
    // Test the profile=true branch in generate_internal
    let config = create_tiny_config();
    let mut model = Qwen2Model::new(&config);

    let prompt = vec![1u32];
    // generate_profiled should trigger profile branch
    let output = model.generate_profiled(&prompt, 3, 0.5);

    assert!(output.len() >= 1);
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
