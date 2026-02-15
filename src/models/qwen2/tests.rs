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

include!("tests_part_02.rs");
include!("tests_part_03.rs");
