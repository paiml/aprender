
// ============================================================================
// infer_q4k_config: Coverage tests (supports build_q4k_metadata)
// ============================================================================

#[test]
fn test_infer_q4k_config_from_tensors() {
    let mut tensors = BTreeMap::new();

    // Embedding: [vocab_size=256, hidden_size=128]
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1f32; 256 * 128], vec![256, 128]),
    );

    // Norm: [hidden_size=128]
    tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        (vec![1.0f32; 128], vec![128]),
    );

    // Layer 0 q_proj: [hidden_size=128, hidden_size=128]
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.1f32; 128 * 128], vec![128, 128]),
    );

    // Layer 0 k_proj: [kv_dim=64, hidden_size=128]
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![0.1f32; 64 * 128], vec![64, 128]),
    );

    // Layer 1 (to test layer counting)
    tensors.insert(
        "model.layers.1.input_layernorm.weight".to_string(),
        (vec![1.0f32; 128], vec![128]),
    );

    // Gate proj: [intermediate=512, hidden=128]
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![0.1f32; 512 * 128], vec![512, 128]),
    );

    let cfg = infer_q4k_config(&tensors);

    assert_eq!(cfg.hidden_size, Some(128));
    assert_eq!(cfg.vocab_size, Some(256));
    assert_eq!(cfg.num_layers, Some(2)); // layers 0 and 1
    assert_eq!(cfg.num_kv_heads, Some(1)); // kv_dim=64, head_dim assumed 64 -> 1 kv_head
    assert_eq!(cfg.num_heads, Some(2)); // q_dim=128, head_dim assumed 64 -> 2 heads
    assert_eq!(cfg.intermediate_size, Some(512));
}

#[test]
fn test_infer_q4k_config_empty_tensors() {
    let tensors = BTreeMap::new();
    let cfg = infer_q4k_config(&tensors);

    assert_eq!(cfg.hidden_size, None);
    assert_eq!(cfg.vocab_size, None);
    assert_eq!(cfg.num_layers, None);
    assert_eq!(cfg.num_kv_heads, None);
    assert_eq!(cfg.num_heads, None);
    assert_eq!(cfg.intermediate_size, None);
}

#[test]
fn test_infer_q4k_config_hidden_from_embedding_not_norm() {
    let mut tensors = BTreeMap::new();
    // Only embedding, no norm
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1f32; 32 * 64], vec![32, 64]),
    );

    let cfg = infer_q4k_config(&tensors);
    assert_eq!(cfg.hidden_size, Some(64));
    assert_eq!(cfg.vocab_size, Some(32));
}

// ============================================================================
// should_quantize_tensor: Coverage tests
// ============================================================================

#[test]
fn test_should_quantize_tensor_large_weight() {
    assert!(should_quantize_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &[128, 128],
        16384
    ));
}

#[test]
fn test_should_quantize_tensor_too_small() {
    assert!(!should_quantize_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &[4, 4],
        16
    ));
}

#[test]
fn test_should_quantize_tensor_bias_excluded() {
    assert!(!should_quantize_tensor(
        "model.layers.0.self_attn.q_proj.bias",
        &[128, 128],
        16384
    ));
}

#[test]
fn test_should_quantize_tensor_norm_excluded() {
    assert!(!should_quantize_tensor(
        "model.layers.0.input_layernorm.weight",
        &[128],
        128
    ));
}

#[test]
fn test_should_quantize_tensor_scale_excluded() {
    assert!(!should_quantize_tensor(
        "model.layers.0.scale",
        &[128, 128],
        16384
    ));
}

#[test]
fn test_should_quantize_tensor_embed_excluded() {
    assert!(!should_quantize_tensor(
        "model.embed_tokens.weight",
        &[256, 128],
        32768
    ));
}

#[test]
fn test_should_quantize_tensor_1d_excluded() {
    // 1D tensor with enough elements but wrong shape
    assert!(!should_quantize_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        &[16384],
        16384
    ));
}

// ============================================================================
// save_model_tensors_with_gguf_config_and_tokenizer: Coverage tests (impact 9.7)
// ============================================================================

#[test]
fn test_save_model_tensors_with_gguf_config_and_tokenizer_basic() {
    use crate::format::gguf::api::{GgufModelConfig, GgufTokenizer};
    use crate::format::v2::AprV2Reader;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model.apr");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1f32; 32], vec![8, 4]),
    );
    tensors.insert("model.norm.weight".to_string(), (vec![1.0f32; 4], vec![4]));

    let config = GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        hidden_size: Some(4),
        num_layers: Some(1),
        num_heads: Some(2),
        num_kv_heads: Some(2),
        vocab_size: Some(8),
        intermediate_size: Some(16),
        max_position_embeddings: Some(2048),
        rope_theta: Some(10000.0),
        rms_norm_eps: Some(1e-5),
        rope_type: Some(2),
    };

    let tokenizer = GgufTokenizer {
        vocabulary: vec!["hello".to_string(), "world".to_string(), "test".to_string()],
        merges: vec!["he llo".to_string(), "wor ld".to_string()],
        model_type: Some("gpt2".to_string()),
        bos_token_id: Some(0),
        eos_token_id: Some(1),
        ..Default::default()
    };

    let result = save_model_tensors_with_gguf_config_and_tokenizer(
        &tensors,
        &output,
        None,
        &config,
        Some(&tokenizer),
        None,
    );
    assert!(result.is_ok(), "save should succeed: {:?}", result.err());
    assert!(output.exists());

    // Read back and verify metadata
    let data = std::fs::read(&output).expect("read output");
    let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
    let meta = reader.metadata();

    assert_eq!(meta.architecture, Some("qwen2".to_string()));
    assert_eq!(meta.hidden_size, Some(4));
    assert_eq!(meta.num_layers, Some(1));
    assert_eq!(meta.num_heads, Some(2));
    assert_eq!(meta.num_kv_heads, Some(2));
    assert_eq!(meta.vocab_size, Some(8));
    assert_eq!(meta.intermediate_size, Some(16));
    assert_eq!(meta.max_position_embeddings, Some(2048));
    assert_eq!(meta.rope_theta, Some(10000.0));
    assert_eq!(meta.rms_norm_eps, Some(1e-5));
    assert_eq!(meta.rope_type, Some(2));
    assert_eq!(meta.original_format, Some("gguf".to_string()));

    // Verify tokenizer was embedded in custom metadata
    assert!(meta.custom.contains_key("tokenizer.vocabulary"));
    assert!(meta.custom.contains_key("tokenizer.vocab_size"));
    assert!(meta.custom.contains_key("tokenizer.model"));
    assert!(meta.custom.contains_key("tokenizer.bos_token_id"));
    assert!(meta.custom.contains_key("tokenizer.eos_token_id"));
    assert!(meta.custom.contains_key("tokenizer.merges"));

    // Verify vocabulary content
    let vocab = meta.custom.get("tokenizer.vocabulary").unwrap();
    let vocab_arr = vocab.as_array().unwrap();
    assert_eq!(vocab_arr.len(), 3);
    assert_eq!(vocab_arr[0].as_str().unwrap(), "hello");

    // Verify merges content
    let merges = meta.custom.get("tokenizer.merges").unwrap();
    let merges_arr = merges.as_array().unwrap();
    assert_eq!(merges_arr.len(), 2);
}

#[test]
fn test_save_model_tensors_with_gguf_config_no_tokenizer() {
    use crate::format::gguf::api::GgufModelConfig;
    use crate::format::v2::AprV2Reader;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model.apr");

    let mut tensors = BTreeMap::new();
    tensors.insert("t1".to_string(), (vec![1.0f32; 4], vec![4]));

    let config = GgufModelConfig {
        architecture: Some("llama".to_string()),
        hidden_size: Some(256),
        ..Default::default()
    };

    let result = save_model_tensors_with_gguf_config_and_tokenizer(
        &tensors, &output, None, &config, None, // no tokenizer
        None,
    );
    assert!(result.is_ok());

    let data = std::fs::read(&output).expect("read output");
    let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
    let meta = reader.metadata();

    assert_eq!(meta.architecture, Some("llama".to_string()));
    assert_eq!(meta.hidden_size, Some(256));
    // No tokenizer metadata
    assert!(!meta.custom.contains_key("tokenizer.vocabulary"));
    assert!(!meta.custom.contains_key("tokenizer.merges"));
}

#[test]
fn test_save_model_tensors_with_gguf_config_empty_tokenizer_vocab() {
    use crate::format::gguf::api::{GgufModelConfig, GgufTokenizer};
    use crate::format::v2::AprV2Reader;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model.apr");

    let mut tensors = BTreeMap::new();
    tensors.insert("t1".to_string(), (vec![1.0f32; 4], vec![4]));

    let config = GgufModelConfig::default();

    // Tokenizer with empty vocabulary
    let tokenizer = GgufTokenizer {
        vocabulary: vec![],
        merges: vec![],
        model_type: Some("gpt2".to_string()),
        ..Default::default()
    };

    let result = save_model_tensors_with_gguf_config_and_tokenizer(
        &tensors,
        &output,
        None,
        &config,
        Some(&tokenizer),
        None,
    );
    assert!(result.is_ok());

    let data = std::fs::read(&output).expect("read output");
    let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
    let meta = reader.metadata();

    // Empty vocab should NOT be stored
    assert!(!meta.custom.contains_key("tokenizer.vocabulary"));
    // But model_type should still be stored
    assert!(meta.custom.contains_key("tokenizer.model"));
}

#[test]
fn test_save_model_tensors_with_gguf_config_default_architecture() {
    use crate::format::gguf::api::GgufModelConfig;
    use crate::format::v2::AprV2Reader;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model.apr");

    let mut tensors = BTreeMap::new();
    tensors.insert("t1".to_string(), (vec![1.0f32; 4], vec![4]));

    // No architecture specified -> defaults to "qwen2"
    let config = GgufModelConfig::default();

    let result = save_model_tensors_with_gguf_config_and_tokenizer(
        &tensors, &output, None, &config, None, None,
    );
    assert!(result.is_ok());

    let data = std::fs::read(&output).expect("read output");
    let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
    let meta = reader.metadata();

    assert_eq!(meta.model_type, "qwen2");
}

#[test]
fn test_save_model_tensors_with_gguf_config_with_quantization() {
    use crate::format::gguf::api::GgufModelConfig;
    use crate::format::v2::AprV2Reader;
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model.apr");

    let mut tensors = BTreeMap::new();
    // Need a tensor large enough to be quantized (>=256 elements, 2D, not bias/norm/embed)
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.1f32; 512], vec![32, 16]),
    );
    // This one should NOT be quantized (1D norm)
    tensors.insert(
        "model.norm.weight".to_string(),
        (vec![1.0f32; 16], vec![16]),
    );

    let config = GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        ..Default::default()
    };

    let result = save_model_tensors_with_gguf_config_and_tokenizer(
        &tensors,
        &output,
        None,
        &config,
        None,
        Some(QuantizationType::Fp16),
    );
    assert!(result.is_ok());

    let data = std::fs::read(&output).expect("read output");
    let reader = AprV2Reader::from_bytes(&data).expect("parse APR");

    // Both tensors should exist
    assert_eq!(reader.tensor_names().len(), 2);
}
