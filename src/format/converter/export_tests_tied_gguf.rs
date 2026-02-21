use super::*;

#[test]
fn test_export_to_gguf_tied_embeddings_creates_output_weight() {
    use crate::format::gguf::GgufReader;

    // When Q4K is requested and no lm_head exists, it should create one from embed_tokens
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 512 * 256], vec![512, 256]),
    );
    // No lm_head.weight — this is a tied-embedding model
    tensors.insert("model.norm.weight".to_string(), (vec![1.0; 256], vec![256]));

    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_tied.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    let quant = QuantizationType::Q4K;
    export_to_gguf(&tensors, &output, &input, Some(&quant)).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    let has_output = reader.tensors.iter().any(|t| t.name == "output.weight");
    // Should have synthesized output.weight from embedding
    assert!(
        has_output,
        "tied embedding model should get synthesized output.weight"
    );
}

#[test]
fn test_export_to_gguf_with_explicit_lm_head_no_duplicate() {
    use crate::format::gguf::GgufReader;

    // When lm_head exists, should NOT create a duplicate output.weight
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 512 * 256], vec![512, 256]),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.2; 512 * 256], vec![512, 256]),
    );

    let dir = tempfile::tempdir().expect("temp dir");
    let output = dir.path().join("test_no_dup.gguf");
    let input = dir.path().join("dummy.safetensors");
    std::fs::write(&input, b"dummy").expect("write dummy");

    let quant = QuantizationType::Q4K;
    export_to_gguf(&tensors, &output, &input, Some(&quant)).expect("export should succeed");

    let reader = GgufReader::from_file(&output).expect("should parse GGUF");
    let output_count = reader
        .tensors
        .iter()
        .filter(|t| t.name == "output.weight")
        .count();
    assert_eq!(output_count, 1, "should have exactly one output.weight");
}

// ========================================================================
// GH-258: resolve_gguf_config edge cases
// ========================================================================

#[test]
fn test_resolve_gguf_config_prefers_apr_over_inferred() {
    use crate::format::gguf::GgufModelConfig;
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = Some("llama".to_string());
    apr.hidden_size = Some(2048);
    apr.num_layers = Some(16);

    let mut inferred = GgufModelConfig::default();
    inferred.architecture = Some("qwen2".to_string());
    inferred.hidden_size = Some(4096);
    inferred.num_layers = Some(32);

    let cfg = resolve_gguf_config(Some(&apr), Some(&inferred));

    // APR metadata should take priority
    assert_eq!(cfg.arch, "llama");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_layers, 16);
}

#[test]
fn test_resolve_gguf_config_falls_back_to_inferred() {
    use crate::format::gguf::GgufModelConfig;

    let mut inferred = GgufModelConfig::default();
    inferred.architecture = Some("phi".to_string());
    inferred.hidden_size = Some(3072);

    let cfg = resolve_gguf_config(None, Some(&inferred));

    assert_eq!(cfg.arch, "phi");
    assert_eq!(cfg.hidden_size, 3072);
}

#[test]
fn test_resolve_gguf_config_head_dim_zero_heads() {
    let cfg = resolve_gguf_config(None, None);
    // Default is 32 heads, 4096 hidden → head_dim = 128
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_heads, 32);
}

#[test]
fn test_resolve_gguf_config_kv_heads_defaults_to_num_heads() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.num_heads = Some(16);
    // num_kv_heads not set → should default to num_heads
    apr.hidden_size = Some(2048);

    let cfg = resolve_gguf_config(Some(&apr), None);
    assert_eq!(cfg.num_kv_heads, 16, "kv_heads should default to num_heads");
}

// ========================================================================
// GH-258: build_gguf_config_metadata verification
// ========================================================================

#[test]
fn test_build_gguf_config_metadata_has_all_required_keys() {
    let cfg = resolve_gguf_config(None, None);
    let metadata = build_gguf_config_metadata(&cfg);

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();

    assert!(keys.contains(&"general.architecture"));
    assert!(keys.contains(&"general.name"));
    assert!(keys.contains(&"general.quantization_version"));
    assert!(keys.contains(&"general.file_type"));
    assert!(keys.contains(&"qwen2.context_length"));
    assert!(keys.contains(&"qwen2.embedding_length"));
    assert!(keys.contains(&"qwen2.block_count"));
    assert!(keys.contains(&"qwen2.feed_forward_length"));
    assert!(keys.contains(&"qwen2.attention.head_count"));
    assert!(keys.contains(&"qwen2.attention.head_count_kv"));
    assert!(keys.contains(&"qwen2.attention.layer_norm_rms_epsilon"));
    assert!(keys.contains(&"qwen2.rope.dimension_count"));
    assert!(keys.contains(&"qwen2.rope.freq_base"));
    assert!(keys.contains(&"qwen2.vocab_size"));
    assert_eq!(metadata.len(), 14, "should have exactly 14 metadata keys");
}

#[test]
fn test_build_gguf_config_metadata_uses_arch_prefix() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = Some("llama".to_string());

    let cfg = resolve_gguf_config(Some(&apr), None);
    let metadata = build_gguf_config_metadata(&cfg);

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();
    // Should use "llama" prefix, not "qwen2"
    assert!(keys.contains(&"llama.context_length"));
    assert!(keys.contains(&"llama.embedding_length"));
    assert!(!keys.contains(&"qwen2.context_length"));
}

// ========================================================================
// GH-258: build_tokenizer_gguf_metadata
// ========================================================================

#[test]
fn test_build_tokenizer_gguf_metadata_with_full_tokenizer() {
    use crate::format::gguf::GgufTokenizer;

    let tok = GgufTokenizer {
        vocabulary: vec!["hello".into(), "world".into()],
        merges: vec!["h e".into()],
        model_type: Some("gpt2".into()),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        architecture: None,
        model_name: None,
        token_type: vec![],
        padding_token_id: None,
        add_bos_token: None,
        chat_template: None,
        pre_type: None,
    };

    let metadata = build_tokenizer_gguf_metadata(&tok, "qwen2", "model");

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"tokenizer.ggml.model"));
    assert!(keys.contains(&"tokenizer.ggml.pre"));
    assert!(keys.contains(&"tokenizer.ggml.bos_token_id"));
    assert!(keys.contains(&"tokenizer.ggml.eos_token_id"));
    assert!(keys.contains(&"tokenizer.ggml.tokens"));
    assert!(keys.contains(&"tokenizer.ggml.merges"));
}

#[test]
fn test_build_tokenizer_gguf_metadata_without_optional_fields() {
    use crate::format::gguf::GgufTokenizer;

    let tok = GgufTokenizer {
        vocabulary: vec![],
        merges: vec![],
        model_type: None, // Should default to "gpt2"
        bos_token_id: None,
        eos_token_id: None,
        architecture: None,
        model_name: None,
        token_type: vec![],
        padding_token_id: None,
        add_bos_token: None,
        chat_template: None,
        pre_type: None,
    };

    let metadata = build_tokenizer_gguf_metadata(&tok, "llama", "model");

    // Should have model and pre, but no bos/eos/tokens/merges
    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"tokenizer.ggml.model"));
    assert!(keys.contains(&"tokenizer.ggml.pre"));
    assert!(!keys.contains(&"tokenizer.ggml.bos_token_id"));
    assert!(!keys.contains(&"tokenizer.ggml.eos_token_id"));
    assert!(!keys.contains(&"tokenizer.ggml.tokens"));
    assert!(!keys.contains(&"tokenizer.ggml.merges"));
}

// ========================================================================
// GH-258: build_tied_output_weight
// ========================================================================

#[test]
fn test_build_tied_output_weight_from_embed_tokens() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 512 * 256], vec![512, 256]),
    );

    let result = build_tied_output_weight(&tensors);
    assert!(
        result.is_some(),
        "should create output.weight from embed_tokens"
    );

    let tensor = result.expect("tied weight");
    assert_eq!(tensor.name, "output.weight");
    // Shape should be reversed: [ne0=256, ne1=512]
    assert_eq!(tensor.shape, vec![256, 512]);
}

#[test]
fn test_build_tied_output_weight_from_token_embedding() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embedding.weight".to_string(),
        (vec![0.1; 512 * 256], vec![512, 256]),
    );

    let result = build_tied_output_weight(&tensors);
    assert!(result.is_some(), "should find token_embedding pattern");
}

#[test]
fn test_build_tied_output_weight_none_without_embedding() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.1; 64 * 64], vec![64, 64]),
    );

    let result = build_tied_output_weight(&tensors);
    assert!(
        result.is_none(),
        "should return None when no embedding tensor found"
    );
}

#[test]
fn test_build_tied_output_weight_none_for_1d() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 64], vec![64]),
    );

    let result = build_tied_output_weight(&tensors);
    assert!(result.is_none(), "should return None for 1D embedding");
}

#[test]
fn test_build_tied_output_weight_none_for_small_data() {
    let mut tensors = BTreeMap::new();
    // data.len() < 256, should return None
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1; 16 * 8], vec![16, 8]),
    );

    let result = build_tied_output_weight(&tensors);
    assert!(
        result.is_none(),
        "should return None when data too small for Q4K"
    );
}

// ========================================================================
// GH-258: append_tokenizer_to_metadata
// ========================================================================

#[test]
fn test_append_tokenizer_prefers_json_over_apr_fallback() {
    use crate::format::gguf::GgufTokenizer;

    let tok = GgufTokenizer {
        vocabulary: vec!["a".into(), "b".into()],
        merges: vec![],
        model_type: Some("gpt2".into()),
        bos_token_id: Some(0),
        eos_token_id: Some(1),
        architecture: None,
        model_name: None,
        token_type: vec![],
        padding_token_id: None,
        add_bos_token: None,
        chat_template: None,
        pre_type: None,
    };

    let mut metadata = Vec::new();
    let dir = tempfile::tempdir().expect("temp dir");
    let input = dir.path().join("dummy.safetensors");

    // When tokenizer is Some, APR metadata should be ignored
    let mut apr = crate::format::v2::AprV2Metadata::new("test");
    apr.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["x", "y"]),
    );

    append_tokenizer_to_metadata(&mut metadata, Some(&tok), Some(&apr), "qwen2", "model", &input);

    // Should have tokenizer metadata from the GgufTokenizer, not APR
    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"tokenizer.ggml.model"));
    assert!(keys.contains(&"tokenizer.ggml.tokens"));
}

#[test]
fn test_append_tokenizer_uses_apr_fallback_when_no_json() {
    let mut metadata = Vec::new();
    let dir = tempfile::tempdir().expect("temp dir");
    let input = dir.path().join("dummy.safetensors");

    let mut apr = crate::format::v2::AprV2Metadata::new("test");
    apr.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["x", "y"]),
    );
    apr.custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));

    append_tokenizer_to_metadata(&mut metadata, None, Some(&apr), "qwen2", "model", &input);

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();
    assert!(
        keys.contains(&"tokenizer.ggml.model"),
        "should have model from APR fallback"
    );
    assert!(
        keys.contains(&"tokenizer.ggml.tokens"),
        "should have tokens from APR fallback"
    );
}

#[test]
fn test_append_tokenizer_no_metadata_when_neither_source() {
    let mut metadata = Vec::new();
    let dir = tempfile::tempdir().expect("temp dir");
    let input = dir.path().join("dummy.safetensors");

    append_tokenizer_to_metadata(&mut metadata, None, None, "qwen2", "model", &input);

    // Should have no tokenizer metadata entries
    let tok_keys: Vec<&str> = metadata
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| k.starts_with("tokenizer."))
        .collect();
    assert!(
        tok_keys.is_empty(),
        "no tokenizer sources → no tokenizer metadata"
    );
}

include!("export_tests_detect_quant.rs");
