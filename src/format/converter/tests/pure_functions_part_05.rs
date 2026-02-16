
// ============================================================================
// should_skip_quantization tests (GH-237)
// ============================================================================

#[test]
fn test_should_skip_quantization_embedding_gh219() {
    // Embeddings should always be skipped
    assert!(should_skip_quantization("model.embed_tokens.weight", 100_000));
    assert!(should_skip_quantization("token_embd.weight", 100_000));
    assert!(should_skip_quantization("wte.weight", 100_000));
    assert!(should_skip_quantization("wpe.weight", 100_000));
    assert!(should_skip_quantization("word_embeddings.weight", 100_000));
    assert!(should_skip_quantization("position_embedding.weight", 100_000));
}

#[test]
fn test_should_skip_quantization_lm_head_gh219() {
    // lm_head should be skipped (GH-234)
    assert!(should_skip_quantization("lm_head.weight", 100_000));
    assert!(should_skip_quantization("output.weight", 100_000));
}

#[test]
fn test_should_skip_quantization_bias_gh219() {
    assert!(should_skip_quantization("model.layers.0.self_attn.q_proj.bias", 4096));
    assert!(should_skip_quantization("model.layers.0.mlp.gate_proj.bias", 4096));
}

#[test]
fn test_should_skip_quantization_norm_gh219() {
    assert!(should_skip_quantization("model.layers.0.input_layernorm.weight", 4096));
    assert!(should_skip_quantization("model.layers.0.post_attention_layernorm.weight", 4096));
    assert!(should_skip_quantization("blk.0.attn_norm.weight", 4096));
    assert!(should_skip_quantization("layer_norm.weight", 4096));
    assert!(should_skip_quantization("model.norm.weight", 4096));
}

#[test]
fn test_should_skip_quantization_small_tensor_gh219() {
    // Tensors with <1024 elements should be skipped
    assert!(should_skip_quantization("model.layers.0.self_attn.q_proj.weight", 1023));
    assert!(should_skip_quantization("some_weight", 100));
    assert!(should_skip_quantization("small_tensor", 0));
}

#[test]
fn test_should_skip_quantization_normal_weight_gh219() {
    // Normal weight tensors >= 1024 elements should NOT be skipped
    assert!(!should_skip_quantization(
        "model.layers.0.self_attn.q_proj.weight",
        4096 * 4096
    ));
    assert!(!should_skip_quantization(
        "model.layers.0.mlp.gate_proj.weight",
        4096 * 11008
    ));
    assert!(!should_skip_quantization(
        "model.layers.0.self_attn.o_proj.weight",
        1024
    ));
}

// ============================================================================
// AprConverter builder pattern tests
// ============================================================================

#[test]
fn test_apr_converter_new_defaults_gh219() {
    let converter = AprConverter::new();
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Auto"));
    assert!(debug.contains("Strict"));
}

#[test]
fn test_apr_converter_default_gh219() {
    let converter = AprConverter::default();
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Auto"));
}

#[test]
fn test_apr_converter_architecture_gh219() {
    let converter = AprConverter::new().architecture(Architecture::Llama);
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Llama"));
}

#[test]
fn test_apr_converter_validate_gh219() {
    let converter = AprConverter::new().validate(ValidationConfig::Basic);
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Basic"));
}

#[test]
fn test_apr_converter_quantize_gh219() {
    let converter = AprConverter::new().quantize(QuantizationType::Int8);
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Int8"));
}

#[test]
fn test_apr_converter_compress_gh219() {
    let converter = AprConverter::new().compress(Compression::Lz4);
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Lz4"));
}

#[test]
fn test_apr_converter_source_local_gh219() {
    let converter = AprConverter::new().source("/tmp/model.safetensors");
    assert!(converter.is_ok());
}

#[test]
fn test_apr_converter_source_hf_gh219() {
    let converter = AprConverter::new().source("hf://Qwen/Qwen2-0.5B");
    assert!(converter.is_ok());
}

#[test]
fn test_apr_converter_convert_no_source_gh219() {
    let result = AprConverter::new().convert();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("No source specified"));
}

#[test]
fn test_apr_converter_convert_local_not_implemented_gh219() {
    let result = AprConverter::new()
        .source("/tmp/nonexistent.safetensors")
        .unwrap()
        .convert();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("GH-80"));
}

#[test]
fn test_apr_converter_full_builder_chain_gh219() {
    let result = AprConverter::new()
        .architecture(Architecture::Qwen2)
        .validate(ValidationConfig::None)
        .quantize(QuantizationType::Q4K)
        .compress(Compression::ZstdDefault)
        .source("/tmp/model.safetensors")
        .unwrap()
        .convert();
    assert!(result.is_err()); // Not yet implemented
}

// ============================================================================
// ConvertOptions tests
// ============================================================================

#[test]
fn test_convert_options_partial_eq_gh219() {
    let opts1 = ConvertOptions {
        quantize: Some(QuantizationType::Int8),
        compress: None,
        validate: true,
    };
    let opts2 = opts1.clone();
    assert_eq!(opts1.quantize, opts2.quantize);
    assert_eq!(opts1.validate, opts2.validate);
}

// ============================================================================
// infer_q4k_single_tensor edge cases
// ============================================================================

#[test]
fn test_infer_q4k_single_tensor_norm_weight_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.0.input_layernorm.weight",
        &[4096],
    );
    assert_eq!(cfg.hidden_size, Some(4096));
}

#[test]
fn test_infer_q4k_single_tensor_embedding_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    infer_q4k_single_tensor(&mut cfg, "model.embed_tokens.weight", &[32000, 4096]);
    assert_eq!(cfg.vocab_size, Some(32000));
    assert_eq!(cfg.hidden_size, Some(4096));
}

#[test]
fn test_infer_q4k_single_tensor_embedding_hidden_already_set_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: Some(4096),
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    // When hidden_size already set, embedding should not overwrite it
    infer_q4k_single_tensor(&mut cfg, "model.embed_tokens.weight", &[32000, 8192]);
    assert_eq!(cfg.vocab_size, Some(32000));
    assert_eq!(cfg.hidden_size, Some(4096)); // Not overwritten
}

#[test]
fn test_infer_q4k_single_tensor_layer_count_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.0.self_attn.q_proj.weight",
        &[4096, 4096],
    );
    assert_eq!(cfg.num_layers, Some(1));

    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.5.self_attn.q_proj.weight",
        &[4096, 4096],
    );
    assert_eq!(cfg.num_layers, Some(6)); // max(1, 6)

    // Earlier layer shouldn't reduce count
    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.2.self_attn.q_proj.weight",
        &[4096, 4096],
    );
    assert_eq!(cfg.num_layers, Some(6)); // Still 6
}

#[test]
fn test_infer_q4k_single_tensor_kv_and_q_heads_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: Some(4096),
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    // k_proj: kv_dim=512, hidden=4096 → num_kv_heads = 512/64 = 8
    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.0.self_attn.k_proj.weight",
        &[512, 4096],
    );
    assert_eq!(cfg.num_kv_heads, Some(8));

    // q_proj: q_dim=4096 → num_heads = 4096/64 = 64
    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.0.self_attn.q_proj.weight",
        &[4096, 4096],
    );
    assert_eq!(cfg.num_heads, Some(64));
}

#[test]
fn test_infer_q4k_single_tensor_intermediate_size_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    infer_q4k_single_tensor(
        &mut cfg,
        "model.layers.0.mlp.gate_proj.weight",
        &[11008, 4096],
    );
    assert_eq!(cfg.intermediate_size, Some(11008));
}

#[test]
fn test_infer_q4k_single_tensor_unrelated_name_gh219() {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    // Unrelated tensor name should not change anything
    infer_q4k_single_tensor(&mut cfg, "some_random_tensor", &[100, 200]);
    assert!(cfg.hidden_size.is_none());
    assert!(cfg.vocab_size.is_none());
    assert!(cfg.num_layers.is_none());
}

// ============================================================================
// build_q4k_metadata tests
// ============================================================================

#[test]
fn test_build_q4k_metadata_basic_gh219() {
    let cfg = InferredQ4kConfig {
        hidden_size: Some(4096),
        vocab_size: Some(32000),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        intermediate_size: Some(11008),
    };

    let metadata = build_q4k_metadata(&cfg, 7_000_000_000);
    assert_eq!(metadata.model_type, "qwen2");
    assert_eq!(metadata.hidden_size, Some(4096));
    assert_eq!(metadata.vocab_size, Some(32000));
    assert_eq!(metadata.num_layers, Some(32));
    assert_eq!(metadata.num_heads, Some(32));
    assert_eq!(metadata.num_kv_heads, Some(8));
    assert_eq!(metadata.intermediate_size, Some(11008));
    assert_eq!(metadata.param_count, 7_000_000_000);
    assert_eq!(metadata.max_position_embeddings, Some(32768));
    assert_eq!(metadata.rope_theta, Some(1000000.0));
    assert_eq!(metadata.rope_type, Some(2)); // NEOX style
    assert_eq!(metadata.rms_norm_eps, Some(1e-6));
    assert!(metadata.quantization.is_some());
    let quant = metadata.quantization.unwrap();
    assert_eq!(quant.quant_type, "q4_k");
    assert_eq!(quant.bits, 4);
    assert_eq!(quant.block_size, Some(256));
}

#[test]
fn test_build_q4k_metadata_empty_config_gh219() {
    let cfg = InferredQ4kConfig {
        hidden_size: None,
        vocab_size: None,
        num_layers: None,
        num_heads: None,
        num_kv_heads: None,
        intermediate_size: None,
    };

    let metadata = build_q4k_metadata(&cfg, 0);
    assert!(metadata.hidden_size.is_none());
    assert!(metadata.vocab_size.is_none());
    assert_eq!(metadata.param_count, 0);
    assert_eq!(metadata.architecture, Some("qwen2".to_string()));
}
