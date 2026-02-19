// ========================================================================
// GH-258: detect_apr_quantization
// ========================================================================

#[test]
fn test_detect_apr_quantization_f32_returns_none() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};

    let dir = tempfile::tempdir().expect("temp dir");
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );

    let path = dir.path().join("test.apr");
    let bytes = writer.write().expect("write");
    std::fs::write(&path, bytes).expect("write file");

    let result = detect_apr_quantization(&path);
    assert_eq!(result, None, "F32-only APR should return None");
}

#[test]
fn test_detect_apr_quantization_nonexistent_file() {
    let result = detect_apr_quantization(std::path::Path::new("/nonexistent/model.apr"));
    assert_eq!(result, None, "nonexistent file should return None");
}

// ========================================================================
// GH-277: Pre-tokenizer type resolution
// ========================================================================

#[test]
fn test_resolve_pre_tokenizer_type_gpt2() {
    assert_eq!(resolve_pre_tokenizer_type("gpt2", ""), "gpt-2");
    assert_eq!(resolve_pre_tokenizer_type("gpt2", "gpt2-124m"), "gpt-2");
}

#[test]
fn test_resolve_pre_tokenizer_type_llama_default() {
    assert_eq!(resolve_pre_tokenizer_type("llama", ""), "default");
    assert_eq!(resolve_pre_tokenizer_type("llama", "LLaMA-7B"), "default");
}

#[test]
fn test_resolve_pre_tokenizer_type_smollm_override() {
    // SmolLM uses "default" despite llama architecture
    assert_eq!(
        resolve_pre_tokenizer_type("llama", "SmolLM-135M"),
        "default"
    );
    assert_eq!(
        resolve_pre_tokenizer_type("llama", "HuggingFaceTB/SmolLM-135M"),
        "default"
    );
}

#[test]
fn test_resolve_pre_tokenizer_type_qwen2() {
    assert_eq!(resolve_pre_tokenizer_type("qwen2", ""), "qwen2");
    assert_eq!(
        resolve_pre_tokenizer_type("qwen2", "Qwen2-0.5B"),
        "qwen2"
    );
}

#[test]
fn test_resolve_pre_tokenizer_type_unknown_fallback() {
    assert_eq!(
        resolve_pre_tokenizer_type("some_new_arch", ""),
        "default"
    );
}

// ========================================================================
// GH-277: uses_rope
// ========================================================================

#[test]
fn test_uses_rope_gpt2_false() {
    assert!(!uses_rope("gpt2"), "GPT-2 uses learned pos embeddings, not RoPE");
}

#[test]
fn test_uses_rope_starcoder_false() {
    assert!(!uses_rope("starcoder"), "StarCoder uses learned pos embeddings");
}

#[test]
fn test_uses_rope_llama_true() {
    assert!(uses_rope("llama"), "LLaMA uses RoPE");
}

#[test]
fn test_uses_rope_qwen2_true() {
    assert!(uses_rope("qwen2"), "Qwen2 uses RoPE");
}

// ========================================================================
// GH-277: GPT-2 architecture-specific metadata keys
// ========================================================================

#[test]
fn test_build_gguf_config_metadata_gpt2_layer_norm_epsilon() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("gpt2-test");
    apr.architecture = Some("gpt2".to_string());
    apr.rms_norm_eps = Some(1e-5);

    let cfg = resolve_gguf_config(Some(&apr), None);
    let metadata = build_gguf_config_metadata(&cfg);

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();

    // GPT-2 should use layer_norm_epsilon, NOT layer_norm_rms_epsilon
    assert!(
        keys.contains(&"gpt2.attention.layer_norm_epsilon"),
        "GPT-2 should have layer_norm_epsilon"
    );
    assert!(
        !keys.contains(&"gpt2.attention.layer_norm_rms_epsilon"),
        "GPT-2 should NOT have layer_norm_rms_epsilon"
    );
}

#[test]
fn test_build_gguf_config_metadata_gpt2_no_rope_keys() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("gpt2-test");
    apr.architecture = Some("gpt2".to_string());

    let cfg = resolve_gguf_config(Some(&apr), None);
    let metadata = build_gguf_config_metadata(&cfg);

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();

    // GPT-2 should NOT have RoPE keys
    assert!(
        !keys.contains(&"gpt2.rope.dimension_count"),
        "GPT-2 should NOT have rope.dimension_count"
    );
    assert!(
        !keys.contains(&"gpt2.rope.freq_base"),
        "GPT-2 should NOT have rope.freq_base"
    );
}

#[test]
fn test_build_gguf_config_metadata_llama_has_rope_keys() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("llama-test");
    apr.architecture = Some("llama".to_string());

    let cfg = resolve_gguf_config(Some(&apr), None);
    let metadata = build_gguf_config_metadata(&cfg);

    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();

    // LLaMA should have RoPE keys and RMSNorm
    assert!(keys.contains(&"llama.rope.dimension_count"));
    assert!(keys.contains(&"llama.rope.freq_base"));
    assert!(keys.contains(&"llama.attention.layer_norm_rms_epsilon"));
    assert!(!keys.contains(&"llama.attention.layer_norm_epsilon"));
}

// ========================================================================
// GH-277: Pre-tokenizer type in tokenizer metadata
// ========================================================================

#[test]
fn test_build_tokenizer_gguf_metadata_pre_type_gpt2() {
    use crate::format::gguf::GgufTokenizer;

    let tok = GgufTokenizer {
        model_type: Some("gpt2".into()),
        pre_type: None,
        ..Default::default()
    };

    let metadata = build_tokenizer_gguf_metadata(&tok, "gpt2", "gpt2-model");

    let pre_val = metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.pre")
        .map(|(_, v)| match v {
            crate::format::gguf::GgufValue::String(s) => s.as_str(),
            _ => "",
        });
    assert_eq!(pre_val, Some("gpt-2"), "GPT-2 pre-tokenizer should be 'gpt-2' with hyphen");
}

#[test]
fn test_build_tokenizer_gguf_metadata_pre_type_llama_default() {
    use crate::format::gguf::GgufTokenizer;

    let tok = GgufTokenizer {
        model_type: Some("gpt2".into()),
        pre_type: None,
        ..Default::default()
    };

    let metadata = build_tokenizer_gguf_metadata(&tok, "llama", "model");

    let pre_val = metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.pre")
        .map(|(_, v)| match v {
            crate::format::gguf::GgufValue::String(s) => s.as_str(),
            _ => "",
        });
    assert_eq!(pre_val, Some("default"), "LLaMA pre-tokenizer should be 'default'");
}

#[test]
fn test_build_tokenizer_gguf_metadata_preserves_roundtrip_pre_type() {
    use crate::format::gguf::GgufTokenizer;

    let tok = GgufTokenizer {
        model_type: Some("gpt2".into()),
        pre_type: Some("custom-pre".to_string()),
        ..Default::default()
    };

    let metadata = build_tokenizer_gguf_metadata(&tok, "llama", "model");

    let pre_val = metadata
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.pre")
        .map(|(_, v)| match v {
            crate::format::gguf::GgufValue::String(s) => s.as_str(),
            _ => "",
        });
    assert_eq!(pre_val, Some("custom-pre"), "Round-trip pre_type should be preserved");
}

// ========================================================================
// GH-277/GH-279: Token table dedup validation
// ========================================================================

#[test]
fn test_validated_metadata_deduplicates_tokens() {
    use crate::format::gguf::GgufValue;

    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec![
                "hello".into(),
                "world".into(),
                "hello".into(), // duplicate — should be deduped to [PAD2]
            ]),
        ),
    ];

    // GH-277: validate() now auto-dedupes using [PAD{id}] format (like HuggingFace)
    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_ok(), "should auto-dedup duplicate tokens: {result:?}");
    let validated = result.expect("validated");
    let tokens = validated.as_slice().iter().find_map(|(k, v)| {
        if k == "tokenizer.ggml.tokens" {
            if let GgufValue::ArrayString(t) = v {
                Some(t.clone())
            } else {
                None
            }
        } else {
            None
        }
    });
    let tokens = tokens.expect("should have token array");
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0], "hello");
    assert_eq!(tokens[1], "world");
    assert_eq!(tokens[2], "[PAD2]", "duplicate should get [PAD<id>] name");
}

#[test]
fn test_validated_metadata_accepts_unique_tokens() {
    use crate::format::gguf::GgufValue;

    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(vec!["hello".into(), "world".into(), "foo".into()]),
        ),
    ];

    let result = ValidatedGgufMetadata::validate(metadata);
    assert!(result.is_ok(), "unique tokens should be accepted");
}

// ========================================================================
// GH-277: build_gguf_arch_metadata (raw passthrough path)
// ========================================================================

#[test]
fn test_build_gguf_arch_metadata_gpt2_keys() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("gpt2-test");
    apr.architecture = Some("gpt2".to_string());
    apr.rms_norm_eps = Some(1e-5);

    let metadata = build_gguf_arch_metadata(&apr);
    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();

    assert!(keys.contains(&"gpt2.attention.layer_norm_epsilon"));
    assert!(!keys.contains(&"gpt2.attention.layer_norm_rms_epsilon"));
    assert!(!keys.contains(&"gpt2.rope.dimension_count"));
    assert!(!keys.contains(&"gpt2.rope.freq_base"));
}

#[test]
fn test_build_gguf_arch_metadata_qwen2_keys() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("qwen2-test");
    apr.architecture = Some("qwen2".to_string());

    let metadata = build_gguf_arch_metadata(&apr);
    let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();

    assert!(keys.contains(&"qwen2.attention.layer_norm_rms_epsilon"));
    assert!(!keys.contains(&"qwen2.attention.layer_norm_epsilon"));
    assert!(keys.contains(&"qwen2.rope.dimension_count"));
    assert!(keys.contains(&"qwen2.rope.freq_base"));
}
